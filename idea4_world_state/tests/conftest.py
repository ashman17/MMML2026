"""Mock vLLM HTTP server for WP4 and WP5.

Fixtures only: nothing here is autouse, and importing this module does not
bind a port or contact a model. Request ``vllm_server`` to get a localhost
OpenAI-compatible server with a scripted ``/v1/chat/completions`` queue.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    block_on_close = False


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError, socket.timeout)):
            return
        super().handle_error(request, client_address)

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        try:
            self._handle()
        except (ConnectionError, TimeoutError, socket.timeout):
            return

    def _handle(self):
        path = urlsplit(self.path).path.rstrip("/") or "/"
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        parsed = None
        if body:
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None
        scripted = self.server.scripted
        scripted.record(self.command, path, self.headers.get("Authorization"), parsed)
        if self.command == "GET" and path == "/v1/models":
            self._send_json(200, scripted.models_payload())
            return
        if self.command == "POST" and path == "/v1/chat/completions":
            self._complete(scripted.next_behavior(), parsed)
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def _complete(self, behavior, request_body):
        name = behavior.get("behavior")
        if name == "drop":
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        if name == "sleep":
            self.server.scripted.wait(float(behavior.get("seconds", 5.0)))
            name = "ok"
            behavior = {**behavior, "behavior": "ok"}
        if name == "http_500":
            self._send_error_body(500, behavior.get("body"))
            return
        if name == "http_429":
            self._send_error_body(429, behavior.get("body"))
            return
        if name not in {"ok", "length"}:
            self._send_error_body(500, "no scripted behavior")
            return
        requested = request_body.get("model") if isinstance(request_body, dict) else None
        model = behavior.get("model", requested or "qwen3-vl-8b")
        content = behavior.get("content")
        if content is None:
            content = "``A``" if name == "ok" else "``"
        prompt_tokens = int(behavior.get("prompt_tokens", 11))
        completion_tokens = int(behavior.get("completion_tokens", 5))
        payload = {
            "id": behavior.get("response_id") or self.server.scripted.next_response_id(),
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length" if name == "length" else "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._send_json(200, payload)

    def _send_error_body(self, status, body):
        if body is None:
            encoded = json.dumps(
                {"error": {"message": f"scripted HTTP {status}", "type": "server_error"}}
            ).encode("utf-8")
        elif isinstance(body, bytes):
            encoded = body
        else:
            encoded = str(body).encode("utf-8")
        self._send_bytes(status, encoded)

    def _send_json(self, status, payload):
        self._send_bytes(status, json.dumps(payload).encode("utf-8"))

    def _send_bytes(self, status, data: bytes):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(data)


class ScriptedVLLMServer:
    """Local ``/v1/models`` and ``/v1/chat/completions`` with a behavior queue.

    Chat behaviors, consumed one per POST:

    - ``"ok"``: HTTP 200, ``finish_reason`` ``stop``, content `` ``A`` ``
    - ``"length"``: HTTP 200, ``finish_reason`` ``length``
    - ``"http_500"`` and ``"http_429"``: that status, body does not echo the request
    - ``"drop"``: read the body, record it, then close the connection
    - ``"sleep"``: record the body, then wait (default 5s, or ``{"behavior": "sleep", "seconds": N}``)

    A dict can override ``content``, ``body``, ``seconds``, ``prompt_tokens``,
    ``completion_tokens``, ``response_id``, and ``model``. ``models`` is the id
    list returned by GET ``/v1/models`` (default ``["qwen3-vl-8b"]``).
    ``requests`` records method, path, parsed JSON body, and the Authorization
    header. ``chat_requests`` is the completion subset.
    """

    def __init__(self):
        self.models = ["qwen3-vl-8b"]
        self._behaviors = []
        self._requests = []
        self._response_ids = 0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._httpd = None
        self._thread = None
        self.base_url = None
        self.port = None

    def start(self) -> None:
        httpd = _Server(("127.0.0.1", 0), _Handler)
        httpd.scripted = self
        self._httpd = httpd
        self.port = httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=httpd.serve_forever, name="scripted-vllm", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._wake.set()
        httpd = self._httpd
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def enqueue(self, *behaviors) -> None:
        with self._lock:
            self._behaviors.extend(_normalize(behavior) for behavior in behaviors)

    def wait(self, seconds: float) -> None:
        self._wake.wait(timeout=seconds)

    def next_behavior(self) -> dict:
        with self._lock:
            if not self._behaviors:
                return {"behavior": "unscripted"}
            return self._behaviors.pop(0)

    def next_response_id(self) -> str:
        with self._lock:
            self._response_ids += 1
            return f"chatcmpl-{self._response_ids}"

    def models_payload(self) -> dict:
        with self._lock:
            models = list(self.models)
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": 0, "owned_by": "vllm"}
                for model_id in models
            ],
        }

    def record(self, method: str, path: str, authorization, body) -> None:
        with self._lock:
            self._requests.append({
                "method": method,
                "path": path,
                "body": body,
                "authorization": authorization,
            })

    @property
    def requests(self) -> list:
        with self._lock:
            return [dict(item) for item in self._requests]

    @property
    def chat_requests(self) -> list:
        return [item for item in self.requests if item["method"] == "POST" and item["path"] == "/v1/chat/completions"]


def _normalize(behavior) -> dict:
    if isinstance(behavior, str):
        return {"behavior": behavior}
    if isinstance(behavior, dict) and "behavior" in behavior:
        return dict(behavior)
    raise AssertionError(f"unsupported scripted behavior {behavior!r}")


def sample_manifest(**overrides) -> dict:
    """Flat Section 8.5 manifest. Unknown keys are rejected."""
    from idea4_world_state.backend import MANIFEST_FIELDS

    manifest = dict.fromkeys(MANIFEST_FIELDS, None)
    manifest.update({
        "created_at": "2026-01-01T00:00:00Z",
        "host": "localhost",
        "gpu_name": "TEST",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "requested_revision": "main",
        "resolved_revision": "0123456789abcdef0123456789abcdef01234567",
        "tokenizer_revision": "0123456789abcdef0123456789abcdef01234567",
        "served_model_name": "qwen3-vl-8b",
        "dtype": "bfloat16",
        "max_model_len": 32768,
        "limit_mm_per_prompt": '{"image":10,"video":0}',
        "mm_processor_kwargs": None,
        "vllm_version": "0.11.0",
        "seed": 0,
        "port": 8000,
        "serve_command": ["vllm", "serve", "Qwen/Qwen3-VL-8B-Instruct"],
    })
    manifest.update(overrides)
    unknown = sorted(set(manifest) - set(MANIFEST_FIELDS))
    if unknown:
        raise AssertionError("sample manifest has unknown fields: " + ", ".join(unknown))
    return manifest


@pytest.fixture
def vllm_server():
    """Scripted localhost vLLM. See ``ScriptedVLLMServer``."""
    server = ScriptedVLLMServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def manifest_file(tmp_path):
    """Write a Section 8.5 manifest and return its path. Overrides are field values."""

    def write(**overrides):
        path = tmp_path / f"server_manifest_{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(sample_manifest(**overrides)), encoding="utf-8")
        return path

    return write

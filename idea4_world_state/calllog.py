"""Shared call cache, per-run attempt log, and transport pause.

The cache key is the Section 8.6 SHA-256. It includes the messages (image bytes
replaced by the SHA-256 of the decoded PNG), ``kind``, ``max_tokens``,
temperature, top_p, seed, concurrency, and the eight ``model_identity`` fields.
It does not include the run id, profile, endpoint, or API key.

Only ``ok`` and ``truncated`` are written to the shared cache. A transport
error is appended to ``attempts.jsonl`` and raised as ``PauseRun``. The next
``CallLog`` in the same run directory tries that key again and keeps every
attempt. Cache hits return the stored entry and do not call the backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from idea4_world_state import contracts
from idea4_world_state.backend import (
    DECODING_SEED,
    MODEL_IDENTITY_FIELDS,
    ConfigurationError,
    TransportError,
    canonical_json,
    index_messages,
)

CACHE_FIELDS = (
    "key",
    "kind",
    "status",
    "text",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "served_model",
    "response_id",
    "created_at",
    "model_identity",
)

ATTEMPT_FIELDS = (
    "key",
    "kind",
    "attempt",
    "started_at",
    "latency_seconds",
    "outcome",
    "error_class",
    "http_status",
    "message",
)


class PauseRun(Exception):
    """A transport failure was logged. Do not start another call in this process.

    WP5 maps this to exit code 75 (``contracts.EXIT_CODES["paused"]``).
    In-flight calls are the runner's concern; this exception covers the call
    that failed. ``attempt`` is the JSON object appended to ``attempts.jsonl``.
    """

    exit_code = contracts.EXIT_CODES["paused"]

    def __init__(self, attempt: dict):
        self.attempt = attempt
        self.key = attempt["key"]
        super().__init__(attempt.get("message") or "transport failure")


def identity_for_key(model_identity) -> dict:
    """Copy of the eight Section 8.6 identity fields, in contract order."""
    if not isinstance(model_identity, dict):
        raise ConfigurationError("model_identity must be an object")
    missing = [field for field in MODEL_IDENTITY_FIELDS if field not in model_identity]
    if missing:
        raise ConfigurationError("model_identity is missing fields: " + ", ".join(missing))
    try:
        return {field: json.loads(canonical_json(model_identity[field])) for field in MODEL_IDENTITY_FIELDS}
    except (TypeError, ValueError):
        raise ConfigurationError("model_identity is not JSON-serializable") from None


def cache_key(messages, max_tokens, kind, model_identity, concurrency) -> str:
    """Section 8.6 cache key. Image parts contribute the SHA-256 of their PNG bytes."""
    if kind not in contracts.CALL_KINDS:
        raise ConfigurationError(f"unknown call kind {kind!r}")
    _require_call_args(max_tokens, concurrency)
    key_messages, _images = index_messages(messages)
    payload = {
        "concurrency": concurrency,
        "kind": kind,
        "max_tokens": max_tokens,
        "messages": key_messages,
        "model_identity": identity_for_key(model_identity),
        "seed": DECODING_SEED,
        "temperature": contracts.TEMPERATURE,
        "top_p": contracts.TOP_P,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def cache_path(cache_dir, key: str) -> Path:
    """``<cache_dir>/<key[:2]>/<key>.json``."""
    return Path(cache_dir) / key[:2] / f"{key}.json"


def _require_call_args(max_tokens, concurrency) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ConfigurationError("max_tokens must be a positive integer")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ConfigurationError("concurrency must be a positive integer")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _pretty(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _archive_messages(key_messages, run_relative_inputs: str = "inputs") -> list:
    archived = []
    for message in key_messages:
        content = message.get("content")
        if not isinstance(content, list):
            archived.append(message)
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url" and "sha256" in part:
                digest = part["sha256"]
                parts.append({
                    "path": f"{run_relative_inputs}/{digest}.png",
                    "sha256": digest,
                    "type": "image",
                })
            else:
                parts.append(part)
        rewritten = {key: value for key, value in message.items() if key != "content"}
        rewritten["content"] = parts
        archived.append(rewritten)
    return archived


class CallLog:
    """Thread-safe attempt log plus a shared terminal-result cache.

    ``concurrency`` is part of the cache key because vLLM batching at
    temperature 0 is not bitwise deterministic across batch sizes. WP5 may
    call ``call`` from a thread pool when concurrency is 2. Appends take a
    lock; cache files are published with ``os.replace``.
    """

    def __init__(self, run_dir, cache_dir, backend, model_identity, concurrency):
        _require_call_args(1, concurrency)
        self.run_dir = Path(run_dir)
        self.cache_dir = Path(cache_dir)
        self.backend = backend
        self.concurrency = concurrency
        self.model_identity = identity_for_key(model_identity)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "requests").mkdir(exist_ok=True)
        (self.run_dir / "inputs").mkdir(exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._append_lock = threading.Lock()
        self._input_lock = threading.Lock()

    def call(self, messages, max_tokens, kind) -> dict:
        """Return the shared cache entry, or raise ``PauseRun`` after logging.

        A hit does not call the backend. A miss writes ``requests/<key>.json``
        and deduplicated ``inputs/<sha>.png``, performs one ``generate``, and
        appends one ``attempts.jsonl`` line. ``ok`` and ``truncated`` are then
        stored at ``cache_path``. Transport errors are not stored.
        """
        if kind not in contracts.CALL_KINDS:
            raise ConfigurationError(f"unknown call kind {kind!r}")
        _require_call_args(max_tokens, self.concurrency)
        key = cache_key(messages, max_tokens, kind, self.model_identity, self.concurrency)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        self._write_request(key, messages, max_tokens, kind)
        started_at = _now()
        started = time.monotonic()
        try:
            response = self.backend.generate(messages, max_tokens, kind)
        except TransportError as exc:
            attempt = self._record_attempt(
                key=key,
                kind=kind,
                started_at=started_at,
                latency_seconds=time.monotonic() - started,
                outcome="transport_error",
                error_class=exc.error_class,
                http_status=exc.http_status,
                message=exc.sanitized_message,
            )
            raise PauseRun(attempt) from None
        outcome = "truncated" if response.get("finish_reason") == "length" else "ok"
        usage = response.get("usage") or {}
        latency = response.get("latency_seconds")
        if latency is None:
            latency = time.monotonic() - started
        entry = {
            "key": key,
            "kind": kind,
            "status": outcome,
            "text": response.get("text") or "",
            "finish_reason": response.get("finish_reason"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "latency_seconds": latency,
            "served_model": response.get("served_model"),
            "response_id": response.get("response_id"),
            "created_at": _now(),
            "model_identity": dict(self.model_identity),
        }
        self._record_attempt(
            key=key,
            kind=kind,
            started_at=started_at,
            latency_seconds=latency,
            outcome=outcome,
            error_class=None,
            http_status=None,
            message=None,
        )
        return self._write_cache(entry)

    def history(self, key: str) -> list:
        """Attempts for ``key`` in this run directory, in append order."""
        with self._append_lock:
            return [dict(row) for row in self._load_attempts() if row.get("key") == key]

    def _write_request(self, key: str, messages, max_tokens, kind) -> None:
        key_messages, images = index_messages(messages)
        with self._input_lock:
            for digest, raw in images:
                path = self.run_dir / "inputs" / f"{digest}.png"
                if not path.exists():
                    _atomic_write(path, raw)
        document = {
            "concurrency": self.concurrency,
            "key": key,
            "kind": kind,
            "max_tokens": max_tokens,
            "messages": _archive_messages(key_messages),
            "model": self.model_identity["served_model_name"],
            "seed": DECODING_SEED,
            "temperature": contracts.TEMPERATURE,
            "top_p": contracts.TOP_P,
        }
        _atomic_write(self.run_dir / "requests" / f"{key}.json", _pretty(document).encode("utf-8"))

    def _record_attempt(self, **fields) -> dict:
        with self._append_lock:
            prior = sum(1 for row in self._load_attempts() if row.get("key") == fields["key"])
            record = {**fields, "attempt": prior + 1}
            missing = [field for field in ATTEMPT_FIELDS if field not in record]
            if missing:
                raise ConfigurationError("attempt is missing fields: " + ", ".join(missing))
            line = canonical_json(record) + "\n"
            path = self.run_dir / "attempts.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return json.loads(line)

    def _load_attempts(self) -> list:
        path = self.run_dir / "attempts.jsonl"
        if not path.exists():
            return []
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ConfigurationError(f"corrupt attempts log at line {line_number}") from None
            if not isinstance(row, dict):
                raise ConfigurationError(f"corrupt attempts log at line {line_number}")
            rows.append(row)
        return rows

    def _read_cache(self, key: str):
        path = cache_path(self.cache_dir, key)
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ConfigurationError("corrupt cache entry") from None
        if not isinstance(value, dict) or value.get("key") != key:
            raise ConfigurationError("corrupt cache entry")
        if value.get("status") not in contracts.TERMINAL_ATTEMPT_OUTCOMES:
            raise ConfigurationError("corrupt cache entry")
        return value

    def _write_cache(self, entry: dict) -> dict:
        text = _pretty(entry)
        _atomic_write(cache_path(self.cache_dir, entry["key"]), text.encode("utf-8"))
        return json.loads(text)

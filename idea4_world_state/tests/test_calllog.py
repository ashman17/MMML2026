"""Shared cache, attempt history, and transport pause."""

import base64
import hashlib
import json
import struct
import threading
import zlib

import pytest

from idea4_world_state import contracts
from idea4_world_state.backend import MODEL_IDENTITY_FIELDS, MockBackend, VLLMBackend
from idea4_world_state.calllog import ATTEMPT_FIELDS, CACHE_FIELDS, CallLog, PauseRun, cache_key, cache_path


def one_pixel_png(red, green, blue) -> bytes:
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw = b"\x00" + bytes((red, green, blue))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def user_messages(pngs, text):
    content = []
    for index, png in enumerate(pngs, 1):
        content.append({"type": "text", "text": f"Image {index}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url(png)}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _open_backend(server, manifest_path, timeout=5):
    return VLLMBackend(
        endpoint=server.base_url,
        served_model="qwen3-vl-8b",
        api_key_env="IDEA4_API_KEY",
        timeout=timeout,
        server_manifest_path=manifest_path,
    )


def _files_contain(root, needle: str) -> list:
    encoded = needle.encode("utf-8")
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and (needle in path.name or encoded in path.read_bytes()):
            hits.append(path)
    return hits


def test_transport_failure_is_logged_not_cached_and_resume_retries_once(
    tmp_path, vllm_server, manifest_file, monkeypatch
):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("http_500", "ok")
    backend = _open_backend(vllm_server, manifest_file())
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    messages = user_messages([one_pixel_png(1, 2, 3)], "resume question")
    try:
        log = CallLog(run, cache, backend, backend.model_identity(), 1)
        with pytest.raises(PauseRun) as caught:
            log.call(messages, 64, "raw_answer")
        pause = caught.value
        assert pause.exit_code == contracts.EXIT_CODES["paused"] == 75
        assert pause.attempt["outcome"] == "transport_error"
        assert pause.attempt["error_class"] == "APIStatusError"
        assert pause.attempt["http_status"] == 500
        assert pause.attempt["message"] == "HTTP 500"
        assert pause.attempt["attempt"] == 1
        assert set(pause.attempt) == set(ATTEMPT_FIELDS)
        assert not cache_path(cache, pause.key).exists()
        assert list(cache.rglob("*.json")) == []
        assert len(vllm_server.chat_requests) == 1
        assert (run / "requests" / f"{pause.key}.json").is_file()

        resumed = CallLog(run, cache, backend, backend.model_identity(), 1)
        result = resumed.call(messages, 64, "raw_answer")
    finally:
        backend.close()

    assert result["status"] == "ok"
    assert result["key"] == pause.key
    assert set(result) == set(CACHE_FIELDS)
    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 5
    assert len(vllm_server.chat_requests) == 2
    history = resumed.history(pause.key)
    assert [row["attempt"] for row in history] == [1, 2]
    assert [row["outcome"] for row in history] == ["transport_error", "ok"]
    assert history[1]["error_class"] is None
    assert history[1]["http_status"] is None
    assert history[1]["message"] is None
    assert cache_path(cache, pause.key).is_file()


def test_a_second_transport_failure_is_still_not_cached(tmp_path, vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("http_500", "http_429")
    backend = _open_backend(vllm_server, manifest_file())
    run = tmp_path / "run"
    cache = tmp_path / "cache"
    messages = [{"role": "user", "content": "still down"}]
    try:
        first = CallLog(run, cache, backend, backend.model_identity(), 1)
        with pytest.raises(PauseRun):
            first.call(messages, 32, "extract")
        second = CallLog(run, cache, backend, backend.model_identity(), 1)
        with pytest.raises(PauseRun) as caught:
            second.call(messages, 32, "extract")
    finally:
        backend.close()
    assert len(vllm_server.chat_requests) == 2
    assert [row["http_status"] for row in second.history(caught.value.key)] == [500, 429]
    assert list(cache.rglob("*.json")) == []


def test_ok_and_truncated_are_reused_across_run_directories(tmp_path, vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("ok", "length")
    backend = _open_backend(vllm_server, manifest_file())
    cache = tmp_path / "cache"
    identity = None
    try:
        identity = backend.model_identity()
        first = CallLog(tmp_path / "run-a", cache, backend, identity, 1)
        ok = first.call([{"role": "user", "content": "alpha"}], 64, "raw_answer")
        truncated = first.call([{"role": "user", "content": "beta"}], 64, "flat_answer")
        assert ok["status"] == "ok"
        assert ok["finish_reason"] == "stop"
        assert truncated["status"] == "truncated"
        assert truncated["finish_reason"] == "length"
        assert len(vllm_server.chat_requests) == 2
        assert len(first.history(ok["key"])) == 1

        second = CallLog(tmp_path / "run-b", cache, backend, identity, 1)
        assert second.call([{"role": "user", "content": "alpha"}], 64, "raw_answer") == ok
        assert second.call([{"role": "user", "content": "beta"}], 64, "flat_answer") == truncated
        assert len(vllm_server.chat_requests) == 2
        assert second.history(ok["key"]) == []
        assert not (tmp_path / "run-b" / "attempts.jsonl").exists()
        assert len(first.history(ok["key"])) == 1
    finally:
        backend.close()


def test_key_changes_with_identity_image_bytes_concurrency_and_max_tokens(tmp_path):
    identity = MockBackend().model_identity()
    png_a = one_pixel_png(1, 2, 3)
    png_b = one_pixel_png(4, 5, 6)
    messages = user_messages([png_a], "where")
    key = cache_key(messages, 64, "raw_answer", identity, 1)
    log = CallLog(tmp_path / "run", tmp_path / "cache", MockBackend(), identity, 1)
    assert log.call(messages, 64, "raw_answer")["key"] == key

    changed = dict(identity)
    changed["resolved_revision"] = "other-revision"
    assert cache_key(messages, 64, "raw_answer", changed, 1) != key
    # max_side is not its own field; a different resize is different PNG bytes.
    assert cache_key(user_messages([png_b], "where"), 64, "raw_answer", identity, 1) != key
    assert cache_key(messages, 64, "raw_answer", identity, 2) != key
    assert cache_key(messages, 128, "raw_answer", identity, 1) != key
    assert cache_key(user_messages([png_a, png_b], "where"), 64, "raw_answer", identity, 1) != cache_key(
        user_messages([png_b, png_a], "where"), 64, "raw_answer", identity, 1
    )

    payload = base64.b64encode(png_a).decode("ascii")
    messy = [{
        "content": [
            {"type": "text", "text": "Image 1:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + payload[:8] + "\n" + payload[8:]}},
            {"type": "text", "text": "where"},
        ],
        "role": "user",
    }]
    assert cache_key(messy, 64, "raw_answer", identity, 1) == key
    reordered = [{"content": "where", "role": "user"}]
    assert cache_key([{"role": "user", "content": "where"}], 64, "raw_answer", identity, 1) == cache_key(
        reordered, 64, "raw_answer", identity, 1
    )

    same_processor = dict(identity)
    other_processor = dict(identity)
    same_processor["mm_processor_kwargs"] = '{"a":2,"b":1}'
    other_processor["mm_processor_kwargs"] = '{"b":1,"a":2}'
    assert cache_key(messages, 64, "raw_answer", same_processor, 1) == cache_key(
        messages, 64, "raw_answer", dict(same_processor), 1
    )
    assert cache_key(messages, 64, "raw_answer", same_processor, 1) != cache_key(
        messages, 64, "raw_answer", other_processor, 1
    )
    assert cache_key(messages, 64, "raw_answer", same_processor, 1) != key

    extra = dict(identity)
    extra["endpoint"] = "http://127.0.0.1:9/v1"
    extra["api_key"] = "not-part-of-the-key"
    assert cache_key(messages, 64, "raw_answer", extra, 1) == key
    assert set(identity) == set(MODEL_IDENTITY_FIELDS)


def test_same_messages_under_two_kinds_have_two_cache_entries(tmp_path, vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("ok", "length")
    backend = _open_backend(vllm_server, manifest_file())
    messages = [{"role": "user", "content": "same body"}]
    cache = tmp_path / "cache"
    try:
        log = CallLog(tmp_path / "run", cache, backend, backend.model_identity(), 1)
        first = log.call(messages, 64, "raw_answer")
        second = log.call(messages, 64, "extract")
    finally:
        backend.close()
    assert first["key"] != second["key"]
    assert first["kind"] == "raw_answer"
    assert second["kind"] == "extract"
    assert cache_path(cache, first["key"]).is_file()
    assert cache_path(cache, second["key"]).is_file()
    assert json.loads(cache_path(cache, first["key"]).read_text(encoding="utf-8"))["kind"] == "raw_answer"
    assert json.loads(cache_path(cache, second["key"]).read_text(encoding="utf-8"))["kind"] == "extract"
    assert len(vllm_server.chat_requests) == 2
    assert [row["kind"] for row in log.history(first["key"])] == ["raw_answer"]
    assert [row["kind"] for row in log.history(second["key"])] == ["extract"]


def test_api_key_never_appears_under_run_or_cache(tmp_path, vllm_server, manifest_file, monkeypatch):
    key_a = "SENTINEL_KEY_idea4_wp4_9f3a7c"
    key_b = "SENTINEL_KEY_idea4_wp4_bb22e1"
    prompt = "PROMPT_SENTINEL_idea4_wp4"
    monkeypatch.setenv("IDEA4_API_KEY", key_a)
    vllm_server.enqueue({"behavior": "http_500", "body": key_a + " echoed body " + prompt}, "ok")
    manifest = manifest_file()
    run = tmp_path / "run"
    other = tmp_path / "other"
    cache = tmp_path / "cache"
    messages = user_messages([one_pixel_png(8, 8, 8)], prompt)
    backend_a = _open_backend(vllm_server, manifest)
    try:
        log = CallLog(run, cache, backend_a, backend_a.model_identity(), 1)
        with pytest.raises(PauseRun) as caught:
            log.call(messages, 64, "extract")
        assert key_a not in str(caught.value)
        assert prompt not in caught.value.attempt["message"]
        assert caught.value.attempt["message"] == "HTTP 500"
        result = CallLog(run, cache, backend_a, backend_a.model_identity(), 1).call(messages, 64, "extract")
        chats_after_success = len(vllm_server.chat_requests)
        monkeypatch.setenv("IDEA4_API_KEY", key_b)
        backend_b = _open_backend(vllm_server, manifest)
        try:
            hit = CallLog(other, cache, backend_b, backend_a.model_identity(), 1).call(messages, 64, "extract")
        finally:
            backend_b.close()
    finally:
        backend_a.close()

    assert hit == result
    assert len(vllm_server.chat_requests) == chats_after_success == 2
    assert vllm_server.chat_requests[0]["authorization"] == f"Bearer {key_a}"
    assert key_a not in json.dumps(vllm_server.chat_requests[0]["body"])
    request_text = (run / "requests" / f"{result['key']}.json").read_text(encoding="utf-8")
    assert prompt in request_text
    assert "base64" not in request_text
    assert "iVBORw0KGgo" not in request_text
    for root in (run, other, cache):
        assert _files_contain(root, key_a) == []
        assert _files_contain(root, key_b) == []
        assert list(root.rglob("*.tmp")) == []


def test_request_archive_deduplicates_png_bytes(tmp_path):
    png = one_pixel_png(9, 9, 9)
    messages = user_messages([png, png], "two copies")
    identity = MockBackend().model_identity()
    run = tmp_path / "run"
    log = CallLog(run, tmp_path / "cache", MockBackend(), identity, 1)
    result = log.call(messages, 64, "extract")
    stored = list((run / "inputs").glob("*.png"))
    assert len(stored) == 1
    digest = hashlib.sha256(png).hexdigest()
    assert stored[0].name == f"{digest}.png"
    assert stored[0].read_bytes() == png
    document = json.loads((run / "requests" / f"{result['key']}.json").read_text(encoding="utf-8"))
    image_parts = [part for part in document["messages"][0]["content"] if part.get("type") == "image"]
    assert image_parts == [
        {"path": f"inputs/{digest}.png", "sha256": digest, "type": "image"},
        {"path": f"inputs/{digest}.png", "sha256": digest, "type": "image"},
    ]
    assert document["temperature"] == 0
    assert document["top_p"] == 1
    assert document["seed"] == 0
    assert "api_key" not in document


def test_concurrent_appends_keep_attempt_numbers_and_valid_jsonl(tmp_path):
    parties = 4
    barrier = threading.Barrier(parties)

    class _BarrierBackend:
        def __init__(self):
            self.inner = MockBackend()

        def model_identity(self):
            return self.inner.model_identity()

        def generate(self, messages, max_tokens, kind):
            barrier.wait(timeout=5)
            return self.inner.generate(messages, max_tokens, kind)

        def close(self):
            return None

    backend = _BarrierBackend()
    messages = [{"role": "user", "content": "shared"}]
    log = CallLog(tmp_path / "run", tmp_path / "cache", backend, backend.model_identity(), 2)
    errors = []

    def worker():
        try:
            log.call(messages, 64, "raw_answer")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    key = cache_key(messages, 64, "raw_answer", backend.model_identity(), 2)
    history = log.history(key)
    assert [row["attempt"] for row in history] == list(range(1, parties + 1))
    lines = (tmp_path / "run" / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == parties
    assert [json.loads(line)["attempt"] for line in lines] == list(range(1, parties + 1))
    assert cache_path(tmp_path / "cache", key).is_file()

    distinct_errors = []

    def other(index):
        try:
            log.call([{"role": "user", "content": f"distinct-{index}"}], 64, "flat_answer")
        except Exception as exc:
            distinct_errors.append(exc)

    more = [threading.Thread(target=other, args=(index,)) for index in range(8)]
    for thread in more:
        thread.start()
    for thread in more:
        thread.join()
    assert distinct_errors == []
    all_lines = (tmp_path / "run" / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(all_lines) == parties + 8
    parsed = [json.loads(line) for line in all_lines]
    assert len({row["key"] for row in parsed}) == 9

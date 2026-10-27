"""VLLMBackend transport mapping and MockBackend determinism."""

import base64
import hashlib
import json
import struct
import zlib

import pytest
from jsonschema import Draft202012Validator

from idea3_consistency.mmsi_consistency import extract
from idea4_world_state import contracts
from idea4_world_state.backend import (
    DECODING_SEED,
    MANIFEST_FIELDS,
    MODEL_IDENTITY_FIELDS,
    ConfigurationError,
    MockBackend,
    TransportError,
    VLLMBackend,
    canonical_json,
)


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


def _backend(server, manifest_path, **kwargs):
    return VLLMBackend(
        endpoint=kwargs.get("endpoint", server.base_url),
        served_model=kwargs.get("served_model", "qwen3-vl-8b"),
        api_key_env=kwargs.get("api_key_env", "IDEA4_API_KEY"),
        timeout=kwargs.get("timeout", 5),
        server_manifest_path=manifest_path,
    )


def test_models_mismatch_is_a_configuration_error(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.models = ["other-model"]
    with pytest.raises(ConfigurationError, match="not listed") as caught:
        _backend(vllm_server, manifest_file())
    assert caught.value.exit_code == contracts.EXIT_CODES["configuration_or_contract"]
    assert vllm_server.chat_requests == []


def test_manifest_mismatch_is_a_configuration_error(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="served_model_name") as caught:
        _backend(vllm_server, manifest_file(served_model_name="other-model"))
    assert caught.value.exit_code == 2
    assert vllm_server.requests == []


def test_manifest_must_be_the_flat_field_set(tmp_path, vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    missing = tmp_path / "missing.json"
    payload = json.loads(manifest_file().read_text())
    del payload["port"]
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="missing port"):
        _backend(vllm_server, missing)

    extra = tmp_path / "extra.json"
    payload["port"] = 8000
    payload["unlisted"] = 1
    extra.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unexpected unlisted"):
        _backend(vllm_server, extra)

    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        _backend(vllm_server, tmp_path / "bad.json")

    (tmp_path / "array.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="JSON object"):
        _backend(vllm_server, tmp_path / "array.json")

    with pytest.raises(ConfigurationError, match="cannot read"):
        _backend(vllm_server, tmp_path / "absent.json")


def test_manifest_containing_the_api_key_is_rejected(vllm_server, manifest_file, monkeypatch):
    sentinel = "SENTINEL_KEY_idea4_wp4_manifest"
    monkeypatch.setenv("IDEA4_API_KEY", sentinel)
    path = manifest_file(host=sentinel)
    with pytest.raises(ConfigurationError, match="API key") as caught:
        _backend(vllm_server, path)
    assert sentinel not in str(caught.value)
    assert vllm_server.requests == []


@pytest.mark.parametrize("command", [
    ["vllm", "serve", "--api-key", "REDACTED"],
    ["vllm", "serve", "--api-key=secret"],
])
def test_serve_command_rejects_api_key_flag(vllm_server, manifest_file, monkeypatch, command):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="--api-key") as caught:
        _backend(vllm_server, manifest_file(serve_command=command))
    assert "REDACTED" not in str(caught.value)
    assert "secret" not in str(caught.value)
    assert vllm_server.requests == []


def test_model_identity_is_the_manifest_subset(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    processor = '{"b":1,"a":[2]}'
    limits = '{"image":10,"video":0}'
    path = manifest_file(mm_processor_kwargs=processor, limit_mm_per_prompt=limits, quantization=None)
    backend = _backend(vllm_server, path)
    try:
        identity = backend.model_identity()
        assert list(identity) == list(MODEL_IDENTITY_FIELDS)
        assert identity["served_model_name"] == "qwen3-vl-8b"
        assert identity["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
        assert identity["mm_processor_kwargs"] == processor
        assert isinstance(identity["mm_processor_kwargs"], str)
        assert identity["quantization"] is None
        stored = json.loads(path.read_text())
        assert stored["mm_processor_kwargs"] == processor
        assert stored["limit_mm_per_prompt"] == limits
        assert set(stored) == set(MANIFEST_FIELDS)
    finally:
        backend.close()
    with pytest.raises(ConfigurationError, match="mm_processor_kwargs"):
        _backend(vllm_server, manifest_file(mm_processor_kwargs={"a": 1}))
    with pytest.raises(ConfigurationError, match="limit_mm_per_prompt"):
        _backend(vllm_server, manifest_file(limit_mm_per_prompt={"image": 10}))


def test_generate_sends_fixed_sampling_and_maps_usage(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue({"behavior": "ok", "content": "``B``", "prompt_tokens": 11, "completion_tokens": 5})
    backend = _backend(vllm_server, manifest_file())
    try:
        assert backend.client.max_retries == contracts.MAX_SDK_RETRIES == 0
        result = backend.generate([{"role": "user", "content": "hello"}], 64, "raw_answer")
    finally:
        backend.close()
    body = vllm_server.chat_requests[0]["body"]
    assert body["temperature"] == contracts.TEMPERATURE == 0
    assert body["top_p"] == contracts.TOP_P == 1
    assert body["seed"] == DECODING_SEED == 0
    assert body["max_tokens"] == 64
    assert body["model"] == "qwen3-vl-8b"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert result["text"] == "``B``"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"input_tokens": 11, "output_tokens": 5}
    assert result["served_model"] == "qwen3-vl-8b"
    assert result["response_id"] == "chatcmpl-1"
    assert result["latency_seconds"] >= 0
    assert len(vllm_server.chat_requests) == 1


def test_length_finish_reason_is_returned(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("length")
    backend = _backend(vllm_server, manifest_file())
    try:
        result = backend.generate([{"role": "user", "content": "hello"}], 64, "extract")
    finally:
        backend.close()
    assert result["finish_reason"] == "length"
    assert result["text"] == "``"


def test_status_errors_are_one_request_and_carry_no_body(vllm_server, manifest_file, monkeypatch):
    sentinel = "SENTINEL_KEY_idea4_wp4_status"
    prompt = "PROMPT_SENTINEL_idea4_wp4_status"
    monkeypatch.setenv("IDEA4_API_KEY", sentinel)
    vllm_server.enqueue(
        {"behavior": "http_500", "body": f"{sentinel} echoed {prompt}"},
        {"behavior": "http_429", "body": f"{sentinel} rate"},
    )
    backend = _backend(vllm_server, manifest_file())
    try:
        with pytest.raises(TransportError) as first:
            backend.generate([{"role": "user", "content": prompt}], 64, "raw_answer")
        with pytest.raises(TransportError) as second:
            backend.generate([{"role": "user", "content": prompt}], 64, "raw_answer")
    finally:
        backend.close()
    assert len(vllm_server.chat_requests) == 2
    for caught, status in ((first, 500), (second, 429)):
        err = caught.value
        assert err.error_class == "APIStatusError"
        assert err.http_status == status
        assert err.sanitized_message == f"HTTP {status}"
        assert sentinel not in str(err)
        assert prompt not in str(err)
    assert vllm_server.chat_requests[0]["authorization"] == f"Bearer {sentinel}"
    assert sentinel not in json.dumps(vllm_server.chat_requests[0]["body"])


def test_drop_and_timeout_map_to_transport_errors(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    vllm_server.enqueue("drop")
    backend = _backend(vllm_server, manifest_file(), timeout=5)
    try:
        with pytest.raises(TransportError) as dropped:
            backend.generate([{"role": "user", "content": "drop"}], 32, "raw_answer")
    finally:
        backend.close()
    assert dropped.value.error_class == "APIConnectionError"
    assert dropped.value.http_status is None
    assert dropped.value.sanitized_message == "connection failed"
    assert len(vllm_server.chat_requests) == 1

    vllm_server.enqueue({"behavior": "sleep", "seconds": 5})
    slow = _backend(vllm_server, manifest_file(), timeout=0.6)
    try:
        with pytest.raises(TransportError) as timed:
            slow.generate([{"role": "user", "content": "sleep"}], 32, "raw_answer")
    finally:
        slow.close()
    assert timed.value.error_class == "APITimeoutError"
    assert timed.value.http_status is None
    assert timed.value.sanitized_message == "request timed out"
    assert len(vllm_server.chat_requests) == 2


def test_missing_key_sends_no_authorization_and_ignores_openai_env(vllm_server, manifest_file, monkeypatch):
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_ENV_SENTINEL_should_not_be_sent")
    vllm_server.enqueue("ok")
    backend = _backend(vllm_server, manifest_file())
    try:
        backend.generate([{"role": "user", "content": "hello"}], 16, "raw_answer")
    finally:
        backend.close()
    for recorded in vllm_server.requests:
        assert recorded["authorization"] is None
        assert "OPENAI_ENV_SENTINEL_should_not_be_sent" not in json.dumps(recorded)


def test_mock_extract_is_schema_valid_and_counts_images():
    schema = json.loads(contracts.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    backend = MockBackend()
    pngs = [one_pixel_png(10, 0, 0), one_pixel_png(20, 0, 0), one_pixel_png(30, 0, 0)]
    messages = user_messages(pngs, "Question stem:\nwhere is the door\n")
    first = backend.generate(messages, 4096, "extract")
    second = MockBackend().generate(messages, 4096, "extract")
    assert first["text"] == second["text"]
    inventory = json.loads(first["text"])
    validator.validate(inventory)
    assert [camera["id"] for camera in inventory["cameras"]] == ["cam1", "cam2", "cam3"]
    assert [camera["image_id"] for camera in inventory["cameras"]] == ["image_1", "image_2", "image_3"]
    assert inventory["schema_version"] == "ws-1"

    empty = json.loads(backend.generate([{"role": "user", "content": "no images"}], 4096, "extract")["text"])
    validator.validate(empty)
    assert empty["cameras"] == []


def test_mock_answer_letter_is_deterministic():
    messages = [{"role": "user", "content": "Which way?"}]
    backend = MockBackend(seed=0)
    text = backend.generate(messages, 64, "raw_answer")["text"]
    digest = canonical_json({"messages": messages, "seed": 0}).encode("utf-8")
    letter = "ABCD"[hashlib.sha256(digest).digest()[0] % 4]
    assert text == f"``{letter}``"
    assert extract(text) == letter
    assert MockBackend(seed=0).generate(messages, 64, "graph_answer")["text"] == text
    assert backend.generate(messages, 64, "flat_answer")["response_id"] == backend.generate(
        messages, 32, "raw_answer"
    )["response_id"]
    assert MockBackend().model_identity()["served_model_name"] == "MOCK"
    letters = {
        MockBackend(seed=seed).generate([{"role": "user", "content": f"q{index}"}], 64, "raw_answer")["text"]
        for seed in range(4)
        for index in range(4)
    }
    assert len(letters) > 1


def test_mock_can_force_truncation_or_invalid_json():
    messages = user_messages([one_pixel_png(1, 1, 1)], "stem")
    truncated = MockBackend(truncate_kinds={"extract"}).generate(messages, 4096, "extract")
    assert truncated["finish_reason"] == "length"
    assert truncated["served_model"] == "MOCK"
    invalid = MockBackend(invalid_json_kinds={"extract"}).generate(messages, 4096, "extract")
    assert invalid["finish_reason"] == "stop"
    with pytest.raises(json.JSONDecodeError):
        json.loads(invalid["text"])
    both = MockBackend(truncate_kinds={"raw_answer"}, invalid_json_kinds={"raw_answer"}).generate(
        messages, 64, "raw_answer"
    )
    assert both["finish_reason"] == "length"
    assert both["text"] == "{not-json"

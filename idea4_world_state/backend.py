"""vLLM OpenAI-compatible client and a deterministic mock.

WP5 constructs one backend, reads ``model_identity()``, and passes that dict to
``CallLog``. Sampling is fixed: temperature 0, top_p 1, seed 0. The SDK retry
budget is ``contracts.MAX_SDK_RETRIES`` (0). Transport failures raise
``TransportError`` and are not cached; ``CallLog`` turns those into ``PauseRun``.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import os
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, omit

from idea4_world_state import contracts

# Decoding seed from Section 4. ``contracts.SEED`` is the dataset seed; both
# are 0 today, and the request always sends this constant.
DECODING_SEED = 0

# Section 8.6 cache identity. Order is the contract order.
MODEL_IDENTITY_FIELDS = (
    "served_model_name",
    "model_id",
    "resolved_revision",
    "dtype",
    "quantization",
    "max_model_len",
    "vllm_version",
    "mm_processor_kwargs",
)

# Section 8.5. Group labels in the plan are not JSON keys.
MANIFEST_FIELDS = (
    "created_at",
    "host",
    "slurm_job_id",
    "gpu_name",
    "gpu_memory_mib",
    "driver_version",
    "cuda_version",
    "vllm_version",
    "torch_version",
    "transformers_version",
    "python_version",
    "pip_freeze_sha256",
    "model_id",
    "requested_revision",
    "resolved_revision",
    "tokenizer_revision",
    "served_model_name",
    "dtype",
    "quantization",
    "max_model_len",
    "limit_mm_per_prompt",
    "mm_processor_kwargs",
    "seed",
    "serve_command",
    "port",
)

_PNG_DATA_PREFIX = "data:image/png;base64,"
_ANSWER_LETTERS = "ABCD"


class ConfigurationError(Exception):
    """Manifest, model-list, or request-shape violation.

    WP5 maps this to exit code 2 (``contracts.EXIT_CODES["configuration_or_contract"]``).
    """

    exit_code = contracts.EXIT_CODES["configuration_or_contract"]


class TransportError(Exception):
    """One failed HTTP attempt. Never cached.

    ``error_class`` is ``APIConnectionError``, ``APITimeoutError``, or
    ``APIStatusError``. ``http_status`` is an int or None. ``sanitized_message``
    is safe to persist: it has no request body and no credential.

    A failure while constructing ``VLLMBackend`` (the ``/v1/models`` probe)
    raises this directly. ``CallLog.call`` catches it and raises ``PauseRun``.
    WP5 should map both to exit code 75.
    """

    def __init__(self, error_class: str, http_status: int | None, sanitized_message: str):
        self.error_class = error_class
        self.http_status = http_status
        self.sanitized_message = " ".join(str(sanitized_message).split())
        super().__init__(self.sanitized_message)


def canonical_json(value) -> str:
    """Canonical JSON used for the Section 8.6 cache key and the mock letter."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def image_bytes_from_part(part) -> bytes | None:
    """Decoded PNG bytes for an OpenAI image part, or None if it is not an image.

    Image parts are ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.
    Whitespace inside the base64 payload is ignored so the hash is of the PNG bytes.
    """
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url.startswith(_PNG_DATA_PREFIX):
        raise ConfigurationError("image part must be a base64 PNG data URL")
    payload = "".join(url[len(_PNG_DATA_PREFIX):].split())
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        raise ConfigurationError("image part is not valid base64") from None


def index_messages(messages) -> tuple[list, list[tuple[str, bytes]]]:
    """Replace each image with the SHA-256 of its PNG bytes.

    Returns ``(key_messages, images)``. ``key_messages`` is JSON-serializable and
    contains no pixel data. ``images`` is ``(sha256, png_bytes)`` in encounter order,
    including duplicates. ``len(images)`` is the image-part count.
    """
    if not isinstance(messages, list):
        raise ConfigurationError("messages must be a list")
    key_messages = []
    images: list[tuple[str, bytes]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ConfigurationError("each message must be an object")
        content = message.get("content")
        if isinstance(content, str) or content is None:
            key_messages.append(copy.deepcopy(message))
            continue
        if not isinstance(content, list):
            raise ConfigurationError("message content must be a string or a list of parts")
        parts = []
        for part in content:
            raw = image_bytes_from_part(part)
            if raw is None:
                parts.append(copy.deepcopy(part) if isinstance(part, dict) else part)
                continue
            digest = hashlib.sha256(raw).hexdigest()
            images.append((digest, raw))
            parts.append({"sha256": digest, "type": "image_url"})
        rewritten = {key: copy.deepcopy(value) for key, value in message.items() if key != "content"}
        rewritten["content"] = parts
        key_messages.append(rewritten)
    return key_messages, images


def iter_image_bytes(messages) -> list[bytes]:
    """PNG bytes of every image part, in order."""
    return [raw for _digest, raw in index_messages(messages)[1]]


def _close_quietly(response) -> None:
    if response is None:
        return
    try:
        response.close()
    except Exception:
        return


def _positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"{label} must be a positive number")
    return float(value)


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{label} must be a positive integer")
    return value


def _contains_secret(value, secret: str) -> bool:
    if not secret:
        return False
    if isinstance(value, str):
        return secret in value
    if isinstance(value, dict):
        return any(_contains_secret(item, secret) for item in value.values())
    if isinstance(value, list):
        return any(_contains_secret(item, secret) for item in value)
    return False


def _load_manifest(path, api_key: str) -> dict:
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read server manifest ({exc.strerror or 'unreadable'})") from None
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError:
        raise ConfigurationError("server manifest is not valid JSON") from None
    if not isinstance(manifest, dict):
        raise ConfigurationError("server manifest must be a JSON object")
    missing = [field for field in MANIFEST_FIELDS if field not in manifest]
    extra = sorted(field for field in manifest if field not in MANIFEST_FIELDS)
    if missing or extra:
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        raise ConfigurationError("server manifest fields: " + "; ".join(detail))
    _validate_manifest_values(manifest)
    if _contains_secret(manifest, api_key):
        raise ConfigurationError("server manifest contains the API key")
    return manifest


def _validate_manifest_values(manifest: dict) -> None:
    """Section 8.5: only ``serve_command`` is a list, and it has no API key."""
    command = manifest["serve_command"]
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ConfigurationError("serve_command must be a list of strings")
    if any("--api-key" in item for item in command):
        raise ConfigurationError("serve_command must not contain --api-key")
    for field in ("limit_mm_per_prompt", "mm_processor_kwargs"):
        value = manifest[field]
        if value is not None and not isinstance(value, str):
            raise ConfigurationError(f"{field} must be a canonical JSON string or null")
    for field, value in manifest.items():
        if field == "serve_command":
            continue
        if isinstance(value, (dict, list)):
            raise ConfigurationError(f"server manifest field {field} must be a scalar")


def _identity_from_manifest(manifest: dict) -> dict:
    return {field: copy.deepcopy(manifest[field]) for field in MODEL_IDENTITY_FIELDS}


def _empty_inventory(n_images: int) -> dict:
    return {
        "cameras": [
            {"id": f"cam{index}", "image_id": f"image_{index}"}
            for index in range(1, n_images + 1)
        ],
        "claims": [],
        "entities": [],
        "identity_links": [],
        "observations": [],
        "query_frame": {"basis": None, "frame": "unknown"},
        "schema_version": contracts.SCHEMA_VERSION,
        "temporal_order": {"basis": None, "order": None, "status": "unknown"},
        "unknowns": ["no entities or relations could be established"],
    }


def _usage_dict(prompt_tokens, completion_tokens) -> dict:
    return {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}


def _generate_result(text, finish_reason, usage, latency_seconds, served_model, response_id) -> dict:
    return {
        "text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "latency_seconds": latency_seconds,
        "served_model": served_model,
        "response_id": response_id,
    }


class VLLMBackend:
    """OpenAI client aimed at a vLLM ``/v1`` endpoint.

    ``endpoint`` includes the ``/v1`` prefix (``contracts.DEFAULT_ENDPOINT``).
    The API key is read from the environment variable ``api_key_env``. An unset
    or empty variable sends no ``Authorization`` header. The key is held only by
    the OpenAI client and is never written by this package.

    On construction the client GETs ``/v1/models``. ``served_model`` must be
    listed there and must equal the manifest's ``served_model_name``.
    """

    def __init__(self, endpoint, served_model, api_key_env, timeout, server_manifest_path):
        if not isinstance(endpoint, str) or not endpoint:
            raise ConfigurationError("endpoint must be a non-empty string")
        if not isinstance(served_model, str) or not served_model:
            raise ConfigurationError("served_model must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ConfigurationError("api_key_env must be a non-empty string")
        self.endpoint = endpoint.rstrip("/")
        self.served_model = served_model
        self.api_key_env = api_key_env
        self.timeout = _positive_number(timeout, "timeout")
        # Empty string is intentional: None would fall back to OPENAI_API_KEY.
        api_key = os.environ.get(api_key_env) or ""
        manifest = _load_manifest(server_manifest_path, api_key)
        manifest_name = manifest["served_model_name"]
        if manifest_name != served_model:
            raise ConfigurationError(
                f"served model {served_model!r} does not match manifest served_model_name {manifest_name!r}"
            )
        self._identity = _identity_from_manifest(manifest)
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            timeout=self.timeout,
            max_retries=contracts.MAX_SDK_RETRIES,
            _enforce_credentials=False,
        )
        # An empty key must not fall back to OPENAI_API_KEY and must not send
        # Authorization. This SDK rejects a missing bearer unless the header is
        # explicitly omitted; the placeholder is the Omit sentinel, not a secret.
        self._no_auth_headers = None if api_key else {"Authorization": omit}
        try:
            self._verify_served_model()
        except Exception:
            self.close()
            raise

    def model_identity(self) -> dict:
        """The Section 8.6 identity fields copied from the flat manifest."""
        return {field: copy.deepcopy(self._identity[field]) for field in MODEL_IDENTITY_FIELDS}

    def generate(self, messages, max_tokens, kind) -> dict:
        """One chat completion. ``kind`` selects no server behavior; it is checked only.

        Returns ``text``, ``finish_reason``, ``usage`` (``input_tokens`` and
        ``output_tokens``), ``latency_seconds``, ``served_model``, and ``response_id``.
        """
        if kind not in contracts.CALL_KINDS:
            raise ConfigurationError(f"unknown call kind {kind!r}")
        max_tokens = _positive_int(max_tokens, "max_tokens")
        started = time.monotonic()
        completion = self._chat(messages, max_tokens)
        latency = time.monotonic() - started
        if not completion.choices:
            raise TransportError("APIStatusError", 200, "malformed completion response")
        choice = completion.choices[0]
        content = choice.message.content
        text = content if isinstance(content, str) else ""
        usage = completion.usage
        return _generate_result(
            text=text,
            finish_reason=choice.finish_reason,
            usage=_usage_dict(
                usage.prompt_tokens if usage is not None else None,
                usage.completion_tokens if usage is not None else None,
            ),
            latency_seconds=latency,
            served_model=completion.model,
            response_id=completion.id,
        )

    def close(self) -> None:
        client = getattr(self, "client", None)
        if client is not None:
            client.close()
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _verify_served_model(self) -> None:
        try:
            page = self.client.models.list(extra_headers=self._no_auth_headers)
        except APITimeoutError:
            raise TransportError("APITimeoutError", None, "request timed out") from None
        except APIConnectionError:
            raise TransportError("APIConnectionError", None, "connection failed") from None
        except APIStatusError as exc:
            _close_quietly(exc.response)
            raise TransportError("APIStatusError", int(exc.status_code), f"HTTP {exc.status_code}") from None
        listed = [model.id for model in page.data]
        if self.served_model not in listed:
            raise ConfigurationError(f"served model {self.served_model!r} is not listed by /v1/models")

    def _chat(self, messages, max_tokens):
        try:
            raw = self.client.chat.completions.with_raw_response.create(
                model=self.served_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=contracts.TEMPERATURE,
                top_p=contracts.TOP_P,
                seed=DECODING_SEED,
                extra_headers=self._no_auth_headers,
            )
        except APITimeoutError:
            raise TransportError("APITimeoutError", None, "request timed out") from None
        except APIConnectionError:
            raise TransportError("APIConnectionError", None, "connection failed") from None
        except APIStatusError as exc:
            _close_quietly(exc.response)
            raise TransportError("APIStatusError", int(exc.status_code), f"HTTP {exc.status_code}") from None
        try:
            status = raw.status_code
            if status != 200:
                raise TransportError("APIStatusError", int(status), f"HTTP {status}")
            return raw.parse()
        finally:
            _close_quietly(getattr(raw, "http_response", None))


class MockBackend:
    """Synthetic deterministic backend. No HTTP.

    ``model_identity()["served_model_name"]`` is ``"MOCK"``. ``extract`` returns
    a schema-shaped empty ws-1 inventory with one camera per image part.
    Answer kinds return a letter in double backticks, for example the characters
    backtick-backtick-A-backtick-backtick. The letter is in ABCD and is chosen by
    SHA-256 of the canonical JSON of ``{"messages", "seed"}``.

    ``truncate_kinds`` forces ``finish_reason="length"``. ``invalid_json_kinds``
    returns the text ``{not-json``. Both are test knobs; WP5 leaves them empty.
    """

    def __init__(self, seed: int = 0, *, truncate_kinds=(), invalid_json_kinds=()):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ConfigurationError("seed must be an integer")
        self.seed = seed
        self.truncate_kinds = frozenset(truncate_kinds)
        self.invalid_json_kinds = frozenset(invalid_json_kinds)
        unknown = (self.truncate_kinds | self.invalid_json_kinds) - set(contracts.CALL_KINDS)
        if unknown:
            raise ConfigurationError(f"unknown call kind {sorted(unknown)!r}")

    def model_identity(self) -> dict:
        return {
            "served_model_name": "MOCK",
            "model_id": "MOCK",
            "resolved_revision": None,
            "dtype": None,
            "quantization": None,
            "max_model_len": None,
            "vllm_version": None,
            "mm_processor_kwargs": None,
        }

    def generate(self, messages, max_tokens, kind) -> dict:
        if kind not in contracts.CALL_KINDS:
            raise ConfigurationError(f"unknown call kind {kind!r}")
        _positive_int(max_tokens, "max_tokens")
        _key_messages, images = index_messages(messages)
        truncated = kind in self.truncate_kinds
        if kind in self.invalid_json_kinds:
            text = "{not-json"
        elif kind == "extract":
            text = json.dumps(_empty_inventory(len(images)), ensure_ascii=False, sort_keys=True)
            if truncated:
                text = text[: max(1, len(text) // 2)]
        else:
            text = f"``{self._letter(messages)}``"
            if truncated:
                text = text[:2]
        digest = self._digest(messages)
        return _generate_result(
            text=text,
            finish_reason="length" if truncated else "stop",
            usage=_usage_dict(None, None),
            latency_seconds=0.0,
            served_model="MOCK",
            response_id=f"mock-{digest[:16]}",
        )

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _digest(self, messages) -> str:
        try:
            encoded = canonical_json({"messages": messages, "seed": self.seed}).encode("utf-8")
        except (TypeError, ValueError):
            raise ConfigurationError("messages are not JSON-serializable") from None
        return hashlib.sha256(encoded).hexdigest()

    def _letter(self, messages) -> str:
        return _ANSWER_LETTERS[bytes.fromhex(self._digest(messages)[:2])[0] % 4]

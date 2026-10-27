"""Write the flat Section 8.5 server manifest.

The manifest is a single JSON object. Missing values are null. Nested
processor and multimodal-limit mappings are stored as canonical JSON
strings so the object itself stays flat. ``serve_command`` is the argv
list as launched. The token reaches vLLM only through ``VLLM_API_KEY``, so
an argv containing ``--api-key`` or the token is refused.

System calls are injected through the callables on ``build_manifest``.
Importing this module does not import vLLM, torch, or huggingface_hub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

_SHA_CHARS = frozenset("0123456789abcdefABCDEF")
_CUDA_RE = re.compile(r"CUDA Version:\s*([0-9.]+)")
_INT_RE = re.compile(r"\d[\d,]*")
_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]
_SKIPPED_POSITIONALS = frozenset({"vllm", "serve"})


def default_run_command(argv: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def default_package_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def default_python_version() -> str:
    return platform.python_version()


def default_read_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def default_model_info(model_id: str, revision: str | None) -> Any:
    from huggingface_hub import model_info

    if revision:
        return model_info(model_id, revision=revision)
    return model_info(model_id)


def default_hostname() -> str:
    return platform.node() or os.uname().nodename


def default_now() -> datetime:
    return datetime.now(timezone.utc)


def build_manifest(
    serve_argv: Sequence[str],
    *,
    requested_revision: str | None = None,
    pip_freeze_path: str | None = None,
    api_key: str | None = None,
    run_command: Callable[[Sequence[str]], Any] | None = None,
    package_version: Callable[[str], str | None] | None = None,
    python_version: Callable[[], str | None] | None = None,
    read_bytes: Callable[[str], bytes] | None = None,
    model_info: Callable[[str, str | None], Any] | None = None,
    hostname: Callable[[], str] | None = None,
    environ: Mapping[str, str] | None = None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Collect one flat manifest. Callers inject the system-call helpers."""
    argv = [str(item) for item in serve_argv]
    flags, positionals, limits, mm_kwargs = _scan_argv(argv)
    env = os.environ if environ is None else environ
    model_id = _model_id(positionals)
    lookup = flags.get("--revision") or requested_revision
    tokenizer_lookup = flags.get("--tokenizer-revision") or lookup
    resolved, tokenizer = _revisions(
        model_info or default_model_info,
        model_id,
        lookup,
        tokenizer_lookup,
    )
    gpu_name, gpu_memory, driver, cuda = _gpu(run_command or default_run_command)
    versions = package_version or default_package_version
    payload: dict[str, Any] = {
        "created_at": _created_at((now or default_now)()),
        "host": _host(env, hostname or default_hostname),
        "slurm_job_id": _blank_to_none(env.get("SLURM_JOB_ID")),
        "gpu_name": gpu_name,
        "gpu_memory_mib": gpu_memory,
        "driver_version": driver,
        "cuda_version": cuda,
        "vllm_version": _call_version(versions, "vllm"),
        "torch_version": _call_version(versions, "torch"),
        "transformers_version": _call_version(versions, "transformers"),
        "python_version": _call_python(python_version or default_python_version),
        "pip_freeze_sha256": _freeze_sha256(
            pip_freeze_path, read_bytes or default_read_bytes
        ),
        "model_id": model_id,
        "requested_revision": _blank_to_none(requested_revision)
        or _blank_to_none(flags.get("--revision")),
        "resolved_revision": resolved,
        "tokenizer_revision": tokenizer,
        "served_model_name": _blank_to_none(flags.get("--served-model-name")),
        "dtype": _blank_to_none(flags.get("--dtype")),
        "quantization": _blank_to_none(flags.get("--quantization")),
        "max_model_len": _maybe_int(flags.get("--max-model-len")),
        "limit_mm_per_prompt": _canonical_or_none(limits),
        "mm_processor_kwargs": _canonical_or_none(mm_kwargs),
        "seed": _maybe_int(flags.get("--seed")),
        "serve_command": check_argv_has_no_key(argv, api_key),
        "port": _maybe_int(flags.get("--port")),
    }
    return {key: payload[key] for key in MANIFEST_FIELDS}


def check_argv_has_no_key(argv: Sequence[str], api_key: str | None) -> list[str]:
    """Return ``argv`` unchanged, or raise ``ValueError`` if it carries a key."""
    items = list(argv)
    for arg in items:
        if arg == "--api-key" or arg.startswith("--api-key="):
            raise ValueError("serve argv contains --api-key; pass the token as VLLM_API_KEY")
        if api_key and api_key in arg:
            raise ValueError("serve argv contains the API token")
    return items


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the flat Section 8.5 server manifest."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--serve-argv-file", type=Path, required=True)
    parser.add_argument("--requested-revision", default=None)
    parser.add_argument("--pip-freeze", type=Path, default=None)
    parser.add_argument("--api-key-file", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        payload = json.loads(args.serve_argv_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read serve argv file: {exc.__class__.__name__}", file=sys.stderr)
        return 2
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        print("serve argv file must be a JSON list of strings", file=sys.stderr)
        return 2

    api_key = None
    if args.api_key_file is not None:
        try:
            api_key = args.api_key_file.read_text(encoding="utf-8").strip() or None
        except OSError as exc:
            print(f"could not read API key file: {exc.__class__.__name__}", file=sys.stderr)
            return 2

    try:
        manifest = build_manifest(
            payload,
            requested_revision=args.requested_revision,
            pip_freeze_path=str(args.pip_freeze) if args.pip_freeze else None,
            api_key=api_key,
        )
    except ValueError as exc:
        print(f"refusing to write manifest: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(manifest, indent=2) + "\n"
    if api_key and len(api_key) >= 8 and api_key in encoded:
        print(
            "refusing to write a manifest that still contains the API key",
            file=sys.stderr,
        )
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="utf-8")
    os.chmod(args.out, 0o644)
    if manifest["resolved_revision"] is None:
        print("resolved_revision is null; model_info did not return a commit SHA", file=sys.stderr)
        return 1
    return 0


def _scan_argv(
    argv: Sequence[str],
) -> tuple[dict[str, str | None], list[str], dict[str, Any] | None, dict[str, Any] | None]:
    flags: dict[str, str | None] = {}
    positionals: list[str] = []
    limits: dict[str, Any] | None = None
    mm_kwargs: dict[str, Any] | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            positionals.extend(argv[index + 1 :])
            break
        if not arg.startswith("--"):
            positionals.append(arg)
            index += 1
            continue
        key, value, consumed = _flag(argv, index)
        flags[key] = value
        if key == "--limit-mm-per-prompt" and value:
            parsed = _json_object(value)
            if parsed is not None:
                limits = {} if limits is None else limits
                limits.update(parsed)
        elif key.startswith("--limit-mm-per-prompt.") and value is not None:
            limits = {} if limits is None else limits
            limits[key.split(".", 1)[1]] = _coerce_scalar(value)
        elif key == "--mm-processor-kwargs" and value:
            parsed = _json_object(value)
            if parsed is not None:
                mm_kwargs = {} if mm_kwargs is None else mm_kwargs
                mm_kwargs.update(parsed)
        elif key.startswith("--mm-processor-kwargs.") and value is not None:
            mm_kwargs = {} if mm_kwargs is None else mm_kwargs
            mm_kwargs[key.split(".", 1)[1]] = _coerce_scalar(value)
        index += consumed
    return flags, positionals, limits, mm_kwargs


def _flag(argv: Sequence[str], index: int) -> tuple[str, str | None, int]:
    arg = argv[index]
    if "=" in arg[2:]:
        key, value = arg.split("=", 1)
        return key, value, 1
    if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
        return arg, argv[index + 1], 2
    return arg, None, 1


def _model_id(positionals: Sequence[str]) -> str | None:
    for item in positionals:
        if item.rsplit("/", 1)[-1] in _SKIPPED_POSITIONALS:
            continue
        return item
    return None


def _revisions(
    model_info: Callable[[str, str | None], Any],
    model_id: str | None,
    lookup: str | None,
    tokenizer_lookup: str | None,
) -> tuple[str | None, str | None]:
    if not model_id or not lookup:
        return None, None
    resolved = _sha_from_info(_call_model_info(model_info, model_id, lookup))
    if tokenizer_lookup == lookup:
        return resolved, resolved
    tokenizer = _sha_from_info(
        _call_model_info(model_info, model_id, tokenizer_lookup)
    )
    return resolved, tokenizer


def _call_model_info(
    model_info: Callable[[str, str | None], Any],
    model_id: str,
    revision: str | None,
) -> Any:
    try:
        return model_info(model_id, revision)
    except Exception as exc:
        print(f"model_info failed ({exc.__class__.__name__})", file=sys.stderr)
        return None


def _sha_from_info(info: Any) -> str | None:
    if info is None:
        return None
    sha = info.get("sha") if isinstance(info, dict) else getattr(info, "sha", None)
    if not isinstance(sha, str) or not _is_sha(sha):
        return None
    return sha.lower()


def _is_sha(value: str) -> bool:
    return len(value) == 40 and all(char in _SHA_CHARS for char in value)


def _gpu(
    run_command: Callable[[Sequence[str]], Any],
) -> tuple[str | None, int | None, str | None, str | None]:
    query = _command_stdout(run_command, _GPU_QUERY)
    header = _command_stdout(run_command, ["nvidia-smi"])
    name = memory = driver = None
    if query:
        for line in query.splitlines():
            if line.strip():
                name, memory, driver = _parse_gpu_line(line)
                break
    cuda = None
    if header:
        match = _CUDA_RE.search(header)
        if match:
            cuda = match.group(1)
    return name, memory, driver, cuda


def _command_stdout(
    run_command: Callable[[Sequence[str]], Any], argv: list[str]
) -> str | None:
    try:
        result = run_command(argv)
    except (OSError, subprocess.SubprocessError):
        return None
    returncode = getattr(result, "returncode", 1)
    if returncode not in (0, None):
        return None
    stdout = getattr(result, "stdout", "")
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if not isinstance(stdout, str) or not stdout.strip():
        return None
    return stdout


def _parse_gpu_line(line: str) -> tuple[str | None, int | None, str | None]:
    try:
        row = next(csv.reader([line]))
    except csv.Error:
        return None, None, None
    if len(row) < 3:
        return None, None, None
    name = row[0].strip() or None
    driver = row[2].strip() or None
    return name, _parse_int(row[1]), driver


def _parse_int(text: str | None) -> int | None:
    if text is None:
        return None
    match = _INT_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _maybe_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    if not re.fullmatch(r"-?\d+", value.strip()):
        return None
    return int(value)


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _coerce_scalar(value: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return value
    if isinstance(parsed, (dict, list, int, float, bool)) or parsed is None:
        return parsed
    return value


def _canonical_or_none(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _freeze_sha256(
    path: str | None, read_bytes: Callable[[str], bytes]
) -> str | None:
    if not path:
        return None
    try:
        data = read_bytes(path)
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _call_version(
    package_version: Callable[[str], str | None], name: str
) -> str | None:
    try:
        value = package_version(name)
    except Exception:
        return None
    return _blank_to_none(value)


def _call_python(python_version: Callable[[], str | None]) -> str | None:
    try:
        return _blank_to_none(python_version())
    except Exception:
        return None


def _host(env: Mapping[str, str], hostname: Callable[[], str]) -> str | None:
    forced = _blank_to_none(env.get("IDEA4_COMPUTE_NODE"))
    if forced:
        return forced
    try:
        return _blank_to_none(hostname())
    except OSError:
        return None


def _created_at(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())

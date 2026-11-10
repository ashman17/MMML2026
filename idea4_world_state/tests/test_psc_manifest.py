"""Manifest written by the PSC launcher, with system calls mocked out."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

SECTION_8_5_FIELDS = (
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

MODULE_PATH = Path(__file__).resolve().parents[1] / "psc" / "write_manifest.py"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SHA = "0123456789abcdef0123456789abcdef01234567"
API_KEY = "not-a-real-key-9f3a"
FIXED_NOW = datetime(2026, 10, 24, 18, 0, 0, tzinfo=timezone.utc)
FREEZE = b"vllm==0.32.0\ntorch==2.9.0\ntransformers==5.16.1\nhuggingface_hub==1.1.0\n"
ARGV = [
    "/ocean/projects/GROUP/USER/idea4_world_state/env/bin/vllm",
    "serve",
    MODEL_ID,
    "--revision",
    SHA,
    "--tokenizer-revision",
    SHA,
    "--served-model-name",
    "qwen3-vl-8b",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "32768",
    "--limit-mm-per-prompt.image",
    "10",
    "--limit-mm-per-prompt.video",
    "0",
    "--seed",
    "0",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
]


def load_write_manifest():
    spec = importlib.util.spec_from_file_location(
        "idea4_psc_write_manifest", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def wm():
    return load_write_manifest()


def fake_run(argv):
    if len(argv) > 1 and argv[1].startswith("--query-gpu="):
        stdout = "NVIDIA L40S, 46068, 550.54.15\n"
    elif list(argv) == ["nvidia-smi"]:
        stdout = "NVIDIA-SMI 550.54.15\nCUDA Version: 12.4\n"
    else:
        raise AssertionError(argv)
    return subprocess.CompletedProcess(list(argv), 0, stdout=stdout, stderr="")


def fake_version(name):
    return {
        "vllm": "0.32.0",
        "torch": "2.9.0",
        "transformers": "5.16.1",
    }[name]


def fake_model_info(model_id, revision):
    assert model_id == MODEL_ID
    assert revision == SHA
    return SimpleNamespace(sha=SHA)


def manifest_from(wm, argv, **overrides):
    params = {
        "serve_argv": argv,
        "requested_revision": "main",
        "pip_freeze_path": "pip-freeze.txt",
        "api_key": API_KEY,
        "run_command": fake_run,
        "package_version": fake_version,
        "python_version": lambda: "3.12.7",
        "read_bytes": lambda path: FREEZE,
        "model_info": fake_model_info,
        "hostname": lambda: "should-not-be-used",
        "environ": {"SLURM_JOB_ID": "12345", "IDEA4_COMPUTE_NODE": "v001"},
        "now": lambda: FIXED_NOW,
    }
    params.update(overrides)
    return wm.build_manifest(**params)


def assert_flat(manifest):
    assert list(manifest) == list(SECTION_8_5_FIELDS)
    for key, value in manifest.items():
        if key == "serve_command":
            assert isinstance(value, list)
            assert all(isinstance(item, str) for item in value)
        else:
            assert value is None or isinstance(value, (str, int))


def test_import_does_not_pull_vllm_or_huggingface(wm):
    del wm
    assert "vllm" not in sys.modules
    assert "torch" not in sys.modules
    assert "huggingface_hub" not in sys.modules


def test_mocked_manifest_is_the_flat_section_8_5_object(wm):
    manifest = manifest_from(wm, ARGV)
    command = list(ARGV)
    expected = {
        "created_at": "2026-10-24T18:00:00Z",
        "host": "v001",
        "slurm_job_id": "12345",
        "gpu_name": "NVIDIA L40S",
        "gpu_memory_mib": 46068,
        "driver_version": "550.54.15",
        "cuda_version": "12.4",
        "vllm_version": "0.32.0",
        "torch_version": "2.9.0",
        "transformers_version": "5.16.1",
        "python_version": "3.12.7",
        "pip_freeze_sha256": hashlib.sha256(FREEZE).hexdigest(),
        "model_id": MODEL_ID,
        "requested_revision": "main",
        "resolved_revision": SHA,
        "tokenizer_revision": SHA,
        "served_model_name": "qwen3-vl-8b",
        "dtype": "bfloat16",
        "quantization": None,
        "max_model_len": 32768,
        "limit_mm_per_prompt": '{"image":10,"video":0}',
        "mm_processor_kwargs": None,
        "seed": 0,
        "serve_command": command,
        "port": 8000,
    }
    assert manifest == expected
    assert tuple(wm.MANIFEST_FIELDS) == SECTION_8_5_FIELDS
    assert_flat(manifest)
    assert not any(arg.startswith("--api-key") for arg in manifest["serve_command"])
    print("SAMPLE MANIFEST (mocked probes)")
    print(json.dumps(manifest, indent=2))
    assert API_KEY not in json.dumps(manifest, indent=2)


def test_unavailable_fields_are_null_and_still_present(wm):
    def fail_run(argv):
        raise FileNotFoundError(argv[0])

    def fail_info(model_id, revision):
        raise OSError("offline")

    def fail_read(path):
        raise FileNotFoundError(path)

    def fail_python():
        raise RuntimeError("no python")

    def fail_host():
        raise OSError("no host")

    manifest = manifest_from(
        wm,
        ["vllm", "serve", MODEL_ID],
        requested_revision="main",
        pip_freeze_path="missing.txt",
        api_key=None,
        run_command=fail_run,
        package_version=lambda name: None,
        python_version=fail_python,
        read_bytes=fail_read,
        model_info=fail_info,
        hostname=fail_host,
        environ={},
    )
    assert list(manifest) == list(SECTION_8_5_FIELDS)
    assert_flat(manifest)
    for key in SECTION_8_5_FIELDS:
        if key in {"created_at", "model_id", "serve_command", "requested_revision"}:
            continue
        assert manifest[key] is None
    assert manifest["requested_revision"] == "main"
    assert manifest["model_id"] == MODEL_ID
    assert manifest["serve_command"] == ["vllm", "serve", MODEL_ID]
    assert manifest["created_at"] == "2026-10-24T18:00:00Z"


def test_json_limits_stay_flat(wm):
    argv = [
        "vllm",
        "serve",
        MODEL_ID,
        "--limit-mm-per-prompt",
        '{"image": 10, "video": 0}',
        "--mm-processor-kwargs",
        '{"min_pixels": 65536}',
        "--seed",
        "0",
    ]
    manifest = manifest_from(wm, argv, model_info=lambda model_id, revision: SimpleNamespace(sha=SHA))
    assert manifest["limit_mm_per_prompt"] == '{"image":10,"video":0}'
    assert manifest["mm_processor_kwargs"] == '{"min_pixels":65536}'
    assert manifest["seed"] == 0
    assert_flat(manifest)


def test_cli_writes_manifest_without_key(wm, tmp_path, monkeypatch):
    monkeypatch.setattr(wm, "default_run_command", fake_run)
    monkeypatch.setattr(wm, "default_model_info", fake_model_info)
    monkeypatch.setattr(wm, "default_package_version", fake_version)
    monkeypatch.setattr(wm, "default_python_version", lambda: "3.12.7")
    monkeypatch.setattr(wm, "default_hostname", lambda: "v001")
    monkeypatch.setattr(wm, "default_now", lambda: FIXED_NOW)
    argv_file = tmp_path / "argv.json"
    argv_file.write_text(json.dumps(ARGV), encoding="utf-8")
    key_file = tmp_path / "api_key"
    key_file.write_text(API_KEY + "\n", encoding="utf-8")
    freeze = tmp_path / "pip-freeze.txt"
    freeze.write_bytes(FREEZE)
    out = tmp_path / "server_manifest.json"
    status = wm.main(
        [
            "--out",
            str(out),
            "--serve-argv-file",
            str(argv_file),
            "--requested-revision",
            "main",
            "--pip-freeze",
            str(freeze),
            "--api-key-file",
            str(key_file),
        ]
    )
    assert status == 0
    text = out.read_text(encoding="utf-8")
    assert API_KEY not in text
    payload = json.loads(text)
    assert list(payload) == list(SECTION_8_5_FIELDS)
    assert payload["serve_command"] == ARGV
    assert payload["resolved_revision"] == SHA
    assert payload["limit_mm_per_prompt"] == '{"image":10,"video":0}'


@pytest.mark.parametrize(
    "extra",
    [["--api-key", API_KEY], ["--api-key=" + API_KEY], ["--chat-template", "/tmp/" + API_KEY]],
)
def test_argv_carrying_a_key_is_refused(wm, tmp_path, extra):
    with pytest.raises(ValueError):
        manifest_from(wm, ARGV + extra)
    argv_file = tmp_path / "argv.json"
    argv_file.write_text(json.dumps(ARGV + extra), encoding="utf-8")
    key_file = tmp_path / "api_key"
    key_file.write_text(API_KEY + "\n", encoding="utf-8")
    out = tmp_path / "server_manifest.json"
    status = wm.main(["--out", str(out), "--serve-argv-file", str(argv_file), "--api-key-file", str(key_file)])
    assert status == 2
    assert not out.exists()


SERVE_COMMON = MODULE_PATH.parent / "serve_common.sh"


def _bash(script, env):
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, timeout=60, env=env, check=False
    )


def _prefix_with_python(tmp_path):
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").symlink_to(sys.executable)
    return prefix


def test_serve_argv_has_no_key_and_vllm_gets_it_from_env(tmp_path):
    prefix = _prefix_with_python(tmp_path)
    stub = prefix / "bin" / "vllm"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$@" > "$STUB_OUT/argv.txt"\n'
        'printf "%s" "${VLLM_API_KEY-unset}" > "$STUB_OUT/key.txt"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    script = f"""
set -euo pipefail
source {SERVE_COMMON}
IDEA4_ENV_PREFIX={prefix}
IDEA4_MODEL_ID={MODEL_ID}
IDEA4_RESOLVED_REVISION={SHA}
IDEA4_SERVED_MODEL=qwen3-vl-8b IDEA4_DTYPE=bfloat16 IDEA4_MAX_MODEL_LEN=32768
IDEA4_MAX_IMAGES=10 IDEA4_VIDEO_LIMIT=0 IDEA4_SEED=0 IDEA4_PORT=8000
IDEA4_TOKEN={API_KEY}
IDEA4_VLLM_LOG={tmp_path}/vllm.log
idea4_build_argv
printf '%s\\n' "${{SERVE_ARGV[@]}}" > {tmp_path}/built.txt
idea4_launch
wait "${{IDEA4_VLLM_PID}}"
printf '%s' "${{VLLM_API_KEY-unset}}" > {tmp_path}/parent_key.txt
"""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "STUB_OUT": str(tmp_path)}
    proc = _bash(script, env)
    assert proc.returncode == 0, proc.stderr
    built = (tmp_path / "built.txt").read_text().split("\n")
    launched = (tmp_path / "argv.txt").read_text().split("\n")
    for argv in (built, launched):
        assert API_KEY not in "\n".join(argv)
        assert not any(arg.startswith("--api-key") for arg in argv)
    assert "--limit-mm-per-prompt.image" in launched
    assert (tmp_path / "key.txt").read_text() == API_KEY
    assert (tmp_path / "parent_key.txt").read_text() == "unset"
    assert API_KEY not in proc.stdout + proc.stderr


@pytest.mark.parametrize("enforce_auth, expected", [(True, 0), (False, 1)])
def test_ready_check_requires_401_without_token(tmp_path, enforce_auth, expected):
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            authorized = self.headers.get("Authorization") == "Bearer " + API_KEY
            self.send_response(200 if authorized or not enforce_auth else 401)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        prefix = _prefix_with_python(tmp_path)
        token_file = tmp_path / "token"
        token_file.write_text(API_KEY + "\n", encoding="utf-8")
        script = f"""
source {SERVE_COMMON}
IDEA4_ENV_PREFIX={prefix}
IDEA4_TOKEN_FILE={token_file}
IDEA4_PORT={server.server_address[1]}
IDEA4_READY_SECONDS=20
IDEA4_VLLM_PID={os.getpid()}
idea4_wait_ready
"""
        proc = _bash(script, {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
    finally:
        server.shutdown()
    assert proc.returncode == expected, proc.stderr
    if not enforce_auth:
        assert "VLLM_API_KEY was not applied" in proc.stderr

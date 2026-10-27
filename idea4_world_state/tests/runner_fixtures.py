"""Shared helpers for the runner tests. They do not call a real model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idea4_world_state import contracts
from idea4_world_state.__main__ import main


PREPARE = contracts.PREPARE_DIR
SECTION_86_FREEZE_FIELDS = (
    "protocol_version",
    "frozen_at",
    "frozen_by",
    "prompt_files",
    "inventory_schema_sha256",
    "serializer_code_sha256",
    "settings",
    "model_identity",
    "server_manifest_sha256",
    "active_profile",
    "id_list_sha256",
    "dev_run_ids",
    "timing_run_ids",
    "pass_criteria_results",
    "timing_summary",
    "budget_inputs",
    "projection_hours",
    "decision",
    "rationale",
    "parser_policy",
    "prompt_revision_history",
)
RECORD_FIELDS = (
    "id",
    "category",
    "n_images",
    "partition",
    "role",
    "condition",
    "status",
    "pred",
    "parse_rule",
    "call_keys",
    "inventory_sha256",
    "serialization_sha256",
    "input_tokens",
    "output_tokens",
    "latency_seconds",
    "synthetic",
    "run_id",
)


def require_dataset():
    if not contracts.DEFAULT_DATASET.is_file():
        pytest.skip("MMSI parquet is not present")


def invoke(argv):
    """Run the CLI and return its exit code.

    ``run`` and ``freeze`` raise ``SystemExit``. ``analyze`` and ``audit``
    return normally on success and raise ``SystemExit`` on failure.
    """
    try:
        main(argv)
    except SystemExit as exc:
        return exc.code
    return 0


def run_argv(tmp, profile, conditions, out, *, mock=True, cache=None, freeze=None, resume=False, endpoint=None, manifest=None, concurrency=None, max_side=None, timeout=None):
    argv = [
        "run",
        "--prepare-dir", str(PREPARE),
        "--profile", profile,
        "--conditions", conditions,
        "--server-manifest", str(manifest or (tmp / "manifest-not-read.json")),
        "--out", str(out),
        "--cache", str(cache or (tmp / "cache")),
    ]
    if endpoint is not None:
        argv.extend(["--endpoint", endpoint])
    if concurrency is not None:
        argv.extend(["--concurrency", str(concurrency)])
    if max_side is not None:
        argv.extend(["--max-side", str(max_side)])
    if timeout is not None:
        argv.extend(["--timeout", str(timeout)])
    if freeze is not None:
        argv.extend(["--freeze", str(freeze)])
    if resume:
        argv.append("--resume")
    if mock:
        argv.append("--mock")
    return argv


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(run_dir):
    found = {}
    for path in sorted((Path(run_dir) / "records").glob("*.json")):
        record = read_json(path)
        found[(record["id"], record["condition"])] = record
    return found


def patch_latencies(run_dir):
    """Give raw calls ``n_images`` seconds and every other kind 1 second.

    Mock cache entries store latency 0. The budget test needs a positive fit
    so eval200 and eval100 have different service-unit costs.
    """
    config = read_json(Path(run_dir) / "run_config.json")
    cache = Path(config["cache_dir"])
    seen = set()
    for path in sorted((Path(run_dir) / "records").glob("*.json")):
        record = read_json(path)
        for kind, key in (record.get("call_keys") or {}).items():
            if key in seen:
                continue
            seen.add(key)
            entry_path = cache / key[:2] / f"{key}.json"
            entry = read_json(entry_path)
            if kind == "raw_answer":
                entry["latency_seconds"] = float(record["n_images"])
            else:
                entry["latency_seconds"] = 1.0
            entry_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def assert_no_gold_key(root):
    """No JSON object under ``root`` has a ``gold`` field."""
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            documents = [read_json(path)]
        elif path.suffix == ".jsonl":
            documents = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            continue
        for document in documents:
            _reject_gold(document, path)


def _reject_gold(value, path):
    if isinstance(value, dict):
        assert "gold" not in value, path
        for item in value.values():
            _reject_gold(item, path)
    elif isinstance(value, list):
        for item in value:
            _reject_gold(item, path)

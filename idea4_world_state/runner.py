"""Run the world-state profiles and freeze the protocol.

Section 8.6 is the record, cache, and freeze contract. Section 5 is the
calibration and budget decision. Pass thresholds that Section 11 still marks
as awaiting confirmation are module constants and are copied into
``protocol_freeze.json``.
"""

from __future__ import annotations

import csv
import getpass
import hashlib
import json
import math
import os
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from idea3_consistency.mmsi_consistency import parse_question
from idea4_visual_reasoning.data import images_of, pixel_hash, sha256_file
from idea4_world_state import contracts, prompts, serialize
from idea4_world_state.backend import (
    ConfigurationError,
    MockBackend,
    TransportError,
    VLLMBackend,
)
from idea4_world_state.calllog import CallLog, PauseRun, cache_path, identity_for_key
from idea4_world_state.inventory import InventoryError, atoms, parse_inventory_with_repairs, parser_policy

# Section 5 / Section 11 item 1. Confirmed only as proposed defaults.
DEV_SCHEMA_VALID_MINIMUM = 20
DEV_ITEM_DENOMINATOR = 22
DEV_EXTRACTION_TRUNCATED_MAXIMUM = 1
ANSWER_PARSE_RATE_MINIMUM = 1.0

# Section 11 item 3.
CONCURRENCY_SPEEDUP_MINIMUM = 1.15

# Section 5 budget. Startup is reserved inside each job, not added afterward.
# usable = walltime - startup_per_job; jobs = ceil(padded / usable).
# Modeled hours are padded inference plus startup per job. Queue delay is not
# part of that total.
PLANNING_ALLOWANCE = 1.3
STARTUP_HOURS_PER_JOB = 0.5
SU_PER_GPU_HOUR = {"l40s": 1, "h100": 2}

PROTOCOL_VERSION = "ws-freeze-1"
ENFORCED_PROFILES = ("raw_all", "eval200", "eval100")
EVAL_PROFILES = ("eval200", "eval100")
SETTING_KEYS = ("extract_tokens", "answer_tokens", "max_side", "timeout", "concurrency")
SHARED_SETTING_KEYS = ("extract_tokens", "answer_tokens", "max_side", "timeout")

# One server process, not the model or the inference settings. Resume may
# change these and must still reject every other manifest field.
SERVER_SESSION_FIELDS = ("created_at", "host", "slurm_job_id")
SERVER_SESSIONS_NAME = "server_sessions.jsonl"
ARTIFACT_FINGERPRINT_FIELDS = ("prompt_sha256", "inventory_schema_sha256", "serializer_code_sha256")
PROMPT_HASH_KEYS = ("raw", "extract", "flat", "graph")
_RERUN_PROFILES = {
    "dev run": "dev",
    "timing run": "timing",
    "synthetic run": "synthetic10",
    "concurrency run": "dev at concurrency 2",
}

SYNTHETIC10_ID = "synthetic10"
SYNTHETIC10_SIDE = 768
SYNTHETIC10_QUESTION = (
    "How many solid-color images are shown?\n"
    "Options: A: ten, B: two, C: one, D: zero"
)
SYNTHETIC10_COLORS = (
    (208, 48, 48),
    (48, 96, 208),
    (48, 160, 72),
    (160, 64, 176),
    (208, 144, 32),
    (32, 144, 160),
    (176, 48, 96),
    (96, 96, 48),
    (64, 64, 64),
    (232, 232, 232),
)
_GLYPHS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
}

FREEZE_FIELDS = (
    "protocol_version",
    "frozen_at",
    "frozen_by",
    "prompt_files",
    "prompt_sha256",
    "inventory_schema_sha256",
    "serializer_code_sha256",
    "parser_policy",
    "settings",
    "model_identity",
    "server_manifest_sha256",
    "active_profile",
    "id_list_sha256",
    "dev_run_ids",
    "timing_run_ids",
    "pass_criteria",
    "pass_criteria_results",
    "timing_summary",
    "budget_inputs",
    "projection_hours",
    "decision",
    "rationale",
    "prompt_revision_history",
)

RUN_DIRS = ("requests", "inputs", "inventories", "serializations", "records")
_PRINT_LOCK = threading.Lock()


class RunnerError(Exception):
    """A run or freeze failure that maps to a Section 8.6 exit code."""

    def __init__(self, message, exit_code=None):
        super().__init__(message)
        self.exit_code = contracts.EXIT_CODES["configuration_or_contract"] if exit_code is None else exit_code


def cli_run(args):
    try:
        code = run(args)
    except RunnerError as exc:
        print(f"run: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from None
    raise SystemExit(code)


def cli_freeze(args):
    try:
        code = freeze(args)
    except RunnerError as exc:
        print(f"freeze: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from None
    raise SystemExit(code)


def run(args):
    """Execute one profile. Returns 0. Transport pause raises ``RunnerError`` (75)."""
    _require_positive(args.extract_tokens, "extract-tokens")
    _require_positive(args.answer_tokens, "answer-tokens")
    _require_positive(args.max_side, "max-side")
    _require_positive(args.timeout, "timeout")
    _require_positive(args.concurrency, "concurrency")
    profile = args.profile
    conditions = _parse_conditions(args.conditions)
    if profile in ENFORCED_PROFILES and not args.mock and args.freeze is None:
        raise RunnerError(f"{profile} requires --freeze")
    selection = load_selection(args.prepare_dir)
    if profile == "synthetic10" and conditions != ["raw"]:
        raise RunnerError("synthetic10 runs the raw condition only")
    ids = profile_ids(selection, profile)
    run_dir = Path(args.out)
    backend = _open_backend(args)
    try:
        identity = backend.model_identity()
        freeze_sha, _freeze_doc = _enforce_freeze(args, selection, profile, identity)
        dataset_sha = _verified_dataset_sha256(selection)
        config = _build_run_config(
            args, selection, profile, ids, conditions, identity, dataset_sha, freeze_sha, backend_manifest(args)
        )
        _refuse_incompatible_config(run_dir, config)
        _refuse_without_resume(run_dir, args.resume)
        prepared = _prepare_items(selection, profile, ids, conditions)
        _write_config(run_dir, config)
        _record_server_session(run_dir, config)
        for name in RUN_DIRS:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        existing = _load_records(run_dir)
        _write_status(run_dir, "running", None, _counts(existing))
        log = CallLog(run_dir, config["cache_dir"], backend, identity, args.concurrency)
        pause_reason = _execute(run_dir, prepared, conditions, existing, config, log, args)
        records = _load_records(run_dir)
        if pause_reason is not None:
            _write_status(run_dir, "paused", pause_reason, _counts(records))
            raise RunnerError(f"paused: {pause_reason}", contracts.EXIT_CODES["paused"])
        pending = any(record.get("status") == "pending_transport" for record in records.values())
        state = "incomplete" if pending else "complete"
        _write_predictions(run_dir, config, records)
        _write_status(run_dir, state, None, _counts(records))
        return contracts.EXIT_CODES["complete"]
    finally:
        backend.close()


def freeze(args):
    """Write ``protocol_freeze.json``. Returns 0, or 2 after recording a shortfall."""
    deadline = _parse_deadline(args.deadline)
    _require_walltime(args.walltime_hours)
    if args.allocation_su < 0:
        raise RunnerError("--allocation-su must be non-negative")
    out = Path(args.out)
    if out.exists():
        raise RunnerError(f"refusing to overwrite {out}")
    dev = _load_run_bundle(args.dev_run, "dev run")
    timing = _load_run_bundle(args.timing_run, "timing run")
    synthetic = _load_run_bundle(args.synthetic_run, "synthetic run")
    concurrency = None
    if args.concurrency_run is not None:
        concurrency = _load_run_bundle(args.concurrency_run, "concurrency run")
    _require_same_protocol(dev, timing, synthetic, concurrency)
    results = _pass_criteria(dev, synthetic)
    if any(not row["passed"] for row in results):
        lines = ["development pass criteria failed"]
        for row in results:
            mark = "PASS" if row["passed"] else "FAIL"
            lines.append(f"  {row['name']}: {mark} {row['detail']}")
        raise RunnerError("\n".join(lines))
    projection = _projection(dev, timing, synthetic, concurrency)
    schedule = _schedule(projection, args.allocation_su, args.gpu, args.walltime_hours, deadline)
    decision = _choose_decision(schedule)
    document = _freeze_document(dev, timing, synthetic, concurrency, results, projection, schedule, decision, args)
    missing = [field for field in FREEZE_FIELDS if field not in document]
    if missing:
        raise RunnerError("freeze document is missing " + ", ".join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_text(out, _pretty(document))
    print(document["rationale"])
    if decision == "shortfall":
        return contracts.EXIT_CODES["configuration_or_contract"]
    return contracts.EXIT_CODES["complete"]


def load_selection(prepare_dir):
    """Load ``selection.json`` and verify ``manifest_sha256``."""
    path = Path(prepare_dir) / "selection.json"
    if not path.is_file():
        raise RunnerError(f"selection.json not found: {path}")
    document = _read_json(path, "selection.json")
    if not isinstance(document, dict) or "manifest_sha256" not in document:
        raise RunnerError("selection.json is missing manifest_sha256")
    body = {key: value for key, value in document.items() if key != "manifest_sha256"}
    digest = _canonical_sha256(body)
    if digest != document["manifest_sha256"]:
        raise RunnerError(
            "selection.json manifest_sha256 does not match the canonical document "
            f"({document['manifest_sha256']} != {digest})"
        )
    return document


def profile_ids(selection, profile):
    """Schedule order. ``synthetic10`` is generated, not stored in the selection."""
    if profile == "synthetic10":
        return [SYNTHETIC10_ID]
    profiles = selection.get("profiles")
    block = profiles.get(profile) if isinstance(profiles, dict) else None
    if not isinstance(block, dict) or not isinstance(block.get("ids"), list) or not block["ids"]:
        raise RunnerError(f"selection.json has no ids for profile {profile}")
    return list(block["ids"])


def synthetic_images():
    """Ten deterministic 768x768 PNGs. Index 1..10 is drawn on image 1..10."""
    return [synthetic_png(index) for index in range(1, 11)]


def synthetic_png(index):
    """One solid-color PNG with ``index`` drawn in a built-in bitmap font."""
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 10:
        raise RunnerError(f"synthetic image index {index} is out of range")
    color = SYNTHETIC10_COLORS[index - 1]
    ink = (0, 0, 0) if sum(color) > 500 else (255, 255, 255)
    image = Image.new("RGB", (SYNTHETIC10_SIDE, SYNTHETIC10_SIDE), color)
    pixels = image.load()
    scale = 20
    origin_x, origin_y = 64, 64
    for glyph_index, char in enumerate(str(index)):
        for row_index, row in enumerate(_GLYPHS[char]):
            for col_index, bit in enumerate(row):
                if bit != "1":
                    continue
                x0 = origin_x + glyph_index * (4 * scale) + col_index * scale
                y0 = origin_y + row_index * scale
                for dy in range(scale):
                    for dx in range(scale):
                        pixels[x0 + dx, y0 + dy] = ink
    buffer = _png_bytes(image)
    return buffer


RUN_CODE_FILES = (
    "idea3_consistency/mmsi_consistency.py",
    "idea4_visual_reasoning/data.py",
    "idea4_world_state/backend.py",
    "idea4_world_state/calllog.py",
    "idea4_world_state/contracts.py",
    "idea4_world_state/data.py",
    "idea4_world_state/inventory.py",
    "idea4_world_state/prompts.py",
    "idea4_world_state/runner.py",
    "idea4_world_state/serialize.py",
)


def code_sha256():
    """SHA-256 of canonical JSON ``{repo-relative path: file sha256}`` over ``RUN_CODE_FILES``.

    Selection.json's ``code_sha256`` hashes ``data.py`` only.
    """
    root = contracts.PACKAGE_DIR.parent
    digests = {name: sha256_file(root / name) for name in RUN_CODE_FILES}
    text = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def serializer_sha256():
    path = Path(serialize.__file__).resolve()
    if path.suffix == ".pyc":
        path = path.with_suffix(".py")
    return sha256_file(path)


def schema_sha256():
    return sha256_file(contracts.SCHEMA_PATH)


def pass_criteria_constants():
    """The Section 5 defaults, under the names freeze records."""
    return {
        "dev_schema_valid_minimum": DEV_SCHEMA_VALID_MINIMUM,
        "dev_item_denominator": DEV_ITEM_DENOMINATOR,
        "dev_extraction_truncated_maximum": DEV_EXTRACTION_TRUNCATED_MAXIMUM,
        "answer_parse_rate_minimum": ANSWER_PARSE_RATE_MINIMUM,
        "concurrency_speedup_minimum": CONCURRENCY_SPEEDUP_MINIMUM,
        "planning_allowance": PLANNING_ALLOWANCE,
        "startup_hours_per_job": STARTUP_HOURS_PER_JOB,
        "su_per_gpu_hour": dict(SU_PER_GPU_HOUR),
        "synthetic10_max_prompt_tokens": contracts.SYNTHETIC10_MAX_PROMPT_TOKENS,
    }


def budget_projection(dev_run, timing_run, synthetic_run, concurrency_run, gpu, walltime_hours, allocation_su, deadline):
    """Projection and per-profile schedule. ``deadline`` is ``YYYY-MM-DD`` or a datetime."""
    if isinstance(deadline, str):
        deadline = _parse_deadline(deadline)
    dev = dev_run if isinstance(dev_run, dict) and "config" in dev_run else _load_run_bundle(dev_run, "dev run")
    timing = timing_run if isinstance(timing_run, dict) and "config" in timing_run else _load_run_bundle(timing_run, "timing run")
    synthetic = (
        synthetic_run
        if isinstance(synthetic_run, dict) and "config" in synthetic_run
        else _load_run_bundle(synthetic_run, "synthetic run")
    )
    concurrency = None
    if concurrency_run is not None:
        concurrency = (
            concurrency_run
            if isinstance(concurrency_run, dict) and "config" in concurrency_run
            else _load_run_bundle(concurrency_run, "concurrency run")
        )
    projection = _projection(dev, timing, synthetic, concurrency)
    schedule = _schedule(projection, allocation_su, gpu, walltime_hours, deadline)
    return {"projection": projection, "schedule": schedule, "decision": _choose_decision(schedule)}


def _parse_conditions(text):
    parts = tuple(text.split(","))
    allowed = {tuple(choice.split(",")) for choice in contracts.CONDITION_CHOICES}
    if parts not in allowed:
        raise RunnerError(f"conditions must be one of {', '.join(contracts.CONDITION_CHOICES)}")
    return list(parts)


def _require_positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RunnerError(f"--{label} must be positive")
    if label != "timeout" and isinstance(value, float) and not value.is_integer():
        raise RunnerError(f"--{label} must be a positive integer")


def _open_backend(args):
    if args.mock:
        return MockBackend(seed=0)
    try:
        return VLLMBackend(
            endpoint=args.endpoint,
            served_model=args.served_model,
            api_key_env=args.api_key_env,
            timeout=args.timeout,
            server_manifest_path=args.server_manifest,
        )
    except TransportError as exc:
        raise RunnerError(f"backend startup failed: {exc}", contracts.EXIT_CODES["paused"]) from None
    except ConfigurationError as exc:
        raise RunnerError(str(exc)) from None


def backend_manifest(args):
    """The flat manifest object, or null for ``--mock`` (the file is not read)."""
    if args.mock:
        return None
    return _read_json(args.server_manifest, "server manifest")


def _enforce_freeze(args, selection, profile, identity):
    """Return ``(freeze_sha256 or None, document or None)``.

    Real ``raw_all`` / ``eval200`` / ``eval100`` runs require the flag. Mock
    runs enforce the same checks only when ``--freeze`` is present. Eval
    profiles also require the active profile and its id-list hash.
    """
    if args.freeze is None:
        return None, None
    path = Path(args.freeze)
    if not path.is_file():
        raise RunnerError(f"freeze file not found: {path}")
    document = _read_json(path, "protocol freeze")
    if not isinstance(document, dict):
        raise RunnerError("protocol freeze must be a JSON object")
    problems = []
    expected_prompts = prompts.prompt_hashes()
    found_prompts = document.get("prompt_sha256")
    if found_prompts != expected_prompts:
        problems.append("prompt hashes differ from the freeze")
    if document.get("inventory_schema_sha256") != schema_sha256():
        problems.append("inventory schema hash differs from the freeze")
    if document.get("serializer_code_sha256") != serializer_sha256():
        problems.append("serialize.py hash differs from the freeze")
    if document.get("parser_policy") != parser_policy():
        problems.append("parser policy differs from the freeze")
    settings = document.get("settings") if isinstance(document.get("settings"), dict) else {}
    actual = _settings_from_args(args)
    for key in SETTING_KEYS:
        if not _numbers_equal(settings.get(key), actual[key]):
            problems.append(f"settings.{key} differs from the freeze")
    try:
        same_identity = identity_for_key(document.get("model_identity")) == identity_for_key(identity)
    except ConfigurationError:
        same_identity = False
    if not same_identity:
        problems.append("model identity differs from the freeze")
    if profile in EVAL_PROFILES:
        if document.get("active_profile") != profile:
            problems.append(
                f"active profile {document.get('active_profile')!r} does not match this {profile} run"
            )
        expected_hash = _profile_hash(selection, profile)
        if document.get("id_list_sha256") != expected_hash:
            problems.append(f"{profile} id-list hash differs from the freeze")
    if problems:
        raise RunnerError("refusing run under this freeze: " + "; ".join(problems))
    return sha256_file(path), document


def _profile_hash(selection, profile):
    block = selection["profiles"][profile]
    stored = block.get("id_list_sha256")
    fresh = hashlib.sha256(json.dumps(list(block["ids"])).encode()).hexdigest()
    if stored != fresh:
        raise RunnerError(f"selection.json {profile} id_list_sha256 does not match its ids")
    return stored


def _settings_from_args(args):
    return {
        "extract_tokens": int(args.extract_tokens),
        "answer_tokens": int(args.answer_tokens),
        "max_side": int(args.max_side),
        "timeout": int(args.timeout),
        "concurrency": int(args.concurrency),
    }


def _verified_dataset_sha256(selection):
    path = contracts.DEFAULT_DATASET
    if not path.is_file():
        raise RunnerError(f"dataset not found: {path}")
    expected = selection.get("dataset_sha256")
    actual = sha256_file(path)
    if actual != expected:
        raise RunnerError(f"dataset sha256 {actual} does not match selection.json dataset_sha256 {expected}")
    pinned = contracts.DATASET_SHA256
    if actual != pinned:
        raise RunnerError(f"dataset sha256 {actual} does not match the pinned dataset")
    return actual


def _build_run_config(args, selection, profile, ids, conditions, identity, dataset_sha, freeze_sha, manifest):
    by_id = {item["id"]: item for item in selection.get("items") or []}
    items = []
    if profile == "synthetic10":
        items.append({
            "id": SYNTHETIC10_ID,
            "category": "synthetic",
            "n_images": 10,
            "partition": "synthetic",
            "role": "synthetic",
        })
    else:
        for item_id in ids:
            source = by_id.get(item_id)
            if source is None:
                raise RunnerError(f"profile id {item_id} is missing from selection.json items")
            items.append({
                "id": source["id"],
                "category": source["category"],
                "n_images": source["n_images"],
                "partition": source["partition"],
                "role": source["role"],
            })
    return {
        "profile": profile,
        "run_id": Path(args.out).name,
        "ids": list(ids),
        "items": items,
        "conditions": list(conditions),
        "settings": _settings_from_args(args),
        "server_manifest": manifest,
        "model_identity": identity,
        "prompt_sha256": prompts.prompt_hashes(),
        "cache_dir": str(Path(args.cache).resolve()),
        "dataset_path": str(contracts.DEFAULT_DATASET.resolve()),
        "dataset_sha256": dataset_sha,
        "prepare_manifest_sha256": selection["manifest_sha256"],
        "code_sha256": code_sha256(),
        "freeze_sha256": freeze_sha,
        "inventory_schema_sha256": schema_sha256(),
        "serializer_code_sha256": serializer_sha256(),
        "parser_policy": parser_policy(),
        "backend": "mock" if args.mock else "vllm",
        "synthetic": bool(args.mock or profile == "synthetic10"),
    }


def _protocol_manifest(manifest):
    """Manifest fields that must stay compatible across server sessions."""
    if not isinstance(manifest, dict):
        return manifest
    return {key: value for key, value in manifest.items() if key not in SERVER_SESSION_FIELDS}


def _stable_config(config):
    if not isinstance(config, dict):
        raise RunnerError("run_config.json must be an object")
    stable = dict(config)
    stable["server_manifest"] = _protocol_manifest(stable.get("server_manifest"))
    return stable


def _changed_fields(saved, current, prefix):
    if saved == current:
        return []
    if isinstance(saved, dict) and isinstance(current, dict):
        fields = []
        for key in sorted(set(saved) | set(current)):
            if saved.get(key) != current.get(key):
                fields.extend(_changed_fields(saved.get(key), current.get(key), f"{prefix}.{key}"))
        return fields
    return [prefix]


def _refuse_incompatible_config(run_dir, config):
    """Reject a resumed run whose model or inference settings changed.

    ``created_at``, ``host``, and ``slurm_job_id`` identify one server
    session. Every other manifest field, including the model revision,
    tokenizer revision, serving settings, dependency fingerprint, and GPU
    identity, stays part of the comparison. The saved ``run_config.json`` is
    not rewritten; a later session is appended to ``server_sessions.jsonl``.
    """
    path = Path(run_dir) / "run_config.json"
    if not path.is_file():
        return
    existing = _read_json(path, "run_config.json")
    saved = _stable_config(existing)
    current = _stable_config(config)
    if saved == current:
        return
    fields = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            fields.extend(_changed_fields(saved.get(key), current.get(key), key))
    listed = ", ".join(fields) if fields else "stable protocol"
    raise RunnerError(
        "run_config.json does not match this invocation (" + listed + "). "
        "A new server session may change only created_at, host, and slurm_job_id. "
        "Model revision and inference settings must match the saved run; "
        "use a new output directory."
    )


def _record_server_session(run_dir, config):
    """Append this invocation's manifest unless that exact session is already recorded.

    The first session also remains in ``run_config.json``. Later sessions are
    not written back over that object.
    """
    path = Path(run_dir) / SERVER_SESSIONS_NAME
    manifest = config.get("server_manifest")
    backend_name = config.get("backend")
    if path.is_file():
        for row in _session_rows(path):
            same_backend = row.get("backend") == backend_name
            same_manifest = _canonical(row.get("server_manifest")) == _canonical(manifest)
            if same_backend and same_manifest:
                return
    row = {
        "backend": backend_name,
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "server_manifest": manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row) + "\n")


def _session_rows(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise RunnerError(f"corrupt server session log at line {line_number}: {path}") from None
        if not isinstance(row, dict):
            raise RunnerError(f"corrupt server session log at line {line_number}: {path}")
        rows.append(row)
    return rows


def _refuse_without_resume(run_dir, resume):
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        if resume:
            raise RunnerError("--resume requires an existing run directory")
        return
    status_path = run_dir / "status.json"
    state = None
    if status_path.is_file():
        status = _read_json(status_path, "status.json")
        if isinstance(status, dict):
            state = status.get("state")
    if state == "complete":
        return
    if resume:
        return
    if state == "paused":
        raise RunnerError("status.json is paused; pass --resume to continue")
    label = state or "incomplete"
    raise RunnerError(f"status.json is {label}; pass --resume to continue")


def _prepare_items(selection, profile, ids, conditions):
    if profile == "synthetic10":
        parsed = parse_question(SYNTHETIC10_QUESTION)
        if parsed is None or len(parsed[1]) != 4:
            raise RunnerError("synthetic10 question is not a four-option question")
        return [{
            "id": SYNTHETIC10_ID,
            "category": "synthetic",
            "n_images": 10,
            "partition": "synthetic",
            "role": "synthetic",
            "question": SYNTHETIC10_QUESTION,
            "images": synthetic_images(),
        }]
    by_id = {item["id"]: item for item in selection["items"]}
    rows = _load_dataset_rows(set(ids))
    prepared = []
    for item_id in ids:
        source = by_id[item_id]
        row = rows.get(item_id)
        if row is None:
            raise RunnerError(f"dataset is missing id {item_id}")
        question = row.get("question")
        if not isinstance(question, str):
            raise RunnerError(f"question {item_id} is missing question text")
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if digest != source.get("question_sha256"):
            raise RunnerError(f"question {item_id} question_sha256 does not match selection.json")
        try:
            raws = images_of(row)
            hashes = [pixel_hash(raw) for raw in raws]
        except (ValueError, OSError, TypeError) as exc:
            raise RunnerError(f"question {item_id} images could not be verified: {exc}") from None
        if hashes != list(source.get("image_hashes") or []):
            raise RunnerError(f"question {item_id} image hashes do not match selection.json")
        if len(raws) != source.get("n_images"):
            raise RunnerError(f"question {item_id} image count does not match selection.json")
        prepared.append({
            "id": source["id"],
            "category": source["category"],
            "n_images": source["n_images"],
            "partition": source["partition"],
            "role": source["role"],
            "question": question,
            "images": raws,
        })
    return prepared


def _load_dataset_rows(needed):
    """Read only the inference columns. The answer column is never requested."""
    columns = list(contracts.INFERENCE_COLUMNS)
    if set(columns) - set(contracts.INFERENCE_COLUMNS):
        raise RunnerError("refusing to read parquet columns outside the inference set")
    found = {}
    parquet = pq.ParquetFile(contracts.DEFAULT_DATASET)
    for batch in parquet.iter_batches(batch_size=8, columns=columns):
        for row in batch.to_pylist():
            row_id = row.get("id")
            if row_id not in needed:
                continue
            if row_id in found:
                raise RunnerError(f"duplicate dataset id {row_id}")
            found[row_id] = row
    return found


def _execute(run_dir, prepared, conditions, existing, config, log, args):
    pause_reason = None
    pending_items = []
    for item in prepared:
        if _item_is_terminal(item["id"], conditions, existing):
            statuses = {condition: existing[(item["id"], condition)]["status"] for condition in conditions}
            _print_progress(bool(args.mock), item["id"], conditions, statuses)
            continue
        pending_items.append(item)

    def worker(item):
        return _process_item(item, conditions, existing, config, log)

    pending = iter(pending_items)
    futures = {}

    def submit_next(pool):
        nonlocal pause_reason
        if pause_reason is not None:
            return
        try:
            item = next(pending)
        except StopIteration:
            return
        futures[pool.submit(worker, item)] = item

    with ThreadPoolExecutor(max_workers=int(args.concurrency)) as pool:
        for _ in range(int(args.concurrency)):
            submit_next(pool)
        while futures:
            done, _not_done = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                outcome = future.result()
                _write_outcome(run_dir, outcome)
                if outcome["paused"] and pause_reason is None:
                    pause_reason = outcome["reason"]
                _print_progress(bool(args.mock), item["id"], conditions, outcome["statuses"])
            if pause_reason is None:
                submit_next(pool)
    return pause_reason


def _item_is_terminal(item_id, conditions, existing):
    return all(
        (item_id, condition) in existing
        and existing[(item_id, condition)].get("status") in contracts.TERMINAL_RECORD_STATUSES
        for condition in conditions
    )


def _process_item(item, conditions, existing, config, log):
    texts = {}
    records = []
    statuses = {}
    paused = False
    reason = None
    for condition in conditions:
        prior = existing.get((item["id"], condition))
        if prior is not None and prior.get("status") in contracts.TERMINAL_RECORD_STATUSES:
            statuses[condition] = prior["status"]

    def remember(record):
        records.append(record)
        statuses[record["condition"]] = record["status"]

    if "raw" in conditions and "raw" not in statuses:
        record, texts, paused, reason = _raw_condition(item, config, log, texts)
        remember(record)
        if paused:
            return _outcome(records, texts, statuses, True, reason)

    evidence = [condition for condition in ("flat", "graph") if condition in conditions and condition not in statuses]
    if evidence and not paused:
        produced, texts, paused, reason = _evidence_conditions(item, evidence, config, log, texts)
        for record in produced:
            remember(record)
    return _outcome(records, texts, statuses, paused, reason)


def _outcome(records, texts, statuses, paused, reason):
    return {
        "records": records,
        "texts": texts,
        "statuses": statuses,
        "paused": paused,
        "reason": reason or "transport failure",
    }


def _raw_condition(item, config, log, texts):
    try:
        messages = prompts.raw_messages(item["images"], item["question"], config["settings"]["max_side"])
    except (ValueError, OSError, TypeError) as exc:
        return _input_record(item, "raw", config, str(exc)), texts, False, None
    try:
        entry = log.call(messages, config["settings"]["answer_tokens"], "raw_answer")
    except PauseRun as exc:
        return _pending_record(item, "raw", config, {"raw_answer": exc.key}), texts, True, str(exc)
    return _answer_record(item, "raw", config, entry, {"raw_answer": entry["key"]}, None, None), texts, False, None


def _evidence_conditions(item, evidence, config, log, texts):
    produced = []
    try:
        messages = prompts.extract_messages(item["images"], item["question"], config["settings"]["max_side"])
    except (ValueError, OSError, TypeError) as exc:
        for condition in evidence:
            produced.append(_input_record(item, condition, config, str(exc)))
        return produced, texts, False, None
    try:
        entry = log.call(messages, config["settings"]["extract_tokens"], "extract")
    except PauseRun as exc:
        keys = {"extract": exc.key}
        for condition in evidence:
            produced.append(_pending_record(item, condition, config, keys))
        return produced, texts, True, str(exc)
    extract_key = entry["key"]
    if entry.get("status") == "truncated":
        for condition in evidence:
            produced.append(_extraction_record(item, condition, config, "extraction_truncated", entry, {"extract": extract_key}))
        return produced, texts, False, None
    try:
        inventory, repairs = parse_inventory_with_repairs(entry.get("text") or "", item["n_images"])
    except InventoryError as exc:
        texts[f"inventories/{item['id']}.error.json"] = _pretty({
            "message": exc.message,
            "rule": exc.rule,
            "text": entry.get("text") or "",
        })
        for condition in evidence:
            produced.append(_extraction_record(
                item, condition, config, "extraction_invalid", entry, {"extract": extract_key}
            ))
        return produced, texts, False, None
    inventory_text = _pretty(inventory)
    inventory_hash = hashlib.sha256(inventory_text.encode("utf-8")).hexdigest()
    texts[f"inventories/{item['id']}.json"] = inventory_text
    texts[f"repairs/{item['id']}.json"] = _pretty(repairs)
    rendered = {
        "flat": serialize.render_flat(inventory),
        "graph": serialize.render_graph(inventory),
    }
    hashes = {}
    for condition, text in rendered.items():
        texts[f"serializations/{item['id']}.{condition}.txt"] = text
        hashes[condition] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    for condition in evidence:
        if condition not in ("flat", "graph"):
            produced.append(_input_record(item, condition, config, f"unknown evidence condition {condition}"))
            continue
        builder = prompts.flat_messages if condition == "flat" else prompts.graph_messages
        try:
            messages = builder(item["images"], item["question"], inventory, config["settings"]["max_side"])
        except (ValueError, OSError, TypeError) as exc:
            record = _input_record(item, condition, config, str(exc))
            record["call_keys"] = {"extract": extract_key}
            record["inventory_sha256"] = inventory_hash
            produced.append(record)
            continue
        kind = f"{condition}_answer"
        try:
            answer = log.call(messages, config["settings"]["answer_tokens"], kind)
        except PauseRun as exc:
            produced.append(_pending_record(
                item, condition, config, {"extract": extract_key, kind: exc.key}
            ))
            for later in evidence[evidence.index(condition) + 1:]:
                produced.append(_pending_record(item, later, config, {"extract": extract_key}))
            return produced, texts, True, str(exc)
        produced.append(_answer_record(
            item,
            condition,
            config,
            answer,
            {"extract": extract_key, kind: answer["key"]},
            inventory_hash,
            hashes[condition],
        ))
    return produced, texts, False, None


def _base_record(item, condition, config):
    return {
        "id": item["id"],
        "category": item["category"],
        "n_images": item["n_images"],
        "partition": item["partition"],
        "role": item["role"],
        "condition": condition,
        "status": None,
        "pred": None,
        "parse_rule": None,
        "call_keys": {},
        "inventory_sha256": None,
        "serialization_sha256": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_seconds": None,
        "synthetic": config["synthetic"],
        "run_id": config["run_id"],
    }


def _with_usage(record, entry):
    if entry is None:
        return record
    record["input_tokens"] = entry.get("input_tokens")
    record["output_tokens"] = entry.get("output_tokens")
    record["latency_seconds"] = entry.get("latency_seconds")
    return record


def _input_record(item, condition, config, _message):
    """``input_error`` stores no free-text cause. Section 8.6 has no message field."""
    record = _base_record(item, condition, config)
    record["status"] = "input_error"
    return record


def _pending_record(item, condition, config, keys):
    record = _base_record(item, condition, config)
    record["status"] = "pending_transport"
    record["call_keys"] = dict(keys)
    return record


def _extraction_record(item, condition, config, status, entry, keys):
    record = _base_record(item, condition, config)
    record["status"] = status
    record["call_keys"] = dict(keys)
    return _with_usage(record, entry)


def _answer_record(item, condition, config, entry, keys, inventory_hash, serialization_hash):
    record = _base_record(item, condition, config)
    record["call_keys"] = dict(keys)
    record["inventory_sha256"] = inventory_hash
    record["serialization_sha256"] = serialization_hash
    _with_usage(record, entry)
    if entry.get("status") == "truncated":
        record["status"] = "answer_truncated"
        return record
    pred, rule = prompts.parse_answer(entry.get("text"))
    if pred is None:
        record["status"] = "answer_parse_error"
        return record
    record["status"] = "ok"
    record["pred"] = pred
    record["parse_rule"] = rule
    return record


def _write_outcome(run_dir, outcome):
    run_dir = Path(run_dir)
    for relative, text in outcome["texts"].items():
        _atomic_text(run_dir / relative, text)
    for record in outcome["records"]:
        _atomic_text(run_dir / "records" / f"{record['id']}__{record['condition']}.json", _pretty(record))


def _write_config(run_dir, config):
    path = Path(run_dir) / "run_config.json"
    if path.is_file():
        return
    _atomic_text(path, _pretty(config))


def _write_status(run_dir, state, reason, counts):
    _atomic_text(Path(run_dir) / "status.json", _pretty({
        "counts": counts,
        "reason": reason,
        "state": state,
    }))


def _counts(records):
    counts = {status: 0 for status in contracts.RECORD_STATUSES}
    for record in records.values():
        status = record.get("status")
        if status not in counts:
            counts[status] = 0
        counts[status] += 1
    return counts


def _load_records(run_dir):
    directory = Path(run_dir) / "records"
    found = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        record = _read_json(path, f"record {path.name}")
        if not isinstance(record, dict) or "id" not in record or "condition" not in record:
            raise RunnerError(f"record {path.name} is missing id or condition")
        found[(record["id"], record["condition"])] = record
    return found


def _write_predictions(run_dir, config, records):
    rows = []
    identity = config.get("model_identity") or {}
    revision = identity.get("resolved_revision")
    max_side = (config.get("settings") or {}).get("max_side")
    prompts_hash = config.get("prompt_sha256") or {}
    cache_dir = _cache_dir(run_dir, config)
    for item_id in config["ids"]:
        for condition in config["conditions"]:
            record = records.get((item_id, condition))
            if record is None:
                continue
            request_key = _request_key(record)
            served = None
            if request_key:
                entry_path = cache_path(cache_dir, request_key)
                if entry_path.is_file():
                    entry = _read_json(entry_path, f"cache entry {request_key}")
                    served = entry.get("served_model") if isinstance(entry, dict) else None
            rows.append({
                "id": record["id"],
                "category": record["category"],
                "n_images": record["n_images"],
                "partition": record["partition"],
                "role": record["role"],
                "condition": record["condition"],
                "pred": record.get("pred"),
                "status": record.get("status"),
                "parse_rule": record.get("parse_rule"),
                "request_key": request_key,
                "served_model": served,
                "model_revision": revision,
                "max_side": max_side,
                "answer_prompt_sha256": prompts_hash.get(condition),
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "latency_seconds": record.get("latency_seconds"),
                "run_id": record.get("run_id"),
                "synthetic": record.get("synthetic"),
            })
    jsonl = "".join(json.dumps({key: row[key] for key in contracts.PREDICTION_COLUMNS}, ensure_ascii=False) + "\n" for row in rows)
    _atomic_text(Path(run_dir) / "predictions.jsonl", jsonl)
    lines = [_csv_line(list(contracts.PREDICTION_COLUMNS))]
    for row in rows:
        lines.append(_csv_line([_csv_cell(row[column]) for column in contracts.PREDICTION_COLUMNS]))
    _atomic_text(Path(run_dir) / "predictions.csv", "".join(lines))


def _request_key(record):
    keys = record.get("call_keys") or {}
    if not isinstance(keys, dict):
        return None
    condition = record.get("condition")
    if condition == "raw":
        return keys.get("raw_answer")
    answer = keys.get(f"{condition}_answer")
    if answer:
        return answer
    return keys.get("extract")


def _csv_line(cells):
    from io import StringIO
    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(cells)
    return buffer.getvalue()


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _print_progress(mock, item_id, conditions, statuses):
    body = " ".join(f"{condition}={statuses.get(condition, 'missing')}" for condition in conditions)
    line = f"MOCK {item_id} {body}" if mock else f"{item_id} {body}"
    with _PRINT_LOCK:
        print(line, flush=True)


def _load_run_bundle(run_dir, label):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunnerError(f"{label} directory not found: {run_dir}")
    config = _read_json(run_dir / "run_config.json", f"{label} run_config.json")
    if not isinstance(config, dict):
        raise RunnerError(f"{label} run_config.json must be an object")
    status = _read_json(run_dir / "status.json", f"{label} status.json")
    if not isinstance(status, dict) or status.get("state") != "complete":
        raise RunnerError(f"{label} is not complete")
    records = _load_records(run_dir)
    return {"dir": run_dir, "config": config, "status": status, "records": records, "label": label}


def _require_same_protocol(dev, timing, synthetic, concurrency):
    _require_profile(dev, "dev", "dev run")
    _require_profile(timing, "timing", "timing run")
    _require_profile(synthetic, "synthetic10", "synthetic run")
    _require_conditions(dev, ("raw", "flat", "graph"), "dev run")
    _require_conditions(timing, ("raw",), "timing run")
    _require_conditions(synthetic, ("raw",), "synthetic run")
    if int(dev["config"]["settings"]["concurrency"]) != 1:
        raise RunnerError("dev run must use concurrency 1; pass the concurrency-2 run as --concurrency-run")
    if int(timing["config"]["settings"]["concurrency"]) != 1 or int(synthetic["config"]["settings"]["concurrency"]) != 1:
        raise RunnerError("timing and synthetic runs must use concurrency 1")
    for other in (timing, synthetic):
        _require_match(dev, other, ignore_concurrency=False)
    if concurrency is not None:
        _require_profile(concurrency, "dev", "concurrency run")
        _require_conditions(concurrency, ("raw", "flat", "graph"), "concurrency run")
        _require_match(dev, concurrency, ignore_concurrency=True)
    manifests = {dev["config"].get("prepare_manifest_sha256")}
    for bundle in (timing, synthetic, concurrency):
        if bundle is not None:
            manifests.add(bundle["config"].get("prepare_manifest_sha256"))
    if len(manifests) != 1:
        raise RunnerError("input runs do not share prepare_manifest_sha256")
    selection_path = contracts.PREPARE_DIR / "selection.json"
    if not selection_path.is_file():
        raise RunnerError(f"selection.json not found: {selection_path}")
    selection = load_selection(contracts.PREPARE_DIR)
    if selection["manifest_sha256"] not in manifests:
        raise RunnerError("input runs do not match artifacts/prepare_v1/selection.json")
    _require_tested_artifacts(dev, timing, synthetic, concurrency)
    _require_parser_policy(dev, timing, synthetic, concurrency)


def _require_parser_policy(dev, timing, synthetic, concurrency):
    """Freeze only runs whose recorded parser behavior is the current one.

    ``medium`` on a repaired ``cross_view`` is ``parser_default``, not a model
    judgment. A run without ``parser_policy`` was produced before that record
    existed and cannot be frozen.
    """
    current = parser_policy()
    for bundle in (dev, timing, synthetic, concurrency):
        if bundle is None:
            continue
        found = bundle["config"].get("parser_policy")
        if found != current:
            raise RunnerError(
                f"{bundle['label']} parser_policy does not match the current parser. "
                "Reusing an entity id across images fills cross_view with status same "
                "and uncertainty medium, and that uncertainty is a parser default."
            )


def _require_profile(bundle, profile, label):
    if bundle["config"].get("profile") != profile:
        raise RunnerError(f"{label} profile is {bundle['config'].get('profile')!r}, expected {profile}")


def _require_conditions(bundle, required, label):
    found = bundle["config"].get("conditions") or []
    missing = [condition for condition in required if condition not in found]
    if missing:
        raise RunnerError(f"{label} is missing conditions: {', '.join(missing)}")


def _require_match(reference, other, ignore_concurrency):
    try:
        same = identity_for_key(reference["config"].get("model_identity")) == identity_for_key(other["config"].get("model_identity"))
    except ConfigurationError as exc:
        raise RunnerError(f"{other['label']} model identity is unusable: {exc}") from None
    if not same:
        raise RunnerError(f"{other['label']} model identity does not match the dev run")
    keys = SHARED_SETTING_KEYS if ignore_concurrency else SETTING_KEYS
    left = reference["config"].get("settings") or {}
    right = other["config"].get("settings") or {}
    for key in keys:
        if not _numbers_equal(left.get(key), right.get(key)):
            raise RunnerError(f"{other['label']} settings.{key} does not match the dev run")


def _pass_criteria(dev, synthetic):
    dev_ids = list(dev["config"]["ids"])
    by_item = {}
    for (item_id, condition), record in dev["records"].items():
        by_item.setdefault(item_id, {})[condition] = record
    transport = sum(1 for row in _attempt_rows(dev["dir"]) if row.get("outcome") == "transport_error")
    valid = 0
    truncated = 0
    for item_id in dev_ids:
        conditions = by_item.get(item_id, {})
        evidence = [conditions.get(name) for name in ("flat", "graph") if name in (dev["config"].get("conditions") or [])]
        statuses = {record.get("status") for record in evidence if record is not None}
        if "extraction_truncated" in statuses:
            truncated += 1
        inventory_path = dev["dir"] / "inventories" / f"{item_id}.json"
        if inventory_path.is_file() and "extraction_truncated" not in statuses and "extraction_invalid" not in statuses:
            valid += 1
    raw_ok = 0
    raw_n = 0
    for item_id in dev_ids:
        record = by_item.get(item_id, {}).get("raw")
        raw_n += 1
        if record is not None and record.get("status") == "ok" and record.get("pred"):
            raw_ok += 1
    evidence_ok = 0
    evidence_n = 0
    for item_id in dev_ids:
        inventory_path = dev["dir"] / "inventories" / f"{item_id}.json"
        if not inventory_path.is_file():
            continue
        for condition in ("flat", "graph"):
            record = by_item.get(item_id, {}).get(condition)
            if record is None or record.get("status") in ("extraction_invalid", "extraction_truncated"):
                continue
            evidence_n += 1
            if record.get("status") == "ok" and record.get("pred"):
                evidence_ok += 1
    atom_ok = 0
    atom_n = 0
    atom_problem = None
    for path in sorted((dev["dir"] / "inventories").glob("*.json")):
        if path.name.endswith(".error.json"):
            continue
        atom_n += 1
        item_id = path.name[:-5]
        n_images = _n_images(dev, item_id)
        try:
            document, _repairs = parse_inventory_with_repairs(path.read_text(encoding="utf-8"), n_images)
            expected = atoms(document)
            same = serialize.parse_flat(serialize.render_flat(document)) == expected == serialize.parse_graph(serialize.render_graph(document))
        except (OSError, UnicodeError, InventoryError, ValueError) as exc:
            same = False
            atom_problem = f"{path.name}: {exc}"
        if same:
            atom_ok += 1
        elif atom_problem is None:
            atom_problem = path.name
    synthetic_row = _synthetic_criterion(synthetic)
    raw_rate = (raw_ok / raw_n) if raw_n else 0.0
    evidence_rate = (evidence_ok / evidence_n) if evidence_n else 1.0
    return [
        _criterion(
            "dev_transport",
            transport == 0,
            f"observed={transport} limit=0 (no transport_error in attempts.jsonl)",
        ),
        _criterion(
            "extraction_schema_valid",
            valid >= DEV_SCHEMA_VALID_MINIMUM,
            f"observed={valid} minimum={DEV_SCHEMA_VALID_MINIMUM} of {DEV_ITEM_DENOMINATOR}",
        ),
        _criterion(
            "extraction_truncated",
            truncated <= DEV_EXTRACTION_TRUNCATED_MAXIMUM,
            f"observed={truncated} maximum={DEV_EXTRACTION_TRUNCATED_MAXIMUM} of {DEV_ITEM_DENOMINATOR}",
        ),
        _criterion(
            "raw_answer_parsing",
            raw_n > 0 and raw_rate + 1e-12 >= ANSWER_PARSE_RATE_MINIMUM,
            f"observed={raw_ok}/{raw_n} minimum={ANSWER_PARSE_RATE_MINIMUM}",
        ),
        _criterion(
            "evidence_answer_parsing",
            evidence_rate + 1e-12 >= ANSWER_PARSE_RATE_MINIMUM,
            f"observed={evidence_ok}/{evidence_n} minimum={ANSWER_PARSE_RATE_MINIMUM} where extraction succeeded",
        ),
        _criterion(
            "fact_atom_equivalence",
            atom_n > 0 and atom_ok == atom_n,
            f"observed={atom_ok}/{atom_n}" + (f" failed={atom_problem}" if atom_problem else ""),
        ),
        synthetic_row,
    ]


def _criterion(name, passed, detail):
    return {"name": name, "passed": bool(passed), "detail": detail}


def _current_artifact_fingerprints():
    return {
        "prompt_sha256": prompts.prompt_hashes(),
        "inventory_schema_sha256": schema_sha256(),
        "serializer_code_sha256": serializer_sha256(),
    }


def _is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in value)


def _missing_fingerprint_names(config):
    missing = []
    prompts_hash = config.get("prompt_sha256") if isinstance(config, dict) else None
    if not isinstance(prompts_hash, dict) or any(not _is_sha256(prompts_hash.get(key)) for key in PROMPT_HASH_KEYS):
        missing.append("prompt_sha256")
    for field in ("inventory_schema_sha256", "serializer_code_sha256"):
        value = config.get(field) if isinstance(config, dict) else None
        if not _is_sha256(value):
            missing.append(field)
    return missing


def _fingerprint_detail(field, found, expected):
    if field != "prompt_sha256" or not isinstance(found, dict) or not isinstance(expected, dict):
        return field
    keys = [key for key in PROMPT_HASH_KEYS if found.get(key) != expected.get(key)]
    if not keys:
        return field
    return "prompt_sha256." + ",".join(keys)


def _rerun_name(label):
    return _RERUN_PROFILES.get(label, label)


def _rerun_list(bundles):
    names = [_rerun_name(bundle["label"]) for bundle in bundles]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + ", and " + names[-1]


def _require_tested_artifacts(dev, timing, synthetic, concurrency):
    """Require every input run to have recorded the artifacts freeze would certify.

    Missing fingerprints are not copied from the current files. A change after
    development has to be rerun; mixed prompt, schema, or serializer versions
    are not frozen together.
    """
    bundles = [bundle for bundle in (dev, timing, synthetic, concurrency) if bundle is not None]
    for bundle in bundles:
        missing = _missing_fingerprint_names(bundle["config"])
        if missing:
            profile = _rerun_name(bundle["label"])
            raise RunnerError(
                f"{bundle['label']} is missing {', '.join(missing)}. "
                "Freeze does not certify a run that did not record these fingerprints, "
                "and it does not fill them in from the current files. "
                f"Rerun {profile} into a new output directory before freeze."
            )
    for bundle in bundles[1:]:
        for field in ARTIFACT_FINGERPRINT_FIELDS:
            found = bundle["config"][field]
            expected = dev["config"][field]
            if found != expected:
                detail = _fingerprint_detail(field, found, expected)
                profile = _rerun_name(bundle["label"])
                raise RunnerError(
                    f"{bundle['label']} {detail} does not match the dev run. "
                    "Development, timing, and synthetic runs must record one tested protocol. "
                    f"Rerun {profile} so its prompt, inventory schema, and serializer "
                    "fingerprints match the dev run before freeze."
                )
    current = _current_artifact_fingerprints()
    for field in ARTIFACT_FINGERPRINT_FIELDS:
        found = dev["config"][field]
        expected = current[field]
        if found != expected:
            detail = _fingerprint_detail(field, found, expected)
            profiles = _rerun_list(bundles)
            raise RunnerError(
                f"current {detail} does not match the tested runs. "
                "The artifact changed after those runs. "
                f"Rerun {profiles} before freeze."
            )


def _synthetic_criterion(synthetic):
    records = [record for (item_id, condition), record in synthetic["records"].items() if condition == "raw"]
    limit = contracts.SYNTHETIC10_MAX_PROMPT_TOKENS
    if len(records) != 1 or records[0].get("status") != "ok":
        return _criterion("synthetic10", False, f"expected one raw ok record, found {len(records)}")
    record = records[0]
    key = (record.get("call_keys") or {}).get("raw_answer")
    if not key:
        return _criterion("synthetic10", False, "raw record has no raw_answer cache key")
    entry = _read_cache(synthetic, key)
    tokens = entry.get("input_tokens")
    if tokens is None:
        config = synthetic["config"] if isinstance(synthetic["config"], dict) else {}
        passed = config.get("backend") == "mock"
        return _criterion("synthetic10", passed, _missing_token_detail(config, limit))
    if isinstance(tokens, bool) or not isinstance(tokens, (int, float)):
        return _criterion("synthetic10", False, f"prompt_tokens={tokens!r} maximum={limit}")
    numeric = float(tokens)
    if not math.isfinite(numeric) or numeric < 0:
        return _criterion("synthetic10", False, f"prompt_tokens={tokens!r} maximum={limit}")
    passed = numeric <= limit
    detail = f"prompt_tokens={numeric} maximum={limit}"
    if synthetic["config"].get("backend") == "mock":
        detail += " (mock backend reported a token count; the mock exemption applies only to missing usage)"
    return _criterion("synthetic10", passed, detail)


def _missing_token_detail(config, limit):
    backend_name = config.get("backend") if isinstance(config, dict) else None
    if backend_name == "mock":
        return (
            f"prompt_tokens=null maximum={limit} "
            "(mock backend exemption; a real synthetic10 response is not exempt)"
        )
    if backend_name is None:
        return (
            f"prompt_tokens=null maximum={limit} "
            "(run_config.json has no backend field; missing usage is not a mock exemption)"
        )
    return (
        f"prompt_tokens=null maximum={limit} "
        f"(backend {backend_name!r} must report a prompt-token count; "
        "synthetic test images are not a usage exemption)"
    )


def _projection(dev, timing, synthetic, concurrency):
    points = _raw_points(dev, "dev") + _raw_points(timing, "timing") + _raw_points(synthetic, "synthetic10")
    grouped = {}
    for point in points:
        grouped.setdefault(point["n_images"], []).append(point["latency_seconds"])
    fit_points = [(n_images, sum(values) / len(values)) for n_images, values in sorted(grouped.items())]
    if len(fit_points) < 2:
        raise RunnerError("raw latency fit needs at least two image-count points")
    intercept, slope = _fit_line(fit_points)
    means = {
        "extract": _mean_kind(dev, "extract"),
        "flat_answer": _mean_kind(dev, "flat_answer"),
        "graph_answer": _mean_kind(dev, "graph_answer"),
    }
    evidence_mean = means["extract"] + means["flat_answer"] + means["graph_answer"]
    selection = load_selection(contracts.PREPARE_DIR)
    raw_total = 0.0
    for item in selection["items"]:
        raw_total += max(0.0, intercept + slope * item["n_images"])
    sizes = {}
    for name in ("eval200", "eval100"):
        n_items = len(selection["profiles"][name]["ids"])
        sizes[name] = raw_total + n_items * evidence_mean
    speedup, concurrency_note = _concurrency_decision(dev, concurrency)
    selected = 2 if speedup is not None else 1
    if speedup is not None:
        sizes = {name: value / speedup for name, value in sizes.items()}
    return {
        "points": points,
        "fit_points": [{"n_images": n_images, "latency_seconds": latency} for n_images, latency in fit_points],
        "aggregation": "mean raw latency at each n_images",
        "intercept": intercept,
        "slope": slope,
        "mean_latency_seconds": means,
        "raw_seconds_over_selection": raw_total,
        "evidence_seconds_per_item": evidence_mean,
        "sequential_seconds": {name: raw_total + len(selection["profiles"][name]["ids"]) * evidence_mean for name in ("eval200", "eval100")},
        "projected_seconds": sizes,
        "concurrency": selected,
        "speedup": speedup,
        "concurrency_note": concurrency_note,
        "n_items": {name: len(selection["profiles"][name]["ids"]) for name in ("eval200", "eval100", "raw_all")},
    }


def _raw_points(bundle, source):
    points = []
    for (item_id, condition), record in bundle["records"].items():
        if condition != "raw":
            continue
        key = (record.get("call_keys") or {}).get("raw_answer")
        if not key:
            raise RunnerError(f"{bundle['label']} raw record {item_id} has no raw_answer key")
        latency = _read_cache(bundle, key).get("latency_seconds")
        if latency is None:
            raise RunnerError(f"{bundle['label']} raw record {item_id} has no latency")
        points.append({
            "source": source,
            "id": item_id,
            "n_images": record["n_images"],
            "latency_seconds": float(latency),
        })
    if not points:
        raise RunnerError(f"{bundle['label']} has no raw latencies")
    return points


def _mean_kind(bundle, kind):
    seen = set()
    values = []
    for record in bundle["records"].values():
        key = (record.get("call_keys") or {}).get(kind)
        if not key or key in seen:
            continue
        seen.add(key)
        latency = _read_cache(bundle, key).get("latency_seconds")
        if latency is None:
            raise RunnerError(f"{bundle['label']} cache entry {key} has no latency")
        values.append(float(latency))
    if not values:
        raise RunnerError(f"{bundle['label']} has no {kind} latencies")
    return sum(values) / len(values)


def _fit_line(points):
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    n_points = len(xs)
    mean_x = sum(xs) / n_points
    mean_y = sum(ys) / n_points
    var_x = sum((value - mean_x) ** 2 for value in xs)
    if var_x == 0:
        return mean_y, 0.0
    cov = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys))
    slope = cov / var_x
    return mean_y - slope * mean_x, slope


def _concurrency_decision(dev, concurrency):
    note = (
        "The server log out-of-memory and preemption check is a human confirmation; "
        "freeze does not read the server log."
    )
    if concurrency is None:
        return None, "Concurrency 2 was not measured. " + note
    failures = sum(1 for row in _attempt_rows(concurrency["dir"]) if row.get("outcome") == "transport_error")
    sequential = _sequential_seconds(dev)
    wall = _wall_seconds(concurrency)
    n_items = len(dev["config"]["ids"])
    if failures:
        return None, f"Concurrency run has {failures} transport_error attempts, so concurrency stays 1. " + note
    if sequential <= 0 or wall is None or wall <= 0:
        return None, "Concurrency speedup is undefined because measured duration is zero, so concurrency stays 1. " + note
    sequential_rate = n_items / (sequential / 3600.0)
    concurrent_rate = len(concurrency["config"]["ids"]) / (wall / 3600.0)
    if concurrent_rate + 1e-9 < CONCURRENCY_SPEEDUP_MINIMUM * sequential_rate:
        return None, (
            f"Concurrency items/hour {concurrent_rate:.4f} is below "
            f"{CONCURRENCY_SPEEDUP_MINIMUM} x sequential {sequential_rate:.4f}, so concurrency stays 1. " + note
        )
    return concurrent_rate / sequential_rate, (
        f"Selected concurrency 2: items/hour {concurrent_rate:.4f} versus sequential {sequential_rate:.4f}. " + note
    )


def _sequential_seconds(bundle):
    seen = set()
    total = 0.0
    for record in bundle["records"].values():
        keys = record.get("call_keys") or {}
        if not isinstance(keys, dict):
            continue
        for key in keys.values():
            if key in seen:
                continue
            seen.add(key)
            latency = _read_cache(bundle, key).get("latency_seconds")
            if latency is None:
                raise RunnerError(f"{bundle['label']} cache entry {key} has no latency")
            total += float(latency)
    return total


def _wall_seconds(bundle):
    rows = [row for row in _attempt_rows(bundle["dir"]) if row.get("outcome") in contracts.TERMINAL_ATTEMPT_OUTCOMES]
    if not rows:
        return None
    spans = []
    for row in rows:
        started = _parse_timestamp(row.get("started_at"))
        if started is None:
            continue
        latency = row.get("latency_seconds") or 0
        spans.append((started, started + timedelta(seconds=float(latency))))
    if not spans:
        return None
    start = min(pair[0] for pair in spans)
    end = max(pair[1] for pair in spans)
    return max(0.0, (end - start).total_seconds())


def _require_walltime(walltime_hours):
    if isinstance(walltime_hours, bool) or not isinstance(walltime_hours, (int, float)):
        raise RunnerError("--walltime-hours must be a positive number")
    if walltime_hours <= 0:
        raise RunnerError("--walltime-hours must be positive")
    if walltime_hours <= STARTUP_HOURS_PER_JOB:
        raise RunnerError(
            f"--walltime-hours ({walltime_hours}) must exceed the {STARTUP_HOURS_PER_JOB} h "
            "startup allowance. Startup consumes part of each job's wall-time limit, so a job "
            "this short has no usable inference time."
        )


def _schedule(projection, allocation_su, gpu, walltime_hours, deadline):
    _require_walltime(walltime_hours)
    usable = walltime_hours - STARTUP_HOURS_PER_JOB
    rate = SU_PER_GPU_HOUR[gpu]
    profiles = {}
    for name in ("eval200", "eval100"):
        projected_hours = projection["projected_seconds"][name] / 3600.0
        padded = PLANNING_ALLOWANCE * projected_hours
        jobs = math.ceil(padded / usable) if padded > 0 else 0
        startup = STARTUP_HOURS_PER_JOB * jobs
        hours = padded + startup
        su = hours * rate
        finish = datetime.now(timezone.utc) + timedelta(hours=hours)
        profiles[name] = {
            "projection_hours": projected_hours,
            "padded_hours": padded,
            "jobs": jobs,
            "startup_hours": startup,
            "hours_with_allowance": hours,
            "su": su,
            "fits_allocation": su <= allocation_su,
            "fits_deadline": finish <= deadline,
        }
    return {
        "allocation_su": allocation_su,
        "gpu": gpu,
        "su_per_gpu_hour": rate,
        "walltime_hours": walltime_hours,
        "usable_inference_hours_per_job": usable,
        "queue_delay_hours": None,
        "deadline": deadline.date().isoformat(),
        "profiles": profiles,
    }


def _choose_decision(schedule):
    for name in ("eval200", "eval100"):
        profile = schedule["profiles"][name]
        if profile["fits_allocation"] and profile["fits_deadline"]:
            return name
    return "shortfall"


def _freeze_document(dev, timing, synthetic, concurrency, results, projection, schedule, decision, args):
    selection = load_selection(contracts.PREPARE_DIR)
    active = None if decision == "shortfall" else decision
    id_hash = None if active is None else selection["profiles"][active]["id_list_sha256"]
    settings = dict(dev["config"]["settings"])
    settings["concurrency"] = projection["concurrency"]
    manifest = dev["config"].get("server_manifest")
    rationale = _rationale(projection, schedule, decision)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_by": _frozen_by(),
        "prompt_files": _prompt_files(),
        "prompt_sha256": dev["config"]["prompt_sha256"],
        "inventory_schema_sha256": dev["config"]["inventory_schema_sha256"],
        "serializer_code_sha256": dev["config"]["serializer_code_sha256"],
        "parser_policy": parser_policy(),
        "settings": settings,
        "model_identity": dev["config"].get("model_identity"),
        "server_manifest_sha256": None if manifest is None else _canonical_sha256(manifest),
        "active_profile": active,
        "id_list_sha256": id_hash,
        "dev_run_ids": list(dev["config"]["ids"]),
        "timing_run_ids": list(timing["config"]["ids"]),
        "pass_criteria": pass_criteria_constants(),
        "pass_criteria_results": results,
        "timing_summary": {
            "intercept": projection["intercept"],
            "slope": projection["slope"],
            "aggregation": projection["aggregation"],
            "fit_points": projection["fit_points"],
            "points": projection["points"],
            "mean_latency_seconds": projection["mean_latency_seconds"],
            "projected_seconds": projection["projected_seconds"],
            "sequential_seconds": projection["sequential_seconds"],
            "concurrency": projection["concurrency"],
            "speedup": projection["speedup"],
            "note": projection["concurrency_note"],
            "runs": {
                "dev": str(dev["dir"]),
                "timing": str(timing["dir"]),
                "synthetic": str(synthetic["dir"]),
                "concurrency": None if concurrency is None else str(concurrency["dir"]),
            },
        },
        "budget_inputs": {
            "allocation_su": args.allocation_su,
            "gpu": args.gpu,
            "walltime_hours": args.walltime_hours,
            "deadline": args.deadline,
            "su_per_gpu_hour": schedule["su_per_gpu_hour"],
            "planning_allowance": PLANNING_ALLOWANCE,
            "startup_hours_per_job": STARTUP_HOURS_PER_JOB,
            "usable_inference_hours_per_job": schedule["usable_inference_hours_per_job"],
            "queue_delay_hours": None,
            "profiles": schedule["profiles"],
        },
        "projection_hours": {
            name: schedule["profiles"][name]["projection_hours"] for name in ("eval200", "eval100")
        },
        "decision": decision,
        "rationale": rationale,
        "prompt_revision_history": _prompt_revision_history(),
    }


def _rationale(projection, schedule, decision):
    eval200 = schedule["profiles"]["eval200"]
    eval100 = schedule["profiles"]["eval100"]
    sentences = [
        (
            f"Projected compute is {eval200['projection_hours']:.4f} h for eval200 and "
            f"{eval100['projection_hours']:.4f} h for eval100 at concurrency {projection['concurrency']}."
        ),
        (
            f"Hours with allowance are {eval200['hours_with_allowance']:.4f} "
            f"({eval200['su']:.4f} SU) and {eval100['hours_with_allowance']:.4f} "
            f"({eval100['su']:.4f} SU) against {schedule['allocation_su']} SU on {schedule['gpu']} "
            f"before {schedule['deadline']}."
        ),
        projection["concurrency_note"],
        (
            "Modeled hours are padded inference plus per-job startup, using usable "
            f"inference time of {schedule['usable_inference_hours_per_job']:.4f} h per job. "
            "Queue delay is not included in the modeled compute time or the deadline comparison."
        ),
    ]
    if decision == "shortfall":
        sentences.append("Neither eval200 nor eval100 fits the allocation and the deadline.")
    else:
        sentences.append(f"Selected {decision}.")
    return " ".join(sentences)


def _prompt_files():
    root = contracts.PACKAGE_DIR / "prompts"
    found = {}
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(contracts.PACKAGE_DIR).as_posix()
            found[relative] = sha256_file(path)
    return found


def _prompt_revision_history():
    root = contracts.PACKAGE_DIR / "prompts"
    history = []
    for path in sorted(root.glob("extract_v*.txt")):
        history.append({
            "path": path.relative_to(contracts.PACKAGE_DIR).as_posix(),
            "sha256": sha256_file(path),
        })
    return history


def _frozen_by():
    user = os.environ.get("USER") or getpass.getuser()
    if not user:
        raise RunnerError("USER is not set; cannot record frozen_by")
    return user


def _parse_deadline(text):
    if not isinstance(text, str) or len(text) != 10 or text[4] != "-" or text[7] != "-":
        raise RunnerError(f"--deadline must be YYYY-MM-DD, got {text!r}")
    try:
        day = datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        raise RunnerError(f"--deadline must be YYYY-MM-DD, got {text!r}") from None
    if day.strftime("%Y-%m-%d") != text:
        raise RunnerError(f"--deadline must be YYYY-MM-DD, got {text!r}")
    return day.replace(tzinfo=timezone.utc) + timedelta(days=1) - timedelta(seconds=1)


def _attempt_rows(run_dir):
    path = Path(run_dir) / "attempts.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise RunnerError(f"corrupt attempts log at line {line_number}: {path}") from None
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _read_cache(bundle, key):
    path = cache_path(_cache_dir(bundle["dir"], bundle["config"]), key)
    if not path.is_file():
        raise RunnerError(f"{bundle['label']} is missing cache entry {key}")
    entry = _read_json(path, f"cache entry {key}")
    if not isinstance(entry, dict):
        raise RunnerError(f"cache entry {key} must be an object")
    return entry


def _cache_dir(run_dir, config):
    raw = config.get("cache_dir")
    if not raw:
        raise RunnerError("run_config.json is missing cache_dir")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(run_dir) / path
    return path


def _n_images(bundle, item_id):
    for item in bundle["config"].get("items") or []:
        if str(item.get("id")) == str(item_id):
            return item["n_images"]
    raise RunnerError(f"run_config has no n_images for {item_id}")


def _parse_timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    text = value
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _numbers_equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _png_bytes(image):
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _canonical_sha256(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _pretty(value):
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_json(path, what):
    path = Path(path)
    if not path.is_file():
        raise RunnerError(f"missing {what}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise RunnerError(f"cannot read {what}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise RunnerError(f"invalid JSON in {what}: {path}") from None

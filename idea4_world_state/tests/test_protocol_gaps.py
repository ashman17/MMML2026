"""Regression tests for resume, freeze fingerprints, job budgeting, and synthetic10 usage.

These cover the gaps that the existing runner tests did not exercise. They use
the mock backend or the scripted localhost server. They do not call a model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from idea4_world_state import contracts, prompts, runner
from idea4_world_state.backend import MockBackend
from idea4_world_state.calllog import cache_path
from idea4_world_state.tests.runner_fixtures import (
    PREPARE,
    invoke,
    patch_latencies,
    read_json,
    records,
    require_dataset,
    run_argv,
)

SERVE_COMMON = Path(__file__).resolve().parents[1] / "psc" / "serve_common.sh"
DEADLINE = datetime(2027, 12, 31, 23, 59, 59, tzinfo=timezone.utc)


def test_resume_keeps_history_when_only_the_server_session_changes(tmp_path, vllm_server, manifest_file, monkeypatch, capsys):
    require_dataset()
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    ids = read_json(PREPARE / "selection.json")["profiles"]["timing"]["ids"]
    vllm_server.enqueue("ok", "ok", "http_500", *["ok"] * 10)
    out = tmp_path / "timing"
    first = manifest_file(created_at="2026-10-24T00:00:00Z", host="r001", slurm_job_id="1001")
    second = manifest_file(created_at="2026-10-24T05:00:00Z", host="r002", slurm_job_id="1002")
    revised = manifest_file(resolved_revision="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    other_gpu = manifest_file(gpu_name="OTHER-GPU")
    common = dict(mock=False, endpoint=vllm_server.base_url, timeout=30, concurrency=1)
    code = invoke(run_argv(tmp_path, "timing", "raw", out, manifest=first, **common))
    captured = capsys.readouterr()
    assert code == 75, captured.err
    paused = _snapshot(out)
    found = records(out)
    assert found[(ids[0], "raw")]["status"] == "ok"
    assert found[(ids[1], "raw")]["status"] == "ok"
    pending_key = found[(ids[2], "raw")]["call_keys"]["raw_answer"]
    assert found[(ids[2], "raw")]["status"] == "pending_transport"
    saved_config = (out / "run_config.json").read_bytes()
    assert read_json(out / "run_config.json")["server_manifest"]["slurm_job_id"] == "1001"
    assert read_json(out / "run_config.json")["backend"] == "vllm"

    rejections = (
        (revised, None, "resolved_revision"),
        (first, 512, "settings.max_side"),
        (other_gpu, None, "server_manifest.gpu_name"),
    )
    for manifest, max_side, expected in rejections:
        code = invoke(run_argv(
            tmp_path, "timing", "raw", out, manifest=manifest, resume=True, max_side=max_side, **common,
        ))
        captured = capsys.readouterr()
        assert code == 2, captured.err
        assert "Traceback" not in captured.err
        assert expected in captured.err
        assert "created_at, host, and slurm_job_id" in captured.err
        assert _snapshot(out) == paused
    assert len(vllm_server.chat_requests) == 3

    code = invoke(run_argv(tmp_path, "timing", "raw", out, manifest=second, resume=True, **common))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (out / "run_config.json").read_bytes() == saved_config
    attempts = [
        json.loads(line)
        for line in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed = [row for row in attempts if row["key"] == pending_key]
    assert [row["attempt"] for row in failed] == [1, 2]
    assert [row["outcome"] for row in failed] == ["transport_error", "ok"]
    assert records(out)[(ids[0], "raw")] == found[(ids[0], "raw")]
    assert records(out)[(ids[1], "raw")] == found[(ids[1], "raw")]
    assert records(out)[(ids[2], "raw")]["status"] == "ok"
    sessions = _sessions(out)
    assert [row["server_manifest"]["slurm_job_id"] for row in sessions] == ["1001", "1002"]
    assert sessions[0]["server_manifest"]["host"] == "r001"
    assert sessions[1]["server_manifest"]["host"] == "r002"
    assert sessions[0]["server_manifest"]["resolved_revision"] == sessions[1]["server_manifest"]["resolved_revision"]
    assert len(vllm_server.chat_requests) == 7

    code = invoke(run_argv(tmp_path, "timing", "raw", out, manifest=second, resume=True, **common))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert len(_sessions(out)) == 2
    assert len(vllm_server.chat_requests) == 7


def test_job_budget_reserves_startup_inside_each_job():
    projection = _hours_projection(3.9, 1.0)
    schedule = runner._schedule(projection, 10**6, "l40s", 4, DEADLINE)
    row = schedule["profiles"]["eval200"]
    assert schedule["usable_inference_hours_per_job"] == pytest.approx(3.5)
    assert row["padded_hours"] == pytest.approx(3.9)
    assert row["jobs"] == 2
    assert row["startup_hours"] == pytest.approx(1.0)
    assert row["hours_with_allowance"] == pytest.approx(4.9)
    assert row["su"] == pytest.approx(4.9)
    assert schedule["queue_delay_hours"] is None
    # The previous formula used the full 4 h wall and then added 0.5 h: 1 job and 4.4 h.
    assert row["jobs"] != 1
    assert row["hours_with_allowance"] != pytest.approx(4.4)

    exact = runner._schedule(_hours_projection(3.5, 1.0), 10**6, "l40s", 4, DEADLINE)
    assert exact["profiles"]["eval200"]["jobs"] == 1
    assert exact["profiles"]["eval200"]["hours_with_allowance"] == pytest.approx(4.0)

    over = runner._schedule(_hours_projection(3.51, 1.0), 10**6, "l40s", 4, DEADLINE)
    assert over["profiles"]["eval200"]["jobs"] == 2
    assert over["profiles"]["eval200"]["hours_with_allowance"] == pytest.approx(3.51 + 1.0)

    idle = runner._schedule(_hours_projection(0.0, 0.0), 10, "h100", 4, DEADLINE)
    assert idle["profiles"]["eval100"]["jobs"] == 0
    assert idle["profiles"]["eval100"]["startup_hours"] == 0
    assert idle["profiles"]["eval100"]["hours_with_allowance"] == 0
    assert idle["queue_delay_hours"] is None


def test_walltime_must_leave_room_for_startup(tmp_path, capsys):
    with pytest.raises(runner.RunnerError, match="startup"):
        runner._schedule(_hours_projection(1.0, 1.0), 10, "l40s", 0.5, DEADLINE)
    with pytest.raises(runner.RunnerError, match="positive"):
        runner._schedule(_hours_projection(1.0, 1.0), 10, "l40s", 0, DEADLINE)

    out = tmp_path / "freeze.json"
    code = invoke(_freeze_argv(tmp_path / "d", tmp_path / "t", tmp_path / "s", 10, out, walltime="0.5"))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "startup" in captured.err
    assert "directory not found" not in captured.err
    assert not out.exists()

    code = invoke(_freeze_argv(tmp_path / "d", tmp_path / "t", tmp_path / "s", 10, tmp_path / "zero.json", walltime="0"))
    captured = capsys.readouterr()
    assert code == 2
    assert "positive" in captured.err
    assert not (tmp_path / "zero.json").exists()


def test_freeze_rejects_a_run_missing_artifact_fingerprints(tmp_path, capsys):
    runs = _write_calibration(tmp_path)
    config_path = runs["synthetic"] / "run_config.json"
    config = read_json(config_path)
    del config["inventory_schema_sha256"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "freeze.json"
    code = invoke(_freeze_argv(runs["dev"], runs["timing"], runs["synthetic"], 10, out))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "missing inventory_schema_sha256" in captured.err
    assert "does not fill them in" in captured.err
    assert "Rerun synthetic10" in captured.err
    assert "new output directory" in captured.err
    assert not out.exists()


def test_freeze_rejects_mixed_artifact_fingerprints(tmp_path, capsys):
    runs = _write_calibration(tmp_path)
    config_path = runs["timing"] / "run_config.json"
    config = read_json(config_path)
    config["prompt_sha256"] = {**config["prompt_sha256"], "raw": "0" * 64}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "freeze.json"
    code = invoke(_freeze_argv(runs["dev"], runs["timing"], runs["synthetic"], 10, out))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "prompt_sha256.raw" in captured.err
    assert "does not match the dev run" in captured.err
    assert "Rerun timing" in captured.err
    assert not out.exists()


def test_freeze_rejects_prompt_schema_and_serializer_edits_after_development(tmp_path, monkeypatch, capsys):
    runs = _write_calibration(tmp_path)
    real_prompts = prompts.prompt_hashes()
    cases = (
        (runner, "schema_sha256", lambda: "0" * 64, "inventory_schema_sha256"),
        (runner, "serializer_sha256", lambda: "ab" * 32, "serializer_code_sha256"),
        (prompts, "prompt_hashes", lambda: {**real_prompts, "extract": "cd" * 32}, "prompt_sha256.extract"),
    )
    for module, name, replacement, expected in cases:
        original = getattr(module, name)
        monkeypatch.setattr(module, name, replacement)
        out = tmp_path / f"freeze-{expected}.json"
        code = invoke(_freeze_argv(runs["dev"], runs["timing"], runs["synthetic"], 10, out))
        captured = capsys.readouterr()
        assert code == 2, captured.err
        assert "Traceback" not in captured.err
        assert expected in captured.err
        assert "changed after those runs" in captured.err
        assert "Rerun dev, timing, and synthetic10" in captured.err
        assert not out.exists()
        monkeypatch.setattr(module, name, original)


def test_synthetic10_usage_check_separates_mock_from_real_inference(tmp_path):
    limit = contracts.SYNTHETIC10_MAX_PROMPT_TOKENS
    mock_missing = runner._synthetic_criterion(_usage_bundle(tmp_path / "mock-missing", "mock", None))
    assert mock_missing["passed"] is True
    assert "mock backend exemption" in mock_missing["detail"]

    real_missing = runner._synthetic_criterion(_usage_bundle(tmp_path / "real-missing", "vllm", None, synthetic=True))
    assert real_missing["passed"] is False
    assert "vllm" in real_missing["detail"]
    assert "not a usage exemption" in real_missing["detail"]

    unlabeled = _usage_bundle(tmp_path / "unlabeled", "vllm", None, synthetic=True)
    del unlabeled["config"]["backend"]
    missing_backend = runner._synthetic_criterion(unlabeled)
    assert missing_backend["passed"] is False
    assert "no backend field" in missing_backend["detail"]

    valid = runner._synthetic_criterion(_usage_bundle(tmp_path / "valid", "vllm", limit))
    assert valid["passed"] is True
    assert f"prompt_tokens={float(limit)}" in valid["detail"]

    over = runner._synthetic_criterion(_usage_bundle(tmp_path / "over", "vllm", limit + 1))
    assert over["passed"] is False
    negative = runner._synthetic_criterion(_usage_bundle(tmp_path / "negative", "vllm", -1))
    assert negative["passed"] is False
    text = runner._synthetic_criterion(_usage_bundle(tmp_path / "text", "vllm", "100"))
    assert text["passed"] is False
    boolean = runner._synthetic_criterion(_usage_bundle(tmp_path / "bool", "vllm", True))
    assert boolean["passed"] is False
    mock_over = runner._synthetic_criterion(_usage_bundle(tmp_path / "mock-over", "mock", limit + 1))
    assert mock_over["passed"] is False
    assert "mock exemption applies only to missing usage" in mock_over["detail"]


def test_freeze_requires_token_usage_for_real_synthetic10(tmp_path, capsys):
    require_dataset()
    dev, timing, synthetic = _run_mock_calibration(tmp_path)
    cloned = _clone_calibration(tmp_path, dev, timing, synthetic, "real-missing")
    _set_usage(cloned["synthetic"], None, backend="vllm")
    missing_out = tmp_path / "missing.json"
    code = invoke(_freeze_argv(cloned["dev"], cloned["timing"], cloned["synthetic"], 10**6, missing_out))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "synthetic10: FAIL" in captured.err
    assert "backend 'vllm'" in captured.err
    assert not missing_out.exists()

    cloned = _clone_calibration(tmp_path, dev, timing, synthetic, "real-valid")
    _set_usage(cloned["synthetic"], 12345, backend="vllm")
    valid_out = tmp_path / "valid.json"
    code = invoke(_freeze_argv(cloned["dev"], cloned["timing"], cloned["synthetic"], 10**6, valid_out))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    document = read_json(valid_out)
    row = next(item for item in document["pass_criteria_results"] if item["name"] == "synthetic10")
    assert row["passed"] is True
    assert "prompt_tokens=12345.0" in row["detail"]
    assert document["inventory_schema_sha256"] == read_json(dev / "run_config.json")["inventory_schema_sha256"]

    cloned = _clone_calibration(tmp_path, dev, timing, synthetic, "real-over")
    _set_usage(cloned["synthetic"], contracts.SYNTHETIC10_MAX_PROMPT_TOKENS + 1, backend="vllm")
    over_out = tmp_path / "over.json"
    code = invoke(_freeze_argv(cloned["dev"], cloned["timing"], cloned["synthetic"], 10**6, over_out))
    captured = capsys.readouterr()
    assert code == 2
    assert "synthetic10: FAIL" in captured.err
    assert not over_out.exists()


def test_batch_resume_restores_prior_state_and_does_not_clobber_it(tmp_path):
    project = tmp_path / "project"
    local_a = tmp_path / "local-a"
    local_b = tmp_path / "local-b"
    local_a.mkdir()
    local_b.mkdir()
    env = _batch_env(project, local=local_a, resume="0", job="job-1")
    first = _bash(
        env,
        """
idea4_prepare_batch_run
mkdir -p "${IDEA4_STAGE_RUN}/records"
printf '%s\n' '{"profile":"dev"}' > "${IDEA4_STAGE_RUN}/run_config.json"
printf '%s\n' '{"attempt":1}' > "${IDEA4_STAGE_RUN}/attempts.jsonl"
printf '%s\n' '{"id":"kept"}' > "${IDEA4_STAGE_RUN}/records/a.json"
idea4_publish_batch_run
printf '%s\n' "${IDEA4_STAGE_RUN}"
""",
    )
    assert first.returncode == 0, first.stderr
    saved = project / "idea4_world_state" / "runs" / "dev1"
    assert (saved / "records" / "a.json").read_text(encoding="utf-8") == '{"id":"kept"}\n'
    assert not (saved.parent / "dev1.partial").exists()
    assert not (saved.parent / "dev1.previous").exists()
    stage_one = Path(first.stdout.strip().splitlines()[-1])
    assert stage_one != saved

    refused = _bash(env, "idea4_prepare_batch_run\n")
    assert refused.returncode == 2
    assert "Refusing to overwrite" in refused.stderr
    assert (saved / "records" / "a.json").read_text(encoding="utf-8") == '{"id":"kept"}\n'

    missing = _batch_env(project, local=local_b, resume="1", job="job-missing", run_id="absent")
    absent = _bash(missing, "idea4_prepare_batch_run\n")
    assert absent.returncode == 2
    assert "no saved run" in absent.stderr
    assert not (project / "idea4_world_state" / "runs" / "absent").exists()

    resume_env = _batch_env(project, local=local_b, resume="1", job="job-2")
    prepared = _bash(
        resume_env,
        """
idea4_prepare_batch_run
printf '%s\n' "${IDEA4_STAGE_RUN}"
""",
    )
    assert prepared.returncode == 0, prepared.stderr
    stage_two = Path(prepared.stdout.strip().splitlines()[-1])
    assert stage_two != stage_one
    assert stage_two != saved
    assert (stage_two / "records" / "a.json").read_text(encoding="utf-8") == '{"id":"kept"}\n'
    assert (saved / "records" / "a.json").read_text(encoding="utf-8") == '{"id":"kept"}\n'
    assert (saved / "attempts.jsonl").read_text(encoding="utf-8") == '{"attempt":1}\n'

    published = _bash(
        resume_env,
        f"""
IDEA4_STAGE_RUN={stage_two}
printf '%s\n' '{{"attempt":2}}' >> "${{IDEA4_STAGE_RUN}}/attempts.jsonl"
printf '%s\n' '{{"id":"added"}}' > "${{IDEA4_STAGE_RUN}}/records/b.json"
idea4_publish_batch_run
""",
    )
    assert published.returncode == 0, published.stderr
    assert (saved / "records" / "a.json").read_text(encoding="utf-8") == '{"id":"kept"}\n'
    assert (saved / "records" / "b.json").read_text(encoding="utf-8") == '{"id":"added"}\n'
    assert (saved / "attempts.jsonl").read_text(encoding="utf-8") == '{"attempt":1}\n{"attempt":2}\n'
    assert not (saved.parent / "dev1.partial").exists()
    assert not (saved.parent / "dev1.previous").exists()

    untouched = (saved / "records" / "a.json").read_bytes()
    empty_stage = tmp_path / "empty-stage"
    empty_stage.mkdir()
    skipped = _bash(
        resume_env,
        f"""
IDEA4_STAGE_RUN={empty_stage}
idea4_publish_batch_run
""",
    )
    assert skipped.returncode == 0, skipped.stderr
    assert "Leaving the saved run unchanged" in skipped.stderr
    assert (saved / "records" / "a.json").read_bytes() == untouched
    assert (saved / "records" / "b.json").is_file()


def test_batch_publish_recovers_an_interrupted_copy(tmp_path):
    project = tmp_path / "project"
    saved = project / "idea4_world_state" / "runs" / "dev1"
    partial = Path(str(saved) + ".partial")
    previous = Path(str(saved) + ".previous")
    _write_marker(previous, "old")
    _write_marker(partial, "new")
    env = _batch_env(project, resume="1", job="job-recover")
    local = tmp_path / "local-recover"
    local.mkdir()
    env["LOCAL"] = str(local)
    recovered = _bash(
        env,
        """
idea4_prepare_batch_run
printf '%s\n' "${IDEA4_STAGE_RUN}"
""",
    )
    assert recovered.returncode == 0, recovered.stderr
    assert (saved / "marker.txt").read_text(encoding="utf-8") == "new\n"
    assert not partial.exists()
    stage = Path(recovered.stdout.strip().splitlines()[-1])
    assert (stage / "marker.txt").read_text(encoding="utf-8") == "new\n"
    assert (saved / "run_config.json").is_file()


def _hours_projection(eval200_padded, eval100_padded):
    scale = 3600.0 / runner.PLANNING_ALLOWANCE
    return {
        "projected_seconds": {
            "eval200": eval200_padded * scale,
            "eval100": eval100_padded * scale,
        }
    }


def _freeze_argv(dev, timing, synthetic, allocation, out, walltime="4"):
    return [
        "freeze",
        "--dev-run", str(dev),
        "--timing-run", str(timing),
        "--synthetic-run", str(synthetic),
        "--allocation-su", str(allocation),
        "--gpu", "l40s",
        "--walltime-hours", str(walltime),
        "--deadline", "2027-12-31",
        "--out", str(out),
    ]


def _write_calibration(root):
    selection = read_json(PREPARE / "selection.json")
    fingerprints = runner._current_artifact_fingerprints()
    identity = MockBackend().model_identity()
    specs = (
        ("dev", "dev", ["raw", "flat", "graph"]),
        ("timing", "timing", ["raw"]),
        ("synthetic", "synthetic10", ["raw"]),
    )
    runs = {}
    for name, profile, conditions in specs:
        path = Path(root) / name
        path.mkdir(parents=True)
        (path / "records").mkdir()
        config = {
            "profile": profile,
            "conditions": conditions,
            "settings": {
                "extract_tokens": contracts.DEFAULT_EXTRACT_TOKENS,
                "answer_tokens": contracts.DEFAULT_ANSWER_TOKENS,
                "max_side": contracts.DEFAULT_MAX_SIDE,
                "timeout": contracts.DEFAULT_TIMEOUT_SECONDS,
                "concurrency": 1,
            },
            "model_identity": identity,
            "prepare_manifest_sha256": selection["manifest_sha256"],
            "prompt_sha256": fingerprints["prompt_sha256"],
            "inventory_schema_sha256": fingerprints["inventory_schema_sha256"],
            "serializer_code_sha256": fingerprints["serializer_code_sha256"],
            "backend": "mock",
            "synthetic": True,
            "ids": [1],
        }
        (path / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        (path / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
        runs[name] = path
    return runs


def _usage_bundle(root, backend, tokens, synthetic=True):
    root = Path(root)
    run = root / "run"
    cache = root / "cache"
    run.mkdir(parents=True)
    (run / "records").mkdir()
    key = "ab" * 32
    record = {
        "id": "synthetic10",
        "condition": "raw",
        "status": "ok",
        "call_keys": {"raw_answer": key},
        "input_tokens": tokens,
    }
    (run / "records" / "synthetic10__raw.json").write_text(json.dumps(record), encoding="utf-8")
    entry_path = cache_path(cache, key)
    entry_path.parent.mkdir(parents=True)
    entry_path.write_text(
        json.dumps({"input_tokens": tokens, "output_tokens": 1, "latency_seconds": 0.1}),
        encoding="utf-8",
    )
    return {
        "dir": run,
        "config": {"cache_dir": str(cache), "backend": backend, "synthetic": synthetic},
        "records": {("synthetic10", "raw"): record},
        "label": "synthetic run",
    }


def _run_mock_calibration(tmp_path):
    dev = tmp_path / "dev"
    timing = tmp_path / "timing"
    synthetic = tmp_path / "synthetic10"
    cache = tmp_path / "cache"
    for profile, conditions, out in (
        ("dev", "raw,flat,graph", dev),
        ("timing", "raw", timing),
        ("synthetic10", "raw", synthetic),
    ):
        code = invoke(run_argv(tmp_path, profile, conditions, out, cache=cache))
        assert code == 0
    for run_dir in (dev, timing, synthetic):
        patch_latencies(run_dir)
    return dev, timing, synthetic


def _clone_calibration(tmp_path, dev, timing, synthetic, name):
    root = tmp_path / name
    cache = root / "cache"
    shutil.copytree(Path(read_json(dev / "run_config.json")["cache_dir"]), cache)
    cloned = {}
    for label, source in (("dev", dev), ("timing", timing), ("synthetic", synthetic)):
        destination = root / label
        shutil.copytree(source, destination)
        config_path = destination / "run_config.json"
        config = read_json(config_path)
        config["cache_dir"] = str(cache.resolve())
        config_path.write_text(json.dumps(config), encoding="utf-8")
        cloned[label] = destination
    return cloned


def _set_usage(run_dir, tokens, backend):
    found = records(run_dir)
    record = found[("synthetic10", "raw")]
    key = record["call_keys"]["raw_answer"]
    config = read_json(run_dir / "run_config.json")
    config["backend"] = backend
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    entry_path = cache_path(config["cache_dir"], key)
    entry = read_json(entry_path)
    entry["input_tokens"] = tokens
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    record["input_tokens"] = tokens
    record_path = run_dir / "records" / "synthetic10__raw.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")


def _snapshot(run_dir):
    found = {}
    for path in sorted(Path(run_dir).rglob("*")):
        if path.is_file():
            found[path.relative_to(run_dir).as_posix()] = path.read_bytes()
    return found


def _sessions(run_dir):
    return [
        json.loads(line)
        for line in (Path(run_dir) / "server_sessions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _batch_env(project, local=None, resume="0", job="job-1", run_id="dev1"):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(project.parent / "home"),
        "PROJECT": str(project),
        "IDEA4_RUN_ID": run_id,
        "IDEA4_JOB_TAG": job,
        "IDEA4_RESUME": resume,
        "SERVE_COMMON": str(SERVE_COMMON),
    }
    if local is not None:
        env["LOCAL"] = str(local)
    return env


def _bash(env, body):
    script = 'set -euo pipefail\nsource "$SERVE_COMMON"\n' + body
    return subprocess.run(["bash", "-c", script], env=env, text=True, capture_output=True)


def _write_marker(directory, text):
    directory.mkdir(parents=True)
    (directory / "run_config.json").write_text('{"profile":"dev"}\n', encoding="utf-8")
    (directory / "marker.txt").write_text(text + "\n", encoding="utf-8")

"""Runner, resume, exports, and protocol freeze."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from idea3_consistency.mmsi_consistency import parse_question
from idea4_world_state import contracts, prompts, runner
from idea4_world_state.backend import MockBackend
from idea4_world_state.calllog import cache_path
from idea4_world_state.tests.runner_fixtures import (
    PREPARE,
    RECORD_FIELDS,
    SECTION_86_FREEZE_FIELDS,
    assert_no_gold_key,
    invoke,
    patch_latencies,
    read_json,
    records,
    require_dataset,
    run_argv,
)


def test_code_sha256_covers_run_path_modules():
    import hashlib

    import idea3_consistency.mmsi_consistency as answer_extraction
    import idea4_visual_reasoning.data as image_data
    from idea4_world_state import backend, calllog, data, inventory, serialize

    root = contracts.PACKAGE_DIR.parent
    modules = (answer_extraction, image_data, backend, calllog, contracts, data, inventory, prompts, runner, serialize)
    names = {Path(module.__file__).resolve().relative_to(root).as_posix() for module in modules}
    assert names == set(runner.RUN_CODE_FILES)
    digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in runner.RUN_CODE_FILES}
    text = json.dumps(digests, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert runner.code_sha256() == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_synthetic_images_are_deterministic_and_numbered():
    first = runner.synthetic_images()
    second = runner.synthetic_images()
    assert first == second
    assert len(first) == 10
    parsed = parse_question(runner.SYNTHETIC10_QUESTION)
    assert parsed is not None and len(parsed[1]) == 4
    colors = set()
    for index, raw in enumerate(first, start=1):
        with Image.open(__import__("io").BytesIO(raw)) as image:
            assert image.format == "PNG"
            assert image.size == (768, 768)
            colors.add(image.getpixel((0, 0)))
            drawn = {image.getpixel((x, y)) for y in range(64, 180) for x in range(64, 220)}
            assert len(drawn) > 1
    assert len(colors) == 10


def test_missing_selection_and_bad_deadline_exit_2(tmp_path, capsys):
    missing = [
        "run", "--prepare-dir", str(tmp_path / "missing"), "--profile", "dev",
        "--conditions", "raw", "--server-manifest", str(tmp_path / "m"),
        "--out", str(tmp_path / "out"), "--mock",
    ]
    code = invoke(missing)
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "selection.json" in captured.err

    freeze = [
        "freeze", "--dev-run", str(tmp_path / "d"), "--timing-run", str(tmp_path / "t"),
        "--synthetic-run", str(tmp_path / "s"), "--allocation-su", "1", "--gpu", "l40s",
        "--walltime-hours", "4", "--deadline", "2026-13-01", "--out", str(tmp_path / "freeze.json"),
    ]
    code = invoke(freeze)
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "deadline" in captured.err
    assert not (tmp_path / "freeze.json").exists()


def test_real_eval_requires_freeze_and_bad_manifest_exits_2(tmp_path, capsys, manifest_file):
    code = invoke([
        "run", "--prepare-dir", str(PREPARE), "--profile", "eval100", "--conditions", "raw",
        "--server-manifest", str(tmp_path / "m"), "--out", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "--freeze" in captured.err
    assert not (tmp_path / "out").exists()

    code = invoke([
        "run", "--prepare-dir", str(PREPARE), "--profile", "synthetic10", "--conditions", "raw",
        "--endpoint", "http://127.0.0.1:9/v1",
        "--server-manifest", str(manifest_file(served_model_name="not-the-served-name")),
        "--out", str(tmp_path / "real"), "--cache", str(tmp_path / "cache"), "--timeout", "5",
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "served model" in captured.err


def test_backend_startup_transport_error_exits_75(tmp_path, capsys, manifest_file):
    code = invoke([
        "run", "--prepare-dir", str(PREPARE), "--profile", "synthetic10", "--conditions", "raw",
        "--endpoint", "http://127.0.0.1:9/v1",
        "--server-manifest", str(manifest_file()),
        "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"), "--timeout", "5",
    ])
    captured = capsys.readouterr()
    assert code == contracts.EXIT_CODES["paused"] == 75
    assert "Traceback" not in captured.err
    assert "backend startup failed" in captured.err


def test_freeze_rejects_a_modified_prompt_hash(tmp_path, monkeypatch, capsys):
    real = prompts.prompt_hashes()
    selection = read_json(PREPARE / "selection.json")
    document = {
        "prompt_sha256": real,
        "inventory_schema_sha256": runner.schema_sha256(),
        "serializer_code_sha256": runner.serializer_sha256(),
        "settings": {
            "extract_tokens": contracts.DEFAULT_EXTRACT_TOKENS,
            "answer_tokens": contracts.DEFAULT_ANSWER_TOKENS,
            "max_side": contracts.DEFAULT_MAX_SIDE,
            "timeout": contracts.DEFAULT_TIMEOUT_SECONDS,
            "concurrency": contracts.DEFAULT_CONCURRENCY,
        },
        "model_identity": MockBackend().model_identity(),
        "active_profile": "eval100",
        "id_list_sha256": selection["profiles"]["eval100"]["id_list_sha256"],
    }
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(prompts, "prompt_hashes", lambda: {**real, "raw": "0" * 64})
    code = invoke(run_argv(
        tmp_path, "eval100", "raw", tmp_path / "out", freeze=freeze,
    ))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    assert "prompt" in captured.err
    assert not (tmp_path / "out" / "run_config.json").exists()


def test_synthetic10_mock_exports_and_config_lock(tmp_path, capsys):
    require_dataset()
    out = tmp_path / "synthetic10"
    code = invoke(run_argv(tmp_path, "synthetic10", "raw", out))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "MOCK synthetic10 raw=ok" in captured.out
    config = read_json(out / "run_config.json")
    assert config["synthetic"] is True
    assert config["backend"] == "mock"
    assert config["server_manifest"] is None
    assert config["inventory_schema_sha256"] == runner.schema_sha256()
    assert config["serializer_code_sha256"] == runner.serializer_sha256()
    assert config["profile"] == "synthetic10"
    assert config["ids"] == ["synthetic10"]
    assert config["dataset_sha256"] == contracts.DATASET_SHA256
    assert config["prepare_manifest_sha256"] == read_json(PREPARE / "selection.json")["manifest_sha256"]
    record = records(out)[("synthetic10", "raw")]
    assert set(record) == set(RECORD_FIELDS)
    assert record["status"] == "ok"
    assert record["pred"] in "ABCD"
    assert record["synthetic"] is True
    assert len(list((out / "inputs").glob("*.png"))) == 10
    status = read_json(out / "status.json")
    assert status["state"] == "complete"
    _assert_prediction_columns(out)
    assert_no_gold_key(out)

    code = invoke(run_argv(tmp_path, "synthetic10", "flat,graph", tmp_path / "bad"))
    captured = capsys.readouterr()
    assert code == 2
    assert "raw condition only" in captured.err

    code = invoke(run_argv(tmp_path, "synthetic10", "raw", out, max_side=512))
    captured = capsys.readouterr()
    assert code == 2
    assert "use a new output directory" in captured.err

    code = invoke(run_argv(tmp_path, "synthetic10", "raw", out))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "MOCK synthetic10 raw=ok" in captured.out
    sessions = [
        json.loads(line)
        for line in (out / "server_sessions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(sessions) == 1
    assert sessions[0]["backend"] == "mock"
    assert sessions[0]["server_manifest"] is None


def test_extraction_invalid_truncated_and_inference_columns(tmp_path, monkeypatch, capsys):
    require_dataset()
    seen = []
    original = runner.pq.ParquetFile.iter_batches

    def wrapped(self, *args, **kwargs):
        seen.append(kwargs.get("columns"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(runner.pq.ParquetFile, "iter_batches", wrapped)
    real = runner.MockBackend

    def invalid(seed=0, **kwargs):
        return real(seed=seed, invalid_json_kinds=("extract",))

    monkeypatch.setattr(runner, "MockBackend", invalid)
    out = tmp_path / "invalid"
    code = invoke(run_argv(tmp_path, "timing", "flat,graph", out))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert seen, "the runner did not read the parquet"
    for columns in seen:
        assert columns is not None
        assert set(columns) <= set(contracts.INFERENCE_COLUMNS)
        assert "answer" not in columns
        assert "thought" not in columns
    found = records(out)
    ids = read_json(PREPARE / "selection.json")["profiles"]["timing"]["ids"]
    assert len(ids) == 6
    for item_id in ids:
        flat = found[(item_id, "flat")]
        graph = found[(item_id, "graph")]
        assert flat["status"] == graph["status"] == "extraction_invalid"
        assert flat["call_keys"] == graph["call_keys"]
        assert set(flat["call_keys"]) == {"extract"}
        error = read_json(out / "inventories" / f"{item_id}.error.json")
        assert set(error) == {"rule", "message", "text"}
        assert error["text"] == "{not-json"
        assert error["rule"]
        assert not (out / "inventories" / f"{item_id}.json").exists()
        assert not (out / "serializations" / f"{item_id}.flat.txt").exists()
    attempts = (out / "attempts.jsonl").read_text(encoding="utf-8")
    assert attempts.count('"kind":"extract"') == 6
    assert "flat_answer" not in attempts
    _assert_prediction_columns(out)
    assert_no_gold_key(out)

    def truncated(seed=0, **kwargs):
        return real(seed=seed, truncate_kinds=("extract", "raw_answer"))

    monkeypatch.setattr(runner, "MockBackend", truncated)
    truncated_out = tmp_path / "truncated"
    code = invoke(run_argv(tmp_path, "timing", "raw,flat,graph", truncated_out, cache=tmp_path / "cache-trunc"))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    found = records(truncated_out)
    for item_id in ids:
        assert found[(item_id, "raw")]["status"] == "answer_truncated"
        assert found[(item_id, "raw")]["pred"] is None
        assert found[(item_id, "flat")]["status"] == "extraction_truncated"
        assert found[(item_id, "graph")]["status"] == "extraction_truncated"
        assert not (truncated_out / "inventories" / f"{item_id}.error.json").exists()


def test_pause_and_resume(tmp_path, vllm_server, manifest_file, monkeypatch, capsys):
    require_dataset()
    monkeypatch.delenv("IDEA4_API_KEY", raising=False)
    ids = read_json(PREPARE / "selection.json")["profiles"]["timing"]["ids"]
    vllm_server.enqueue("ok", "ok", "http_500", *["ok"] * 10)
    out = tmp_path / "timing"
    manifest = manifest_file()
    common = dict(mock=False, manifest=manifest, endpoint=vllm_server.base_url, timeout=30, concurrency=1)
    code = invoke(run_argv(tmp_path, "timing", "raw", out, **common))
    captured = capsys.readouterr()
    assert code == 75, captured.err
    assert "paused" in captured.err
    status = read_json(out / "status.json")
    assert status["state"] == "paused"
    assert status["reason"]
    assert status["counts"]["pending_transport"] == 1
    assert status["counts"]["ok"] == 2
    found = records(out)
    assert found[(ids[0], "raw")]["status"] == "ok"
    assert found[(ids[1], "raw")]["status"] == "ok"
    pending = found[(ids[2], "raw")]
    assert pending["status"] == "pending_transport"
    failed_key = pending["call_keys"]["raw_answer"]
    assert not cache_path(tmp_path / "cache", failed_key).exists()
    for item_id in ids[3:]:
        assert (item_id, "raw") not in found
    assert not (out / "predictions.jsonl").exists()

    code = invoke(run_argv(tmp_path, "timing", "raw", out, **common))
    captured = capsys.readouterr()
    assert code == 2
    assert "--resume" in captured.err
    assert len(vllm_server.chat_requests) == 3

    code = invoke(run_argv(tmp_path, "timing", "raw", out, resume=True, **common))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert read_json(out / "status.json")["state"] == "complete"
    found = records(out)
    assert [found[(item_id, "raw")]["status"] for item_id in ids] == ["ok"] * 6
    history = [
        json.loads(line)
        for line in (out / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["key"] == failed_key
    ]
    assert [row["attempt"] for row in history] == [1, 2]
    assert [row["outcome"] for row in history] == ["transport_error", "ok"]
    assert cache_path(tmp_path / "cache", failed_key).is_file()
    # 3 calls before the pause, then 4 on resume (the failed item plus three unattempted).
    # The two completed calls are cache hits and are not sent again.
    assert len(vllm_server.chat_requests) == 7
    _assert_prediction_columns(out)


def test_freeze_on_mock_runs(tmp_path, capsys):
    require_dataset()
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
        captured = capsys.readouterr()
        assert code == 0, captured.err
    for run_dir in (dev, timing, synthetic):
        patch_latencies(run_dir)
    preview = runner.budget_projection(dev, timing, synthetic, None, "l40s", 4, 10**6, "2027-12-31")
    su200 = preview["schedule"]["profiles"]["eval200"]["su"]
    su100 = preview["schedule"]["profiles"]["eval100"]["su"]
    assert su100 < su200
    mid = (su100 + su200) / 2.0

    chosen = tmp_path / "freeze-eval100.json"
    code = invoke(_freeze_argv(dev, timing, synthetic, mid, chosen))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    document = read_json(chosen)
    for field in SECTION_86_FREEZE_FIELDS:
        assert field in document
    assert document["decision"] == "eval100"
    assert document["active_profile"] == "eval100"
    selection = read_json(PREPARE / "selection.json")
    assert document["id_list_sha256"] == selection["profiles"]["eval100"]["id_list_sha256"]
    assert document["dev_run_ids"] == selection["profiles"]["dev"]["ids"]
    assert document["timing_run_ids"] == selection["profiles"]["timing"]["ids"]
    assert document["server_manifest_sha256"] is None
    assert document["model_identity"]["served_model_name"] == "MOCK"
    assert document["frozen_by"]
    assert document["pass_criteria"]["dev_schema_valid_minimum"] == runner.DEV_SCHEMA_VALID_MINIMUM
    assert document["pass_criteria"]["concurrency_speedup_minimum"] == runner.CONCURRENCY_SPEEDUP_MINIMUM
    assert document["settings"]["concurrency"] == 1
    assert {row["name"] for row in document["pass_criteria_results"]} >= {
        "dev_transport", "extraction_schema_valid", "extraction_truncated",
        "raw_answer_parsing", "evidence_answer_parsing", "fact_atom_equivalence", "synthetic10",
    }
    assert all(row["passed"] for row in document["pass_criteria_results"])
    assert document["parser_policy"]["cross_view_repair"]["uncertainty"] == "medium"
    assert document["parser_policy"]["cross_view_repair"]["uncertainty_source"] == "parser_default"
    assert document["prompt_revision_history"]
    assert "prompts/extract_v1.txt" in document["prompt_files"]
    assert "human confirmation" in document["rationale"]
    assert "Queue delay is not included" in document["rationale"]
    assert set(document["projection_hours"]) == {"eval200", "eval100"}
    dev_config = read_json(dev / "run_config.json")
    assert document["prompt_sha256"] == dev_config["prompt_sha256"] == prompts.prompt_hashes()
    assert document["inventory_schema_sha256"] == dev_config["inventory_schema_sha256"] == runner.schema_sha256()
    assert document["serializer_code_sha256"] == dev_config["serializer_code_sha256"] == runner.serializer_sha256()
    assert read_json(timing / "run_config.json")["prompt_sha256"] == document["prompt_sha256"]
    assert read_json(synthetic / "run_config.json")["inventory_schema_sha256"] == document["inventory_schema_sha256"]
    synthetic_row = next(row for row in document["pass_criteria_results"] if row["name"] == "synthetic10")
    assert "mock backend exemption" in synthetic_row["detail"]

    shortfall = tmp_path / "freeze-shortfall.json"
    code = invoke(_freeze_argv(dev, timing, synthetic, 0, shortfall))
    captured = capsys.readouterr()
    assert code == 2
    assert "Traceback" not in captured.err
    short = read_json(shortfall)
    assert short["decision"] == "shortfall"
    assert short["active_profile"] is None
    assert short["budget_inputs"]["profiles"]["eval200"]["su"] == su200
    assert short["budget_inputs"]["profiles"]["eval100"]["su"] == su100

    broken = tmp_path / "dev-broken"
    shutil.copytree(dev, broken)
    with (broken / "attempts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"outcome": "transport_error", "key": "not-a-call"}) + "\n")
    failed = tmp_path / "freeze-failed.json"
    code = invoke(_freeze_argv(broken, timing, synthetic, mid, failed))
    captured = capsys.readouterr()
    assert code == 2
    assert not failed.exists()
    assert "development pass criteria failed" in captured.err
    assert "dev_transport" in captured.err and "FAIL" in captured.err
    assert "extraction_schema_valid" in captured.err


def _freeze_argv(dev, timing, synthetic, allocation, out):
    return [
        "freeze",
        "--dev-run", str(dev),
        "--timing-run", str(timing),
        "--synthetic-run", str(synthetic),
        "--allocation-su", str(allocation),
        "--gpu", "l40s",
        "--walltime-hours", "4",
        "--deadline", "2027-12-31",
        "--out", str(out),
    ]


def _assert_prediction_columns(run_dir):
    import csv
    path = Path(run_dir) / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert tuple(header) == contracts.PREDICTION_COLUMNS
    assert rows
    jsonl = [
        json.loads(line)
        for line in (Path(run_dir) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert jsonl
    for row in jsonl:
        assert list(row) == list(contracts.PREDICTION_COLUMNS)
        assert "gold" not in row

"""Scoring tests for world-state analysis, on hand-built run directories."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from idea4_world_state.analysis import (
    cli_analyze,
    load_gold,
    paired_bootstrap,
    stratified_resample_indices,
)
from idea4_world_state.contracts import BOOTSTRAP_RESAMPLES, FIXTURES_DIR, SEED

_HELPER_PATH = Path(__file__).with_name("analysis_fixtures.py")
_SPEC = importlib.util.spec_from_file_location("analysis_fixtures", _HELPER_PATH)
analysis_fixtures = importlib.util.module_from_spec(_SPEC)
sys.modules["analysis_fixtures"] = analysis_fixtures
_SPEC.loader.exec_module(analysis_fixtures)

build_known = analysis_fixtures.build_known
build_known_join = analysis_fixtures.build_known_join
write_dataset = analysis_fixtures.write_dataset
write_run = analysis_fixtures.write_run


def _rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary(out):
    return json.loads((Path(out) / "summary.json").read_text(encoding="utf-8"))


def _report(out):
    return (Path(out) / "report.md").read_text(encoding="utf-8")


def _by_condition(out):
    return {row["condition"]: row for row in _rows(Path(out) / "accuracy_by_condition.csv")}


def _run_known(tmp_path):
    run, dataset = build_known(tmp_path)
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    return out


def _assert_exit_2(tmp_path, evidence, raw, dataset):
    with pytest.raises(SystemExit) as caught:
        cli_analyze(SimpleNamespace(
            run=evidence,
            raw_run=raw,
            dataset=dataset,
            out=tmp_path / "analysis",
        ))
    assert caught.value.code == 2


def test_known_answer_accuracy_macro_and_per_category(tmp_path):
    out = _run_known(tmp_path)
    for name in (
        "summary.json",
        "accuracy_by_condition.csv",
        "accuracy_by_category.csv",
        "paired_differences.csv",
        "failures.csv",
        "item_outcomes.csv",
        "coverage.csv",
        "costs.csv",
        "report.md",
    ):
        assert (out / name).is_file()
    summary = _summary(out)
    assert summary["completeness"] == "COMPLETE"
    assert summary["n_items"] == 3
    table = _by_condition(out)
    assert int(table["raw"]["n"]) == 3
    assert int(table["raw"]["correct"]) == 2
    assert float(table["raw"]["accuracy"]) == pytest.approx(2 / 3)
    assert float(table["raw"]["macro_accuracy"]) == pytest.approx(0.5)
    assert int(table["flat"]["n"]) == 3
    assert int(table["flat"]["correct"]) == 2
    assert float(table["flat"]["accuracy"]) == pytest.approx(2 / 3)
    assert float(table["flat"]["macro_accuracy"]) == pytest.approx(0.75)
    assert int(table["graph"]["correct"]) == 3
    assert float(table["graph"]["accuracy"]) == pytest.approx(1)
    assert float(table["graph"]["macro_accuracy"]) == pytest.approx(1)
    # Macro is unweighted, so it is not the sample accuracy on this profile.
    assert float(table["flat"]["macro_accuracy"]) != pytest.approx(float(table["flat"]["accuracy"]))
    categories = {(row["condition"], row["category"]): row for row in _rows(out / "accuracy_by_category.csv")}
    assert int(categories["raw", "spatial"]["n"]) == 2
    assert int(categories["raw", "spatial"]["correct"]) == 2
    assert float(categories["raw", "spatial"]["accuracy"]) == pytest.approx(1)
    assert int(categories["raw", "motion"]["n"]) == 1
    assert int(categories["raw", "motion"]["correct"]) == 0
    assert float(categories["raw", "motion"]["accuracy"]) == pytest.approx(0)
    assert int(categories["flat", "spatial"]["correct"]) == 1
    assert float(categories["flat", "spatial"]["accuracy"]) == pytest.approx(0.5)
    assert int(categories["flat", "motion"]["correct"]) == 1
    assert float(categories["graph", "spatial"]["correct"]) == 2
    assert int(categories["graph", "motion"]["correct"]) == 1
    assert summary["by_condition"]["flat"]["correct"] == 2
    assert summary["by_condition"]["raw"]["macro_accuracy"] == pytest.approx(0.5)
    report = _report(out)
    assert "Completeness: COMPLETE" in report
    assert "INCOMPLETE" not in report
    assert "MOCK / SYNTHETIC" not in report


def test_item_outcomes_match_accuracy_with_missing_and_non_ok(tmp_path):
    items = [
        {"id": "b", "category": "motion"},
        {"id": "a", "category": "spatial"},
    ]
    records = [
        # Lenient correct: parse_rule 2 still counts.
        {"id": "b", "condition": "graph", "status": "ok", "pred": "A", "parse_rule": 2},
        # Letter matches gold, but the status is not ok.
        {"id": "a", "condition": "graph", "status": "answer_parse_error", "pred": "B", "parse_rule": 0},
        {"id": "b", "condition": "flat", "status": "ok", "pred": "C", "parse_rule": 0},
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["graph", "flat"])
    dataset = write_dataset(tmp_path / "gold.parquet", {"b": "A", "a": "B"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    rows = _rows(out / "item_outcomes.csv")
    assert list(rows[0]) == ["id", "category", "condition", "status", "pred", "gold", "correct"]
    assert [(row["id"], row["condition"]) for row in rows] == [
        ("b", "graph"),
        ("b", "flat"),
        ("a", "graph"),
        ("a", "flat"),
    ]
    assert len(rows) == 4
    by_key = {(row["id"], row["condition"]): row for row in rows}
    assert by_key["b", "graph"] == {
        "id": "b", "category": "motion", "condition": "graph",
        "status": "ok", "pred": "A", "gold": "A", "correct": "1",
    }
    assert by_key["b", "flat"]["pred"] == "C"
    assert by_key["b", "flat"]["gold"] == "A"
    assert by_key["b", "flat"]["correct"] == "0"
    assert by_key["a", "graph"]["status"] == "answer_parse_error"
    assert by_key["a", "graph"]["pred"] == "B"
    assert by_key["a", "graph"]["gold"] == "B"
    assert by_key["a", "graph"]["correct"] == "0"
    assert by_key["a", "flat"]["status"] == "missing"
    assert by_key["a", "flat"]["pred"] == ""
    assert by_key["a", "flat"]["gold"] == "B"
    assert by_key["a", "flat"]["correct"] == "0"
    accuracy = _by_condition(out)
    for condition in ("graph", "flat"):
        summed = sum(int(row["correct"]) for row in rows if row["condition"] == condition)
        assert summed == int(accuracy[condition]["correct"])
        assert int(accuracy[condition]["n"]) == 2
    summary = _summary(out)
    assert summary["item_outcomes"] == {"rows": 4, "by_condition": {"graph": 2, "flat": 2}}
    assert "item_outcomes.csv" in _report(out)
    assert "gold" not in summary["item_outcomes"]


def test_failures_count_as_incorrect_with_full_denominator(tmp_path):
    items = [{"id": "a", "category": "spatial"}, {"id": "b", "category": "spatial"}]
    records = [
        {"id": "a", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
        # The letter matches gold, but the status is not ok.
        {"id": "b", "condition": "raw", "status": "extraction_invalid", "pred": "B", "parse_rule": None},
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["raw"], profile="eval200")
    dataset = write_dataset(tmp_path / "gold.parquet", {"a": "A", "b": "B"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    row = _by_condition(out)["raw"]
    assert int(row["n"]) == 2
    assert int(row["correct"]) == 1
    assert float(row["accuracy"]) == pytest.approx(0.5)
    failures = {(r["condition"], r["status"]): int(r["count"]) for r in _rows(out / "failures.csv")}
    assert failures["raw", "extraction_invalid"] == 1
    assert failures["raw", "ok"] == 1
    assert _summary(out)["completeness"] == "COMPLETE"


def test_missing_record_marks_incomplete(tmp_path):
    items = [{"id": "a", "category": "spatial"}, {"id": "b", "category": "motion"}]
    records = [
        {"id": "a", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["raw"], profile="eval200")
    dataset = write_dataset(tmp_path / "gold.parquet", {"a": "A", "b": "B"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    summary = _summary(out)
    report = _report(out)
    row = _by_condition(out)["raw"]
    assert summary["completeness"] == "INCOMPLETE"
    assert summary["missing_records"] == 1
    assert summary["pending_transport_records"] == 0
    assert "Completeness: INCOMPLETE" in report
    assert "Missing records: 1." in report
    assert "Pending transport records: 0." in report
    assert int(row["n"]) == 2
    assert int(row["correct"]) == 1
    assert float(row["accuracy"]) == pytest.approx(0.5)
    failures = {(r["status"]): int(r["count"]) for r in _rows(out / "failures.csv")}
    assert failures["missing"] == 1


def test_pending_transport_marks_incomplete(tmp_path):
    items = [{"id": "a", "category": "spatial"}, {"id": "b", "category": "spatial"}]
    records = [
        {"id": "a", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
        # Pending transport is incorrect even when the letter matches gold.
        {"id": "b", "condition": "raw", "status": "pending_transport", "pred": "B", "parse_rule": 0},
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["raw"], profile="eval200")
    dataset = write_dataset(tmp_path / "gold.parquet", {"a": "A", "b": "B"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    summary = _summary(out)
    report = _report(out)
    row = _by_condition(out)["raw"]
    assert summary["completeness"] == "INCOMPLETE"
    assert summary["missing_records"] == 0
    assert summary["pending_transport_records"] == 1
    assert "Completeness: INCOMPLETE" in report
    assert "Pending transport records: 1." in report
    assert "Missing records: 0." in report
    assert int(row["n"]) == 2
    assert int(row["correct"]) == 1
    failures = {r["status"]: int(r["count"]) for r in _rows(out / "failures.csv")}
    assert failures["pending_transport"] == 1


def test_bootstrap_resample_preserves_category_sizes():
    categories = ["spatial", "spatial", "motion", "size", "size", "size"]
    pools = defaultdict(set)
    for index, category in enumerate(categories):
        pools[category].add(index)
    rng = np.random.default_rng(0)
    for _ in range(40):
        chosen = stratified_resample_indices(categories, rng)
        assert len(chosen) == len(categories)
        counts = Counter()
        for index in chosen:
            owner = next(category for category, pool in pools.items() if int(index) in pool)
            counts[owner] += 1
        assert counts == Counter(categories)


def test_bootstrap_interval_reproducible_with_seed_0(tmp_path):
    assert BOOTSTRAP_RESAMPLES == 2000
    assert SEED == 0
    categories = ["spatial", "spatial", "motion"]
    graph = [1, 1, 1]
    flat = [1, 0, 1]
    first = paired_bootstrap(graph, flat, categories, reps=2000, seed=0)
    second = paired_bootstrap(graph, flat, categories, reps=2000, seed=0)
    assert first == second
    assert first["n"] == 3
    assert first["diff"] == pytest.approx(1 / 3)
    assert first["corrected"] == 1
    assert first["introduced"] == 0
    # Sorted-category choice with seed 0, unrolled independently of the helper.
    rng = np.random.default_rng(0)
    got = stratified_resample_indices(["b", "a", "b"], rng)
    rng = np.random.default_rng(0)
    expect = np.concatenate([
        rng.choice(np.asarray([1], dtype=np.int64), size=1, replace=True),
        rng.choice(np.asarray([0, 2], dtype=np.int64), size=2, replace=True),
    ])
    assert np.array_equal(got, expect)
    out = _run_known(tmp_path)
    stored = next(row for row in _summary(out)["comparisons"] if row["comparison"] == "graph-flat")
    assert stored["ci_low"] == first["ci_low"]
    assert stored["ci_high"] == first["ci_high"]
    assert stored["diff"] == pytest.approx(first["diff"])
    # An item-wise resample is a different interval on this profile.
    delta = np.asarray(graph, dtype=np.int64) - np.asarray(flat, dtype=np.int64)
    iid = np.empty(2000, dtype=np.float64)
    iid_rng = np.random.default_rng(0)
    for i in range(2000):
        iid[i] = float(delta[iid_rng.integers(0, len(delta), len(delta))].mean())
    iid_low, iid_high = np.percentile(iid, [2.5, 97.5], method="linear")
    assert first["ci_low"] != pytest.approx(float(iid_low)) or first["ci_high"] != pytest.approx(float(iid_high))


def test_corrected_and_introduced_counts(tmp_path):
    out = _run_known(tmp_path)
    paired = {row["comparison"]: row for row in _rows(out / "paired_differences.csv")}
    assert set(paired) == {"graph-flat", "flat-raw", "graph-raw"}
    assert paired["graph-flat"]["role"] == "primary"
    assert paired["flat-raw"]["role"] == "exploratory"
    assert paired["graph-raw"]["role"] == "exploratory"
    assert int(paired["graph-flat"]["n"]) == 3
    assert int(paired["graph-flat"]["corrected"]) == 1
    assert int(paired["graph-flat"]["introduced"]) == 0
    assert float(paired["graph-flat"]["diff"]) == pytest.approx(1 / 3)
    assert int(paired["flat-raw"]["corrected"]) == 1
    assert int(paired["flat-raw"]["introduced"]) == 1
    assert float(paired["flat-raw"]["diff"]) == pytest.approx(0)
    assert int(paired["graph-raw"]["corrected"]) == 1
    assert int(paired["graph-raw"]["introduced"]) == 0
    assert float(paired["graph-raw"]["diff"]) == pytest.approx(1 / 3)


def test_physical_calls_versus_per_method_costs(tmp_path):
    items = [{"id": "q1", "category": "spatial"}]
    records = [
        {
            "id": "q1",
            "condition": "raw",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [_explicit("raw_answer", "raw-q1", 10, 2, 0.5)],
        },
        {
            "id": "q1",
            "condition": "flat",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [
                _explicit("extract", "ext-q1", 100, 40, 1.5),
                _explicit("flat_answer", "flt-q1", 30, 4, 0.25),
            ],
        },
        {
            "id": "q1",
            "condition": "graph",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [
                _explicit("extract", "ext-q1", 100, 40, 1.5),
                _explicit("graph_answer", "gra-q1", 35, 5, 0.5),
            ],
        },
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["raw", "flat", "graph"])
    dataset = write_dataset(tmp_path / "gold.parquet", {"q1": "A"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    costs = {(row["accounting"], row["name"]): row for row in _rows(out / "costs.csv")}
    assert int(costs["physical", "extract"]["n_calls"]) == 1
    assert int(costs["physical", "extract"]["input_tokens"]) == 100
    assert int(costs["physical", "flat_answer"]["n_calls"]) == 1
    assert int(costs["physical", "graph_answer"]["n_calls"]) == 1
    assert int(costs["physical", "raw_answer"]["n_calls"]) == 1
    assert _summary(out)["costs"]["unique_physical_calls"] == 4
    assert int(costs["method", "flat"]["n_calls"]) == 2
    assert int(costs["method", "flat"]["input_tokens"]) == 130
    assert int(costs["method", "flat"]["output_tokens"]) == 44
    assert float(costs["method", "flat"]["latency_seconds"]) == pytest.approx(1.75)
    assert int(costs["method", "graph"]["n_calls"]) == 2
    assert int(costs["method", "graph"]["input_tokens"]) == 135
    assert int(costs["method", "graph"]["output_tokens"]) == 45
    assert float(costs["method", "graph"]["latency_seconds"]) == pytest.approx(2.0)
    assert int(costs["method", "raw"]["n_calls"]) == 1
    assert int(costs["method", "raw"]["input_tokens"]) == 10
    overhead = _summary(out)["token_overhead"]
    assert overhead["flat"]["per_item"] == [{"id": "q1", "overhead": 20}]
    assert overhead["graph"]["per_item"] == [{"id": "q1", "overhead": 25}]
    assert overhead["flat"]["mean"] == pytest.approx(20)
    assert overhead["graph"]["mean"] == pytest.approx(25)


def test_null_tokens_stay_null(tmp_path):
    items = [
        {"id": "q1", "category": "spatial"},
        {"id": "q2", "category": "spatial"},
    ]
    records = [
        {
            "id": "q1",
            "condition": "raw",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [_explicit("raw_answer", "raw-q1", 10, 4, 0.5)],
        },
        {
            "id": "q2",
            "condition": "raw",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [_explicit("raw_answer", "raw-q2", None, None, 0.5)],
        },
        {
            "id": "q1",
            "condition": "flat",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [
                _explicit("extract", "ext-q1", 5, 1, 0.25),
                _explicit("flat_answer", "flt-q1", 30, 4, 0.25),
            ],
        },
        {
            "id": "q2",
            "condition": "flat",
            "status": "ok",
            "pred": "A",
            "parse_rule": 0,
            "calls": [
                _explicit("extract", "ext-q2", 5, 1, 0.25),
                _explicit("flat_answer", "flt-q2", 30, 4, 0.25),
            ],
        },
    ]
    run = write_run(tmp_path / "run", items, records, conditions=["raw", "flat"])
    dataset = write_dataset(tmp_path / "gold.parquet", {"q1": "A", "q2": "A"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    costs = {(row["accounting"], row["name"]): row for row in _rows(out / "costs.csv")}
    # 10 plus null is null, not 10 and not 0. Latency is still summed.
    assert costs["physical", "raw_answer"]["input_tokens"] == ""
    assert costs["physical", "raw_answer"]["output_tokens"] == ""
    assert float(costs["physical", "raw_answer"]["latency_seconds"]) == pytest.approx(1.0)
    assert costs["method", "raw"]["input_tokens"] == ""
    assert costs["method", "raw"]["output_tokens"] == ""
    assert int(costs["physical", "extract"]["input_tokens"]) == 10
    assert int(costs["method", "flat"]["input_tokens"]) == 70
    by_id = {row["id"]: row["overhead"] for row in _summary(out)["token_overhead"]["flat"]["per_item"]}
    assert by_id["q1"] == 20
    assert by_id["q2"] is None


def test_raw_run_config_model_identity_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {
        **config,
        "model_identity": {**config["model_identity"], "resolved_revision": "other-revision"},
    })
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "raw run model_identity" in capsys.readouterr().err


def test_raw_cache_model_identity_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_cache(raw, lambda entry: {**entry, "model_identity": {
        **entry["model_identity"],
        "resolved_revision": "not-the-evidence-revision",
    }})
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    err = capsys.readouterr().err
    assert "raw cache entry" in err
    assert "model_identity" in err


def test_raw_run_prompt_hash_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {
        **config,
        "prompt_sha256": {**config["prompt_sha256"], "raw": "other-raw-prompt"},
    })
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "prompt_sha256.raw" in capsys.readouterr().err


def test_raw_run_concurrency_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {
        **config,
        "settings": {**config["settings"], "concurrency": 2},
    })
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "settings.concurrency" in capsys.readouterr().err


def test_raw_run_max_side_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {
        **config,
        "settings": {**config["settings"], "max_side": 512},
    })
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "settings.max_side" in capsys.readouterr().err


def test_raw_run_synthetic_mismatch_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {**config, "synthetic": True})
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "synthetic" in capsys.readouterr().err


def test_raw_run_without_raw_condition_exits_2(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    _rewrite_raw_config(raw, lambda config: {**config, "conditions": ["flat"]})
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "conditions" in capsys.readouterr().err


def test_missing_raw_record_is_incomplete(tmp_path):
    evidence, raw, dataset = build_known_join(tmp_path)
    (raw / "records" / "m1__raw.json").unlink()
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=evidence, raw_run=raw, dataset=dataset, out=out))
    summary = _summary(out)
    assert summary["completeness"] == "INCOMPLETE"
    assert summary["missing_records"] == 1
    assert summary["pending_transport_records"] == 0
    row = _by_condition(out)["raw"]
    assert int(row["n"]) == 3
    assert int(row["correct"]) == 2
    assert "Completeness: INCOMPLETE" in _report(out)
    assert "Missing records: 1." in _report(out)


def test_raw_run_ids_must_cover_evidence_profile(tmp_path, capsys):
    evidence, raw, dataset = build_known_join(tmp_path)
    config = json.loads((raw / "run_config.json").read_text(encoding="utf-8"))
    config["ids"] = config["ids"][:-1]
    config["items"] = config["items"][:-1]
    (raw / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    _assert_exit_2(tmp_path, evidence, raw, dataset)
    assert "cover" in capsys.readouterr().err


def test_raw_run_join_scores_raw_from_the_other_directory(tmp_path):
    single, dataset = build_known(tmp_path / "single")
    evidence, raw, joined_dataset = build_known_join(tmp_path / "joined")
    # Size, concurrency, and prompt hashes are not cache-entry fields.
    _rewrite_raw_cache(raw, lambda entry: {
        **entry,
        "max_side": 1,
        "concurrency": 9,
        "prompt_sha256": "not-a-cache-field",
    })
    inline = tmp_path / "inline"
    joined = tmp_path / "joined-out"
    cli_analyze(SimpleNamespace(run=single, raw_run=None, dataset=dataset, out=inline))
    cli_analyze(SimpleNamespace(run=evidence, raw_run=raw, dataset=joined_dataset, out=joined))
    assert _by_condition(inline) == _by_condition(joined)
    assert _rows(inline / "paired_differences.csv") == _rows(joined / "paired_differences.csv")
    assert _summary(joined)["conditions"] == ["raw", "flat", "graph"]


def test_strict_parse_sensitivity(tmp_path):
    out = _run_known(tmp_path)
    summary = _summary(out)
    # Flat s1 is correct under pattern 2, so it drops out of the strict count.
    assert summary["by_condition"]["flat"]["correct"] == 2
    assert summary["by_condition"]["flat"]["strict_correct"] == 1
    assert summary["by_condition"]["flat"]["strict_accuracy"] == pytest.approx(1 / 3)
    assert summary["by_condition"]["flat"]["strict_macro_accuracy"] == pytest.approx(0.5)
    assert summary["by_condition"]["raw"]["strict_correct"] == 2
    assert summary["by_condition"]["graph"]["strict_correct"] == 3
    assert "Strict accuracy" in _report(out)


def test_scoring_function_reads_only_id_and_answer(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    run, dataset = build_known(tmp_path)
    seen = []
    real = pq.read_table

    def spy(source, *, columns=None, **kwargs):
        seen.append(list(columns) if columns is not None else None)
        return real(source, columns=columns, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    gold = load_gold(dataset)
    assert gold == {"s1": "A", "s2": "B", "m1": "C"}
    assert seen == [["id", "answer"]]
    seen.clear()
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    assert seen == [["id", "answer"]]
    assert "should-not-be-read" not in (out / "summary.json").read_text(encoding="utf-8")
    assert "should-not-be-read" not in _report(out)


def test_coverage_counts_from_inventory_json(tmp_path):
    full = json.loads((FIXTURES_DIR / "valid_full.json").read_text(encoding="utf-8"))
    empty = json.loads((FIXTURES_DIR / "valid_empty.json").read_text(encoding="utf-8"))
    tentative = {
        "query_frame": {"frame": "unknown", "basis": None},
        "entities": [
            {"id": "e1", "type": "object", "cross_view": None},
            {"id": "e2", "type": "object", "cross_view": None},
        ],
        "observations": [{"id": "o1", "entity": "cam1"}],
        "claims": [],
        "unknowns": [],
        "identity_links": [
            {"id": "i1", "entities": ["e1", "e2"], "status": "tentative", "uncertainty": "high"}
        ],
        "cameras": [{"id": "cam1", "image_id": "image_1"}, {"id": "cam2", "image_id": "image_2"}],
    }
    items = [
        {"id": "full", "category": "spatial"},
        {"id": "empty", "category": "spatial"},
        {"id": "tentative", "category": "motion"},
        {"id": "gone", "category": "motion"},
    ]
    records = [
        {"id": item["id"], "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0}
        for item in items
    ]
    run = write_run(
        tmp_path / "run",
        items,
        records,
        conditions=["raw"],
        inventories={"full": full, "empty": empty, "tentative": tentative},
    )
    dataset = write_dataset(tmp_path / "gold.parquet", {item["id"]: "A" for item in items})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    rows = {row["id"]: row for row in _rows(out / "coverage.csv")}
    assert rows["full"]["n_entities"] == "2"
    assert rows["full"]["n_observations"] == "3"
    assert rows["full"]["n_relation_claims"] == "1"
    assert rows["full"]["n_attribute_claims"] == "1"
    assert rows["full"]["n_unknowns"] == "1"
    assert rows["full"]["query_frame_known"] == "true"
    assert rows["full"]["cross_view_matched"] == "true"
    assert rows["full"]["has_claim"] == "true"
    assert rows["empty"]["n_entities"] == "0"
    assert rows["empty"]["n_observations"] == "0"
    assert rows["empty"]["n_relation_claims"] == "0"
    assert rows["empty"]["n_attribute_claims"] == "0"
    assert rows["empty"]["n_unknowns"] == "1"
    assert rows["empty"]["query_frame_known"] == "false"
    assert rows["empty"]["cross_view_matched"] == "false"
    assert rows["empty"]["has_claim"] == "false"
    # A tentative identity link is not a cross-view match. Cameras stay in their own list.
    assert rows["tentative"]["n_entities"] == "2"
    assert rows["tentative"]["n_observations"] == "1"
    assert rows["tentative"]["cross_view_matched"] == "false"
    assert rows["tentative"]["query_frame_known"] == "false"
    assert rows["gone"]["n_entities"] == ""
    assert rows["gone"]["query_frame_known"] == ""
    assert rows["gone"]["has_claim"] == ""
    assert _summary(out)["coverage"]["n_with_inventory"] == 3


def test_synthetic_report_is_marked_mock(tmp_path):
    run, dataset = build_known(tmp_path, synthetic=True)
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    summary = _summary(out)
    report = _report(out)
    assert summary["synthetic"] is True
    assert summary["banner"] == "MOCK / SYNTHETIC"
    assert "MOCK / SYNTHETIC" in report
    assert "not model findings" in report
    assert summary["completeness"] == "COMPLETE"
    assert "INCOMPLETE" not in report


def test_raw_only_profile_writes_raw_restricted_files(tmp_path):
    items = [
        {"id": "a", "category": "spatial"},
        {"id": "b", "category": "motion"},
    ]
    records = [
        {"id": "a", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
        {"id": "b", "condition": "raw", "status": "ok", "pred": "C", "parse_rule": 1},
    ]
    run = write_run(
        tmp_path / "run",
        items,
        records,
        conditions=["raw"],
        profile="raw_all",
    )
    dataset = write_dataset(tmp_path / "gold.parquet", {"a": "A", "b": "B"})
    out = tmp_path / "analysis"
    cli_analyze(SimpleNamespace(run=run, raw_run=None, dataset=dataset, out=out))
    for name in (
        "summary.json",
        "accuracy_by_condition.csv",
        "accuracy_by_category.csv",
        "paired_differences.csv",
        "failures.csv",
        "item_outcomes.csv",
        "coverage.csv",
        "costs.csv",
        "report.md",
    ):
        assert (out / name).is_file()
    conditions = _rows(out / "accuracy_by_condition.csv")
    assert [row["condition"] for row in conditions] == ["raw"]
    assert int(conditions[0]["n"]) == 2
    assert int(conditions[0]["correct"]) == 1
    assert _rows(out / "paired_differences.csv") == []
    assert _summary(out)["conditions"] == ["raw"]
    assert _summary(out)["token_overhead"] == {}
    assert "No paired comparison" in _report(out)


def _explicit(kind, key, input_tokens, output_tokens, latency_seconds):
    return {
        "kind": kind,
        "key": key,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
    }


def _rewrite_raw_cache(raw, mutate):
    for path in sorted((raw / "cache").rglob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps(mutate(entry)), encoding="utf-8")


def _rewrite_raw_config(raw, mutate):
    path = raw / "run_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(mutate(config)), encoding="utf-8")

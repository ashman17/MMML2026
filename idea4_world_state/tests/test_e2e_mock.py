"""Mock end-to-end run through analysis and the audit packet.

Skipped when ``data/MMSI_Bench.parquet`` is absent. Every output is synthetic.
"""

from __future__ import annotations

from idea4_world_state import runner
from idea4_world_state.tests.runner_fixtures import (
    assert_no_gold_key,
    invoke,
    read_json,
    records,
    require_dataset,
    run_argv,
)

RUN_CONFIG_FIELDS = (
    "profile",
    "run_id",
    "ids",
    "items",
    "conditions",
    "settings",
    "server_manifest",
    "model_identity",
    "prompt_sha256",
    "cache_dir",
    "dataset_path",
    "dataset_sha256",
    "prepare_manifest_sha256",
    "code_sha256",
    "freeze_sha256",
    "inventory_schema_sha256",
    "serializer_code_sha256",
    "backend",
    "synthetic",
)


def test_dev_and_eval100_mock_analyze_and_audit(tmp_path, capsys):
    require_dataset()
    cache = tmp_path / "cache"
    dev = tmp_path / "dev"
    raw = tmp_path / "eval100-raw"
    evidence = tmp_path / "eval100-evidence"
    for profile, conditions, out in (
        ("dev", "raw,flat,graph", dev),
        ("eval100", "raw", raw),
        ("eval100", "flat,graph", evidence),
    ):
        code = invoke(run_argv(tmp_path, profile, conditions, out, cache=cache))
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert captured.out.startswith("MOCK ") or "\nMOCK " in "\n" + captured.out
        config = read_json(out / "run_config.json")
        for field in RUN_CONFIG_FIELDS:
            assert field in config
        assert config["synthetic"] is True
        assert config["backend"] == "mock"
        assert config["server_manifest"] is None
        assert config["inventory_schema_sha256"] == runner.schema_sha256()
        assert config["serializer_code_sha256"] == runner.serializer_sha256()
        assert config["freeze_sha256"] is None
        assert read_json(out / "status.json")["state"] == "complete"
        assert_no_gold_key(out)

    found = records(evidence)
    assert len(found) == 200
    sample = next(iter(found.values()))
    assert sample["synthetic"] is True
    assert isinstance(sample["call_keys"], dict)
    paired = {}
    for (item_id, condition), record in found.items():
        paired.setdefault(item_id, {})[condition] = record
    one = next(iter(paired.values()))
    assert one["flat"]["call_keys"]["extract"] == one["graph"]["call_keys"]["extract"]
    assert (evidence / "inventories").glob("*.json")
    assert any(not path.name.endswith(".error.json") for path in (evidence / "inventories").glob("*.json"))
    assert list((evidence / "serializations").glob("*.flat.txt"))
    assert list((evidence / "serializations").glob("*.graph.txt"))
    for name in ("requests", "inputs", "attempts.jsonl", "predictions.jsonl", "predictions.csv"):
        assert (evidence / name).exists()
    assert len(records(raw)) == 100
    assert len(records(dev)) == 66

    analysis = evidence / "analysis"
    code = invoke([
        "analyze", "--run", str(evidence), "--raw-run", str(raw), "--out", str(analysis),
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    report = (analysis / "report.md").read_text(encoding="utf-8")
    head = "\n".join(report.splitlines()[:8])
    assert "MOCK" in head
    assert "COMPLETE" in head
    summary = read_json(analysis / "summary.json")
    assert summary["synthetic"] is True
    assert summary["completeness"] == "COMPLETE"
    assert summary["conditions"] == ["raw", "flat", "graph"]

    audit = evidence / "audit"
    code = invoke(["audit", "make", "--run", str(evidence), "--n", "20", "--out", str(audit)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    packet = read_json(audit / "packet.json")
    assert len(packet["drawn_ids"]) == 20
    assert packet["excluded_count"] == 0
    assert (audit / "index.html").is_file()
    assert (audit / "items.csv").is_file()

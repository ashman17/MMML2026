"""Hand-built run directories and a tiny gold parquet for analysis tests.

The files follow the Section 8.6 record and cache-entry fields that analysis
reads. They are not produced by the runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_IDENTITY = {
    "served_model_name": "qwen3-vl-8b",
    "model_id": "Qwen/Qwen3-VL-8B-Instruct",
    "resolved_revision": "abc123",
    "dtype": "bfloat16",
    "quantization": None,
    "max_model_len": 32768,
    "vllm_version": "0.11.0",
    "mm_processor_kwargs": None,
}


def write_dataset(path, gold_by_id, extra_columns=True):
    """Write a parquet with ``id`` and ``answer``. A decoy column is included."""
    ids = [str(item_id) for item_id in gold_by_id]
    answers = [gold_by_id[item_id] for item_id in ids]
    columns = {
        "id": pa.array(ids, type=pa.string()),
        "answer": pa.array(answers, type=pa.string()),
    }
    if extra_columns:
        columns["question"] = pa.array(["should-not-be-read"] * len(ids), type=pa.string())
        columns["thought"] = pa.array(["should-not-be-read"] * len(ids), type=pa.string())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)
    return path


def write_run(
    path,
    items,
    records,
    *,
    profile="eval200",
    conditions=None,
    synthetic=False,
    identity=None,
    max_side=768,
    concurrency=1,
    raw_prompt="raw-prompt-hash",
    inventories=None,
    run_id=None,
):
    """Write ``run_config.json``, records, and cache entries.

    ``items`` is a list of ``{id, category}``. ``records`` is a list of
    ``{id, condition, status, pred, parse_rule, calls}``. Each call is
    ``{kind, key, input_tokens, output_tokens, latency_seconds}``. Omitted
    ``calls`` get one deterministic call per kind for that condition. Flat and
    graph share ``ext-<id>``.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    identity = json.loads(json.dumps(identity or DEFAULT_IDENTITY))
    items = [{"id": str(item["id"]), "category": item["category"]} for item in items]
    if conditions is None:
        conditions = list(dict.fromkeys(record["condition"] for record in records))
    run_id = run_id or path.name
    cache_dir = path / "cache"
    categories = {item["id"]: item["category"] for item in items}
    cache_entries = {}
    record_dir = path / "records"
    record_dir.mkdir(exist_ok=True)
    for spec in records:
        item_id = str(spec["id"])
        condition = spec["condition"]
        calls = spec["calls"] if "calls" in spec else _default_calls(item_id, condition)
        call_keys = {}
        for call in calls:
            key = call["key"]
            call_keys[call["kind"]] = key
            entry = _cache_entry(call, identity=identity)
            previous = cache_entries.get(key)
            if previous is not None and previous != entry:
                raise ValueError(f"conflicting cache payloads for {key}")
            cache_entries[key] = entry
        record = {
            "id": item_id,
            "category": spec.get("category", categories[item_id]),
            "n_images": spec.get("n_images", 2),
            "partition": spec.get("partition", "explore"),
            "role": spec.get("role", "eval"),
            "condition": condition,
            "status": spec["status"],
            "pred": spec.get("pred"),
            "parse_rule": spec.get("parse_rule"),
            "call_keys": call_keys,
            "inventory_sha256": spec.get("inventory_sha256"),
            "serialization_sha256": spec.get("serialization_sha256"),
            "input_tokens": spec.get("input_tokens"),
            "output_tokens": spec.get("output_tokens"),
            "latency_seconds": spec.get("latency_seconds"),
            "synthetic": synthetic,
            "run_id": run_id,
        }
        (record_dir / f"{item_id}__{condition}.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
    for key, entry in cache_entries.items():
        destination = cache_dir / key[:2] / f"{key}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")
    if inventories:
        inventory_dir = path / "inventories"
        inventory_dir.mkdir(exist_ok=True)
        for item_id, inventory in inventories.items():
            (inventory_dir / f"{item_id}.json").write_text(
                json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
            )
    config = {
        "profile": profile,
        "run_id": run_id,
        "ids": [item["id"] for item in items],
        "items": [
            {
                "id": item["id"],
                "category": item["category"],
                "n_images": 2,
                "partition": "explore",
                "role": "eval",
            }
            for item in items
        ],
        "conditions": list(conditions),
        "synthetic": synthetic,
        "model_identity": identity,
        "settings": {
            "max_side": max_side,
            "concurrency": concurrency,
            "extract_tokens": 4096,
            "answer_tokens": 64,
            "timeout": 300,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        },
        "prompt_sha256": {
            "raw": raw_prompt,
            "extract": "extract-prompt-hash",
            "flat": "flat-prompt-hash",
            "graph": "graph-prompt-hash",
        },
        "prepare_manifest_sha256": "prepare-manifest",
        "code_sha256": "code",
        "freeze_sha256": None,
        "server_manifest": {"served_model_name": identity.get("served_model_name")},
        "cache_dir": str(cache_dir),
    }
    (path / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def known_spec():
    """Three items with a hand-checkable pattern of correct and incorrect records."""
    items = [
        {"id": "s1", "category": "spatial"},
        {"id": "s2", "category": "spatial"},
        {"id": "m1", "category": "motion"},
    ]
    gold = {"s1": "A", "s2": "B", "m1": "C"}
    records = [
        {"id": "s1", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
        {"id": "s2", "condition": "raw", "status": "ok", "pred": "B", "parse_rule": 0},
        {"id": "m1", "condition": "raw", "status": "ok", "pred": "A", "parse_rule": 0},
        {"id": "s1", "condition": "flat", "status": "ok", "pred": "A", "parse_rule": 2},
        {"id": "s2", "condition": "flat", "status": "answer_parse_error", "pred": "B", "parse_rule": 0},
        {"id": "m1", "condition": "flat", "status": "ok", "pred": "C", "parse_rule": 1},
        {"id": "s1", "condition": "graph", "status": "ok", "pred": "A", "parse_rule": 0},
        {"id": "s2", "condition": "graph", "status": "ok", "pred": "B", "parse_rule": 1},
        {"id": "m1", "condition": "graph", "status": "ok", "pred": "C", "parse_rule": 0},
    ]
    return items, gold, records


def build_known(root, *, synthetic=False):
    root = Path(root)
    items, gold, records = known_spec()
    run = write_run(root / "run", items, records, synthetic=synthetic, profile="eval200")
    dataset = write_dataset(root / "gold.parquet", gold)
    return run, dataset


def build_known_join(root):
    """Evidence run has flat and graph. The raw run covers the same ids."""
    root = Path(root)
    items, gold, records = known_spec()
    evidence = write_run(
        root / "evidence",
        items,
        [record for record in records if record["condition"] != "raw"],
        conditions=["flat", "graph"],
        profile="eval200",
    )
    raw = write_run(
        root / "raw",
        items,
        [record for record in records if record["condition"] == "raw"],
        conditions=["raw"],
        profile="raw_all",
    )
    dataset = write_dataset(root / "gold.parquet", gold)
    return evidence, raw, dataset


def _default_calls(item_id, condition):
    if condition == "raw":
        return [_call("raw_answer", f"raw-{item_id}", 10, 2, 0.1)]
    if condition == "flat":
        return [
            _call("extract", f"ext-{item_id}", 100, 40, 1.0),
            _call("flat_answer", f"flt-{item_id}", 20, 2, 0.2),
        ]
    if condition == "graph":
        return [
            _call("extract", f"ext-{item_id}", 100, 40, 1.0),
            _call("graph_answer", f"gra-{item_id}", 25, 2, 0.3),
        ]
    raise ValueError(f"no default calls for {condition}")


def _call(kind, key, input_tokens, output_tokens, latency_seconds):
    return {
        "kind": kind,
        "key": key,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
    }


def _cache_entry(call, *, identity):
    """Section 8.6 cache fields only. Size, concurrency, and prompt hashes stay on run_config."""
    return {
        "key": call["key"],
        "kind": call["kind"],
        "status": "ok",
        "text": "",
        "finish_reason": "stop",
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "latency_seconds": call["latency_seconds"],
        "served_model": identity.get("served_model_name"),
        "response_id": f"resp-{call['key']}",
        "created_at": "2026-01-01T00:00:00Z",
        "model_identity": identity,
    }

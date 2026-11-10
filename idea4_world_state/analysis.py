"""Score a world-state run and write the Section 8.8 analysis outputs.

`load_gold` is the only function in this package that reads the dataset
`answer` column. It reads `id` and `answer` and nothing else.

A record is correct only when its status is `ok` and `pred` equals that gold
letter. Every other status is incorrect. Denominators are the profile `ids`
in `run_config.json`, including missing records.

Paired intervals resample items with replacement inside each category, at
that category's original size. Categories are visited in sorted order so
`numpy.random.default_rng(0)` is reproducible. This is not an image-group
bootstrap.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .contracts import (
    BOOTSTRAP_RESAMPLES,
    CALL_KINDS,
    COMPARISONS,
    CONDITIONS,
    PRIMARY_COMPARISON,
    RECORD_STATUSES,
    SEED,
)

# Columns of the scoring read. Nothing else in the dataset is requested.
GOLD_COLUMNS = ("id", "answer")

COMPARISON_SIDES = {
    "graph-flat": ("graph", "flat"),
    "flat-raw": ("flat", "raw"),
    "graph-raw": ("graph", "raw"),
}
METHOD_KINDS = {
    "raw": ("raw_answer",),
    "flat": ("extract", "flat_answer"),
    "graph": ("extract", "graph_answer"),
}
COVERAGE_FIELDS = (
    "n_entities",
    "n_observations",
    "n_relation_claims",
    "n_attribute_claims",
    "n_unknowns",
    "query_frame_known",
    "cross_view_matched",
    "has_claim",
)
COVERAGE_NUMERIC = (
    "n_entities",
    "n_observations",
    "n_relation_claims",
    "n_attribute_claims",
    "n_unknowns",
)
COVERAGE_FLAGS = ("query_frame_known", "cross_view_matched", "has_claim")
FAILURE_STATUSES = tuple(RECORD_STATUSES) + ("missing",)
ANALYSIS_FILES = (
    "summary.json",
    "accuracy_by_condition.csv",
    "accuracy_by_category.csv",
    "paired_differences.csv",
    "failures.csv",
    "item_outcomes.csv",
    "coverage.csv",
    "costs.csv",
    "report.md",
)

LIMITS = (
    "Automatic screening and public benchmark exposure.",
    "Rare two-image multi-step items and small category samples.",
    "A fixed representation budget.",
    "Exact-image groups do not establish independent physical scenes.",
    "Unweighted accuracy on an evidence profile is not an estimate for all 810 two-image questions or all 1,000 questions.",
    "Identical predictions or a degenerate interval do not establish population equivalence.",
    "Summing per-method costs double-counts the shared physical extraction.",
    "Corrected human graphs and extension beyond two images are later research options.",
)


class AnalysisError(Exception):
    """Configuration or contract violation. The CLI exits 2."""


def load_gold(dataset):
    """Return ``{id: gold letter}``.

    This is the scoring boundary: the only dataset read in the package, and
    it requests only ``id`` and ``answer``.
    """
    table = pq.read_table(dataset, columns=list(GOLD_COLUMNS))
    names = set(table.column_names)
    if names != set(GOLD_COLUMNS):
        raise AnalysisError(f"scoring read unexpected columns: {sorted(names)}")
    gold = {}
    for item_id, answer in zip(table.column("id").to_pylist(), table.column("answer").to_pylist()):
        key = str(item_id)
        if key in gold:
            raise AnalysisError(f"duplicate dataset id {key}")
        gold[key] = None if answer is None else str(answer)
    return gold


def is_strict_parse(parse_rule):
    """True for extract patterns 0 and 1. Booleans are not patterns."""
    return type(parse_rule) is int and parse_rule in (0, 1)


def record_is_correct(status, pred, gold, parse_rule=None, *, strict=False):
    """Correct only for status ``ok`` and an exact gold match."""
    if status != "ok":
        return False
    if pred is None or gold is None or str(pred) != str(gold):
        return False
    if strict and not is_strict_parse(parse_rule):
        return False
    return True


def stratified_resample_indices(categories, rng):
    """Draw one stratified resample.

    Items are drawn with replacement within each category, and each category
    keeps its original count. The result is concatenated in sorted category
    order. Returned values are indices into ``categories``.
    """
    groups = defaultdict(list)
    for index, category in enumerate(categories):
        groups[category].append(index)
    if not groups:
        return np.asarray([], dtype=np.int64)
    parts = []
    for category in sorted(groups):
        pool = np.asarray(groups[category], dtype=np.int64)
        parts.append(rng.choice(pool, size=pool.size, replace=True))
    return np.concatenate(parts)


def paired_bootstrap(correct_x, correct_y, categories, reps=BOOTSTRAP_RESAMPLES, seed=SEED):
    """Mean of correct_X minus correct_Y, with a stratified percentile interval.

    Corrected: Y wrong and X right. Introduced: Y right and X wrong.
    """
    x = np.asarray(list(correct_x), dtype=np.int64)
    y = np.asarray(list(correct_y), dtype=np.int64)
    categories = list(categories)
    if len(x) != len(y) or len(x) != len(categories):
        raise AnalysisError("paired comparison lengths differ")
    corrected = int(np.sum((y == 0) & (x == 1)))
    introduced = int(np.sum((y == 1) & (x == 0)))
    if len(x) == 0:
        return {
            "n": 0,
            "diff": None,
            "ci_low": None,
            "ci_high": None,
            "corrected": corrected,
            "introduced": introduced,
        }
    delta = x - y
    rng = np.random.default_rng(seed)
    boots = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        chosen = stratified_resample_indices(categories, rng)
        boots[i] = float(delta[chosen].mean())
    low, high = np.percentile(boots, [2.5, 97.5], method="linear")
    return {
        "n": int(len(x)),
        "diff": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "corrected": corrected,
        "introduced": introduced,
    }


def coverage_metrics(inventory):
    """Section 6 coverage from an inventory object. Does not validate ws-1.

    ``n_entities`` is the length of ``entities``. Cameras are a separate
    ``cameras`` list. A query frame is known when ``query_frame.frame`` is
    present and is not ``unknown``. An entity is matched across views when
    its ``cross_view.status`` is ``same``. Identity links are not matches.
    """
    if not isinstance(inventory, dict):
        raise AnalysisError("inventory JSON must be an object")
    entities = _list_field(inventory, "entities")
    observations = _list_field(inventory, "observations")
    claims = _list_field(inventory, "claims")
    unknowns = _list_field(inventory, "unknowns")
    matched = False
    for entity in entities:
        if not isinstance(entity, dict):
            raise AnalysisError("inventory entities must be objects")
        cross = entity.get("cross_view")
        if isinstance(cross, dict) and cross.get("status") == "same":
            matched = True
    n_entities = len(entities)
    n_relation = 0
    n_attribute = 0
    for claim in claims:
        if not isinstance(claim, dict):
            raise AnalysisError("inventory claims must be objects")
        kind = claim.get("kind")
        if kind == "relation":
            n_relation += 1
        elif kind == "attribute":
            n_attribute += 1
    frame = None
    query = inventory.get("query_frame")
    if isinstance(query, dict):
        frame = query.get("frame")
    return {
        "n_entities": n_entities,
        "n_observations": len(observations),
        "n_relation_claims": n_relation,
        "n_attribute_claims": n_attribute,
        "n_unknowns": len(unknowns),
        "query_frame_known": frame is not None and frame != "unknown",
        "cross_view_matched": matched,
        "has_claim": len(claims) > 0,
    }


def analyze(run, dataset, out, raw_run=None):
    """Score ``run`` and write every Section 8.8 analysis file under ``out``."""
    run = Path(run)
    out = Path(out)
    config = _load_config(run)
    gold = load_gold(dataset)
    ids, conditions, records, raw_config = _assemble_records(run, config, raw_run)
    categories = [_category_for(item_id, config, records, conditions) for item_id in ids]
    for item_id in ids:
        if item_id not in gold:
            raise AnalysisError(f"dataset has no answer for {item_id}")

    cache_dirs = _cache_dirs(run, config, raw_run, raw_config)
    entries, calls_of = _resolve_calls(records, cache_dirs)

    correct = {}
    strict = {}
    for item_id, category in zip(ids, categories):
        for condition in conditions:
            record = records.get((item_id, condition))
            status = None if record is None else record.get("status")
            pred = None if record is None else record.get("pred")
            parse_rule = None if record is None else record.get("parse_rule")
            correct[item_id, condition] = record_is_correct(status, pred, gold[item_id], parse_rule)
            strict[item_id, condition] = record_is_correct(
                status, pred, gold[item_id], parse_rule, strict=True
            )

    by_condition = {}
    category_rows = []
    failure_rows = []
    missing = 0
    pending = 0
    for condition in conditions:
        values = [int(correct[item_id, condition]) for item_id in ids]
        strict_values = [int(strict[item_id, condition]) for item_id in ids]
        cat_acc = []
        strict_cat_acc = []
        for category in sorted(set(categories)):
            member = [i for i, cat in zip(ids, categories) if cat == category]
            n_cat = len(member)
            n_correct = sum(correct[i, condition] for i in member)
            n_strict = sum(strict[i, condition] for i in member)
            accuracy = n_correct / n_cat
            cat_acc.append(accuracy)
            strict_cat_acc.append(n_strict / n_cat)
            category_rows.append({
                "condition": condition,
                "category": category,
                "n": n_cat,
                "correct": n_correct,
                "accuracy": accuracy,
            })
        statuses = {name: 0 for name in FAILURE_STATUSES}
        for item_id in ids:
            record = records.get((item_id, condition))
            if record is None:
                statuses["missing"] += 1
                missing += 1
            else:
                status = record.get("status")
                if status not in statuses:
                    statuses[status] = 0
                statuses[status] += 1
                if status == "pending_transport":
                    pending += 1
        for status in list(FAILURE_STATUSES) + [s for s in statuses if s not in FAILURE_STATUSES]:
            if status in FAILURE_STATUSES or statuses[status]:
                failure_rows.append({
                    "condition": condition,
                    "status": status,
                    "count": statuses.get(status, 0),
                })
        n = len(ids)
        by_condition[condition] = {
            "n": n,
            "correct": int(sum(values)),
            "accuracy": (sum(values) / n) if n else None,
            "macro_accuracy": (sum(cat_acc) / len(cat_acc)) if cat_acc else None,
            "strict_correct": int(sum(strict_values)),
            "strict_accuracy": (sum(strict_values) / n) if n else None,
            "strict_macro_accuracy": (sum(strict_cat_acc) / len(strict_cat_acc)) if strict_cat_acc else None,
        }

    comparisons = []
    for name in COMPARISONS:
        x_name, y_name = COMPARISON_SIDES[name]
        if x_name not in conditions or y_name not in conditions:
            continue
        stats = paired_bootstrap(
            [int(correct[i, x_name]) for i in ids],
            [int(correct[i, y_name]) for i in ids],
            categories,
        )
        comparisons.append({
            "comparison": name,
            "role": "primary" if name == PRIMARY_COMPARISON else "exploratory",
            **stats,
        })

    outcome_conditions = _item_outcome_conditions(config, conditions)
    outcome_rows = _item_outcome_rows(ids, categories, outcome_conditions, records, correct, gold)
    coverage_rows = _coverage_rows(run, ids, categories)
    costs = _cost_rows(ids, conditions, calls_of, entries)
    overhead = _token_overhead(ids, conditions, calls_of, entries)
    present = sum(1 for item_id in ids for condition in conditions if (item_id, condition) in records)
    incomplete = missing > 0 or pending > 0
    synthetic = bool(config["synthetic"])
    summary = {
        "completeness": "INCOMPLETE" if incomplete else "COMPLETE",
        "synthetic": synthetic,
        "banner": "MOCK / SYNTHETIC" if synthetic else None,
        "missing_records": missing,
        "pending_transport_records": pending,
        "expected_records": len(ids) * len(conditions),
        "present_records": present,
        "profile": config.get("profile"),
        "run_id": config.get("run_id"),
        "n_items": len(ids),
        "conditions": list(conditions),
        "by_condition": by_condition,
        "by_category": category_rows,
        "comparisons": comparisons,
        "failures": failure_rows,
        "item_outcomes": {
            "rows": len(outcome_rows),
            "by_condition": {condition: len(ids) for condition in outcome_conditions},
        },
        "costs": costs,
        "token_overhead": overhead,
        "coverage": _coverage_summary(coverage_rows),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": SEED,
            "within": "category",
            "category_order": "sorted",
            "percentile": [2.5, 97.5],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "summary.json", summary)
    _write_csv(
        out / "accuracy_by_condition.csv",
        ["condition", "n", "correct", "accuracy", "macro_accuracy"],
        [
            {"condition": condition, **{k: by_condition[condition][k] for k in ("n", "correct", "accuracy", "macro_accuracy")}}
            for condition in conditions
        ],
    )
    _write_csv(
        out / "accuracy_by_category.csv",
        ["condition", "category", "n", "correct", "accuracy"],
        category_rows,
    )
    _write_csv(
        out / "paired_differences.csv",
        ["comparison", "role", "n", "diff", "ci_low", "ci_high", "corrected", "introduced"],
        comparisons,
    )
    _write_csv(out / "failures.csv", ["condition", "status", "count"], failure_rows)
    _write_csv(
        out / "item_outcomes.csv",
        ["id", "category", "condition", "status", "pred", "gold", "correct"],
        outcome_rows,
    )
    _write_csv(
        out / "coverage.csv",
        ["id", "category", *COVERAGE_FIELDS],
        coverage_rows,
    )
    _write_csv(
        out / "costs.csv",
        ["accounting", "name", "n_calls", "input_tokens", "output_tokens", "latency_seconds"],
        costs["rows"],
    )
    (out / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def cli_analyze(args):
    try:
        analyze(args.run, args.dataset, args.out, raw_run=getattr(args, "raw_run", None))
    except AnalysisError as exc:
        print(f"analyze: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def _item_outcome_conditions(config, scored):
    """Profile condition order, then any joined condition the config did not list."""
    listed = []
    for condition in config.get("conditions") or []:
        if condition in scored and condition not in listed:
            listed.append(condition)
    return listed + [condition for condition in scored if condition not in listed]


def _item_outcome_rows(ids, categories, conditions, records, correct, gold):
    rows = []
    for item_id, category in zip(ids, categories):
        for condition in conditions:
            record = records.get((item_id, condition))
            if record is None:
                status, pred = "missing", None
            else:
                status, pred = record.get("status"), record.get("pred")
            rows.append({
                "id": item_id,
                "category": category,
                "condition": condition,
                "status": status,
                "pred": pred,
                "gold": gold[item_id],
                "correct": int(bool(correct[item_id, condition])),
            })
    return rows


def _assemble_records(run, config, raw_run):
    ids = _profile_ids(config)
    evidence_conditions = _condition_list(config)
    evidence_records = _load_records(run)
    raw_config = None
    if raw_run is None:
        records = evidence_records
        selected = set(evidence_conditions)
    else:
        raw_path = Path(raw_run)
        raw_config = _load_config(raw_path)
        _verify_raw_run(config, raw_config, raw_path, ids)
        raw_records = _load_records(raw_path)
        selected = {condition for condition in evidence_conditions if condition != "raw"}
        selected.add("raw")
        records = {}
        for key, record in evidence_records.items():
            if key[1] == "raw":
                continue
            records[key] = record
        for item_id in ids:
            record = raw_records.get((item_id, "raw"))
            if record is not None:
                records[(item_id, "raw")] = record
        for record in records.values():
            if record.get("synthetic") != config["synthetic"]:
                raise AnalysisError(
                    f"record {record.get('id')} {record.get('condition')} synthetic flag does not match the evidence run"
                )
    conditions = [condition for condition in CONDITIONS if condition in selected]
    expected = {(item_id, condition) for item_id in ids for condition in conditions}
    for key, record in records.items():
        if key not in expected:
            raise AnalysisError(f"unexpected record {key[0]} condition {key[1]}")
        if raw_run is None and record.get("synthetic") != config["synthetic"]:
            raise AnalysisError(
                f"record {record.get('id')} {record.get('condition')} synthetic flag does not match the run"
            )
        if "status" not in record:
            raise AnalysisError(f"record {key[0]} {key[1]} is missing status")
    return ids, conditions, records, raw_config


def _verify_raw_run(evidence, raw_config, raw_path, evidence_ids):
    """Join checks from Section 8.7. Failures are contract errors (exit 2).

    Image size, concurrency, and the raw prompt hash are compared on
    ``run_config.json``. Cache entries keep the Section 8.6 fields, so the
    only cache comparison is ``model_identity``.
    """
    if bool(raw_config.get("synthetic")) != bool(evidence.get("synthetic")):
        raise AnalysisError("raw run synthetic flag does not match the evidence run")
    raw_ids = {str(item_id) for item_id in raw_config.get("ids") or []}
    missing_ids = [item_id for item_id in evidence_ids if item_id not in raw_ids]
    if missing_ids:
        raise AnalysisError(
            "raw run ids do not cover the evidence profile: " + ", ".join(missing_ids)
        )
    if "raw" not in _condition_list(raw_config):
        raise AnalysisError("raw run conditions do not include raw")
    if raw_config.get("model_identity") != evidence.get("model_identity"):
        raise AnalysisError("raw run model_identity does not match the evidence run")
    for field in ("max_side", "concurrency"):
        if _setting(raw_config, field) != _setting(evidence, field):
            raise AnalysisError(f"raw run settings.{field} does not match the evidence run")
    if _raw_prompt_hash(raw_config) != _raw_prompt_hash(evidence):
        raise AnalysisError("raw run prompt_sha256.raw does not match the evidence run")
    identity = evidence.get("model_identity")
    raw_records = _load_records(raw_path)
    cache_dir = _resolve_cache_dir(raw_path, raw_config)
    for item_id in evidence_ids:
        record = raw_records.get((item_id, "raw"))
        if record is None:
            continue
        mapping = _call_key_map(record.get("call_keys"), item_id, "raw")
        raw_keys = [key for kind, key in mapping.items() if kind == "raw_answer"]
        if not raw_keys:
            raise AnalysisError(f"raw record {item_id} has no raw_answer cache entry to verify")
        for key in raw_keys:
            entry = _read_cache_entry(cache_dir, key)
            if entry.get("model_identity") != identity:
                raise AnalysisError(f"raw cache entry {key} model_identity does not match the evidence run")


def _setting(config, field):
    settings = config.get("settings")
    if not isinstance(settings, dict) or field not in settings:
        return None
    return settings[field]


def _raw_prompt_hash(config):
    prompts = config.get("prompt_sha256")
    if isinstance(prompts, dict) and "raw" in prompts:
        return prompts["raw"]
    return None


def _profile_ids(config):
    if "ids" not in config or not isinstance(config["ids"], list):
        raise AnalysisError("run_config.json is missing ids")
    ids = [str(item_id) for item_id in config["ids"]]
    if len(ids) != len(set(ids)):
        raise AnalysisError("run_config.json ids are not unique")
    return ids


def _condition_list(config):
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise AnalysisError("run_config.json is missing conditions")
    unknown = [condition for condition in conditions if condition not in CONDITIONS]
    if unknown:
        raise AnalysisError(f"unknown conditions: {', '.join(map(str, unknown))}")
    if len(conditions) != len(set(conditions)):
        raise AnalysisError("run_config.json conditions are not unique")
    return list(conditions)


def _category_for(item_id, config, records, conditions):
    listed = None
    for item in config.get("items") or []:
        if str(item.get("id")) == item_id:
            listed = item.get("category")
            break
    found = None
    for condition in conditions:
        record = records.get((item_id, condition))
        if record is not None and record.get("category") is not None:
            found = record["category"]
            break
    if listed is None and found is None:
        raise AnalysisError(f"no category for {item_id}")
    if listed is not None and found is not None and listed != found:
        raise AnalysisError(f"category mismatch for {item_id}")
    return listed if listed is not None else found


def _cache_dirs(run, config, raw_run, raw_config):
    evidence_cache = _resolve_cache_dir(run, config)
    raw_cache = evidence_cache if raw_config is None else _resolve_cache_dir(Path(raw_run), raw_config)
    return {"evidence": evidence_cache, "raw": raw_cache, "joined": raw_config is not None}


def _resolve_cache_dir(run, config):
    raw = config.get("cache_dir")
    if not raw:
        raise AnalysisError("run_config.json is missing cache_dir")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(run) / path
    return path


def _resolve_calls(records, cache_dirs):
    entries = {}
    calls_of = {}
    for key, record in records.items():
        item_id, condition = key
        mapping = _call_key_map(record.get("call_keys"), item_id, condition)
        use_raw = cache_dirs["joined"] and condition == "raw"
        cache_dir = cache_dirs["raw"] if use_raw else cache_dirs["evidence"]
        resolved = {}
        for kind, cache_key in mapping.items():
            if kind not in CALL_KINDS:
                raise AnalysisError(f"unknown call kind {kind} on {item_id} {condition}")
            entry = entries.get(cache_key)
            if entry is None:
                entry = _read_cache_entry(cache_dir, cache_key)
                if entry.get("kind") != kind:
                    raise AnalysisError(
                        f"cache entry {cache_key} kind {entry.get('kind')!r} does not match call_keys {kind}"
                    )
                entries[cache_key] = entry
            elif entry.get("kind") != kind:
                raise AnalysisError(f"cache key {cache_key} is used for more than one kind")
            resolved[kind] = cache_key
        calls_of[key] = resolved
    return entries, calls_of


def _call_key_map(value, item_id, condition):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AnalysisError(
            f"record {item_id} {condition} call_keys must be an object mapping call kind to cache key"
        )
    return {str(kind): str(cache_key) for kind, cache_key in value.items()}


def _read_cache_entry(cache_dir, key):
    path = Path(cache_dir) / key[:2] / f"{key}.json"
    entry = _load_json(path, f"cache entry {key}")
    if not isinstance(entry, dict):
        raise AnalysisError(f"cache entry {key} must be an object")
    if "key" in entry and entry["key"] != key:
        raise AnalysisError(f"cache entry {key} key field does not match the path")
    return entry


def _coverage_rows(run, ids, categories):
    rows = []
    for item_id, category in zip(ids, categories):
        path = Path(run) / "inventories" / f"{item_id}.json"
        row = {"id": item_id, "category": category}
        if not path.is_file():
            for field in COVERAGE_FIELDS:
                row[field] = None
        else:
            inventory = _load_json(path, f"inventory {item_id}")
            row.update(coverage_metrics(inventory))
        rows.append(row)
    return rows


def _coverage_summary(rows):
    present = [row for row in rows if row["n_entities"] is not None]

    def numeric(field, group):
        values = [row[field] for row in group]
        if not values:
            return {"n": 0, "mean": None, "min": None, "max": None}
        return {
            "n": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def flags(field, group):
        values = [row[field] for row in group if row[field] is not None]
        return {"n": len(values), "true": sum(bool(value) for value in values)}

    by_category = []
    for category in sorted({row["category"] for row in rows}):
        member = [row for row in rows if row["category"] == category]
        known = [row for row in member if row["n_entities"] is not None]
        by_category.append({
            "category": category,
            "n_items": len(member),
            "n_with_inventory": len(known),
            "numeric": {field: numeric(field, known) for field in COVERAGE_NUMERIC},
            "flags": {field: flags(field, known) for field in COVERAGE_FLAGS},
        })
    return {
        "n_items": len(rows),
        "n_with_inventory": len(present),
        "numeric": {field: numeric(field, present) for field in COVERAGE_NUMERIC},
        "flags": {field: flags(field, present) for field in COVERAGE_FLAGS},
        "by_category": by_category,
    }


def _cost_rows(ids, conditions, calls_of, entries):
    physical_keys = {}
    method_keys = {condition: {} for condition in conditions}
    for item_id in ids:
        for condition in conditions:
            for kind, key in calls_of.get((item_id, condition), {}).items():
                physical_keys[key] = entries[key]
                if condition in METHOD_KINDS and kind in METHOD_KINDS[condition]:
                    method_keys[condition][key] = entries[key]
    rows = []
    for kind in CALL_KINDS:
        grouped = [entry for entry in physical_keys.values() if entry.get("kind") == kind]
        if grouped:
            rows.append({"accounting": "physical", "name": kind, **_usage(grouped)})
    for condition in conditions:
        grouped = list(method_keys[condition].values())
        rows.append({"accounting": "method", "name": condition, **_usage(grouped)})
    return {"unique_physical_calls": len(physical_keys), "rows": rows}


def _usage(grouped):
    return {
        "n_calls": len(grouped),
        "input_tokens": _sum_nullable(entry.get("input_tokens") for entry in grouped),
        "output_tokens": _sum_nullable(entry.get("output_tokens") for entry in grouped),
        "latency_seconds": _sum_nullable(entry.get("latency_seconds") for entry in grouped),
    }


def _token_overhead(ids, conditions, calls_of, entries):
    """Answer-call prompt tokens minus raw prompt tokens on the same item.

    A null on either side stays null and is not treated as zero. The mean uses
    only items where both counts are known.
    """
    if "raw" not in conditions:
        return {}
    overhead = {}
    for condition in conditions:
        if condition not in ("flat", "graph"):
            continue
        answer_kind = f"{condition}_answer"
        per_item = []
        known = []
        for item_id in ids:
            answer = _input_tokens(calls_of, entries, item_id, condition, answer_kind)
            raw = _input_tokens(calls_of, entries, item_id, "raw", "raw_answer")
            value = None if answer is None or raw is None else answer - raw
            per_item.append({"id": item_id, "overhead": value})
            if value is not None:
                known.append(value)
        overhead[condition] = {
            "n": len(ids),
            "n_known": len(known),
            "mean": (sum(known) / len(known)) if known else None,
            "per_item": per_item,
        }
    return overhead


def _input_tokens(calls_of, entries, item_id, condition, kind):
    key = calls_of.get((item_id, condition), {}).get(kind)
    if key is None:
        return None
    value = entries[key].get("input_tokens")
    return None if value is None else value


def _sum_nullable(values):
    """Sum numbers. An empty group or any null makes the total null, not zero."""
    total = 0
    seen = False
    for value in values:
        if value is None:
            return None
        total += value
        seen = True
    if not seen:
        return None
    return total


def _list_field(inventory, name):
    value = inventory.get(name, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisError(f"inventory {name} must be a list")
    return value


def _load_config(run):
    config = _load_json(Path(run) / "run_config.json", "run_config.json")
    if not isinstance(config, dict):
        raise AnalysisError("run_config.json must be an object")
    if "synthetic" not in config:
        raise AnalysisError("run_config.json is missing synthetic")
    return config


def _load_records(run):
    directory = Path(run) / "records"
    if not directory.is_dir():
        return {}
    records = {}
    for path in sorted(directory.glob("*.json")):
        record = _load_json(path, f"record {path.name}")
        if not isinstance(record, dict):
            raise AnalysisError(f"record {path.name} must be an object")
        if "id" not in record or "condition" not in record:
            raise AnalysisError(f"record {path.name} is missing id or condition")
        key = (str(record["id"]), record["condition"])
        if key in records:
            raise AnalysisError(f"duplicate record {key[0]} {key[1]}")
        record = dict(record)
        record["id"] = key[0]
        records[key] = record
    return records


def _load_json(path, what):
    path = Path(path)
    if not path.is_file():
        raise AnalysisError(f"missing {what}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"invalid JSON in {what}: {path}") from exc


def _write_json(path, payload):
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _cell(row.get(name)) for name in fieldnames})


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _render_report(summary):
    lines = ["# World-state analysis", ""]
    if summary["banner"]:
        lines.extend([summary["banner"], "", "These results are synthetic and are not model findings.", ""])
    lines.append(f"Completeness: {summary['completeness']}")
    lines.append(
        f"Missing records: {summary['missing_records']}. "
        f"Pending transport records: {summary['pending_transport_records']}."
    )
    lines.append(
        f"Profile: `{summary['profile']}`. Run: `{summary['run_id']}`. "
        f"Items: {summary['n_items']}. Conditions: {', '.join(summary['conditions']) or 'none'}."
    )
    lines.append(
        f"Records: {summary['present_records']} / {summary['expected_records']}."
    )
    lines.extend([
        "",
        "Denominators are the full profile. A record counts as correct only when its status is `ok` and `pred` equals the gold letter. Every other status, including missing records, counts as incorrect.",
        "",
        "Per-item status, prediction, gold, and correctness are in `item_outcomes.csv`.",
        "",
        "## Accuracy",
        "",
        "| Condition | Correct / N | Accuracy | Macro accuracy | Strict correct / N | Strict accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for condition in summary["conditions"]:
        row = summary["by_condition"][condition]
        lines.append(
            f"| {condition} | {row['correct']} / {row['n']} | {_fmt(row['accuracy'])} | "
            f"{_fmt(row['macro_accuracy'])} | {row['strict_correct']} / {row['n']} | {_fmt(row['strict_accuracy'])} |"
        )
    lines.extend([
        "",
        "Strict accuracy counts a record only when it is correct and `parse_rule` is 0 or 1.",
        "",
        "## Per category",
        "",
        "| Condition | Category | Correct / N | Accuracy |",
        "| --- | --- | ---: | ---: |",
    ])
    for row in summary["by_category"]:
        lines.append(
            f"| {row['condition']} | {row['category']} | {row['correct']} / {row['n']} | {_fmt(row['accuracy'])} |"
        )
    lines.extend([
        "",
        "## Paired differences",
        "",
        "Each comparison is the mean of correct_X minus correct_Y. "
        "`graph-flat` is primary. `flat-raw` and `graph-raw` are exploratory.",
        "",
        f"Intervals are the 2.5 and 97.5 percentiles of {summary['bootstrap']['resamples']} "
        f"stratified bootstrap resamples (`numpy.random.default_rng({summary['bootstrap']['seed']})`). "
        "Each resample draws items with replacement within each category, at that category's original size. "
        "Categories are visited in sorted order. Image groups are not the resampling unit.",
        "",
        "Corrected items are wrong under Y and right under X. Introduced items are right under Y and wrong under X.",
        "",
        "| Comparison | Role | N | Mean difference | 95% interval | Corrected | Introduced |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ])
    if not summary["comparisons"]:
        lines.append("")
        lines.append("No paired comparison: both sides are not in this run.")
    for row in summary["comparisons"]:
        interval = f"{_fmt(row['ci_low'])} to {_fmt(row['ci_high'])}"
        lines.append(
            f"| {row['comparison']} | {row['role']} | {row['n']} | {_fmt(row['diff'])} | "
            f"{interval} | {row['corrected']} | {row['introduced']} |"
        )
    lines.extend([
        "",
        "## Failures",
        "",
        "| Condition | Status | Count |",
        "| --- | --- | ---: |",
    ])
    for row in summary["failures"]:
        lines.append(f"| {row['condition']} | {row['status']} | {row['count']} |")
    lines.extend([
        "",
        "## Evidence coverage",
        "",
        "One row per profile item in `coverage.csv`, read from `inventories/<id>.json`. "
        "Items without an inventory are blank, not zero. Counts are entities, observations, "
        "relation claims, attribute claims, and unknowns, plus whether the query frame is known, "
        "whether any entity has `cross_view.status` `same`, and whether the inventory has at least one claim.",
        "",
        f"Inventories present: {summary['coverage']['n_with_inventory']} / {summary['coverage']['n_items']}.",
        "",
        "| Field | N | Mean | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for field, stats in summary["coverage"]["numeric"].items():
        lines.append(
            f"| {field} | {stats['n']} | {_fmt(stats['mean'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )
    lines.extend(["", "| Field | N | True |", "| --- | ---: | ---: |"])
    for field, stats in summary["coverage"]["flags"].items():
        lines.append(f"| {field} | {stats['n']} | {stats['true']} |")
    lines.extend(["", "### Per category", ""])
    for category_row in summary["coverage"]["by_category"]:
        lines.append(
            f"- `{category_row['category']}`: {category_row['n_with_inventory']} / {category_row['n_items']} inventories."
        )
    lines.extend([
        "",
        "## Costs",
        "",
        f"Unique physical calls: {summary['costs']['unique_physical_calls']}. "
        "Physical rows deduplicate cache keys. Each evidence method includes extraction once. "
        "Adding method rows together double-counts that shared extraction. "
        "Null token usage stays null and is not summed as zero.",
        "",
        "| Accounting | Name | Calls | Input tokens | Output tokens | Latency (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in summary["costs"]["rows"]:
        lines.append(
            f"| {row['accounting']} | {row['name']} | {row['n_calls']} | "
            f"{_fmt(row['input_tokens'])} | {_fmt(row['output_tokens'])} | {_fmt(row['latency_seconds'])} |"
        )
    if summary["token_overhead"]:
        lines.extend([
            "",
            "## Evidence token overhead",
            "",
            "Answer-call prompt tokens minus raw prompt tokens on the same item. "
            "If either count is null, that item's overhead is null and is omitted from the mean.",
            "",
            "| Method | Known items | Mean overhead |",
            "| --- | ---: | ---: |",
        ])
        for name, row in summary["token_overhead"].items():
            lines.append(f"| {name} | {row['n_known']} / {row['n']} | {_fmt(row['mean'])} |")
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {line}" for line in LIMITS)
    lines.append("")
    return "\n".join(lines)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.4f}"

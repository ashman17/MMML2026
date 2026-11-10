"""Prepare the world-state sample and write artifacts/prepare_v1.

Sampling follows IMPLEMENTATION_PLAN.md Section 2. Inference reads only
``contracts.INFERENCE_COLUMNS``. ``answer`` and ``thought`` are never read.
"""

from __future__ import annotations

import hashlib
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from idea3_consistency.mmsi_consistency import parse_question
from idea4_visual_reasoning.data import image_groups, images_of, pixel_hash, read_json, sha256_file
from idea4_visual_reasoning.render import decode_image

from .contracts import DATASET_SHA256, EXIT_CODES, INFERENCE_COLUMNS, SEED

VERSION = "prepare-v1"
EVAL_LIMIT = 200
EVAL100_LIMIT = 100
DEV_PER_CATEGORY = 2
TIMING_IMAGE_COUNTS = range(3, 11)
THUMB_SIDE = 384
# sha256 of this file. WP5 should reuse the stored value rather than hashing
# the whole package: other work packages edit sibling modules.
CODE_PATH = Path(__file__).resolve()

# Section 2 read-only facts for the pinned parquet. Recomputed and compared.
_PINNED_CATEGORIES = [
    "Attribute (Appr.)",
    "Attribute (Meas.)",
    "MSR",
    "Motion (Cam.)",
    "Motion (Obj.)",
    "Positional Relationship (Cam.–Cam.)",
    "Positional Relationship (Cam.–Obj.)",
    "Positional Relationship (Cam.–Reg.)",
    "Positional Relationship (Obj.–Obj.)",
    "Positional Relationship (Obj.–Reg.)",
    "Positional Relationship (Reg.–Reg.)",
]
_PINNED_IMAGE_COUNTS = {
    "2": 810,
    "3": 58,
    "4": 40,
    "5": 19,
    "6": 38,
    "7": 15,
    "8": 15,
    "9": 2,
    "10": 3,
}
_PINNED_FACTS = {
    "all_questions_have_four_options": True,
    "categories": _PINNED_CATEGORIES,
    "explore_max_images": 8,
    "image_counts": _PINNED_IMAGE_COUNTS,
    "n_categories": 11,
    "n_eligible": 810,
    "n_eligible_confirm": 405,
    "n_eligible_groups": 627,
    "n_eligible_msr": 12,
    "n_eligible_msr_confirm": 6,
    "n_questions": 1000,
    "nine_and_ten_image_questions_in_confirm": True,
    "ten_image_ids": [140, 644, 975],
}


class PrepareError(Exception):
    """The dataset, split, or output path violates the prepare contract."""


def cli_prepare(args):
    try:
        prepare(args.dataset, args.split, args.out)
    except PrepareError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(EXIT_CODES["configuration_or_contract"]) from exc


def prepare(dataset, split_path, out, *, verify_pin=True):
    """Sample profiles and write ``selection.json`` plus the inspection packet.

    ``verify_pin`` checks the pinned dataset hash, the 500/500 split, and the
    Section 2 facts. The CLI leaves it on. Tests of synthetic parquet pass
    False; the column read and the sampling rules are the same either way.
    """
    dataset = Path(dataset)
    split_path = Path(split_path)
    out = Path(out)
    _refuse_nonempty(out)
    if not dataset.is_file():
        raise PrepareError(f"dataset file not found: {dataset}")
    dataset_hash = sha256_file(dataset)
    if verify_pin and dataset_hash != DATASET_SHA256:
        raise PrepareError(
            f"dataset sha256 {dataset_hash} does not match pinned {DATASET_SHA256}"
        )
    explore, confirm = _read_split(split_path)
    if verify_pin and (len(explore) != 500 or len(confirm) != 500):
        raise PrepareError(
            f"split is not 500/500 (explore {len(explore)}, confirm {len(confirm)})"
        )
    records = _load_records(dataset, explore, confirm)
    facts = _facts(records)
    if verify_pin:
        _require_pinned_facts(facts)
        if facts["n_questions"] != 1000:
            raise PrepareError(f"pinned dataset does not have 1,000 questions ({facts['n_questions']})")
    selection = _selection(records, facts, dataset_hash, sha256_file(split_path))
    _write_outputs(dataset, out, records, selection)
    return selection


def _refuse_nonempty(out):
    if not out.exists():
        return
    if not out.is_dir():
        raise PrepareError(f"output path is not a directory: {out}")
    if any(out.iterdir()):
        raise PrepareError(f"refusing to write into a non-empty output directory: {out}")


def _read_split(path):
    if not path.is_file():
        raise PrepareError(f"split file not found: {path}")
    try:
        value = read_json(path)
    except json.JSONDecodeError as exc:
        raise PrepareError(f"split file is not valid JSON: {path}") from exc
    if not isinstance(value, dict) or "explore" not in value or "confirm" not in value:
        raise PrepareError("split must contain explore and confirm lists")
    explore = _integer_ids(value["explore"], "explore")
    confirm = _integer_ids(value["confirm"], "confirm")
    if len(explore) != len(set(explore)) or len(confirm) != len(set(confirm)):
        raise PrepareError("split contains duplicate ids")
    if set(explore) & set(confirm):
        raise PrepareError("split explore and confirm overlap")
    return explore, confirm


def _integer_ids(value, name):
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise PrepareError(f"split {name} must be a list of integer ids")
    return value


def _iter_rows(dataset, columns):
    if set(columns) - set(INFERENCE_COLUMNS):
        raise PrepareError("refusing to read parquet columns outside the inference set")
    parquet = pq.ParquetFile(dataset)
    for batch in parquet.iter_batches(batch_size=8, columns=list(columns)):
        yield from batch.to_pylist()


def _load_records(dataset, explore, confirm):
    explore_ids = set(explore)
    confirm_ids = set(confirm)
    records = []
    seen = set()
    for row in _iter_rows(dataset, INFERENCE_COLUMNS):
        row_id = row["id"]
        if isinstance(row_id, bool) or not isinstance(row_id, int):
            raise PrepareError(f"dataset id is not an integer: {row_id!r}")
        if row_id in seen:
            raise PrepareError(f"duplicate dataset id {row_id}")
        seen.add(row_id)
        if row_id in explore_ids:
            partition = "explore"
        elif row_id in confirm_ids:
            partition = "confirm"
        else:
            raise PrepareError(f"dataset id {row_id} is not in the split")
        question = row["question"]
        if not isinstance(question, str):
            raise PrepareError(f"question {row_id} is missing question text")
        stem = _stem(row_id, question)
        try:
            raws = images_of(row)
            hashes = [pixel_hash(raw) for raw in raws]
        except (ValueError, OSError, TypeError) as exc:
            raise PrepareError(f"question {row_id} images could not be hashed: {exc}") from exc
        records.append({
            "id": row_id,
            "category": row["question_type"],
            "partition": partition,
            "n_images": len(hashes),
            "image_hashes": hashes,
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "stem": stem,
        })
    missing = (explore_ids | confirm_ids) - seen
    if missing:
        sample = ", ".join(str(item) for item in sorted(missing)[:8])
        raise PrepareError(f"split ids missing from the dataset: {sample}")
    records.sort(key=lambda record: record["id"])
    return records


def _stem(row_id, question):
    parsed = parse_question(question)
    if parsed is None:
        raise PrepareError(f"question {row_id} does not have exactly four A-D options")
    stem, options = parsed
    if len(options) != 4 or not all(options):
        raise PrepareError(f"question {row_id} does not have exactly four A-D options")
    return stem.strip()


def _facts(records):
    categories = sorted({record["category"] for record in records})
    image_counts = Counter(record["n_images"] for record in records)
    eligible = [record for record in records if record["n_images"] == 2]
    eligible_groups = image_groups(
        [{"id": record["id"], "image_hashes": record["image_hashes"]} for record in eligible]
    )
    msr = [record for record in eligible if record["category"] == "MSR"]
    explore_sizes = [record["n_images"] for record in records if record["partition"] == "explore"]
    nine_and_ten = [record for record in records if record["n_images"] >= 9]
    return {
        "all_questions_have_four_options": True,
        "categories": categories,
        "explore_max_images": max(explore_sizes) if explore_sizes else None,
        "image_counts": {str(count): image_counts[count] for count in sorted(image_counts)},
        "n_categories": len(categories),
        "n_eligible": len(eligible),
        "n_eligible_confirm": sum(record["partition"] == "confirm" for record in eligible),
        "n_eligible_groups": len(set(eligible_groups.values())),
        "n_eligible_msr": len(msr),
        "n_eligible_msr_confirm": sum(record["partition"] == "confirm" for record in msr),
        "n_questions": len(records),
        "nine_and_ten_image_questions_in_confirm": all(
            record["partition"] == "confirm" for record in nine_and_ten
        ),
        "ten_image_ids": sorted(record["id"] for record in records if record["n_images"] == 10),
    }


def _require_pinned_facts(facts):
    if facts == _PINNED_FACTS:
        return
    diffs = []
    for key in sorted(set(facts) | set(_PINNED_FACTS)):
        if facts.get(key) != _PINNED_FACTS.get(key):
            diffs.append(f"{key}: got {facts.get(key)!r} expected {_PINNED_FACTS.get(key)!r}")
    raise PrepareError("pinned dataset facts do not match Section 2: " + "; ".join(diffs))


def _selection(records, facts, dataset_hash, split_hash):
    groups = image_groups(
        [{"id": record["id"], "image_hashes": record["image_hashes"]} for record in records]
    )
    group_of = {record_id: int(group) for record_id, group in groups.items()}
    eligible = [record for record in records if record["n_images"] == 2]
    rng = random.Random(SEED)
    category_order = sorted({record["category"] for record in eligible})
    rng.shuffle(category_order)

    confirm_pools = {}
    for category in category_order:
        pool = sorted(
            (record for record in eligible if record["category"] == category and record["partition"] == "confirm"),
            key=lambda record: record["id"],
        )
        rng.shuffle(pool)
        confirm_pools[category] = pool

    occupied = set()
    eval_selected, eval_discarded = _round_robin(confirm_pools, category_order, groups, occupied, EVAL_LIMIT)
    dev_selected, dev_discarded, dev_unreached, dev_counts = _development(
        eligible, category_order, groups, occupied, rng
    )
    timing_selected, timing_by_count = _timing(records, groups, occupied)

    eval_ids = [record["id"] for record in eval_selected]
    dev_ids = [record["id"] for record in dev_selected]
    timing_ids = [record["id"] for record in timing_selected]
    scheduled = set(eval_ids) | set(dev_ids) | set(timing_ids)
    rest_ids = sorted(record["id"] for record in records if record["id"] not in scheduled)
    raw_ids = eval_ids + dev_ids + timing_ids + rest_ids
    if len(raw_ids) != len(records) or len(set(raw_ids)) != len(records):
        raise PrepareError("raw_all schedule does not cover each question once")

    rank_of = {record_id: rank for rank, record_id in enumerate(eval_ids, start=1)}
    eval_id_set = set(eval_ids)
    dev_id_set = set(dev_ids)
    timing_id_set = set(timing_ids)
    discarded = set(eval_discarded) | set(dev_discarded)
    unreached = set(dev_unreached)
    for pool in confirm_pools.values():
        unreached.update(record["id"] for record in pool)
    if discarded & unreached:
        raise PrepareError("a question was both discarded for a shared group and left unreached")

    items = []
    exclusions = []
    role_of = {}
    for record in records:
        row_id = record["id"]
        if row_id in eval_id_set:
            role = "eval"
        elif row_id in dev_id_set:
            role = "dev"
        elif row_id in timing_id_set:
            role = "timing"
        else:
            role = "raw_only"
        role_of[row_id] = role
        rank = rank_of.get(row_id)
        items.append({
            "id": row_id,
            "category": record["category"],
            "partition": record["partition"],
            "role": role,
            "rank": rank,
            "in_eval100": rank is not None and rank <= EVAL100_LIMIT,
            "n_images": record["n_images"],
            "image_hashes": list(record["image_hashes"]),
            "image_group": group_of[row_id],
            "question_sha256": record["question_sha256"],
        })
    items.sort(key=lambda item: item["id"])
    for record in records:
        row_id = record["id"]
        role = role_of[row_id]
        if role in ("eval", "dev"):
            continue
        if record["n_images"] != 2:
            reason = "not_two_images"
        elif row_id in discarded:
            reason = "shared_image_group"
        elif row_id in unreached:
            reason = "not_reached"
        else:
            raise PrepareError(f"question {row_id} has no exclusion reason")
        exclusions.append({"id": row_id, "reason": reason})
    exclusions.sort(key=lambda entry: entry["id"])

    shortfalls = {}
    for category in sorted(category_order):
        missing = DEV_PER_CATEGORY - dev_counts[category]
        if missing > 0:
            shortfalls[category] = missing

    timing_ids_with_nulls = [timing_by_count[count] for count in TIMING_IMAGE_COUNTS]
    profiles = {
        "raw_all": _profile(raw_ids),
        "eval200": _profile(eval_ids),
        "eval100": _profile(eval_ids[:EVAL100_LIMIT]),
        "dev": _profile(dev_ids),
        "timing": _profile(timing_ids),
    }
    # Image-count keys are strings in JSON. Null means explore had no free question.
    profiles["timing"]["by_image_count"] = {str(count): timing_by_count[count] for count in TIMING_IMAGE_COUNTS}
    if timing_ids != [row_id for row_id in timing_ids_with_nulls if row_id is not None]:
        raise PrepareError("timing profile ids are not in image-count order")

    body = {
        "version": VERSION,
        "seed": SEED,
        "dataset_sha256": dataset_hash,
        "split_sha256": split_hash,
        "code_sha256": sha256_file(CODE_PATH),
        "category_order": category_order,
        "facts": facts,
        "items": items,
        "exclusions": exclusions,
        "shortfalls": shortfalls,
        "profiles": profiles,
    }
    body["manifest_sha256"] = _digest({key: value for key, value in body.items() if key != "manifest_sha256"})
    return body


def _round_robin(pools, category_order, groups, occupied, limit):
    """Pop until a free candidate or the pool is empty. One acceptance ends the turn."""
    selected = []
    discarded = []
    while len(selected) < limit and any(pools[category] for category in category_order):
        for category in category_order:
            if len(selected) >= limit:
                break
            pool = pools[category]
            while pool:
                candidate = pool.pop()
                group = groups[candidate["id"]]
                if group in occupied:
                    discarded.append(candidate["id"])
                    continue
                occupied.add(group)
                selected.append(candidate)
                break
    return selected, discarded


def _development(eligible, category_order, groups, occupied, rng):
    """Category-by-category, not round-robin. Same RNG, after eval shuffles."""
    selected = []
    discarded = []
    unreached = []
    counts = {}
    for category in category_order:
        pool = sorted(
            (record for record in eligible if record["category"] == category and record["partition"] == "explore"),
            key=lambda record: record["id"],
        )
        rng.shuffle(pool)
        taken = 0
        for candidate in reversed(pool):
            if taken >= DEV_PER_CATEGORY:
                unreached.append(candidate["id"])
                continue
            group = groups[candidate["id"]]
            if group in occupied:
                discarded.append(candidate["id"])
                continue
            occupied.add(group)
            selected.append(candidate)
            taken += 1
        counts[category] = taken
    return selected, discarded, unreached, counts


def _timing(records, groups, occupied):
    by_count = {count: [] for count in TIMING_IMAGE_COUNTS}
    for record in records:
        if record["partition"] == "explore" and record["n_images"] in by_count:
            by_count[record["n_images"]].append(record)
    selected = []
    chosen = {}
    for count in TIMING_IMAGE_COUNTS:
        pick = None
        for candidate in sorted(by_count[count], key=lambda record: record["id"]):
            if groups[candidate["id"]] in occupied:
                continue
            pick = candidate
            break
        if pick is None:
            chosen[count] = None
            continue
        occupied.add(groups[pick["id"]])
        selected.append(pick)
        chosen[count] = pick["id"]
    return selected, chosen


def _profile(ids):
    return {"ids": list(ids), "id_list_sha256": _id_list_sha256(ids)}


def _id_list_sha256(ids):
    if any(isinstance(row_id, bool) or not isinstance(row_id, int) for row_id in ids):
        raise PrepareError("id-list hash requires a list of integers")
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def _digest(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_outputs(dataset, out, records, selection):
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "selection.json", selection)
    stems = {record["id"]: record["stem"] for record in records}
    _write_inspection(dataset, out / "inspection", selection, stems)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _write_inspection(dataset, inspection, selection, stems):
    wanted = []
    by_id = {item["id"]: item for item in selection["items"]}
    for row_id in selection["profiles"]["eval200"]["ids"]:
        wanted.append(by_id[row_id])
    for row_id in selection["profiles"]["dev"]["ids"]:
        wanted.append(by_id[row_id])
    for row_id in selection["profiles"]["timing"]["ids"]:
        wanted.append(by_id[row_id])
    wanted_ids = {item["id"] for item in wanted}
    images = {}
    if wanted_ids:
        for row in _iter_rows(dataset, ("id", "images")):
            row_id = int(row["id"])
            if row_id not in wanted_ids:
                continue
            try:
                images[row_id] = images_of(row)
            except ValueError as exc:
                raise PrepareError(f"question {row_id} images could not be decoded: {exc}") from exc
    missing = wanted_ids - set(images)
    if missing:
        raise PrepareError(f"inspection images missing for ids: {sorted(missing)[:8]}")
    image_dir = inspection / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    for item in wanted:
        raws = images[item["id"]]
        if len(raws) != item["n_images"]:
            raise PrepareError(f"question {item['id']} image count changed while writing the inspection packet")
        figures = []
        for index, raw in enumerate(raws, start=1):
            name = f"{item['id']}_{index}.jpg"
            try:
                _write_thumbnail(raw, image_dir / name)
            except (OSError, ValueError) as exc:
                raise PrepareError(f"question {item['id']} thumbnail failed: {exc}") from exc
            figures.append(
                f'<figure><figcaption>Image {index}</figcaption>'
                f'<img src="images/{name}" alt="Image {index}"></figure>'
            )
        rank = "" if item["rank"] is None else str(item["rank"])
        rank_label = rank if rank else "none"
        sections.append(
            f'<section data-id="{item["id"]}" data-role="{html.escape(item["role"])}" '
            f'data-rank="{html.escape(rank)}">'
            f"<h2>{item['id']} | {html.escape(item['role'])} | rank {html.escape(rank_label)} | "
            f"{html.escape(item['category'])}</h2>"
            f"<p>{html.escape(stems[item['id']])}</p>"
            f'<div class="views">{"".join(figures)}</div></section>'
        )
    document = (
        "<!doctype html>\n<html lang=\"en\">\n<meta charset=\"utf-8\">\n"
        "<title>World-state inspection packet</title>\n<style>\n"
        "body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:16px;background:#f4f6f8;color:#17202a}\n"
        "section{background:#fff;padding:20px;margin:16px 0}\n"
        ".views{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;width:280px}img{width:100%;height:auto}\n"
        "</style>\n<h1>Inspection packet</h1>\n"
        "<p>Data-integrity check for eval, development, and timing items. "
        "Question stems and images only. This packet does not gate sampling.</p>\n"
        + "\n".join(sections)
        + "\n</html>\n"
    )
    (inspection / "index.html").write_text(document)


def _write_thumbnail(raw, path):
    image = decode_image(raw)
    image.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
    image.save(path, format="JPEG", quality=95, optimize=False, subsampling=0)

"""Blind state audit and outcome-aware case lists.

``audit make`` draws from the run profile in ``run_config.json``. It keeps an
id only when ``inventories/<id>.json`` exists and ``parse_inventory`` accepts
it. The draw is one ``random.Random(SEED)`` stream: shuffle the sorted
category labels, then, in that shuffled order, shuffle each category's
numerically sorted ids, then round-robin ``list.pop()`` until ``n``
items or every category is empty. Make does not read ``records/``,
predictions, or the dataset ``answer`` column. Images and question text come
from ``dataset_path`` after its SHA-256 matches ``dataset_sha256``, and only
``id``, ``question``, and ``images`` are requested (a subset of
``INFERENCE_COLUMNS``). The packet shows the stem, never the options.

The packet is written to ``--out``. ``audit outcomes`` does not search for
it. It reads only ``<run>/audit``, which is the directory produced by
``audit make --out <run>/audit``, and it exits 2 unless that directory
contains ``imported.json`` from ``audit import``.

``audit outcomes`` reads gold, predictions, status, and ``correct`` only from
``<analysis>/item_outcomes.csv``. A row is correct only when ``correct``
agrees with the Section 6 rule (status ``ok`` and ``pred`` equal to ``gold``;
a blank pred or gold is not correct). Corrected means Y is wrong and X is
right; introduced means Y is right and X is wrong. The lists cover every id
in ``item_outcomes.csv``, not only the blind sample. ``primary_error_type``,
``reviewer``, and ``notes`` are left empty.

``packet.json`` records ``run_id``, ``run_dir``, ``run_config_sha256``,
``seed``, ``drawn_ids``, ``excluded_count``, ``inventory_sha256``, and
per-item ``n_images``. Import re-derives the expected atom rows from those
inventories, so an in-place edit of the label CSVs cannot drop a row without
being detected.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import numbers
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from idea3_consistency.mmsi_consistency import parse_question
from idea4_visual_reasoning.data import images_of
from idea4_visual_reasoning.render import decode_image

from .contracts import (
    ATOM_ACCURACY,
    ATOM_RELEVANCE,
    AUDIT_ATOM_COLUMNS,
    AUDIT_ITEM_COLUMNS,
    AUDIT_ITEM_LABEL_COLUMNS,
    AUDIT_OUTCOME_COLUMNS,
    COMPARISONS,
    DEFAULT_MAX_SIDE,
    EXIT_CODES,
    INFERENCE_COLUMNS,
    QUERY_FRAME_LABELS,
    SEED,
    YES_NO,
)
from .inventory import InventoryError, atom_id, atom_sort_key, atom_text, atoms, parse_inventory

# Requested parquet columns. Never add ``answer`` or ``thought``.
DATASET_COLUMNS = ("id", "question", "images")
ITEM_OUTCOME_COLUMNS = ("id", "category", "condition", "status", "pred", "gold", "correct")
ATOM_LABEL_FIELDS = {
    "accuracy": ATOM_ACCURACY,
    "relevance": ATOM_RELEVANCE,
}
ITEM_LABEL_FIELDS = {
    "query_frame_correct": QUERY_FRAME_LABELS,
    "identity_error": YES_NO,
    "frame_error": YES_NO,
    "missing_critical_evidence": YES_NO,
}
PACKET_KEYS = (
    "run_id",
    "run_dir",
    "run_config_sha256",
    "seed",
    "drawn_ids",
    "excluded_count",
    "inventory_sha256",
    "items",
)
IMPORTED_KEYS = (
    "atom_labels_sha256",
    "item_labels_sha256",
    "atom_count",
    "item_count",
    "imported_at",
)

_PAGE_STYLE = (
    "body{font:16px system-ui,sans-serif;margin:24px auto;max-width:1100px;"
    "color:#17202a;background:#f4f6f8}"
    "section{background:#fff;padding:20px;margin:16px 0}"
    ".views{display:flex;flex-wrap:wrap;gap:12px}"
    "figure{margin:0;width:280px}img{width:100%;height:auto}"
    "ol.atoms{padding-left:1.2rem}code{font-size:.9em}"
    ".stem,.question{white-space:pre-wrap}"
)


class AuditError(Exception):
    """The audit command cannot proceed. The CLI turns this into exit 2."""


def cli_audit_make(args):
    try:
        make_packet(args.run, args.n, args.out)
    except AuditError as exc:
        print(f"audit: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CODES["configuration_or_contract"]) from None


def cli_audit_import(args):
    try:
        import_labels(args.packet, args.labels)
    except AuditError as exc:
        print(f"audit: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CODES["configuration_or_contract"]) from None


def cli_audit_outcomes(args):
    try:
        write_outcomes(args.run, args.analysis, args.out)
    except AuditError as exc:
        print(f"audit: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_CODES["configuration_or_contract"]) from None


def make_packet(run, n, out):
    """Draw a blinded audit packet. See the module docstring for the draw."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise AuditError(f"--n must be a non-negative integer, got {n!r}")
    run, config, config_path = _load_run(run)
    out = _refuse_nonempty(out)
    dataset = _verify_dataset(config)
    profile = _profile(config)
    eligible = []
    loaded = {}
    excluded = 0
    for item in profile:
        found = _read_inventory(run, item["item_id"], item["n_images"])
        if found is None:
            excluded += 1
            continue
        eligible.append(item)
        loaded[item["item_id"]] = found
    drawn_ids = _draw(eligible, n)
    by_id = {item["item_id"]: item for item in profile}
    rows_by_id = _load_dataset_rows(dataset, set(drawn_ids)) if drawn_ids else {}

    prepared = []
    for item_id in drawn_ids:
        item = by_id[item_id]
        dataset_row = rows_by_id[item_id]
        stem = _stem(item_id, dataset_row.get("question"))
        raws = _image_bytes(item_id, dataset_row, item["n_images"])
        figures = []
        for index, raw in enumerate(raws, start=1):
            figures.append((f"{item_id}_{index}.jpg", _jpeg_thumbnail(item_id, raw)))
        prepared.append({
            "item_id": item_id,
            "category": item["category"],
            "n_images": item["n_images"],
            "stem": stem,
            "inventory_sha256": loaded[item_id]["sha256"],
            "atoms": _ordered_atoms(loaded[item_id]["document"]),
            "figures": figures,
        })

    packet = {
        "run_id": config["run_id"],
        "run_dir": str(run),
        "run_config_sha256": _sha256_file(config_path),
        "seed": SEED,
        "drawn_ids": drawn_ids,
        "excluded_count": excluded,
        "inventory_sha256": {item["item_id"]: item["inventory_sha256"] for item in prepared},
        "items": [
            {
                "item_id": item["item_id"],
                "category": item["category"],
                "n_images": item["n_images"],
                "inventory_sha256": item["inventory_sha256"],
            }
            for item in prepared
        ],
    }
    _write_packet(out, prepared, packet)
    print(f"drew {len(drawn_ids)}; excluded {excluded}")
    return packet


def import_labels(packet_dir, labels):
    """Validate completed label CSVs and write ``imported.json`` into the packet.

    Expected atom rows are re-derived from the inventories named in
    ``packet.json``. The label file must match that set: no missing row, no
    duplicate, no extra row, and the same ``atom_text``. Label cells must be
    members of their enums. Reviewer cells must be non-blank. Nothing in the
    label files is rewritten.
    """
    packet_dir, packet = _load_packet(packet_dir)
    items = _packet_items(packet)
    run = _packet_run_dir(packet)
    expected_atoms = []
    for item in items:
        found = _inventory_matching_hash(run, item)
        expected_atoms.extend(
            (item["item_id"], atom["atom_id"], atom["atom_text"])
            for atom in _ordered_atoms(found["document"])
        )
    labels = Path(labels)
    if not labels.is_dir():
        raise AuditError(
            "labels path must be a directory containing atom_labels.csv and "
            f"item_labels.csv: {labels}"
        )
    atom_path = labels / "atom_labels.csv"
    item_path = labels / "item_labels.csv"
    atom_rows = _read_csv(atom_path, AUDIT_ATOM_COLUMNS)
    item_rows = _read_csv(item_path, AUDIT_ITEM_LABEL_COLUMNS)
    _check_atom_labels(atom_rows, expected_atoms)
    _check_item_labels(item_rows, [item["item_id"] for item in items])
    imported = {
        "atom_labels_sha256": _sha256_file(atom_path),
        "item_labels_sha256": _sha256_file(item_path),
        "atom_count": len(atom_rows),
        "item_count": len(item_rows),
        "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "labels_dir": str(labels.resolve()),
    }
    _write_json(packet_dir / "imported.json", imported)
    print(f"imported {imported['atom_count']} atom labels and {imported['item_count']} item labels")
    return imported


def write_outcomes(run, analysis, out):
    """Write corrected and introduced rows for every ``COMPARISONS`` token.

    Requires ``<run>/audit/imported.json``. Does not read dataset ``answer``.
    """
    run, config, _config_path = _load_run(run)
    dataset = _verify_dataset(config)
    _packet_dir, _imported, packet = _require_imported(run)
    _assert_packet_hashes(run, packet)
    out = _refuse_nonempty(out)
    table, ids = _load_item_outcomes(Path(analysis))
    meta = {item["item_id"]: item for item in _profile(config)}
    outcome_rows = _comparison_rows(table, ids)
    wanted = {row["item_id"] for row in outcome_rows}
    dataset_rows = _load_dataset_rows(dataset, wanted) if wanted else {}
    rendered = {}
    for item_id in wanted:
        info = meta.get(item_id)
        n_images = info["n_images"] if info else None
        raws = _image_bytes(item_id, dataset_rows[item_id], n_images)
        if n_images is None:
            n_images = len(raws)
        found = _read_inventory(run, item_id, n_images)
        rendered[item_id] = {
            "question": _question_text(item_id, dataset_rows[item_id].get("question")),
            "figures": [
                (f"{item_id}_{index}.jpg", _jpeg_thumbnail(item_id, raw))
                for index, raw in enumerate(raws, start=1)
            ],
            "atoms": None if found is None else _ordered_atoms(found["document"]),
        }
    _write_outcome_packet(out, outcome_rows, rendered)
    print(f"wrote {len(outcome_rows)} corrected or introduced rows")
    return outcome_rows


def _id_sort_key(item_id):
    text = str(item_id)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _draw(eligible, n):
    """Seeded category round-robin. ``pop()`` takes the last shuffled id."""
    if n == 0 or not eligible:
        return []
    by_category = {}
    for item in eligible:
        by_category.setdefault(item["category"], []).append(item["item_id"])
    rng = random.Random(SEED)
    categories = sorted(by_category)
    rng.shuffle(categories)
    buckets = []
    for category in categories:
        ids = sorted(by_category[category], key=_id_sort_key)
        rng.shuffle(ids)
        buckets.append(ids)
    drawn = []
    while len(drawn) < n:
        progressed = False
        for bucket in buckets:
            if not bucket:
                continue
            drawn.append(bucket.pop())
            progressed = True
            if len(drawn) >= n:
                break
        if not progressed:
            break
    return drawn


def _load_run(run):
    run = Path(run).resolve()
    if not run.is_dir():
        raise AuditError(f"run directory not found: {run}")
    path = run / "run_config.json"
    config = _read_json(path, "run_config.json")
    if not isinstance(config, dict):
        raise AuditError("run_config.json must be an object")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise AuditError("run_config is missing run_id")
    return run, config, path


def _profile(config):
    raw_ids = config.get("ids")
    raw_items = config.get("items")
    if not isinstance(raw_ids, list):
        raise AuditError("run_config ids must be a list")
    if not isinstance(raw_items, list):
        raise AuditError("run_config items must be a list")
    meta = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise AuditError("run_config items must be objects")
        item_id = _canonical_id(item.get("id"))
        if item_id in meta:
            raise AuditError(f"duplicate item {item_id} in run_config items")
        category = item.get("category")
        n_images = item.get("n_images")
        if not isinstance(category, str) or not category:
            raise AuditError(f"item {item_id} is missing category")
        if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 1:
            raise AuditError(f"item {item_id} is missing a positive integer n_images")
        meta[item_id] = {"item_id": item_id, "category": category, "n_images": n_images}
    ordered = []
    seen = set()
    for raw in raw_ids:
        item_id = _canonical_id(raw)
        if item_id in seen:
            raise AuditError(f"duplicate profile id {item_id}")
        seen.add(item_id)
        if item_id not in meta:
            raise AuditError(f"profile id {item_id} is missing from run_config items")
        ordered.append(meta[item_id])
    return ordered


def _canonical_id(value):
    if isinstance(value, bool) or value is None:
        raise AuditError(f"invalid item id {value!r}")
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, str) and value and "/" not in value and "\\" not in value and value not in {".", ".."}:
        return value
    raise AuditError(f"invalid item id {value!r}")


def _read_inventory(run, item_id, n_images):
    path = Path(run) / "inventories" / f"{item_id}.json"
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
        document = parse_inventory(data.decode("utf-8"), n_images)
    except (OSError, UnicodeError, InventoryError):
        return None
    return {"document": document, "sha256": hashlib.sha256(data).hexdigest(), "path": path}


def _inventory_matching_hash(run, item):
    path = Path(run) / "inventories" / f"{item['item_id']}.json"
    if not path.is_file():
        raise AuditError(f"inventory hash for item {item['item_id']} does not match the packet")
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != item["inventory_sha256"]:
        raise AuditError(f"inventory hash for item {item['item_id']} does not match the packet")
    try:
        document = parse_inventory(data.decode("utf-8"), item["n_images"])
    except (UnicodeError, InventoryError) as exc:
        raise AuditError(f"inventory for item {item['item_id']} no longer parses: {exc}") from exc
    return {"document": document, "sha256": actual, "path": path}


def _ordered_atoms(document):
    ordered = sorted(atoms(document), key=atom_sort_key)
    return [
        {"number": f"A{index}", "atom_id": atom_id(atom), "atom_text": atom_text(atom)}
        for index, atom in enumerate(ordered, start=1)
    ]


def _verify_dataset(config):
    raw_path = config.get("dataset_path")
    expected = config.get("dataset_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise AuditError("run_config is missing dataset_path")
    if not isinstance(expected, str) or not expected:
        raise AuditError("run_config is missing dataset_sha256")
    path = Path(raw_path)
    if not path.is_file():
        raise AuditError(f"dataset not found: {path}")
    actual = _sha256_file(path)
    if actual.lower() != expected.lower():
        raise AuditError(
            f"dataset sha256 {actual} does not match run_config dataset_sha256 {expected}"
        )
    return path


def _iter_dataset_rows(dataset):
    columns = list(DATASET_COLUMNS)
    if set(columns) - set(INFERENCE_COLUMNS):
        raise AuditError("refusing to read parquet columns outside the inference set")
    parquet = pq.ParquetFile(dataset)
    for batch in parquet.iter_batches(batch_size=32, columns=columns):
        yield from batch.to_pylist()


def _load_dataset_rows(dataset, wanted):
    found = {}
    for row in _iter_dataset_rows(dataset):
        try:
            item_id = _canonical_id(row.get("id"))
        except AuditError as exc:
            raise AuditError(f"dataset id is not usable: {exc}") from exc
        if item_id not in wanted:
            continue
        if item_id in found:
            raise AuditError(f"duplicate dataset id {item_id}")
        found[item_id] = row
    missing = wanted - set(found)
    if missing:
        sample = ", ".join(sorted(missing)[:8])
        raise AuditError(f"dataset is missing ids: {sample}")
    return found


def _stem(item_id, question):
    if not isinstance(question, str):
        raise AuditError(f"dataset question for {item_id} is not text")
    parsed = parse_question(question)
    if parsed is None:
        raise AuditError(f"dataset question for {item_id} has no parseable stem")
    stem = parsed[0].strip()
    if not stem:
        raise AuditError(f"dataset question for {item_id} has an empty stem")
    return stem


def _question_text(item_id, question):
    if not isinstance(question, str) or not question.strip():
        raise AuditError(f"dataset question for {item_id} is missing")
    return question


def _image_bytes(item_id, row, n_images):
    try:
        raws = images_of(row)
    except (TypeError, ValueError, KeyError) as exc:
        raise AuditError(f"dataset images for {item_id} could not be read: {exc}") from exc
    if n_images is not None and len(raws) != n_images:
        raise AuditError(
            f"dataset image count for {item_id} is {len(raws)} "
            f"but the run item has n_images {n_images}"
        )
    return raws


def _jpeg_thumbnail(item_id, raw):
    try:
        image = decode_image(raw)
        image.thumbnail((DEFAULT_MAX_SIDE, DEFAULT_MAX_SIDE), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95, optimize=False, subsampling=0)
    except (OSError, ValueError) as exc:
        raise AuditError(f"could not thumbnail an image for {item_id}: {exc}") from exc
    return buffer.getvalue()


def _write_packet(out, prepared, packet):
    out.mkdir(parents=True, exist_ok=True)
    image_dir = out / "images"
    image_dir.mkdir()
    sections = []
    item_rows = []
    atom_rows = []
    label_rows = []
    for item in prepared:
        for name, blob in item["figures"]:
            (image_dir / name).write_bytes(blob)
        item_rows.append({
            "item_id": item["item_id"],
            "category": item["category"],
            "inventory_sha256": item["inventory_sha256"],
        })
        label_rows.append({"item_id": item["item_id"]})
        for atom in item["atoms"]:
            atom_rows.append({
                "item_id": item["item_id"],
                "atom_id": atom["atom_id"],
                "atom_text": atom["atom_text"],
            })
        figures = "".join(
            _figure(name, index) for index, (name, _blob) in enumerate(item["figures"], start=1)
        )
        sections.append(
            "<section "
            f'data-item-id="{html.escape(item["item_id"], quote=True)}" '
            f'data-category="{html.escape(item["category"], quote=True)}">'
            f"<h2>{html.escape(item['item_id'])}</h2>"
            f"<p class=\"category\">{html.escape(item['category'])}</p>"
            f"<p class=\"stem\">{html.escape(item['stem'])}</p>"
            f"<div class=\"views\">{figures}</div>"
            f"{_atom_list(item['atoms'])}</section>"
        )
    _write_csv(out / "items.csv", AUDIT_ITEM_COLUMNS, item_rows)
    _write_csv(out / "atom_labels.csv", AUDIT_ATOM_COLUMNS, atom_rows)
    _write_csv(out / "item_labels.csv", AUDIT_ITEM_LABEL_COLUMNS, label_rows)
    _write_json(out / "packet.json", packet)
    body = (
        f"<h1>Blind state audit</h1><p>Run {html.escape(packet['run_id'])}. "
        f"Drew {len(prepared)} items. Excluded for invalid inventory: {packet['excluded_count']}. "
        f"Seed {packet['seed']}. Question stems and state atoms only.</p>"
        + "".join(sections)
    )
    (out / "index.html").write_text(_page("Blind state audit", body), encoding="utf-8")


def _write_outcome_packet(out, outcome_rows, rendered):
    out.mkdir(parents=True, exist_ok=True)
    image_dir = out / "images"
    image_dir.mkdir()
    written = set()
    sections = []
    for row in outcome_rows:
        item_id = row["item_id"]
        view = rendered[item_id]
        for name, blob in view["figures"]:
            if name not in written:
                (image_dir / name).write_bytes(blob)
                written.add(name)
        figures = "".join(
            _figure(name, index) for index, (name, _blob) in enumerate(view["figures"], start=1)
        )
        atoms_html = (
            _atom_list(view["atoms"])
            if view["atoms"] is not None
            else "<p>No schema-valid inventory.</p>"
        )
        x_name = html.escape(row["x_name"])
        y_name = html.escape(row["y_name"])
        sections.append(
            "<section "
            f'data-comparison="{html.escape(row["comparison"], quote=True)}" '
            f'data-item-id="{html.escape(item_id, quote=True)}" '
            f'data-direction="{html.escape(row["direction"], quote=True)}">'
            f"<h2>{html.escape(row['comparison'])} · {html.escape(row['direction'])} · "
            f"{html.escape(item_id)}</h2>"
            f"<div class=\"views\">{figures}</div>"
            f"<p class=\"question\">{html.escape(view['question'])}</p>"
            f"<p class=\"gold\">Gold: {html.escape(row['gold'])}</p>"
            f"<p class=\"pred\" data-condition=\"{x_name}\">{x_name} prediction: "
            f"{html.escape(row['x_pred'])} ({html.escape(row['x_status'])})</p>"
            f"<p class=\"pred\" data-condition=\"{y_name}\">{y_name} prediction: "
            f"{html.escape(row['y_pred'])} ({html.escape(row['y_status'])})</p>"
            f"{atoms_html}</section>"
        )
    _write_csv(out / "outcomes.csv", AUDIT_OUTCOME_COLUMNS, outcome_rows)
    intro = (
        "<h1>Outcome-aware audit</h1>"
        "<p>Corrected means Y is wrong and X is right. "
        "Introduced means Y is right and X is wrong. "
        "Primary error type, reviewer, and notes are left blank.</p>"
    )
    if not sections:
        intro += "<p>No corrected or introduced items.</p>"
    (out / "index.html").write_text(_page("Outcome-aware audit", intro + "".join(sections)), encoding="utf-8")


def _figure(name, index):
    label = f"Image {index}"
    src = html.escape(name, quote=True)
    return (
        f"<figure><figcaption>{label}</figcaption>"
        f'<img src="images/{src}" alt="{label}"></figure>'
    )


def _atom_list(atom_rows):
    items = []
    for atom in atom_rows:
        items.append(
            "<li "
            f'data-atom="{html.escape(atom["number"], quote=True)}" '
            f'data-atom-id="{html.escape(atom["atom_id"], quote=True)}">'
            f'<span class="atom-no">{html.escape(atom["number"])}</span> '
            f"<code>{html.escape(atom['atom_id'])}</code> "
            f"{html.escape(atom['atom_text'])}</li>"
        )
    return '<ol class="atoms">' + "".join(items) + "</ol>"


def _page(title, body):
    return (
        "<!doctype html>\n<html lang=\"en\">\n<meta charset=\"utf-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>\n{_PAGE_STYLE}\n</style>\n"
        f"{body}\n</html>\n"
    )


def _load_packet(packet_dir):
    packet_dir = Path(packet_dir)
    if not packet_dir.is_dir():
        raise AuditError(f"packet directory not found: {packet_dir}")
    packet = _read_json(packet_dir / "packet.json", "packet.json")
    if not isinstance(packet, dict):
        raise AuditError("packet.json must be an object")
    missing = [key for key in PACKET_KEYS if key not in packet]
    if missing:
        raise AuditError(f"packet.json is missing {', '.join(missing)}")
    return packet_dir, packet


def _packet_items(packet):
    raw_items = packet["items"]
    drawn = packet["drawn_ids"]
    hashes = packet["inventory_sha256"]
    if not isinstance(raw_items, list) or not isinstance(drawn, list) or not isinstance(hashes, dict):
        raise AuditError("packet.json items, drawn_ids, and inventory_sha256 have the wrong shape")
    items = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise AuditError("packet.json items must be objects")
        item_id = _canonical_id(raw.get("item_id"))
        n_images = raw.get("n_images")
        digest = raw.get("inventory_sha256")
        if isinstance(n_images, bool) or not isinstance(n_images, int) or n_images < 1:
            raise AuditError(f"packet item {item_id} is missing n_images")
        if not isinstance(digest, str) or not digest:
            raise AuditError(f"packet item {item_id} is missing inventory_sha256")
        items.append({
            "item_id": item_id,
            "category": raw.get("category"),
            "n_images": n_images,
            "inventory_sha256": digest,
        })
    if [item["item_id"] for item in items] != [_canonical_id(item_id) for item_id in drawn]:
        raise AuditError("packet.json drawn_ids do not match items")
    for item in items:
        if hashes.get(item["item_id"]) != item["inventory_sha256"]:
            raise AuditError(f"packet.json inventory hash for {item['item_id']} is inconsistent")
    return items


def _packet_run_dir(packet):
    raw = packet.get("run_dir")
    if not isinstance(raw, str) or not raw:
        raise AuditError("packet.json is missing run_dir")
    path = Path(raw)
    if not path.is_dir():
        raise AuditError(f"packet run_dir does not exist: {path}")
    return path


def _check_atom_labels(rows, expected_atoms):
    found = {}
    for row in rows:
        item_id = _required_id(row, "atom label")
        atom = _cell(row, "atom_id").strip()
        if not atom:
            raise AuditError(f"atom label row for {item_id} is missing atom_id")
        key = (item_id, atom)
        if key in found:
            raise AuditError(f"duplicate atom label row for item {item_id} atom {atom}")
        found[key] = row
    expected = {(item_id, atom_id): text for item_id, atom_id, text in expected_atoms}
    missing = sorted(expected.keys() - found.keys())
    extra = sorted(found.keys() - expected.keys())
    if missing:
        item_id, atom = missing[0]
        raise AuditError(f"missing atom label row for item {item_id} atom {atom}")
    if extra:
        item_id, atom = extra[0]
        raise AuditError(f"unexpected atom label row for item {item_id} atom {atom}")
    for key, text in expected.items():
        item_id, atom = key
        actual = _cell(found[key], "atom_text")
        if actual != text:
            raise AuditError(f"atom_text for item {item_id} atom {atom} does not match the packet")
        where = f"item {item_id} atom {atom}"
        for field, allowed in ATOM_LABEL_FIELDS.items():
            _require_enum(_cell(found[key], field), allowed, f"{field} for {where}")
        _require_reviewer(_cell(found[key], "reviewer"), where)


def _check_item_labels(rows, expected_ids):
    found = {}
    for row in rows:
        item_id = _required_id(row, "item label")
        if item_id in found:
            raise AuditError(f"duplicate item label row for item {item_id}")
        found[item_id] = row
    missing = sorted(set(expected_ids) - found.keys())
    extra = sorted(found.keys() - set(expected_ids))
    if missing:
        raise AuditError(f"missing item label row for item {missing[0]}")
    if extra:
        raise AuditError(f"unexpected item label row for item {extra[0]}")
    for item_id in expected_ids:
        row = found[item_id]
        where = f"item {item_id}"
        for field, allowed in ITEM_LABEL_FIELDS.items():
            _require_enum(_cell(row, field), allowed, f"{field} for {where}")
        _require_reviewer(_cell(row, "reviewer"), where)


def _required_id(row, kind):
    raw = _cell(row, "item_id").strip()
    if not raw:
        raise AuditError(f"{kind} row is missing item_id")
    return _canonical_id(raw)


def _require_enum(value, allowed, what):
    text = value.strip()
    if text not in allowed:
        choices = ", ".join(allowed)
        raise AuditError(f"{what} {value!r} is outside the allowed labels ({choices})")
    return text


def _require_reviewer(value, what):
    if not value.strip():
        raise AuditError(f"reviewer is blank for {what}")


def _require_imported(run):
    packet_dir = Path(run) / "audit"
    imported_path = packet_dir / "imported.json"
    if not imported_path.is_file():
        raise AuditError(
            "state labels have not been imported; "
            f"expected {imported_path}. "
            "audit outcomes reads only the packet at <run>/audit, "
            "the directory written by audit make --out <run>/audit"
        )
    payload = _read_json(imported_path, "imported.json")
    if not isinstance(payload, dict):
        raise AuditError("imported.json must be an object")
    missing = [key for key in IMPORTED_KEYS if key not in payload]
    if missing:
        raise AuditError(f"imported.json is missing {', '.join(missing)}")
    _packet_dir, packet = _load_packet(packet_dir)
    return packet_dir, payload, packet


def _assert_packet_hashes(run, packet):
    hashes = packet.get("inventory_sha256")
    if not isinstance(hashes, dict):
        raise AuditError("packet.json inventory_sha256 must be an object")
    for item_id, expected in hashes.items():
        path = Path(run) / "inventories" / f"{item_id}.json"
        if not path.is_file() or _sha256_file(path) != expected:
            raise AuditError(f"inventory hash for item {item_id} does not match the audit packet")


def _load_item_outcomes(analysis):
    analysis = Path(analysis)
    if analysis.is_file() or not analysis.is_dir():
        raise AuditError(f"analysis path must be a directory: {analysis}")
    path = analysis / "item_outcomes.csv"
    rows = _read_csv(path, ITEM_OUTCOME_COLUMNS)
    table = {}
    for row in rows:
        raw_id = _cell(row, "id").strip()
        if not raw_id:
            raise AuditError("item_outcomes row is missing id")
        item_id = _canonical_id(raw_id)
        condition = _cell(row, "condition").strip()
        if not condition:
            raise AuditError(f"item_outcomes row for {item_id} is missing condition")
        key = (item_id, condition)
        if key in table:
            raise AuditError(f"duplicate item_outcomes row for {item_id} {condition}")
        table[key] = {
            "status": _cell(row, "status").strip(),
            "pred": _cell(row, "pred"),
            "gold": _cell(row, "gold"),
            "correct": _cell(row, "correct"),
            "category": _cell(row, "category").strip(),
        }
    ids = sorted({item_id for item_id, _condition in table})
    return table, ids


def _comparison_rows(table, ids):
    result = []
    for comparison in COMPARISONS:
        x_name, y_name = comparison.split("-", 1)
        for item_id in ids:
            if (item_id, x_name) not in table or (item_id, y_name) not in table:
                missing = x_name if (item_id, x_name) not in table else y_name
                raise AuditError(f"item {item_id} is missing an item_outcomes row for {missing}")
            x_row = table[(item_id, x_name)]
            y_row = table[(item_id, y_name)]
            if x_row["gold"] != y_row["gold"]:
                raise AuditError(f"item {item_id} has conflicting gold values in item_outcomes.csv")
            x_right = _checked_correct(x_row, item_id, x_name)
            y_right = _checked_correct(y_row, item_id, y_name)
            if x_right and not y_right:
                direction = "corrected"
            elif y_right and not x_right:
                direction = "introduced"
            else:
                continue
            result.append({
                "comparison": comparison,
                "item_id": item_id,
                "x_pred": x_row["pred"],
                "y_pred": y_row["pred"],
                "gold": x_row["gold"],
                "direction": direction,
                "primary_error_type": "",
                "reviewer": "",
                "notes": "",
                "category": x_row["category"],
                "x_name": x_name,
                "y_name": y_name,
                "x_status": x_row["status"],
                "y_status": y_row["status"],
            })
    return result


def _checked_correct(row, item_id, condition):
    flag = _parse_correct(row["correct"], item_id, condition)
    expected = row["status"] == "ok" and row["pred"] != "" and row["gold"] != "" and row["pred"] == row["gold"]
    if flag != expected:
        raise AuditError(
            f"item_outcomes correct for {item_id} {condition} disagrees with status, pred, and gold"
        )
    return flag


def _parse_correct(value, item_id, condition):
    text = value.strip().lower()
    if text in {"1", "true"}:
        return True
    if text in {"0", "false"}:
        return False
    raise AuditError(
        f"item_outcomes correct for {item_id} {condition} must be 0 or 1, got {value!r}"
    )


def _refuse_nonempty(out):
    out = Path(out)
    if out.exists():
        if not out.is_dir():
            raise AuditError(f"output path exists and is not a directory: {out}")
        if any(out.iterdir()):
            raise AuditError(f"refusing to write a non-empty directory: {out}")
    return out


def _read_json(path, what):
    path = Path(path)
    if not path.is_file():
        raise AuditError(f"missing {what}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON in {what}: {path}") from exc


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_csv(path, columns):
    path = Path(path)
    if not path.is_file():
        raise AuditError(f"missing {path.name}: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise AuditError(f"{path.name} is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise AuditError(f"{path.name} has no header")
    missing = [name for name in columns if name not in reader.fieldnames]
    if missing:
        raise AuditError(f"{path.name} is missing columns: {', '.join(missing)}")
    rows = []
    for row in reader:
        if all(not _cell(row, name).strip() for name in row):
            continue
        rows.append(row)
    return rows


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row, name) for name in fieldnames})


def _csv_value(row, name):
    value = row.get(name)
    if value is None:
        return ""
    return value


def _cell(row, name):
    value = row.get(name)
    return "" if value is None else str(value)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

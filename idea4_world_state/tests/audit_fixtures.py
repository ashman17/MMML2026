"""Hand-built run directories for audit tests.

The layout follows the Section 8.6 fields ``audit`` reads: ``run_id``, ``ids``,
``items`` (``id``, ``category``, ``n_images``), ``dataset_path``, and
``dataset_sha256``. Inventories are ``inventories/<id>.json``. Record files,
when present, are decoys. The parquet ``answer`` and ``thought`` columns are
poison strings; audit must not request them.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

OPTION_TEXTS = (
    "SENTINEL_OPTION_ALPHA",
    "SENTINEL_OPTION_BRAVO",
    "SENTINEL_OPTION_CHARLIE",
    "SENTINEL_OPTION_DELTA",
)
PARQUET_ANSWER = "SENTINEL_PARQUET_ANSWER"
PARQUET_THOUGHT = "SENTINEL_PARQUET_THOUGHT"
PARQUET_CATEGORY = "PARQUET_CATEGORY_SHOULD_NOT_APPEAR"
RECORD_PRED = "SENTINEL_RECORD_PRED"
ERROR_INVENTORY = "SENTINEL_ERROR_INVENTORY"


def question_text(stem, options=OPTION_TEXTS):
    body = ", ".join(f"{letter}: {option}" for letter, option in zip("ABCD", options))
    return f"{stem}\nOptions: {body}"


def valid_inventory(item_id, n_images=2):
    return {
        "schema_version": "ws-1",
        "query_frame": {"frame": "unknown", "basis": None},
        "temporal_order": {"status": "unknown", "order": None, "basis": None},
        "cameras": [
            {"id": f"cam{index}", "image_id": f"image_{index}"}
            for index in range(1, n_images + 1)
        ],
        "entities": [],
        "observations": [],
        "identity_links": [],
        "claims": [],
        "unknowns": [f"inventory note for {item_id}, unchecked"],
    }


def write_audit_run(root, specs, *, run_id=None, int_ids=False, records=True):
    """Write a run directory and a parquet whose SHA-256 is stored on the run.

    Each spec needs ``id`` and ``category``. Optional keys: ``n_images``
    (default 2), ``inventory`` (``valid``, ``invalid``, or ``missing``),
    ``stem``, and ``sizes`` (a ``(width, height)`` per image).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = root / "run"
    run.mkdir()
    inventory_dir = run / "inventories"
    inventory_dir.mkdir()
    dataset = root / "dataset.parquet"
    stored_ids = []
    categories = []
    questions = []
    images = []
    items = []
    for index, spec in enumerate(specs):
        n_images = spec.get("n_images", 2)
        item_key = int(spec["id"]) if int_ids else str(spec["id"])
        item_id = str(item_key)
        stored_ids.append(item_key)
        categories.append(spec["category"])
        stem = spec.get("stem", f"Where is the door in item {item_id}?")
        questions.append(question_text(stem, spec.get("options", OPTION_TEXTS)))
        sizes = spec.get("sizes") or [(8, 8)] * n_images
        if len(sizes) != n_images:
            raise ValueError(f"{item_id} has {len(sizes)} sizes for {n_images} images")
        images.append([
            _png(size, ((index * 40 + slot * 15) % 256, (index * 70) % 256, (slot * 90) % 256))
            for slot, size in enumerate(sizes, start=1)
        ])
        items.append({
            "id": item_key,
            "category": spec["category"],
            "n_images": n_images,
            "partition": "explore",
            "role": "eval",
        })
        kind = spec.get("inventory", "valid")
        if kind == "valid":
            _write_json(inventory_dir / f"{item_id}.json", valid_inventory(item_id, n_images))
        elif kind == "invalid":
            (inventory_dir / f"{item_id}.json").write_text("{not-json\n", encoding="utf-8")
        elif kind == "missing":
            _write_json(inventory_dir / f"{item_id}.error.json", {"error": ERROR_INVENTORY})
        else:
            raise ValueError(f"unknown inventory kind {kind}")
    id_type = pa.int64() if int_ids else pa.string()
    table = pa.table({
        "id": pa.array(stored_ids, type=id_type),
        "question_type": pa.array([PARQUET_CATEGORY] * len(specs)),
        "question": pa.array(questions),
        "images": pa.array(images, type=pa.list_(pa.binary())),
        "answer": pa.array([PARQUET_ANSWER] * len(specs)),
        "thought": pa.array([PARQUET_THOUGHT] * len(specs)),
    })
    pq.write_table(table, dataset)
    run_id = run_id or "audit-run"
    if records:
        record_dir = run / "records"
        record_dir.mkdir()
        for item_id in (str(item_id) for item_id in stored_ids):
            for condition in ("raw", "flat", "graph"):
                _write_json(record_dir / f"{item_id}__{condition}.json", {
                    "id": item_id,
                    "condition": condition,
                    "status": "ok",
                    "pred": RECORD_PRED,
                    "gold": PARQUET_ANSWER,
                })
        (run / "predictions.csv").write_text(
            f"id,pred\n{stored_ids[0]},{RECORD_PRED}\n",
            encoding="utf-8",
        )
    config = {
        "profile": "eval100",
        "run_id": run_id,
        "ids": stored_ids,
        "items": items,
        "conditions": ["raw", "flat", "graph"],
        "dataset_path": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "synthetic": True,
    }
    _write_json(run / "run_config.json", config)
    return run, dataset


def _png(size, color):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

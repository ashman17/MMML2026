"""Extraction-only retest. Not an evaluation profile and not a freeze input.

Uses the same image check, ``extract_messages``, ``CallLog``, and
``parse_inventory`` path as ``run``. The cache is inside ``--out``, so a new
prompt cannot read the pilot cache. Answer choices and gold labels are not
written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idea4_world_state import contracts, prompts, runner
from idea4_world_state.backend import VLLMBackend
from idea4_world_state.calllog import CallLog, PauseRun
from idea4_world_state.inventory import InventoryError, parse_inventory_with_repairs

# First empty-inventory probe. Kept so the original four-item check stays named.
DEBUG_IDS = (583, 575, 809, 356)
# v4 probe, chosen before looking at answer letters:
# 356 has a visible within-image layout (stove beside refrigerator);
# 575 and 583 are uncertain cross-view comparisons;
# 14, 531, and 912 previously reused one entity id across both images.
V4_PROBE_IDS = (356, 575, 583, 14, 531, 912)
COPIED_EMPTY_PHRASE = "nothing established from the image"


def require_dev_ids(selection, ids):
    allowed = set(runner.profile_ids(selection, "dev"))
    missing = [item_id for item_id in ids if item_id not in allowed]
    if missing:
        raise runner.RunnerError(f"debug ids are not in the development set: {missing}")
    return list(ids)


def run_extracts(items, backend, out_dir, max_side, extract_tokens):
    """Write one inventory per item. Returns the summary rows."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise runner.RunnerError(f"debug output already exists: {out_dir}")
    if COPIED_EMPTY_PHRASE in prompts.EXTRACTION_INSTRUCTION:
        raise runner.RunnerError("live extraction instruction still contains the copied empty example")
    out_dir.mkdir(parents=True)
    log = CallLog(out_dir, out_dir / "cache", backend, backend.model_identity(), 1)
    rows = []
    for item in items:
        messages = prompts.extract_messages(item["images"], item["question"], max_side)
        image_parts = sum(
            1
            for part in messages[0]["content"]
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        try:
            entry = log.call(messages, extract_tokens, "extract")
        except PauseRun as exc:
            raise runner.RunnerError(f"extraction transport failure on {item['id']}: {exc}") from None
        text = entry.get("text") or ""
        row = {
            "id": item["id"],
            "category": item["category"],
            "n_images": item["n_images"],
            "n_image_parts": image_parts,
            "finish_reason": entry.get("finish_reason"),
            "input_tokens": entry.get("input_tokens"),
            "output_tokens": entry.get("output_tokens"),
            "latency_seconds": entry.get("latency_seconds"),
        }
        try:
            inventory, repairs = parse_inventory_with_repairs(text, item["n_images"])
        except InventoryError as exc:
            (out_dir / "inventories").mkdir(exist_ok=True)
            _write(out_dir / "inventories" / f"{item['id']}.error.json", {
                "message": exc.message,
                "rule": exc.rule,
                "text": text,
            })
            row.update({
                "status": "extraction_invalid",
                "rule": exc.rule,
                "n_entities": None,
                "n_observations": None,
                "n_claims": None,
                "n_unknowns": None,
            })
        else:
            _write(out_dir / "inventories" / f"{item['id']}.json", inventory)
            _write(out_dir / "repairs" / f"{item['id']}.json", repairs)
            row.update({
                "status": "ok" if entry.get("status") != "truncated" else "extraction_truncated",
                "rule": None,
                "n_entities": len(inventory["entities"]),
                "n_observations": len(inventory["observations"]),
                "n_claims": len(inventory["claims"]),
                "n_unknowns": len(inventory["unknowns"]),
                "cross_view_repairs": [repair["entity_id"] for repair in repairs],
            })
        rows.append(row)
        print(
            f"{row['id']} status={row['status']} images={row['n_image_parts']} "
            f"entities={row['n_entities']} observations={row['n_observations']} "
            f"claims={row['n_claims']} unknowns={row['n_unknowns']} "
            f"output_tokens={row['output_tokens']}"
        )
    _write(out_dir / "summary.json", {
        "prompt_sha256": prompts.prompt_hashes(),
        "instruction": prompts._INSTRUCTION_PATH.name,
        "ids": [item["id"] for item in items],
        "rows": rows,
    })
    return rows


def _write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extraction-only debug retest. Not an evaluation profile.")
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--ids", default=",".join(str(item_id) for item_id in DEBUG_IDS))
    parser.add_argument("--endpoint", default=contracts.DEFAULT_ENDPOINT)
    parser.add_argument("--served-model", default=contracts.DEFAULT_SERVED_MODEL)
    parser.add_argument("--server-manifest", type=Path, required=True)
    parser.add_argument("--api-key-env", default=contracts.DEFAULT_API_KEY_ENV)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extract-tokens", type=int, default=contracts.DEFAULT_EXTRACT_TOKENS)
    parser.add_argument("--max-side", type=int, default=contracts.DEFAULT_MAX_SIDE)
    parser.add_argument("--timeout", type=int, default=contracts.DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        ids = [int(part) for part in args.ids.split(",") if part]
    except ValueError:
        parser.error("--ids must be comma-separated integers")
    selection = runner.load_selection(args.prepare_dir)
    require_dev_ids(selection, ids)
    items = runner._prepare_items(selection, "dev", ids, ["flat"])
    backend = VLLMBackend(
        args.endpoint,
        args.served_model,
        args.api_key_env,
        args.timeout,
        args.server_manifest,
    )
    try:
        run_extracts(items, backend, args.out, args.max_side, args.extract_tokens)
    finally:
        backend.close()
    return contracts.EXIT_CODES["complete"]


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path


def item_key(item: dict[str, object]) -> object:
    return item.get("id", (item.get("source_id"), item.get("target_id")))


def index(items: list[dict[str, object]]) -> dict[object, dict[str, object]]:
    return {item_key(item): item for item in items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dino", type=Path)
    parser.add_argument("lightglue", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    dino = json.loads(args.dino.read_text(encoding="utf-8"))
    lightglue = json.loads(args.lightglue.read_text(encoding="utf-8"))
    summary: dict[str, object] = {
        "dino_auc": dino["summary"]["geometric_score_auc_vs_mismatched_pairs"],
        "lightglue_auc": lightglue["summary"]["geometric_score_auc_vs_mismatched_pairs"],
    }

    for label, field in (("within", "measurements"), ("negative", "negative_controls")):
        dino_items = index(dino[field])
        lightglue_items = index(lightglue[field])
        keys = sorted(dino_items.keys() & lightglue_items.keys(), key=str)
        both = [
            key for key in keys
            if dino_items[key]["passes_negative_control_p95"]
            and lightglue_items[key]["passes_negative_control_p95"]
        ]
        either = [
            key for key in keys
            if dino_items[key]["passes_negative_control_p95"]
            or lightglue_items[key]["passes_negative_control_p95"]
        ]
        summary[f"{label}_pairs"] = len(keys)
        summary[f"{label}_both_p95"] = len(both)
        summary[f"{label}_either_p95"] = len(either)
        summary[f"{label}_both_ids"] = both

    payload = {
        "warning": (
            "Exploratory same-sample comparison. The joint rule must be evaluated on an "
            "independent, manually labeled overlap set before use as proof evidence."
        ),
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

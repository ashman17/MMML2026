from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def stable_order(key: str) -> str:
    return hashlib.sha256(("fvpg-overlap-v1:" + key).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("dino", type=Path)
    parser.add_argument("lightglue", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = {r["id"]: r for r in json.loads(args.records.read_text(encoding="utf-8"))}
    dino = json.loads(args.dino.read_text(encoding="utf-8"))
    lightglue = json.loads(args.lightglue.read_text(encoding="utf-8"))
    lg_within = {m["id"]: m for m in lightglue["measurements"]}
    lg_negative = {
        (m["source_id"], m["target_id"]): m for m in lightglue["negative_controls"]
    }

    items = []
    for measurement in dino["measurements"]:
        qid = measurement["id"]
        record = records[qid]
        items.append({
            "key": f"within-{qid}",
            "pair_type": "within_question",
            "source_question_id": qid,
            "target_question_id": qid,
            "images": record["images"][:2],
            "category": record["category"],
            "difficulty": record["difficulty"],
            "dino_score": measurement["geometric_score"],
            "lightglue_score": lg_within[qid]["geometric_score"],
        })
    for measurement in dino["negative_controls"]:
        source_id = measurement["source_id"]
        target_id = measurement["target_id"]
        items.append({
            "key": f"control-{source_id}-{target_id}",
            "pair_type": "mismatched_question_control",
            "source_question_id": source_id,
            "target_question_id": target_id,
            "images": [records[source_id]["images"][0], records[target_id]["images"][1]],
            "category": "blinded control",
            "difficulty": "hidden",
            "dino_score": measurement["geometric_score"],
            "lightglue_score": lg_negative[(source_id, target_id)]["geometric_score"],
        })
    items.sort(key=lambda item: stable_order(item["key"]))
    payload = {
        "schema_version": 1,
        "instructions": (
            "Label visible geometric overlap, not merely similar content. Pair type and tool scores "
            "are hidden by the annotation interface."
        ),
        "items": items,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Prepared {len(items)} blinded pairs at {args.output}")


if __name__ == "__main__":
    main()

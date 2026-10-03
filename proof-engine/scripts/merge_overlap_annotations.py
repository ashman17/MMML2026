from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import cohen_kappa


def load_export(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["key"]: item for item in payload["annotations"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.annotations) < 2:
        raise SystemExit("At least two independent annotation exports are required")

    exports = [load_export(path) for path in args.annotations]
    common_keys = set.intersection(*(set(export) for export in exports))
    annotated_keys = sorted(
        key for key in common_keys
        if all((export[key].get("annotation") or {}).get("relationship") for export in exports)
    )

    pairwise = []
    for left, right in itertools.combinations(range(len(exports)), 2):
        labels_left = [(exports[left][key]["annotation"] or {})["relationship"] for key in annotated_keys]
        labels_right = [(exports[right][key]["annotation"] or {})["relationship"] for key in annotated_keys]
        binary_left = ["overlap" if label == "overlap" else "not_overlap" for label in labels_left]
        binary_right = ["overlap" if label == "overlap" else "not_overlap" for label in labels_right]
        pairwise.append({
            "annotators": [args.annotations[left].name, args.annotations[right].name],
            "four_way_kappa": cohen_kappa(labels_left, labels_right),
            "binary_overlap_kappa": cohen_kappa(binary_left, binary_right),
        })

    items = []
    for key in annotated_keys:
        source = exports[0][key]
        relationships = [(export[key]["annotation"] or {})["relationship"] for export in exports]
        fractions = [(export[key]["annotation"] or {}).get("fraction") for export in exports]
        agreement = len(set(relationships)) == 1
        items.append({
            **{field: source[field] for field in (
                "key", "pair_type", "source_question_id", "target_question_id",
                "images", "category", "difficulty", "dino_score", "lightglue_score",
            )},
            "annotator_relationships": relationships,
            "consensus_relationship": relationships[0] if agreement else None,
            "consensus_fraction": fractions[0] if agreement and len(set(fractions)) == 1 else None,
            "needs_adjudication": not agreement,
        })

    payload = {
        "schema_version": 1,
        "annotators": [path.name for path in args.annotations],
        "fully_annotated_common_pairs": len(annotated_keys),
        "consensus_pairs": sum(not item["needs_adjudication"] for item in items),
        "adjudication_pairs": sum(item["needs_adjudication"] for item in items),
        "pairwise_agreement": pairwise,
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "fully_annotated_common_pairs", "consensus_pairs", "adjudication_pairs", "pairwise_agreement"
    )}, indent=2))


if __name__ == "__main__":
    main()

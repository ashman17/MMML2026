from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import PairItem, classification_metrics, component_split, select_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("consensus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--maximum-calibration-fpr", type=float, default=0.05)
    args = parser.parse_args()

    payload = json.loads(args.consensus.read_text(encoding="utf-8"))
    rows = [
        item for item in payload["items"]
        if item["consensus_relationship"] in {
            "overlap", "same_scene_no_overlap", "different_scene"
        }
    ]
    if len(rows) < 20:
        raise SystemExit("At least 20 adjudicated binary labels are required")
    splits = component_split([
        PairItem(item["key"], item["source_question_id"], item["target_question_id"])
        for item in rows
    ])
    calibration = [item for item in rows if splits[item["key"]] == "calibration"]
    evaluation = [item for item in rows if splits[item["key"]] == "evaluation"]
    if not calibration or not evaluation:
        raise SystemExit("Component split produced an empty partition")

    results = {}
    for name, field in (("dino", "dino_score"), ("lightglue", "lightglue_score")):
        train_labels = np.asarray([item["consensus_relationship"] == "overlap" for item in calibration])
        train_scores = np.asarray([item[field] for item in calibration], dtype=float)
        test_labels = np.asarray([item["consensus_relationship"] == "overlap" for item in evaluation])
        test_scores = np.asarray([item[field] for item in evaluation], dtype=float)
        if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
            raise SystemExit("Both splits must contain overlap and non-overlap labels")
        threshold = select_threshold(train_labels, train_scores, args.maximum_calibration_fpr)
        results[name] = {
            "calibration": classification_metrics(train_labels, train_scores, threshold),
            "evaluation": classification_metrics(test_labels, test_scores, threshold),
        }

    # Agreement requires both independently selected thresholds. It is evaluated
    # as a verifier policy, not retrospectively retuned on the test split.
    dino_threshold = results["dino"]["calibration"]["threshold"]
    lightglue_threshold = results["lightglue"]["calibration"]["threshold"]
    for split_name, split_rows in (("calibration", calibration), ("evaluation", evaluation)):
        labels = np.asarray([item["consensus_relationship"] == "overlap" for item in split_rows])
        scores = np.asarray([
            min(item["dino_score"] - dino_threshold,
                item["lightglue_score"] - lightglue_threshold)
            for item in split_rows
        ])
        results.setdefault("agreement", {})[split_name] = classification_metrics(labels, scores, 0.0)

    report = {
        "warning": "Thresholds selected on the calibration split only; uncertain labels excluded.",
        "split": {
            "calibration_pairs": len(calibration),
            "evaluation_pairs": len(evaluation),
            "image_leakage_prevention": "Connected question-ID components remain in one split.",
        },
        "maximum_calibration_fpr": args.maximum_calibration_fpr,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["category"], row["difficulty"])].append(row)

    def summarize(items):
        raw_valid = [
            item for item in items
            if item.get("raw_semantic_valid", item.get("semantic_valid", not item.get("error")))
        ]
        valid = [
            item for item in items
            if item.get("compiled_semantic_valid", item.get("semantic_valid", not item.get("error")))
        ]
        confidences = [
            item["parsed"].get("confidence") for item in items
            if item.get("parsed") and isinstance(item["parsed"].get("confidence"), (int, float))
        ]
        return {
            "n": len(items),
            "accuracy": sum(item["correct"] for item in items) / len(items),
            "parse_rate": sum(item["predicted_answer"] is not None for item in items) / len(items),
            "raw_semantic_valid_rate": len(raw_valid) / len(items),
            "compiled_semantic_valid_rate": len(valid) / len(items),
            "mean_compiler_repairs": statistics.mean(
                len(item.get("compiler_repairs", [])) for item in items
            ),
            "valid_accuracy": (
                sum(item["correct"] for item in valid) / len(valid) if valid else None
            ),
            "mean_confidence": statistics.mean(confidences) if confidences else None,
            "mean_wall_seconds": statistics.mean(item["wall_seconds"] for item in items),
            "mean_output_tokens": statistics.mean(
                item["ollama_metrics"]["eval_count"] or 0 for item in items
            ),
        }

    report = {
        "overall": {
            condition: summarize([row for row in rows if row["condition"] == condition])
            for condition in sorted(set(row["condition"] for row in rows))
        },
        "strata": [
            {"condition": key[0], "category": key[1], "difficulty": key[2], **summarize(items)}
            for key, items in sorted(groups.items())
        ],
        "errors": [
            {"id": row["id"], "condition": row["condition"], "error": row["error"]}
            for row in rows if row["error"]
        ],
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()

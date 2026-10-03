from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import MMSIProofCompiler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    compiler = MMSIProofCompiler()
    positional = [r for r in records if "Positional Relationship" in r["category"]]
    compiled = [compiler.compile_record(record) for record in positional]

    fully_compiled = [item for item in compiled if len(item.hypotheses) == 4]
    exact_entities = [item for item in compiled if not any("parse" in w or "unresolved" in w for w in item.warnings)]
    usable_entities = [item for item in compiled if item.target != "target_entity"]
    payload = {
        "summary": {
            "positional_questions": len(compiled),
            "four_option_relation_coverage": len(fully_compiled),
            "exact_entity_template_coverage": len(exact_entities),
            "target_extraction_coverage": len(usable_entities),
        },
        "questions": [item.to_dict() for item in compiled],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()

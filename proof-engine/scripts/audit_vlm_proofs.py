from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import compile_proof_program, validate_proof_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run semantic proof contracts on saved VLM outputs.")
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = []
    for line in args.results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("condition") == "proof" and row.get("parsed"):
            raw_errors = validate_proof_payload(row["parsed"], row["image_count"])
            compiled, repairs = compile_proof_program(row["parsed"], row["image_count"])
            compiled_errors = validate_proof_payload(compiled, row["image_count"])
            row["raw_semantic_errors"] = raw_errors
            row["raw_semantic_valid"] = not raw_errors
            row["compiled_proof"] = compiled
            row["compiler_repairs"] = repairs
            row["compiled_semantic_errors"] = compiled_errors
            row["compiled_semantic_valid"] = not compiled_errors
            row["semantic_errors"] = compiled_errors
            row["semantic_valid"] = not compiled_errors
            row["error"] = "; ".join(compiled_errors) if compiled_errors else None
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "raw_semantic_valid": sum(row.get("raw_semantic_valid", False) for row in rows),
        "compiled_semantic_valid": sum(row.get("compiled_semantic_valid", False) for row in rows),
    }))


if __name__ == "__main__":
    main()

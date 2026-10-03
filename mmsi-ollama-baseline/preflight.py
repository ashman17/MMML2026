from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline import api_json, model_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("../mmsi-explorer/dist/records.json"))
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()

    records_path = args.records.resolve()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    missing = [
        str(records_path.parent / image)
        for record in records for image in record.get("images", [])
        if not (records_path.parent / image).is_file()
    ]
    version = api_json(args.base_url, "/api/version")
    model = model_provenance(args.base_url, args.model)
    print(json.dumps({
        "ready": not missing,
        "ollama_version": version.get("version"),
        "model": model,
        "records": len(records),
        "images": sum(len(record.get("images", [])) for record in records),
        "missing_images": missing[:10],
    }, indent=2))
    if missing:
        raise SystemExit(f"{len(missing)} image files are missing")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from baseline import (
    OFFICIAL_REPO_COMMIT,
    PROTOCOL_VERSION,
    api_json,
    build_request,
    encode_images,
    extract_official_choice,
    model_provenance,
    official_prompt,
    select_records,
    summarize,
)


def completed_keys(path: Path) -> set[tuple[int, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("error"):
                keys.add((int(row["id"]), row["model"]["digest"], row["protocol_version"]))
    return keys


def latest_rows(path: Path) -> list[dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (int(row["id"]), row["model"]["digest"], row["protocol_version"])
            latest[key] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Official-style MMSI-Bench one-pass baseline via Ollama")
    parser.add_argument("--records", type=Path, default=Path("../mmsi-explorer/dist/records.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3vl_4b.jsonl"))
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category", action="append")
    parser.add_argument("--difficulty", action="append")
    parser.add_argument("--id", type=int, action="append")
    parser.add_argument("--sampling", choices=("official-order", "stratified"), default="official-order")
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--num-ctx", type=int, default=8192)
    args = parser.parse_args()

    records_path = args.records.resolve()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    selected = select_records(
        records,
        limit=args.limit,
        categories=set(args.category or []),
        difficulties=set(args.difficulty or []),
        ids=set(args.id or []),
        sampling=args.sampling,
        seed=args.sample_seed,
    )
    if not selected:
        raise SystemExit("No records match the requested filters")

    provenance = model_provenance(args.base_url, args.model)
    done = completed_keys(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(selected, start=1):
        key = (int(record["id"]), provenance["digest"], PROTOCOL_VERSION)
        if key in done:
            continue
        prompt = official_prompt(record["question"])
        started = time.perf_counter()
        response = None
        try:
            images = encode_images(records_path.parent, record["images"])
            request = build_request(args.model, prompt, images, num_ctx=args.num_ctx)
            response = api_json(args.base_url, "/api/chat", request)
            raw = response["message"]["content"]
            extracted = extract_official_choice(raw)
            error = None
        except Exception as exception:
            raw = None
            extracted = None
            error = f"{type(exception).__name__}: {exception}"
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "id": int(record["id"]),
            "category": record["category"],
            "difficulty": record["difficulty"],
            "question": record["question"],
            "images": record["images"],
            "gold_answer": record["answer"],
            "raw_prediction": raw,
            "extracted_answer": extracted,
            "correct": extracted == record["answer"],
            "error": error,
            "model": provenance,
            "sampling": args.sampling,
            "sample_seed": args.sample_seed,
            "num_ctx": args.num_ctx,
            "prompt_sha256": hashlib.sha256(
                request["messages"][0]["content"].encode("utf-8")
            ).hexdigest() if response is not None else None,
            "wall_seconds": time.perf_counter() - started,
            "ollama_metrics": {
                metric: response.get(metric) if response else None
                for metric in (
                    "total_duration", "load_duration", "prompt_eval_count",
                    "prompt_eval_duration", "eval_count", "eval_duration",
                )
            },
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done.add(key)
        print(json.dumps({
            "progress": f"{index}/{len(selected)}", "id": row["id"],
            "prediction": extracted, "gold": row["gold_answer"],
            "correct": row["correct"], "seconds": round(row["wall_seconds"], 2),
            "error": error,
        }))

    rows = latest_rows(args.output)
    report = summarize(rows)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **report["overall"]}, indent=2))


if __name__ == "__main__":
    main()

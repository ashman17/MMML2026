from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg import (
    PROOF_COMPILER_VERSION,
    PROTOCOL_VERSION,
    build_ollama_request,
    compile_proof_program,
    parse_structured_response,
    stratified_sample,
    validate_proof_payload,
)


def api_json(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def model_provenance(base_url: str, model: str) -> dict[str, Any]:
    tags = api_json(base_url, "/api/tags")
    matches = [item for item in tags.get("models", []) if item.get("name") == model]
    if not matches:
        raise RuntimeError(f"Model {model!r} is not installed at {base_url}")
    item = matches[0]
    return {
        "name": item.get("name"),
        "digest": item.get("digest"),
        "size": item.get("size"),
        "details": item.get("details"),
    }


def load_completed(path: Path) -> set[tuple[int, str, str]]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed.add((row["id"], row["condition"], row["model"]["digest"]))
    return completed


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--condition", choices=("direct", "proof", "both"), default="direct")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--category", action="append")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    records_path = args.records.resolve()
    image_root = records_path.parent
    records = json.loads(records_path.read_text(encoding="utf-8"))
    categories = set(args.category) if args.category else {
        "Positional Relationship (Cam.–Cam.)",
        "Positional Relationship (Cam.–Obj.)",
        "Positional Relationship (Obj.–Obj.)",
    }
    selected = stratified_sample(records, args.limit, categories)
    provenance = model_provenance(args.base_url, args.model)
    completed = load_completed(args.output)
    conditions = ("direct", "proof") if args.condition == "both" else (args.condition,)

    for record in selected:
        images = [
            base64.b64encode((image_root / image).read_bytes()).decode("ascii")
            for image in record["images"]
        ]
        for condition in conditions:
            key = (record["id"], condition, provenance["digest"])
            if key in completed:
                continue
            request = build_ollama_request(args.model, record["question"], images, condition, args.seed)
            started = time.perf_counter()
            try:
                response = api_json(args.base_url, "/api/chat", request)
                content = response["message"]["content"]
                parsed = parse_structured_response(content)
                if condition == "proof":
                    raw_semantic_errors = validate_proof_payload(parsed, len(images))
                    compiled_proof, compiler_repairs = compile_proof_program(parsed, len(images))
                    compiled_semantic_errors = validate_proof_payload(compiled_proof, len(images))
                else:
                    raw_semantic_errors = []
                    compiled_proof = None
                    compiler_repairs = []
                    compiled_semantic_errors = []
                semantic_errors = compiled_semantic_errors
                error = "; ".join(compiled_semantic_errors) if compiled_semantic_errors else None
            except Exception as exception:
                response = None
                content = None
                parsed = None
                semantic_errors = []
                raw_semantic_errors = []
                compiled_proof = None
                compiler_repairs = []
                compiled_semantic_errors = []
                error = f"{type(exception).__name__}: {exception}"
            row = {
                "id": record["id"],
                "category": record["category"],
                "difficulty": record["difficulty"],
                "condition": condition,
                "model": provenance,
                "seed": args.seed,
                "protocol_version": PROTOCOL_VERSION,
                "proof_compiler_version": PROOF_COMPILER_VERSION,
                "question": record["question"],
                "images": record["images"],
                "image_count": len(images),
                "gold_answer": record["answer"],
                "predicted_answer": parsed.get("answer") if parsed else None,
                "correct": parsed.get("answer") == record["answer"] if parsed else False,
                "parsed": parsed,
                "raw_semantic_errors": raw_semantic_errors,
                "raw_semantic_valid": parsed is not None and not raw_semantic_errors,
                "compiled_proof": compiled_proof,
                "compiler_repairs": compiler_repairs,
                "compiled_semantic_errors": compiled_semantic_errors,
                "compiled_semantic_valid": parsed is not None and not compiled_semantic_errors,
                "semantic_errors": semantic_errors,
                "semantic_valid": parsed is not None and not semantic_errors,
                "raw_content": content,
                "error": error,
                "wall_seconds": time.perf_counter() - started,
                "ollama_metrics": {
                    key: response.get(key) if response else None
                    for key in (
                        "total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration",
                    )
                },
            }
            append_jsonl(args.output, row)
            completed.add(key)
            print(json.dumps({
                "id": row["id"], "condition": condition,
                "prediction": row["predicted_answer"], "gold": row["gold_answer"],
                "correct": row["correct"], "seconds": round(row["wall_seconds"], 2),
                "error": error,
            }))


if __name__ == "__main__":
    main()

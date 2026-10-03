from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from baseline import (
    api_json,
    api_stream_json,
    build_claim_verification_request,
    build_inspection_request,
    encode_images,
    extract_official_choice,
    extract_inspection_answer,
    extract_numbered_claims,
    model_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect model output on selected MMSI examples")
    parser.add_argument("--id", type=int, action="append", required=True)
    parser.add_argument("--records", type=Path, default=Path("../mmsi-explorer/dist/records.json"))
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument(
        "--temperature", type=float,
        help="Sampling temperature; defaults to 0.6 for rationale modes and 0 for --official-direct",
    )
    parser.add_argument("--show-reference-rationale", action="store_true")
    parser.add_argument("--no-stream", action="store_true", help="Wait for the complete response")
    parser.add_argument("--official-direct", action="store_true", help="Use the official direct-answer prompt")
    parser.add_argument("--rationale", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--structured", action="store_true",
        help="Use the non-official JSON rationale schema",
    )
    parser.add_argument(
        "--verify-claims", action="store_true",
        help="Run a second, reference-grounded audit of the generated claims",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/inspection.jsonl"))
    args = parser.parse_args()
    if args.official_direct and (args.structured or args.verify_claims):
        parser.error("--official-direct cannot be combined with --structured or --verify-claims")
    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.official_direct else 0.6

    records_path = args.records.resolve()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_id = {int(record["id"]): record for record in records}
    missing = [question_id for question_id in args.id if question_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown question IDs: {missing}")
    provenance = model_provenance(args.base_url, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for question_id in args.id:
        record = by_id[question_id]
        encoded = encode_images(records_path.parent, record["images"])
        request = build_inspection_request(
            args.model, record["question"], encoded, args.num_ctx,
            stream=not args.no_stream,
            rationale=not args.official_direct and not args.structured,
            structured=args.structured,
            temperature=temperature,
        )
        print(f"\n=== MMSI question {question_id} ===")
        print(record["question"])
        if args.structured:
            mode = "structured-rationale"
        elif args.official_direct:
            mode = "official-direct"
        else:
            mode = "claim-trace"
        print(f"\n--- Model output ({mode}) ---")
        started = time.perf_counter()
        if args.no_stream:
            response = api_json(args.base_url, "/api/chat", request)
            raw = response["message"]["content"]
            print(raw, flush=True)
        else:
            pieces = []
            response = {}
            for event in api_stream_json(args.base_url, "/api/chat", request):
                piece = event.get("message", {}).get("content", "")
                if piece:
                    print(piece, end="", flush=True)
                    pieces.append(piece)
                if event.get("done"):
                    response = event
            print(flush=True)
            raw = "".join(pieces)
        if args.structured:
            generated_output = json.loads(raw)
            extracted_answer = generated_output["answer"]
        elif args.official_direct:
            generated_output = raw
            extracted_answer = extract_official_choice(raw)
        else:
            generated_output = raw
            extracted_answer = extract_inspection_answer(raw)
        claim_audit = None
        generated_claims = extract_numbered_claims(raw) if mode == "claim-trace" else []
        if args.verify_claims:
            print("\n--- Reference-grounded claim audit ---")
            audit_request = build_claim_verification_request(
                args.model,
                record["question"],
                encoded,
                raw,
                record["answer"],
                record["thought"],
                args.num_ctx,
            )
            audit_response = api_json(args.base_url, "/api/chat", audit_request)
            claim_audit = json.loads(audit_response["message"]["content"])
            for claim in claim_audit["claims"]:
                print(
                    f"Claim {claim['claim_number']}: {claim['verdict']} "
                    f"[{claim['error_type']}] — {claim['explanation']}"
                )
            expected = {claim["claim_number"] for claim in generated_claims}
            returned = {claim["claim_number"] for claim in claim_audit["claims"]}
            if expected != returned:
                print(f"Audit warning: missing claim verdicts {sorted(expected - returned)}")
            print(f"Primary failure: {claim_audit['primary_failure']}")
        row = {
            "id": question_id,
            "category": record["category"],
            "difficulty": record["difficulty"],
            "question": record["question"],
            "images": record["images"],
            "model": provenance,
            "inspection_mode": mode,
            "generation": {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 20,
                "num_ctx": args.num_ctx,
                "num_predict": 2048,
            },
            "generated_output": generated_output,
            "raw_response": raw,
            "generated_claims": generated_claims,
            "claim_audit": claim_audit,
            "extracted_answer": extracted_answer,
            "gold_answer": record["answer"],
            "correct": extracted_answer == record["answer"],
            "wall_seconds": time.perf_counter() - started,
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\nPredicted: {extracted_answer} | Gold: {record['answer']} | Correct: {row['correct']}")
        print(f"Runtime: {row['wall_seconds']:.2f} seconds")
        if args.show_reference_rationale:
            print("\n--- Human reference rationale (not shown to model) ---")
            print(record["thought"])


if __name__ == "__main__":
    main()

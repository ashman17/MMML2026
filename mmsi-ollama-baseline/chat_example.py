from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from baseline import (
    api_stream_json,
    claim_trace_prompt,
    encode_images,
    extract_inspection_answer,
    extract_numbered_claims,
    model_provenance,
    prompt_with_image_placeholders,
)


SYSTEM_PROMPT = """You are participating in an iterative audit of a multi-image spatial problem.
On every response, produce a complete revised claim trace using the format requested in the first
user message: observations for every image in order, common aspects, then inferences, uncertainty,
and a final answer. Consider later user corrections carefully, but do not accept them blindly:
recheck them against the images. Begin revised responses with a short REVISION NOTES section that
states which feedback you accepted or rejected and why. Never omit an image merely because the
feedback concerns another image."""


def request_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    num_ctx: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 20,
            "num_predict": 2048,
            "num_ctx": num_ctx,
        },
    }


def rebuild_messages(
    question: str,
    encoded_images: list[str],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": prompt_with_image_placeholders(
                claim_trace_prompt(question, len(encoded_images)), len(encoded_images)
            ),
            "images": encoded_images,
        },
    ]
    for turn in turns:
        messages.append({"role": "assistant", "content": turn["assistant"]})
        if turn.get("feedback"):
            messages.append({"role": "user", "content": turn["feedback"]})
    return messages


def stream_response(base_url: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pieces = []
    final_event: dict[str, Any] = {}
    for event in api_stream_json(base_url, "/api/chat", payload):
        piece = event.get("message", {}).get("content", "")
        if piece:
            print(piece, end="", flush=True)
            pieces.append(piece)
        if event.get("done"):
            final_event = event
    print(flush=True)
    return "".join(pieces), final_event


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def image_manifest(image_root: Path, image_paths: list[str]) -> list[dict[str, Any]]:
    manifest = []
    for image_number, relative_path in enumerate(image_paths, 1):
        path = image_root / relative_path
        raw = path.read_bytes()
        manifest.append({
            "image_number": image_number,
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Iteratively correct an MMSI model claim trace")
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--records", type=Path, default=Path("../mmsi-explorer/dist/records.json"))
    parser.add_argument("--model", default="qwen3-vl:4b-instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature; Qwen recommends 0.6 for thinking models",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--debug-inputs", action="store_true",
        help="Print the ordered image manifest without printing base64 data",
    )
    args = parser.parse_args()

    records_path = args.records.resolve()
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_id = {int(record["id"]): record for record in records}
    if args.id not in by_id:
        raise SystemExit(f"Unknown question ID: {args.id}")
    record = by_id[args.id]
    output = args.output or Path(f"outputs/chat_question_{args.id}.json")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not output.is_file():
            raise SystemExit(f"Cannot resume; transcript does not exist: {output}")
        state = json.loads(output.read_text(encoding="utf-8"))
        if int(state["id"]) != args.id:
            raise SystemExit("Transcript question ID does not match --id")
        model = state["model"]["name"]
    else:
        if output.exists():
            raise SystemExit(f"Transcript already exists; use --resume or another --output: {output}")
        model = args.model
        state = {
            "id": args.id,
            "question": record["question"],
            "images": record["images"],
            "model": model_provenance(args.base_url, model),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "turns": [],
        }
        save_state(output, state)

    encoded_images = encode_images(records_path.parent, record["images"])
    manifest = image_manifest(records_path.parent, record["images"])
    print(f"\n=== MMSI correction chat: question {args.id} ===")
    print(record["question"])
    print(f"Images supplied every turn through retained history: {len(encoded_images)}")
    print(f"Sampling: temperature={args.temperature}, top_p=0.95, top_k=20")
    if args.debug_inputs:
        for item in manifest:
            print(
                f"Image {item['image_number']}: {item['path']} | "
                f"{item['bytes']} bytes | sha256={item['sha256']}"
            )
    print("Commands: /quit, /show, /claims, /help")
    print(f"Transcript: {output}")

    while True:
        turns = state["turns"]
        if not turns or turns[-1].get("feedback"):
            messages = rebuild_messages(record["question"], encoded_images, turns)
            print(f"\n--- Model revision {len(turns) + 1} ---")
            started = time.perf_counter()
            raw, event = stream_response(
                args.base_url,
                request_payload(
                    model, messages, num_ctx=args.num_ctx, temperature=args.temperature
                ),
            )
            turn = {
                "revision": len(turns) + 1,
                "assistant": raw,
                "answer": extract_inspection_answer(raw),
                "claims": extract_numbered_claims(raw),
                "feedback": None,
                "wall_seconds": time.perf_counter() - started,
                "ollama_metrics": {
                    key: event.get(key)
                    for key in (
                        "done_reason", "prompt_eval_count", "prompt_eval_duration",
                        "eval_count", "eval_duration", "total_duration",
                    )
                },
                "generation": {
                    "temperature": args.temperature,
                    "top_p": 0.95,
                    "top_k": 20,
                    "num_ctx": args.num_ctx,
                    "num_predict": 2048,
                },
            }
            turns.append(turn)
            save_state(output, state)
            print(f"Parsed answer: {turn['answer']} | Claims: {len(turn['claims'])}")

        try:
            feedback = input("\nfeedback> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChat saved.")
            break
        if not feedback:
            continue
        if feedback == "/quit":
            print("Chat saved.")
            break
        if feedback == "/help":
            print("Enter corrections such as: Claim 4 is wrong; Image 2 shows the shower, not the toilet.")
            print("Commands: /quit, /show, /claims, /help")
            continue
        if feedback == "/show":
            print(turns[-1]["assistant"])
            continue
        if feedback == "/claims":
            for claim in turns[-1]["claims"]:
                print(f"{claim['claim_number']}. {claim['text']}")
            continue
        turns[-1]["feedback"] = feedback
        save_state(output, state)


if __name__ == "__main__":
    main()

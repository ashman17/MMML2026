from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Iterator
from urllib.request import Request, urlopen
from urllib.error import HTTPError


PROTOCOL_VERSION = "mmsi-official-vanilla-ollama-v4-separated-images"
OFFICIAL_REPO_COMMIT = "13e58a2b8b30d880d7e8a1e4a6aa1c0feda94cac"
OFFICIAL_POST_PROMPT = (
    "Answer with the option's letter from the given choices directly. "
    "Enclose the option's letter within ``."
)

CLAIM_TRACE_INSTRUCTION = (
    "Let's think step by step before answering. Follow this order exactly:\n"
    "\n"
    "IMAGE-BY-IMAGE OBSERVATIONS\n"
    "For each supplied image in order, create a separate IMAGE N OBSERVATIONS section, "
    "starting with Image 1 and finishing it before examining Image 2, then continuing for "
    "all remaining images. Record only directly visible facts relevant to the question. "
    "Do not make spatial inferences in these sections.\n"
    "\n"
    "COMMON ASPECTS\n"
    "After all per-image sections, identify objects, regions, landmarks, or visual features "
    "that correspond across images. Explicitly cite every supporting image. Do not assume "
    "two things are the same without stating the visual evidence.\n"
    "\n"
    "INFERENCES\n"
    "Only after the common-aspects section, infer viewpoint changes, camera pose, scene "
    "layout, and the requested spatial relation. State which earlier claims support each "
    "inference.\n"
    "\n"
    "Number every observation, common-aspect claim, and inference sequentially. Keep each "
    "claim atomic: do not combine multiple factual assertions. Use one-based image labels "
    "throughout (the first supplied image is Image 1). Then write UNCERTAIN CLAIMS followed "
    "by the least-certain claim numbers, or NONE. Do not claim to have used tools. End with "
    "exactly FINAL ANSWER: ``X``, where X is A, B, C, or D."
)

INSPECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "view": {"type": "integer"},
                    "claim": {"type": "string"},
                },
                "required": ["view", "claim"],
                "additionalProperties": False,
            },
        },
        "cross_view_inferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "evidence_views": {"type": "array", "items": {"type": "integer"}},
                    "uncertainty": {"type": "string"},
                },
                "required": ["claim", "evidence_views", "uncertainty"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["observations", "cross_view_inferences", "answer", "confidence"],
    "additionalProperties": False,
}

CLAIM_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_number": {"type": "integer"},
                    "verdict": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "not_established"],
                    },
                    "explanation": {"type": "string"},
                    "error_type": {
                        "type": "string",
                        "enum": [
                            "none", "grounding", "cross_view_reconstruction",
                            "situation_transformation", "spatial_logic",
                        ],
                    },
                },
                "required": ["claim_number", "verdict", "explanation", "error_type"],
                "additionalProperties": False,
            },
        },
        "primary_failure": {"type": "string"},
    },
    "required": ["claims", "primary_failure"],
    "additionalProperties": False,
}


def official_prompt(question: str) -> str:
    """Match MMSIBenchDataset.build_prompt at the pinned official commit."""
    return f"{question}\n{OFFICIAL_POST_PROMPT}"


def claim_trace_prompt(question: str, image_count: int | None = None) -> str:
    if image_count is None:
        prefix = "You are given multiple separate images in a fixed order."
    else:
        prefix = (
            f"You are given exactly {image_count} separate images in a fixed order, numbered "
            f"Image 1 through Image {image_count}. You must inspect and report on every image."
        )
    return f"{prefix}\n\n{CLAIM_TRACE_INSTRUCTION}\n\n{question}"


def prompt_with_image_placeholders(prompt: str, image_count: int) -> str:
    """Keep same-sized Qwen-VL images distinct in Ollama's renderer."""
    image_blocks = "\n\n".join(
        f"IMAGE {image_number}:\n[img]" for image_number in range(1, image_count + 1)
    )
    return f"{image_blocks}\n\n{prompt}" if image_blocks else prompt


def extract_official_choice(prediction: Any) -> str | None:
    """Match MMSIBenchDataset.extract_single_choice_with_word_boundary."""
    if prediction is None:
        return None
    try:
        prediction = str(prediction)
    except Exception:
        return None

    match = re.search(r"``([^`]*)``", prediction)
    if match:
        prediction = match.group(1)
    match = re.search(r"`([^`]*)`", prediction)
    if match:
        prediction = match.group(1)
    match = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", prediction)
    return match.group() if match else None


def extract_inspection_answer(prediction: Any) -> str | None:
    """Extract only an explicitly marked final answer from a natural rationale."""
    if prediction is None:
        return None
    text = str(prediction)
    matches = re.findall(
        r"FINAL\s+ANSWER\s*:\s*[`*\[(]*([A-D])[`*\])\.]*",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()
    final_nonempty_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    match = re.fullmatch(r"[`*\[(]*([A-D])[`*\])\.]*", final_nonempty_line, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_numbered_claims(prediction: Any) -> list[dict[str, Any]]:
    """Extract top-level numbered claims from a visible claim trace."""
    if prediction is None:
        return []
    claims = []
    for match in re.finditer(r"(?m)^\s*(\d+)\.\s+(.+)$", str(prediction)):
        claims.append({"claim_number": int(match.group(1)), "text": match.group(2).strip()})
    return claims


def api_json(
    base_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 900,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exception:
        body = exception.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exception.code}: {body}") from exception


def api_stream_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: int = 900,
) -> Iterator[dict[str, Any]]:
    """Yield Ollama newline-delimited JSON events as they arrive."""
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            for line in response:
                if line.strip():
                    yield json.loads(line)
    except HTTPError as exception:
        body = exception.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exception.code}: {body}") from exception


def model_provenance(base_url: str, model: str) -> dict[str, Any]:
    tags = api_json(base_url, "/api/tags")
    candidates = [item for item in tags.get("models", []) if item.get("name") == model]
    if not candidates:
        installed = [item.get("name") for item in tags.get("models", [])]
        raise RuntimeError(f"Model {model!r} is not installed. Installed models: {installed}")
    item = candidates[0]
    return {
        "name": item.get("name"),
        "digest": item.get("digest"),
        "size": item.get("size"),
        "details": item.get("details"),
    }


def encode_images(image_root: Path, image_paths: Iterable[str]) -> list[str]:
    encoded = []
    for relative_path in image_paths:
        path = image_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded.append(base64.b64encode(path.read_bytes()).decode("ascii"))
    return encoded


def build_request(
    model: str, prompt: str, encoded_images: list[str], num_ctx: int = 8192
) -> dict[str, Any]:
    prompt = prompt_with_image_placeholders(prompt, len(encoded_images))
    return {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": encoded_images,
        }],
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": 2048,
            "num_ctx": num_ctx,
        },
    }


def build_inspection_request(
    model: str,
    question: str,
    encoded_images: list[str],
    num_ctx: int = 8192,
    stream: bool = True,
    rationale: bool = False,
    structured: bool = False,
    temperature: float = 0.0,
) -> dict[str, Any]:
    if structured:
        instruction = (
            "Solve the spatial question using concise, checkable claims. Separate direct per-image "
            "observations from cross-image inferences. For every inference, cite the image indices "
            "that support it and state any uncertainty."
        )
        prompt = (
            f"There are {len(encoded_images)} images, numbered Image 1 through Image {len(encoded_images)}. "
            + instruction + "\n\n" + question
        )
    elif rationale:
        prompt = claim_trace_prompt(question, len(encoded_images))
    else:
        # Exact text construction used by MMSIBenchDataset.build_prompt.
        prompt = official_prompt(question)
    prompt = prompt_with_image_placeholders(prompt, len(encoded_images))
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": encoded_images}],
        "stream": stream,
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
    if structured:
        request["format"] = INSPECTION_SCHEMA
    return request


def build_claim_verification_request(
    model: str,
    question: str,
    encoded_images: list[str],
    model_output: str,
    gold_answer: str,
    reference_rationale: str,
    num_ctx: int = 8192,
) -> dict[str, Any]:
    """Build a post-hoc audit request; gold/reference are never sent to the solver."""
    claims = extract_numbered_claims(model_output)
    claims_text = "\n".join(
        f"{claim['claim_number']}. {claim['text']}" for claim in claims
    ) or model_output
    prompt = f"""Audit the numbered claims in a model's spatial reasoning.

Judge each claim against the supplied images. The human reference is useful evidence but is not
the only possible valid reasoning path. Use:
- supported: directly supported by the images or a valid inference;
- contradicted: conflicts with the images or reference geometry;
- not_established: plausible but lacks sufficient evidence.

Assign the most relevant MMSI error type to every failed claim: grounding,
cross_view_reconstruction, situation_transformation, or spatial_logic. Use none for supported
claims. Be concise.

You must return exactly one verdict for every numbered claim below, retaining its claim number.
Do not merge claims and do not omit conclusions.

QUESTION:
{question}

NUMBERED CLAIMS TO AUDIT:
{claims_text}

CORRECT ANSWER: {gold_answer}

HUMAN REFERENCE REASONING:
{reference_rationale}
"""
    prompt = prompt_with_image_placeholders(prompt, len(encoded_images))
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": encoded_images}],
        "stream": False,
        "think": False,
        "format": CLAIM_VERIFICATION_SCHEMA,
        "keep_alive": "10m",
        "options": {"temperature": 0, "num_predict": 2048, "num_ctx": num_ctx},
    }


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    categories: set[str] | None,
    difficulties: set[str] | None,
    ids: set[int] | None,
    sampling: str,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        record for record in records
        if (not categories or record["category"] in categories)
        and (not difficulties or record["difficulty"] in difficulties)
        and (not ids or int(record["id"]) in ids)
    ]
    if sampling == "official-order":
        selected = eligible
    elif sampling == "stratified":
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in eligible:
            groups[(record["category"], record["difficulty"])].append(record)
        for group in groups.values():
            group.sort(key=lambda record: hashlib.sha256(
                f"{seed}:{record['id']}".encode("utf-8")
            ).hexdigest())
        selected = []
        ordered_keys = sorted(
            groups,
            key=lambda key: hashlib.sha256(
                f"{seed}:{key[0]}:{key[1]}".encode("utf-8")
            ).hexdigest(),
        )
        depth = 0
        while any(depth < len(group) for group in groups.values()):
            for key in ordered_keys:
                if depth < len(groups[key]):
                    selected.append(groups[key][depth])
            depth += 1
    else:
        raise ValueError(f"Unknown sampling mode: {sampling}")
    return selected[:limit] if limit is not None else selected


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(item.get("correct")) for item in items)
        parsed = sum(item.get("extracted_answer") is not None for item in items)
        low, high = wilson_interval(correct, len(items))
        return {
            "n": len(items),
            "correct": correct,
            "accuracy": correct / len(items) if items else None,
            "accuracy_wilson_95": [low, high],
            "parse_rate": parsed / len(items) if items else None,
            "successful_requests": sum(not item.get("error") for item in items),
            "request_success_rate": (
                sum(not item.get("error") for item in items) / len(items) if items else None
            ),
            "mean_wall_seconds": statistics.mean(
                item["wall_seconds"] for item in items
            ) if items else None,
            "mean_output_tokens": statistics.mean(
                item.get("ollama_metrics", {}).get("eval_count") or 0 for item in items
            ) if items else None,
        }

    strata = []
    for field in ("category", "difficulty"):
        values = sorted({row[field] for row in rows})
        strata.extend({"group": field, "value": value, **aggregate([
            row for row in rows if row[field] == value
        ])} for value in values)
    return {"protocol_version": PROTOCOL_VERSION, "overall": aggregate(rows), "strata": strata}

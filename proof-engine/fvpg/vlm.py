from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = "fvpg-vlm-v3"

PREDICATES = [
    "SAME_ENTITY",
    "VIEW_OVERLAP",
    "RELATIVE_IMAGE_POSITION",
    "DEPTH_ORDER",
    "CAMERA_TRANSLATION",
    "CAMERA_ROTATION",
    "OBJECT_FACING",
    "FRAME_TRANSFORM",
    "TARGET_RELATION",
]

VERIFIERS = [
    "grounding",
    "correspondence",
    "depth",
    "camera_pose",
    "geometry_checker",
    "vlm_judge",
]

VERIFIER_BY_PREDICATE = {
    "SAME_ENTITY": {"correspondence"},
    "VIEW_OVERLAP": {"correspondence"},
    "RELATIVE_IMAGE_POSITION": {"grounding", "vlm_judge"},
    "DEPTH_ORDER": {"depth"},
    "CAMERA_TRANSLATION": {"camera_pose"},
    "CAMERA_ROTATION": {"camera_pose"},
    "OBJECT_FACING": {"vlm_judge"},
    "FRAME_TRANSFORM": {"camera_pose", "geometry_checker"},
    "TARGET_RELATION": {"geometry_checker"},
}


ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}


PROOF_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_frame": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "forward": {"type": "string"},
                "unresolved": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["origin", "forward", "unresolved"],
            "additionalProperties": False,
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "views": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "integer"},
                    },
                },
                "required": ["name", "role", "views"],
                "additionalProperties": False,
            },
        },
        "proofs": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "option": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "claims": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "predicate": {"type": "string", "enum": PREDICATES},
                                "arguments": {"type": "array", "items": {"type": "string"}},
                                "frame": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                                "evidence_request": {
                                    "type": "object",
                                    "properties": {
                                        "verifier": {"type": "string", "enum": VERIFIERS},
                                        "views": {
                                            "type": "array", "minItems": 1,
                                            "items": {"type": "integer"},
                                        },
                                        "test": {"type": "string"},
                                    },
                                    "required": ["verifier", "views", "test"],
                                    "additionalProperties": False,
                                },
                                "status": {"type": "string", "enum": ["unknown"]},
                            },
                            "required": [
                                "id", "predicate", "arguments", "frame",
                                "depends_on", "evidence_request", "status"
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "conclusion": {"type": "string"},
                    "critical_claim_id": {"type": "string"},
                },
                "required": ["option", "conclusion", "claims", "critical_claim_id"],
                "additionalProperties": False,
            },
        },
        "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["reference_frame", "entities", "proofs", "answer", "confidence"],
    "additionalProperties": False,
}


DIRECT_SYSTEM = (
    "Solve multi-image spatial multiple-choice questions. Images are supplied in numbered "
    "order. Resolve the requested reference frame carefully. Return only the required JSON."
)

PROOF_SYSTEM = (
    "Propose falsifiable visual proof graphs for multi-image spatial questions. Build one proof "
    "for each answer option. Every atomic claim must name its coordinate frame and an independent "
    "visual or geometric check routed to one named verifier. Use only supplied view indices. "
    "Do not invent compass directions or a world axis unless the question explicitly defines one. "
    "SAME_ENTITY, VIEW_OVERLAP, CAMERA_TRANSLATION, CAMERA_ROTATION, and FRAME_TRANSFORM "
    "are cross-view claims and must request at least two distinct views. Route camera motion to "
    "camera_pose, overlap/identity to correspondence, depth order to depth, and final relation "
    "composition to geometry_checker. "
    "A model judgment is allowed only as a fallback and is not geometric proof. You are only "
    "proposing claims: their status must remain unknown until an external verifier runs. Identify "
    "the claim whose verification would best distinguish the options. Return only the required JSON."
)


def build_ollama_request(
    model: str,
    question: str,
    encoded_images: list[str],
    condition: str,
    seed: int = 7,
) -> dict[str, Any]:
    if condition not in {"direct", "proof"}:
        raise ValueError(f"Unknown condition: {condition}")
    proof = condition == "proof"
    user_prefix = (
        "Create four competing proof proposals, one matching each option's exact meaning, then "
        "select your provisional answer. In depends_on, write only IDs of earlier claims in the "
        "same proof (never image/view IDs). "
        if proof else
        "Select the best answer. "
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": PROOF_SYSTEM if proof else DIRECT_SYSTEM},
            {
                "role": "user",
                "content": (
                    user_prefix
                    + f"There are {len(encoded_images)} images indexed from 0 to "
                    + f"{len(encoded_images) - 1}. Do not assume image-right equals world-right.\n\n"
                    + question
                ),
                "images": encoded_images,
            },
        ],
        "format": PROOF_SCHEMA if proof else ANSWER_SCHEMA,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": 1800 if proof else 96,
        },
    }


def parse_structured_response(content: str) -> dict[str, Any]:
    payload = json.loads(content)
    answer = payload.get("answer")
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError("Response does not contain a valid answer")
    return payload


def validate_proof_payload(payload: dict[str, Any], image_count: int) -> list[str]:
    """Return semantic errors that JSON Schema alone cannot express."""
    errors: list[str] = []
    proofs = payload.get("proofs", [])
    options = [proof.get("option") for proof in proofs]
    if sorted(options) != ["A", "B", "C", "D"]:
        errors.append("proof options must contain A, B, C, and D exactly once")
    for entity in payload.get("entities", []):
        if any(
            not isinstance(view, int) or view < 0 or view >= image_count
            for view in entity.get("views", [])
        ):
            errors.append(f"entity {entity.get('name', '?')} references an invalid view")
    for proof in proofs:
        option = proof.get("option", "?")
        claims = proof.get("claims", [])
        ids = [claim.get("id") for claim in claims]
        if not claims:
            errors.append(f"option {option} has no claims")
        if len(ids) != len(set(ids)):
            errors.append(f"option {option} has duplicate claim ids")
        critical = proof.get("critical_claim_id")
        if critical not in ids:
            errors.append(f"option {option} critical claim is missing")
        seen: set[str] = set()
        for claim in claims:
            claim_id = claim.get("id", "?")
            for dependency in claim.get("depends_on", []):
                if dependency not in seen:
                    errors.append(f"claim {claim_id} has missing or forward dependency {dependency}")
            seen.add(claim_id)
            views = claim.get("evidence_request", {}).get("views", [])
            verifier = claim.get("evidence_request", {}).get("verifier")
            predicate = claim.get("predicate")
            if any(not isinstance(view, int) or view < 0 or view >= image_count for view in views):
                errors.append(f"claim {claim_id} references an invalid view")
            if predicate in {
                "SAME_ENTITY", "VIEW_OVERLAP", "CAMERA_TRANSLATION",
                "CAMERA_ROTATION", "FRAME_TRANSFORM",
            } and len(set(views)) < 2:
                errors.append(f"claim {claim_id} requires at least two distinct views")
            allowed = VERIFIER_BY_PREDICATE.get(predicate, set())
            if verifier not in allowed:
                errors.append(
                    f"claim {claim_id} routes {predicate} to incompatible verifier {verifier}"
                )
    return errors


def stratified_sample(
    records: list[dict[str, Any]],
    limit: int,
    categories: set[str] | None = None,
    salt: str = "fvpg-vlm-v1",
) -> list[dict[str, Any]]:
    eligible = [
        record for record in records
        if (categories is None or record["category"] in categories)
        and record.get("images")
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in eligible:
        groups.setdefault((record["category"], record["difficulty"]), []).append(record)
    for key in groups:
        groups[key].sort(key=lambda record: hashlib.sha256(
            f"{salt}:{record['id']}".encode()
        ).hexdigest())
    selected: list[dict[str, Any]] = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for key in sorted(groups):
            if index < len(groups[key]):
                selected.append(groups[key][index])
                if len(selected) == limit:
                    return selected
    return selected

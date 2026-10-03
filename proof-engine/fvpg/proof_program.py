from __future__ import annotations

from copy import deepcopy
from typing import Any


PROOF_COMPILER_VERSION = "proof-compiler-v1"

CANONICAL_VERIFIER = {
    "SAME_ENTITY": "correspondence",
    "VIEW_OVERLAP": "correspondence",
    "RELATIVE_IMAGE_POSITION": "grounding",
    "DEPTH_ORDER": "depth",
    "CAMERA_TRANSLATION": "camera_pose",
    "CAMERA_ROTATION": "camera_pose",
    "OBJECT_FACING": "vlm_judge",
    "FRAME_TRANSFORM": "camera_pose",
    "TARGET_RELATION": "geometry_checker",
}

CROSS_VIEW_PREDICATES = {
    "SAME_ENTITY",
    "VIEW_OVERLAP",
    "CAMERA_TRANSLATION",
    "CAMERA_ROTATION",
    "FRAME_TRANSFORM",
}


def compile_proof_program(
    payload: dict[str, Any], image_count: int
) -> tuple[dict[str, Any], list[str]]:
    """Compile untrusted VLM scheduling hints into deterministic tool contracts.

    This may repair execution metadata (tool routing, views, dependencies), but
    never changes predicates, arguments, conclusions, answer options, or the
    provisional answer. Every repair is returned for audit.
    """
    if image_count < 1:
        raise ValueError("image_count must be positive")
    compiled = deepcopy(payload)
    repairs: list[str] = []
    all_views = list(range(image_count))

    for proof in compiled.get("proofs", []):
        seen: set[str] = set()
        option = proof.get("option", "?")
        for claim in proof.get("claims", []):
            claim_id = claim.get("id", "?")
            prefix = f"option {option} claim {claim_id}"
            request = claim.setdefault("evidence_request", {})
            predicate = claim.get("predicate")

            canonical = CANONICAL_VERIFIER.get(predicate)
            if canonical is not None and request.get("verifier") != canonical:
                repairs.append(
                    f"{prefix}: verifier {request.get('verifier')} -> {canonical}"
                )
                request["verifier"] = canonical

            original_views = request.get("views", [])
            valid_views = sorted({
                view for view in original_views
                if isinstance(view, int) and 0 <= view < image_count
            })
            desired_views = all_views if predicate in CROSS_VIEW_PREDICATES else valid_views
            if not desired_views:
                desired_views = [0]
            if original_views != desired_views:
                repairs.append(f"{prefix}: views {original_views} -> {desired_views}")
                request["views"] = desired_views

            original_dependencies = claim.get("depends_on", [])
            valid_dependencies = [
                dependency for dependency in original_dependencies
                if dependency in seen
            ]
            if original_dependencies != valid_dependencies:
                repairs.append(
                    f"{prefix}: dependencies {original_dependencies} -> {valid_dependencies}"
                )
                claim["depends_on"] = valid_dependencies
            seen.add(claim_id)

    return compiled, repairs

from __future__ import annotations

from typing import Any

from .models import Evidence, Status


def one_sided_support_evidence(
    *,
    evidence_id: str,
    verifier: str,
    score: float,
    support_threshold: float,
    calibrated_precision: float,
    provenance: dict[str, Any],
    reliability_group: str,
) -> Evidence:
    """Convert a high-precision detector into support-or-unknown evidence.

    A failed correspondence or grounding test is not a counterexample: lack of
    detected evidence can be caused by occlusion, texture, or model failure.
    """
    if not 0 <= calibrated_precision <= 1:
        raise ValueError("calibrated_precision must be in [0, 1]")
    full_provenance = {
        **provenance,
        "raw_score": score,
        "support_threshold": support_threshold,
        "calibrated_precision": calibrated_precision,
        "evidence_policy": "one-sided-support-v1",
    }
    if score >= support_threshold:
        return Evidence(
            id=evidence_id,
            verifier=verifier,
            status=Status.SUPPORTED,
            support_probability=calibrated_precision,
            contradiction_probability=0.0,
            unknown_probability=1.0 - calibrated_precision,
            provenance=full_provenance,
            reliability_group=reliability_group,
        )
    return Evidence(
        id=evidence_id,
        verifier=verifier,
        status=Status.UNKNOWN,
        support_probability=0.0,
        contradiction_probability=0.0,
        unknown_probability=1.0,
        provenance=full_provenance,
        reliability_group=reliability_group,
    )

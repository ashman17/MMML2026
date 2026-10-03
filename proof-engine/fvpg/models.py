from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class Frame(str, Enum):
    IMAGE = "image"
    CAMERA = "camera"
    OBJECT = "object"
    GLOBAL = "global"
    GEOGRAPHIC = "geographic"


@dataclass(frozen=True)
class EvidenceContract:
    verifier: str
    accepted_predicates: tuple[str, ...]
    required_inputs: tuple[str, ...]
    output_type: str
    failure_modes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    id: str
    verifier: str
    status: Status
    support_probability: float
    contradiction_probability: float
    unknown_probability: float
    provenance: dict[str, Any]
    residual: float | None = None
    reliability_group: str | None = None

    def __post_init__(self) -> None:
        probabilities = (
            self.support_probability,
            self.contradiction_probability,
            self.unknown_probability,
        )
        if any(p < 0.0 or p > 1.0 for p in probabilities):
            raise ValueError("Evidence probabilities must be in [0, 1]")
        if abs(sum(probabilities) - 1.0) > 1e-6:
            raise ValueError("Evidence probabilities must sum to one")
        if not self.provenance:
            raise ValueError("Evidence requires provenance")


@dataclass
class Claim:
    id: str
    predicate: str
    arguments: tuple[str, ...]
    frame: Frame
    views: tuple[str, ...] = ()
    tolerance: float | None = None
    status: Status = Status.UNKNOWN
    evidence_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Derivation:
    id: str
    rule: str
    premises: tuple[str, ...]
    conclusion: str


@dataclass
class ProofGraph:
    claims: dict[str, Claim] = field(default_factory=dict)
    evidence: dict[str, Evidence] = field(default_factory=dict)
    derivations: list[Derivation] = field(default_factory=list)

    def add_claim(self, claim: Claim) -> None:
        if claim.id in self.claims:
            raise ValueError(f"Duplicate claim id: {claim.id}")
        self.claims[claim.id] = claim

    def add_evidence(self, evidence: Evidence, claim_id: str) -> None:
        if evidence.id in self.evidence:
            raise ValueError(f"Duplicate evidence id: {evidence.id}")
        if claim_id not in self.claims:
            raise KeyError(f"Unknown claim: {claim_id}")
        self.evidence[evidence.id] = evidence
        self.claims[claim_id].evidence_ids.append(evidence.id)

    def add_derivation(self, derivation: Derivation) -> None:
        self.derivations.append(derivation)

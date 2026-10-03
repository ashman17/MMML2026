"""Falsifiable Visual Proof Graph primitives."""

from .models import (
    Claim,
    Derivation,
    Evidence,
    EvidenceContract,
    Frame,
    ProofGraph,
    Status,
)
from .checker import CheckResult, ProofChecker
from .compiler import AnswerHypothesis, CompiledQuestion, MMSIProofCompiler
from .correspondence import CorrespondenceMeasurement, measure_patch_correspondence
from .evaluation import (
    PairItem,
    binary_auc,
    classification_metrics,
    cohen_kappa,
    component_split,
    select_threshold,
)
from .vlm import (
    PROTOCOL_VERSION,
    build_ollama_request,
    parse_structured_response,
    stratified_sample,
    validate_proof_payload,
)
from .search import (
    ClaimTest,
    TestChoice,
    expected_information_gain,
    posterior_for_outcome,
    select_next_test,
)
from .adapters import one_sided_support_evidence
from .proof_program import PROOF_COMPILER_VERSION, compile_proof_program
from .camera_geometry import (
    RelativeCameraMotion,
    cardinal_after_yaw,
    inverse_yaw_consistent,
    relative_camera_motion,
)

__all__ = [
    "CheckResult",
    "AnswerHypothesis",
    "Claim",
    "Derivation",
    "Evidence",
    "EvidenceContract",
    "Frame",
    "CompiledQuestion",
    "CorrespondenceMeasurement",
    "MMSIProofCompiler",
    "PairItem",
    "ProofChecker",
    "ProofGraph",
    "Status",
    "measure_patch_correspondence",
    "binary_auc",
    "classification_metrics",
    "cohen_kappa",
    "component_split",
    "select_threshold",
    "build_ollama_request",
    "parse_structured_response",
    "stratified_sample",
    "validate_proof_payload",
    "PROTOCOL_VERSION",
    "ClaimTest",
    "TestChoice",
    "expected_information_gain",
    "posterior_for_outcome",
    "select_next_test",
    "one_sided_support_evidence",
    "PROOF_COMPILER_VERSION",
    "compile_proof_program",
    "RelativeCameraMotion",
    "cardinal_after_yaw",
    "inverse_yaw_consistent",
    "relative_camera_motion",
]

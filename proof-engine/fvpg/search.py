from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ClaimTest:
    """A calibrated atomic test and the options it separates."""

    id: str
    supports_options: frozenset[str]
    contradicts_options: frozenset[str]
    true_positive_rate: float
    false_positive_rate: float
    unknown_probability: float
    cost: float
    reliability_group: str

    def __post_init__(self) -> None:
        for name, value in (
            ("true_positive_rate", self.true_positive_rate),
            ("false_positive_rate", self.false_positive_rate),
            ("unknown_probability", self.unknown_probability),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.cost <= 0:
            raise ValueError("cost must be positive")
        if self.supports_options & self.contradicts_options:
            raise ValueError("an option cannot be both supported and contradicted")


@dataclass(frozen=True)
class TestChoice:
    test_id: str
    information_gain: float
    utility: float
    targets_leader: bool


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        raise ValueError("option weights must contain positive mass")
    return {key: max(value, 0.0) / total for key, value in weights.items()}


def _entropy(probabilities: Mapping[str, float]) -> float:
    return -sum(value * math.log2(value) for value in probabilities.values() if value > 0)


def _likelihood(test: ClaimTest, option: str, outcome: str) -> float:
    unknown = test.unknown_probability
    if outcome == "unknown":
        return unknown
    if option in test.supports_options:
        positive = test.true_positive_rate
    elif option in test.contradicts_options:
        positive = test.false_positive_rate
    else:
        positive = 0.5
    return (1 - unknown) * (positive if outcome == "supported" else 1 - positive)


def posterior_for_outcome(
    priors: Mapping[str, float], test: ClaimTest, outcome: str
) -> dict[str, float]:
    if outcome not in {"supported", "contradicted", "unknown"}:
        raise ValueError(f"unknown outcome: {outcome}")
    normalized = _normalize(priors)
    weighted = {
        option: probability * _likelihood(test, option, outcome)
        for option, probability in normalized.items()
    }
    if sum(weighted.values()) <= 0:
        return normalized
    return _normalize(weighted)


def expected_information_gain(priors: Mapping[str, float], test: ClaimTest) -> float:
    priors = _normalize(priors)
    expected_entropy = 0.0
    for outcome in ("supported", "contradicted", "unknown"):
        outcome_probability = sum(
            prior * _likelihood(test, option, outcome)
            for option, prior in priors.items()
        )
        if outcome_probability > 0:
            expected_entropy += outcome_probability * _entropy(
                posterior_for_outcome(priors, test, outcome)
            )
    return max(0.0, _entropy(priors) - expected_entropy)


def select_next_test(
    priors: Mapping[str, float],
    tests: Sequence[ClaimTest],
    used_test_ids: set[str] | None = None,
    used_reliability_groups: set[str] | None = None,
    counterproof_bonus: float = 0.25,
) -> TestChoice | None:
    """Select a cheap discriminative test, favoring attacks on the current leader.

    Reusing a reliability group is disallowed so correlated evidence is not
    mistakenly counted as independent confirmation.
    """
    priors = _normalize(priors)
    leader = max(priors, key=priors.get)
    used_test_ids = used_test_ids or set()
    used_reliability_groups = used_reliability_groups or set()
    choices: list[TestChoice] = []
    for test in tests:
        if test.id in used_test_ids or test.reliability_group in used_reliability_groups:
            continue
        gain = expected_information_gain(priors, test)
        targets_leader = leader in test.supports_options or leader in test.contradicts_options
        utility = gain / test.cost
        if targets_leader:
            utility *= 1 + counterproof_bonus
        choices.append(TestChoice(test.id, gain, utility, targets_leader))
    return max(choices, key=lambda choice: (choice.utility, choice.information_gain)) if choices else None

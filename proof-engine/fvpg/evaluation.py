from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label lists must have equal length")
    if not labels_a:
        return None
    classes = sorted(set(labels_a) | set(labels_b))
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    expected = sum(
        (labels_a.count(label) / len(labels_a)) * (labels_b.count(label) / len(labels_b))
        for label in classes
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        return None
    differences = positive[:, None] - negative[None, :]
    return float(
        ((differences > 0).sum() + 0.5 * (differences == 0).sum())
        / differences.size
    )


def select_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    maximum_false_positive_rate: float = 0.05,
) -> float:
    """Choose the highest-recall threshold satisfying train-split FPR."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not (~labels).any():
        raise ValueError("Threshold selection requires negative examples")
    candidates = np.r_[np.inf, np.unique(scores)[::-1], -np.inf]
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted = scores >= threshold
        false_positive_rate = float(predicted[~labels].mean())
        true_positive_rate = float(predicted[labels].mean()) if labels.any() else 0.0
        if false_positive_rate <= maximum_false_positive_rate:
            feasible.append((true_positive_rate, -false_positive_rate, threshold))
    return float(max(feasible)[2])


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    labels = np.asarray(labels, dtype=bool)
    predicted = np.asarray(scores, dtype=float) >= threshold
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    tn = int((~predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "true_positive_rate": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "accuracy": (tp + tn) / len(labels) if len(labels) else None,
        "auc": binary_auc(labels, np.asarray(scores, dtype=float)),
    }


@dataclass(frozen=True)
class PairItem:
    key: str
    source_question_id: int
    target_question_id: int


def component_split(
    items: list[PairItem],
    train_fraction: float = 0.6,
    salt: str = "fvpg-calibration-v1",
) -> dict[str, str]:
    """Split connected question-ID components to prevent image leakage."""
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for item in items:
        union(item.source_question_id, item.target_question_id)
    assignment: dict[int, str] = {}
    for value in parent:
        root = find(value)
        digest = hashlib.sha256(f"{salt}:{root}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        assignment[root] = "calibration" if fraction < train_fraction else "evaluation"
    return {item.key: assignment[find(item.source_question_id)] for item in items}

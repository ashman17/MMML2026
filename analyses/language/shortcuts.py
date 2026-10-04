"""Reusable helpers for Step D answer-choice shortcut analysis."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from common import LETTERS
from lexicon import ABSTAIN, ANGLE, COMPASS, EGO, NUMERIC_OPTION, ORDER_OPTION, SOMETIMES

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SINGLE_ORDER_OPTION = re.compile(
    r"^(?:first|second|third|fourth)(?:\s+(?:image|picture|photo|one))?$",
    re.IGNORECASE,
)


def tokens(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def ngrams(text: str) -> list[str]:
    words = tokens(text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


def option_family(text: str) -> str:
    """Coarse answer format used only for the format-outlier elimination rule."""
    text = str(text).strip()
    if SOMETIMES.search(text):
        return "sometimes"
    if ABSTAIN.search(text):
        return "abstain"
    if NUMERIC_OPTION.search(text):
        return "number"
    if ORDER_OPTION.search(text) or SINGLE_ORDER_OPTION.search(text):
        return "order"
    if COMPASS.search(text):
        return "compass"
    if ANGLE.search(text):
        return "angle"
    if EGO.search(text):
        return "ego"
    if len(tokens(text)) >= 5:
        return "statement"
    return "entity"


def candidate_eliminations(opts: dict[str, str]) -> dict[str, set[str]]:
    """Letters eliminated by each pre-stated transparent candidate rule."""
    families = {letter: option_family(text) for letter, text in opts.items()}
    counts = Counter(families.values())
    modal, n_modal = counts.most_common(1)[0]
    outlier = {letter for letter, family in families.items() if family != modal} if n_modal == 3 else set()
    return {
        "sometimes": {letter for letter, text in opts.items() if SOMETIMES.search(text)},
        "abstain": {letter for letter, text in opts.items() if ABSTAIN.search(text)},
        "format_outlier": outlier,
    }


def tied_policy(values: dict[str, float], maximize: bool = True) -> np.ndarray:
    """Probability vector for random tie-breaking among extrema."""
    array = np.array([values[letter] for letter in LETTERS], dtype=float)
    target = np.nanmax(array) if maximize else np.nanmin(array)
    mask = np.isclose(array, target)
    return mask / mask.sum()


def heuristic_policies(stem: str, opts: dict[str, str]) -> dict[str, np.ndarray]:
    stem_set = set(tokens(stem))
    lengths = {letter: len(tokens(text)) for letter, text in opts.items()}
    overlaps = {}
    for letter, text in opts.items():
        option_set = set(tokens(text))
        union = stem_set | option_set
        overlaps[letter] = len(stem_set & option_set) / len(union) if union else 0.0
    return {
        "longest_option": tied_policy(lengths),
        "shortest_option": tied_policy(lengths, maximize=False),
        "greatest_overlap": tied_policy(overlaps),
    }


def elimination_policy(opts: dict[str, str], selected_rules: set[str]) -> np.ndarray:
    candidates = candidate_eliminations(opts)
    eliminated = set().union(*(candidates[name] for name in selected_rules)) if selected_rules else set()
    remaining = [letter for letter in LETTERS if letter not in eliminated]
    if not remaining:  # Defensive fallback; no current candidate can eliminate all four.
        remaining = list(LETTERS)
    policy = np.zeros(len(LETTERS), dtype=float)
    for letter in remaining:
        policy[LETTERS.index(letter)] = 1 / len(remaining)
    return policy


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm family-wise adjustment, preserving the input keys."""
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (m - rank) * p_values[key])
        adjusted[key] = min(1.0, running)
    return {key: adjusted[key] for key in p_values}


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            item, self.parent[item] = self.parent[item], root
        return root

    def union(self, left, right) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


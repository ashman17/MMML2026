"""Cue and landmark helpers for the reasoning-trace analysis.

Traces are a separate descriptive lens. Nothing here is used as a feature for
an answer-shortcut test.
"""

from __future__ import annotations

import re

from entities import IMAGE_NOUNS, SKIP_HEADS

DIFFICULTY_ORDINAL = {"easy": 0, "medium": 1, "hard": 2}
STEP_CUES = {
    "numbered_steps": re.compile(r"(?:^|\n)\s*1[\.\)]\s+.+(?:\n|\s)+2[\.\)]\s+", re.IGNORECASE),
    "sequence": re.compile(r"\b(?:then|in sequence|after that|next)\b", re.IGNORECASE),
    "conclusion": re.compile(r"\b(?:so|therefore|thus|hence)\b|\bthis (?:indicates|means|shows)\b",
                             re.IGNORECASE),
    "cross_view": re.compile(r"\b(?:figure|image|photo|picture|frame)\s*\d+\b|"
                             r"\bthe other (?:image|photo|picture|figure)\b", re.IGNORECASE),
    "perspective": re.compile(r"\b(?:relative to you|when you|you can see|your (?:left|right|front))\b",
                              re.IGNORECASE),
    "comparison": re.compile(r"\b(?:closer|farther|further|between|larger|smaller|higher|lower)\b",
                             re.IGNORECASE),
    "motion": re.compile(r"\b(?:moving|moves|moved|rotating|rotates|rotated)\b", re.IGNORECASE),
}


def cue_flags(text: str) -> dict[str, bool]:
    return {name: pattern.search(text) is not None for name, pattern in STEP_CUES.items()}


def content_lemmas(doc) -> set[str]:
    """Content-noun lemmas, excluding image references and spatial query words."""
    lemmas = set()
    for token in doc:
        lemma = token.lemma_.lower()
        if (token.pos_ in {"NOUN", "PROPN"} and lemma.isalpha() and lemma not in IMAGE_NOUNS
                and lemma not in SKIP_HEADS):
            lemmas.add(lemma)
    return lemmas


def bridging_lemmas(trace_doc, question_doc) -> set[str]:
    """Nouns introduced by the trace and absent from the stem and options."""
    return content_lemmas(trace_doc) - content_lemmas(question_doc)

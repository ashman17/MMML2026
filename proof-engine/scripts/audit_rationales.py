from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvpg.compiler import MMSIProofCompiler, OPTION_RE


# Ordered from specific to general. Overlapping cardinal matches inside a
# diagonal phrase are discarded.
MENTION_PATTERNS = (
    (re.compile(r"\b(?:front|upper)[ -]?left\b|\bnorthwest\b", re.I), "front_left_of"),
    (re.compile(r"\b(?:front|upper)[ -]?right\b|\bnortheast\b", re.I), "front_right_of"),
    (re.compile(r"\b(?:back|rear|lower)[ -]?left\b|\bsouthwest\b", re.I), "back_left_of"),
    (re.compile(r"\b(?:back|rear|lower)[ -]?right\b|\bsoutheast\b", re.I), "back_right_of"),
    (re.compile(r"\b(?:directly |to |on )?(?:the |your )?left\b|\bwest(?:ern)?\b", re.I), "left_of"),
    (re.compile(r"\b(?:directly |to |on )?(?:the |your )?right\b|\beast(?:ern)?\b", re.I), "right_of"),
    (re.compile(r"\b(?:directly )?(?:in )?front\b|\bforward\b|\bahead\b|\bnorth\b", re.I), "front_of"),
    (re.compile(r"\b(?:directly )?(?:behind|back|rear)\b|\bbackward\b|\bsouth\b", re.I), "behind"),
)


def conclusion_clause(thought: str) -> str:
    clauses = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", thought) if part.strip()]
    if not clauses:
        return thought
    marked: list[str] = []
    marker = re.compile(r"\b(?:therefore|thus|hence|so|correct answer|conclusion)\b", re.I)
    for clause in clauses:
        matches = list(marker.finditer(clause))
        if matches:
            # Earlier parts of the sentence often describe an intermediate
            # direction; audit only the explicit concluding fragment.
            marked.append(clause[matches[-1].start():].strip())
    return marked[-1] if marked else clauses[-1]


def relation_mentions(text: str) -> set[str]:
    if re.search(r"\b(?:cannot|unable|impossible|not possible)\b.{0,35}\bdetermin", text, re.I):
        return {"undetermined"}
    occupied: list[tuple[int, int]] = []
    found: set[str] = set()
    for pattern, predicate in MENTION_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            found.add(predicate)
    combinations = {
        frozenset(("front_of", "left_of")): "front_left_of",
        frozenset(("front_of", "right_of")): "front_right_of",
        frozenset(("behind", "left_of")): "back_left_of",
        frozenset(("behind", "right_of")): "back_right_of",
    }
    for components, combined in combinations.items():
        if components.issubset(found):
            found.difference_update(components)
            found.add(combined)
    return found


def parse_options(question: str) -> dict[str, str]:
    if "Options:" not in question:
        return {}
    return {
        label: text.strip().rstrip(".")
        for label, text in OPTION_RE.findall(question.split("Options:", 1)[1].strip())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    flagged: list[dict[str, object]] = []
    auditable = 0
    for record in records:
        if "Positional Relationship" not in record["category"]:
            continue
        options = parse_options(record["question"])
        answer_text = options.get(str(record["answer"]).strip())
        answer_predicate = MMSIProofCompiler.normalize_relation(answer_text or "")
        if answer_predicate is None:
            continue
        clause = conclusion_clause(str(record.get("thought") or ""))
        mentions = relation_mentions(clause)
        if len(mentions) != 1:
            continue
        auditable += 1
        rationale_predicate = next(iter(mentions))
        if rationale_predicate != answer_predicate:
            flagged.append({
                "id": record["id"],
                "category": record["category"],
                "difficulty": record["difficulty"],
                "answer": record["answer"],
                "answer_text": answer_text,
                "answer_predicate": answer_predicate,
                "rationale_predicate": rationale_predicate,
                "conclusion_clause": clause,
                "question": record["question"],
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(flagged[0]) if flagged else ["id"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flagged)
    print(json.dumps({
        "position_questions": sum("Positional Relationship" in r["category"] for r in records),
        "heuristically_auditable": auditable,
        "candidate_inconsistencies": len(flagged),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()

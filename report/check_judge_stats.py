"""Fisher exact tests behind Table 2 of Research Idea 2: are judge-flagged answers less often correct?

Run from the repo root: python report/check_judge_stats.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import workflow

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
CATEGORIES = ("overall", "Motion (Cam.)", "Positional Relationship (Cam.–Cam.)", "Positional Relationship (Cam.–Obj.)")


def fisher(a, b, c, d):
    """Two-sided Fisher exact p-value for the 2x2 table [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    prob = lambda x: math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    observed = prob(a)
    return min(1.0, sum(prob(x) for x in range(max(0, c1 - (n - r1)), min(r1, c1) + 1) if prob(x) <= observed * (1 + 1e-9)))


items = {x["id"]: x for x in workflow.read_jsonl(workflow.DATA / "items.jsonl")}
keys = {x["id"]: str(x["answer"]).strip().upper() for x in workflow.read_jsonl(workflow.DATA / "keys.jsonl")}
for condition in ("images", "images_vggt"):
    answers = {x["id"]: x["parsed_answer"] for x in workflow.read_jsonl(workflow.run_path("compatible", MODEL, condition))}
    verdicts = {x["id"]: x["inconsistent"] for x in workflow.read_jsonl(workflow.run_path("compatible", MODEL, f"judge_{condition}"))}
    print(f"\nTraces judged: {condition}")
    for category in CATEGORIES:
        ids = [i for i in verdicts if category == "overall" or items[i]["question_type"] == category]
        flagged_ok = sum(verdicts[i] is True and answers[i] == keys[i] for i in ids)
        flagged_bad = sum(verdicts[i] is True and answers[i] != keys[i] for i in ids)
        clean_ok = sum(verdicts[i] is False and answers[i] == keys[i] for i in ids)
        clean_bad = sum(verdicts[i] is False and answers[i] != keys[i] for i in ids)
        print(f"  {category:<40} flagged {flagged_ok}/{flagged_ok + flagged_bad} correct, "
              f"not flagged {clean_ok}/{clean_ok + clean_bad} correct, "
              f"Fisher p = {fisher(flagged_ok, flagged_bad, clean_ok, clean_bad):.4f}")

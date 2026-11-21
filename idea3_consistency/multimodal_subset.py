"""Split MMSI questions by whether the text alone can answer them.

Uses the no-image letter scores in tgcd.py's scores.jsonl. A question is kept in the
multimodal subset when the no-image prediction is not confident (max probability below
--max_conf). This rule does not look at the gold answer, so it can be used as an
evaluation subset. The label-based rule (no-image prediction wrong) is also written,
but the image model's errors are correlated with the no-image errors, so accuracy on
that subset is biased downward and it is only meant for filtering training data.

    python multimodal_subset.py --scores results_dev/scores.jsonl results_test/scores.jsonl --out multimodal_subset
"""
import argparse
import collections
import json
import math
import os


def probs(scores):
    top = max(scores.values())
    exp = {k: math.exp(v - top) for k, v in scores.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def argmax(scores):
    return max(scores, key=scores.get)


def accuracy(records, pred):
    return 100.0 * sum(pred(r) == r["answer"] for r in records) / max(1, len(records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--max_conf", type=float, default=0.7)
    ap.add_argument("--out", default="multimodal_subset")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    orig = lambda r: argmax(r["variants"]["orig"]["scores"])
    noimg = lambda r: r["noimage"]["pred"]
    lines = ["# Multimodal subset of MMSI-Bench", "",
             f"Rule: keep a question when the no-image max probability < {args.max_conf:g}.", "",
             "| split | rule | n | orig acc | no-image acc | orig = no-image |",
             "|---|---|---:|---:|---:|---:|"]
    summary = {}
    for path in args.scores:
        split = os.path.basename(os.path.dirname(os.path.abspath(path)))
        records = [json.loads(line) for line in open(path)]
        subsets = {
            "all": records,
            "conf": [r for r in records if max(probs(r["noimage"]["scores"]).values()) < args.max_conf],
            "noimage_wrong": [r for r in records if noimg(r) != r["answer"]],
        }
        for rule, recs in subsets.items():
            same = 100.0 * sum(orig(r) == noimg(r) for r in recs) / max(1, len(recs))
            lines.append(f"| {split} | {rule} | {len(recs)} | {accuracy(recs, orig):.1f}% | "
                         f"{accuracy(recs, noimg):.1f}% | {same:.1f}% |")
            summary[f"{split}/{rule}"] = sorted(r["id"] for r in recs)

        by_type = collections.defaultdict(lambda: [0, 0])
        for r in records:
            by_type[r["question_type"]][0] += 1
            by_type[r["question_type"]][1] += r in subsets["conf"]
        lines += ["", f"### {split}: kept per question type", "| type | n | kept |", "|---|---:|---:|"]
        lines += [f"| {t} | {n} | {k} |" for t, (n, k) in sorted(by_type.items())]
        lines.append("")

    with open(os.path.join(args.out, "subset_ids.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

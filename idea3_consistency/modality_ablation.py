"""
Modality ablation for Idea 3: how much does the model answer without looking at the images?

Variants per question (NOT answer-preserving, so kept out of the consistency/voting metrics):
  orig       : as in mmsi_consistency.py
  noimage    : same text, all images (and their "Image k:" labels) removed
  randimage  : same text and labels, images taken from a different random question

If accuracy and answers barely change without the right images, the model relies on a
language prior. Joined with the baseline run (--baseline), we also test whether the mirror
"lexical anchoring" failures are the questions the model answers the same way with no images.

Usage:
  python modality_ablation.py --parquet MMSI_Bench.parquet --n 1000 \
      --exclude dev1/predictions.jsonl dev2/predictions.jsonl --out results_ablation
  python modality_ablation.py --analyze results_ablation/predictions.jsonl \
      --baseline results_test/predictions.jsonl
"""
import argparse, json, os, random, sys
from collections import Counter

from mmsi_consistency import (POST_PROMPT, LETTERS, MockBackend, VLLMBackend, build_messages,
                              extract, load_rows, mcnemar_exact, stratified_sample)

VARIANTS = ["orig", "noimage", "randimage"]
pct = lambda k, n: 100.0 * k / n if n else float("nan")


def ablation_messages(rows, seed, max_side):
    """Yield (row_idx, variant, conv). Donors for randimage come from the same sampled pool."""
    rng = random.Random(seed)
    for i, r in enumerate(rows):
        n_img = len(r["images"])
        base = dict(question=r["question"], order=list(range(n_img)), flip=False, note="")
        yield i, "orig", build_messages(r, base, max_side)
        yield i, "noimage", [{"role": "user", "content": [
            {"type": "text", "text": r["question"] + POST_PROMPT}]}]
        donors = [d for d in rows if d is not r and len(d["images"]) >= n_img]
        donor = rng.choice(donors)
        yield i, "randimage", build_messages(dict(r, images=donor["images"][:n_img]), base, max_side)


def run(args):
    rows = load_rows(args.parquet)
    if args.exclude:
        seen = {json.loads(l)["id"] for p in args.exclude for l in open(p)}
        rows = [r for r in rows if int(r["id"]) not in seen]
    rows = stratified_sample(rows, args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k not in ("analyze", "baseline")}, f, indent=2)
    backend = MockBackend(args.seed) if args.backend == "mock" else VLLMBackend(
        args.model, hf_overrides=json.loads(args.hf_overrides) if args.hf_overrides else None)
    jobs = list(ablation_messages(rows, args.seed, args.max_side))
    print(f"{len(rows)} questions, {len(jobs)} greedy prompts", file=sys.stderr)
    outs = backend.generate([c for _, _, c in jobs], n=1, temperature=0.0)
    recs = {}
    for (i, v, _), out in zip(jobs, outs):
        r = rows[i]
        rec = recs.setdefault(i, dict(id=int(r["id"]), question_type=r["question_type"],
                                      answer=r["answer"], n_images=len(r["images"]), variants={}))
        rec["variants"][v] = dict(raw=out[0], pred=extract(out[0]))
    path = os.path.join(args.out, "predictions.jsonl")
    with open(path, "w") as f:
        for i in sorted(recs):
            f.write(json.dumps(recs[i]) + "\n")
    print(f"wrote {path}", file=sys.stderr)
    analyze(path, args.baseline)


def analyze(path, baseline=None):
    recs = [json.loads(l) for l in open(path)]
    N = len(recs)
    pred = lambda r, v: r["variants"][v]["pred"]
    ok = {v: [pred(r, v) == r["answer"] for r in recs] for v in VARIANTS}
    lines = [f"# Modality ablation  (N = {N} questions, source `{path}`)\n",
             "| variant | accuracy | same answer as orig | unparsable |", "|---|---|---|---|"]
    for v in VARIANTS:
        same = sum(pred(r, v) == pred(r, "orig") for r in recs)
        lines.append(f"| {v} | {pct(sum(ok[v]), N):.1f}% | {pct(same, N):.1f}% | "
                     f"{sum(pred(r, v) is None for r in recs)} |")
    lines.append("\nChance accuracy with four options is about 25%.")
    for v in ["noimage", "randimage"]:
        a, b, p = mcnemar_exact(ok["orig"], ok[v])
        lines.append(f"- orig vs {v}: orig-only correct={a}, {v}-only correct={b}, exact McNemar p={p:.4f}")

    lines.append("\n## By question type")
    lines.append("| type | n | orig | noimage | randimage | noimage same as orig |")
    lines.append("|---|---|---|---|---|---|")
    for t in sorted({r["question_type"] for r in recs}):
        ix = [i for i, r in enumerate(recs) if r["question_type"] == t]
        same = sum(pred(recs[i], "noimage") == pred(recs[i], "orig") for i in ix)
        lines.append(f"| {t} | {len(ix)} | " + " | ".join(
            f"{pct(sum(ok[v][i] for i in ix), len(ix)):.1f}%" for v in VARIANTS) +
            f" | {pct(same, len(ix)):.1f}% |")

    if baseline:
        # Join with the consistency run of the same model on the same questions.
        from failure_analysis import category, mirror_anchoring
        base = {r["id"]: r for r in map(json.loads, open(baseline))}
        joined = [(r, base[r["id"]]) for r in recs if r["id"] in base]
        lines.append(f"\n## Joined with baseline run (`{baseline}`, {len(joined)} shared questions)")
        lines.append("Share of questions whose no-image answer equals the baseline original answer:")
        lines.append("| group | n | no-image answer = original answer |")
        lines.append("|---|---|---|")
        groups = Counter()
        hits = Counter()
        for r, b in joined:
            same = pred(r, "noimage") == b["variants"]["orig"]["pred"]
            for g in [f"mirror: {mirror_anchoring(b)}" if mirror_anchoring(b) else None,
                      f"taxonomy: {category(b)}", "all"]:
                if g:
                    groups[g] += 1; hits[g] += same
        for g in sorted(groups):
            lines.append(f"| {g} | {groups[g]} | {pct(hits[g], groups[g]):.1f}% |")
        lines.append("\nIf 'mirror: lexical' is well above 'mirror: same', the questions where the model "
                     "ignored the flip are the ones it can answer from text alone.")

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(os.path.dirname(path) or ".", "ablation_report.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet")
    ap.add_argument("--analyze", help="only analyze an existing predictions.jsonl")
    ap.add_argument("--baseline", help="predictions.jsonl of the consistency run on the same questions")
    ap.add_argument("--exclude", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--hf_overrides", help="model config overrides as JSON (see mmsi_consistency.py)")
    ap.add_argument("--out", default="results_ablation")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, a.baseline)
    else:
        run(a)

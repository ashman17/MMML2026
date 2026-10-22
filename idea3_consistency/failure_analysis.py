"""
Midterm failure analysis for Idea 3, computed offline from an existing predictions.jsonl
(no model calls).

  1. Transform-subset ablation: vote accuracy, AUROC and consistency for every subset of
     {mirror, reorder, shuffle}, against self-consistency with the same number of calls.
  2. Error taxonomy: original prediction correct/wrong x variants consistent/inconsistent.
     "Confident errors" (wrong but consistent) are the cases a consistency signal cannot catch.
  3. Flip outcomes per transform: correct->wrong, wrong->correct, wrong->other wrong.
  4. Mirror lexical anchoring: after mirroring, does the model follow the image (same letter)
     or the original word (the option whose swapped text equals its original answer text)?
  5. Shuffle position anchoring: after permuting options, does the model keep the same
     displayed letter instead of following the content?
  6. Reorder flip rate by number of images and by explicit image references.

Usage:
  python failure_analysis.py results_x/predictions.jsonl      # writes results_x/failure_analysis/
"""
import argparse, itertools, json, os, re, sys
from collections import Counter

from mmsi_consistency import (LETTERS, auroc, mcnemar_exact, parse_question, vote,
                              _write_csv)

TRANSFORMS = ["mirror", "reorder", "shuffle"]
IMAGE_REF = re.compile(r"\b(image|figure|picture|view|photo)\s*\d", re.IGNORECASE)
CATEGORIES = [("correct_consistent", "Correct, consistent"),
              ("correct_inconsistent", "Correct, inconsistent"),
              ("wrong_inconsistent", "Wrong, inconsistent (detectable)"),
              ("wrong_consistent", "Wrong, consistent (confident error)")]

pct = lambda k, n: 100.0 * k / n if n else float("nan")


def option_text(question, letter):
    parsed = parse_question(question)
    if not parsed or letter is None or LETTERS.find(letter) >= len(parsed[1]):
        return None
    return parsed[1][LETTERS.find(letter)].strip().lower()


def mirror_anchoring(rec):
    """Classify the mirror prediction as 'same' (same letter as orig, i.e. follows the image),
    'lexical' (picks the option whose mirrored text equals the original answer's text, i.e.
    follows the word and ignores the flip), 'other', or None when the test is not defined
    (no mirror variant, unparsable, or the original answer's text has no swapped twin)."""
    v = rec["variants"]
    if "mirror" not in v or v["orig"]["pred"] is None or v["mirror"]["pred"] is None:
        return None
    orig_txt = option_text(v["orig"]["question"], v["orig"]["pred"])
    parsed = parse_question(v["mirror"]["question"])
    if orig_txt is None or not parsed:
        return None
    twins = [LETTERS[i] for i, t in enumerate(parsed[1])
             if t.strip().lower() == orig_txt and LETTERS[i] != v["orig"]["pred"]]
    if not twins:
        return None
    p = v["mirror"]["pred"]
    return "same" if p == v["orig"]["pred"] else "lexical" if p in twins else "other"


def shuffle_anchoring(rec):
    """Classify the shuffle prediction as 'content' (same original option), 'position' (same
    displayed letter, different content), 'other', or None when undefined (unparsable, or the
    permutation left the original answer at the same position)."""
    v = rec["variants"]
    if "shuffle" not in v or v["orig"]["pred"] is None or v["shuffle"]["pred_raw"] is None:
        return None
    perm = v["shuffle"]["perm"]
    i = LETTERS.find(v["orig"]["pred"])
    if i >= len(perm) or perm[i] == i:
        return None
    if v["shuffle"]["pred"] == v["orig"]["pred"]:
        return "content"
    return "position" if v["shuffle"]["pred_raw"] == v["orig"]["pred"] else "other"


def category(rec):
    ok = rec["variants"]["orig"]["pred"] == rec["answer"]
    cons = len({x["pred"] for x in rec["variants"].values()}) == 1
    return ("correct" if ok else "wrong") + ("_consistent" if cons else "_inconsistent")


def ablation(recs):
    rows = []
    for k in range(1, len(TRANSFORMS) + 1):
        for subset in itertools.combinations(TRANSFORMS, k):
            tv, sc, dis, wrong, cons, calls = [], [], [], [], [], 0
            for r in recs:
                v = r["variants"]
                names = ["orig"] + [t for t in subset if t in v]
                preds = [v[n]["pred"] for n in names]
                orig = v["orig"]["pred"]
                tv.append(vote(preds, orig) == r["answer"])
                sc.append(vote(r["sc_preds"][:len(names)], orig) == r["answer"])
                dis.append(sum(p != orig for p in preds[1:]) / max(1, len(preds) - 1))
                wrong.append(orig != r["answer"])
                cons.append(len(set(preds)) == 1)
                calls += len(names)
            better_tv, better_sc, p = mcnemar_exact(tv, sc)
            rows.append(dict(transforms="+".join(subset), calls_per_question=calls / len(recs),
                             transform_vote_acc=pct(sum(tv), len(tv)),
                             self_consistency_acc=pct(sum(sc), len(sc)),
                             mcnemar_p=p, disagreement_auroc=auroc(dis, wrong),
                             consistency_rate=pct(sum(cons), len(cons))))
    return rows


def make_figure(outdir, recs, types):
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(outdir, ".mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[failure] figure skipped: {e}", file=sys.stderr)
        return None
    groups = ["All"] + types
    shares = {c: [] for c, _ in CATEGORIES}
    for g in groups:
        sub = recs if g == "All" else [r for r in recs if r["question_type"] == g]
        cnt = Counter(category(r) for r in sub)
        for c, _ in CATEGORIES:
            shares[c].append(pct(cnt[c], len(sub)))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    bottom = [0.0] * len(groups)
    for (c, label), color in zip(CATEGORIES, ["#54A24B", "#A0CBE8", "#F58518", "#E45756"]):
        ax.bar(groups, shares[c], bottom=bottom, label=label, color=color)
        bottom = [b + s for b, s in zip(bottom, shares[c])]
    ax.set_xticks(range(len(groups)), [g.replace("Positional Relationship", "PR") for g in groups],
                  rotation=35, ha="right")
    ax.set_ylabel("Share of questions (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_title("Error taxonomy: original correctness x transformation consistency")
    fig.tight_layout()
    p = os.path.join(outdir, "fig_error_taxonomy.png")
    fig.savefig(p, dpi=180); plt.close(fig)
    return p


def main(path):
    recs = [json.loads(l) for l in open(path)]
    outdir = os.path.join(os.path.dirname(path) or ".", "failure_analysis")
    os.makedirs(outdir, exist_ok=True)
    N = len(recs)
    types = sorted({r["question_type"] for r in recs})
    lines = [f"# Idea 3 failure analysis  (N = {N} questions, source `{path}`)\n"]

    # 1. ablation
    abl = ablation(recs)
    lines.append("## 1. Transform-subset ablation (equal-compute self-consistency)")
    lines.append("| transforms | calls/q | transform vote | self-consistency | McNemar p | AUROC | consistency |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in abl:
        lines.append(f"| {r['transforms']} | {r['calls_per_question']:.2f} | "
                     f"{r['transform_vote_acc']:.1f}% | {r['self_consistency_acc']:.1f}% | "
                     f"{r['mcnemar_p']:.3f} | {r['disagreement_auroc']:.3f} | "
                     f"{r['consistency_rate']:.1f}% |")
    _write_csv(os.path.join(outdir, "ablation_transform_subsets.csv"), list(abl[0]), abl)

    # 2. taxonomy
    cats = [category(r) for r in recs]
    tax_rows = []
    lines.append("\n## 2. Error taxonomy")
    lines.append("| question type | n | " + " | ".join(l for _, l in CATEGORIES) + " |")
    lines.append("|---|---|" + "---|" * len(CATEGORIES))
    for g in ["All"] + types:
        ix = [i for i, r in enumerate(recs) if g == "All" or r["question_type"] == g]
        cnt = Counter(cats[i] for i in ix)
        lines.append(f"| {g} | {len(ix)} | " +
                     " | ".join(f"{pct(cnt[c], len(ix)):.1f}%" for c, _ in CATEGORIES) + " |")
        tax_rows.append(dict(question_type=g, n=len(ix),
                             **{c: pct(cnt[c], len(ix)) for c, _ in CATEGORIES}))
    _write_csv(os.path.join(outdir, "error_taxonomy.csv"), list(tax_rows[0]), tax_rows)
    n_wrong = sum(c.startswith("wrong") for c in cats)
    lines.append(f"\nOf {n_wrong} wrong original answers, {cats.count('wrong_consistent')} "
                 f"({pct(cats.count('wrong_consistent'), n_wrong):.1f}%) are confident errors "
                 "that no transformation exposes.")

    # 3. flip outcomes
    flip_rows = []
    lines.append("\n## 3. Flip outcomes per transform")
    lines.append("| transform | n | flips | correct->wrong | wrong->correct | wrong->other wrong |")
    lines.append("|---|---|---|---|---|---|")
    for t in TRANSFORMS:
        sub = [r for r in recs if t in r["variants"]]
        flips = [r for r in sub if r["variants"][t]["pred"] != r["variants"]["orig"]["pred"]]
        c2w = sum(r["variants"]["orig"]["pred"] == r["answer"] for r in flips)
        w2c = sum(r["variants"][t]["pred"] == r["answer"] for r in flips)
        w2w = len(flips) - c2w - w2c
        lines.append(f"| {t} | {len(sub)} | {len(flips)} | {c2w} | {w2c} | {w2w} |")
        flip_rows.append(dict(transform=t, n=len(sub), flips=len(flips), correct_to_wrong=c2w,
                              wrong_to_correct=w2c, wrong_to_other_wrong=w2w))
    _write_csv(os.path.join(outdir, "flip_outcomes.csv"), list(flip_rows[0]), flip_rows)

    # 4 + 5. anchoring
    mir = Counter(mirror_anchoring(r) for r in recs)
    shu = Counter(shuffle_anchoring(r) for r in recs)
    m_n = sum(v for k, v in mir.items() if k)
    s_n = sum(v for k, v in shu.items() if k)
    lines.append("\n## 4. Mirror: image-following vs lexical anchoring")
    lines.append("Defined on questions whose original answer text has a left/right-swapped twin "
                 f"among the mirrored options (n={m_n}).")
    lines.append(f"- same letter (follows the flipped image): {mir['same']} ({pct(mir['same'], m_n):.1f}%)")
    lines.append(f"- swapped twin (keeps the original word, ignores the flip): {mir['lexical']} "
                 f"({pct(mir['lexical'], m_n):.1f}%)")
    lines.append(f"- other option: {mir['other']} ({pct(mir['other'], m_n):.1f}%)")
    lines.append(f"- among changed answers, share going to the swapped twin: "
                 f"{pct(mir['lexical'], mir['lexical'] + mir['other']):.1f}% (about 33% if random)")
    lines.append("\n## 5. Shuffle: content vs position anchoring")
    lines.append(f"Defined when the permutation moved the original answer (n={s_n}).")
    lines.append(f"- same content (consistent): {shu['content']} ({pct(shu['content'], s_n):.1f}%)")
    lines.append(f"- same displayed letter, different content: {shu['position']} "
                 f"({pct(shu['position'], s_n):.1f}%)")
    lines.append(f"- other: {shu['other']} ({pct(shu['other'], s_n):.1f}%)")
    lines.append(f"- among changed answers, share keeping the displayed letter: "
                 f"{pct(shu['position'], shu['position'] + shu['other']):.1f}% (about 33% if random)")

    # 6. reorder
    lines.append("\n## 6. Reorder flip rate by image count and explicit image references")
    lines.append("| subset | n | flip rate |")
    lines.append("|---|---|---|")
    sub = [r for r in recs if "reorder" in r["variants"]]
    flip = lambda r: r["variants"]["reorder"]["pred"] != r["variants"]["orig"]["pred"]
    for label, keep in [("2 images", lambda r: r["n_images"] == 2),
                        ("3+ images", lambda r: r["n_images"] >= 3),
                        ("mentions 'Image/Figure k'", lambda r: bool(IMAGE_REF.search(r["variants"]["orig"]["question"]))),
                        ("no explicit image index", lambda r: not IMAGE_REF.search(r["variants"]["orig"]["question"]))]:
        s = [r for r in sub if keep(r)]
        lines.append(f"| {label} | {len(s)} | {pct(sum(map(flip, s)), len(s)):.1f}% |")

    # case list for picking qualitative examples
    cases = []
    for r, c in zip(recs, cats):
        v = r["variants"]
        cases.append(dict(id=r["id"], question_type=r["question_type"], gold=r["answer"],
                          category=c, mirror_anchoring=mirror_anchoring(r) or "",
                          shuffle_anchoring=shuffle_anchoring(r) or "",
                          **{f"pred_{t}": v[t]["pred"] if t in v else "" for t in ["orig"] + TRANSFORMS},
                          sc_preds="".join(p or "?" for p in r["sc_preds"]),
                          question=" ".join(v["orig"]["question"].split())))
    _write_csv(os.path.join(outdir, "cases.csv"), list(cases[0]), cases)

    fig = make_figure(outdir, recs, types)
    lines.append(f"\nFigure: {os.path.basename(fig) if fig else 'skipped'}. "
                 "Per-question tags for picking qualitative examples: `cases.csv`.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(outdir, "failure_report.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="predictions.jsonl from mmsi_consistency.py")
    main(ap.parse_args().predictions)

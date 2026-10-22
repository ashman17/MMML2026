"""
Image-side probes for Idea 3, kept out of the baseline voting/consistency metrics.

Variants per question:
  orig   : as in mmsi_consistency.py
  jpeg   : noise floor. Every image resized to 97% and re-encoded; content is unchanged, so its
           flip rate measures how unstable the model is under a perturbation that means nothing.
  split  : every image cut into two overlapping crops (left 0-60%, right 40-100% of the width),
           shown as "Image k (left part):" / "Image k (right part):". Tests whether the model can
           re-associate overlapping views it is told come from one image. Skipped for counting
           questions (objects in the overlap appear twice) and camera-motion questions (crops
           narrow the field of view the motion is judged from).

A transform only "confuses" the model if its flip rate is clearly above the jpeg noise floor,
so the report compares the two on the same questions with an exact McNemar test.

Usage:
  python image_probes.py --parquet MMSI_Bench.parquet --n 1000 \
      --exclude dev1/predictions.jsonl dev2/predictions.jsonl --out results_probes
  python image_probes.py --analyze results_probes/predictions.jsonl
"""
import argparse, io, json, os, random, re, sys

from mmsi_consistency import (POST_PROMPT, MockBackend, VLLMBackend, bootstrap_ci, build_messages,
                              extract, img_to_url, load_rows, mcnemar_exact, stratified_sample)

VARIANTS = ["orig", "jpeg", "split"]
OVERLAP = 0.2  # crops are [0, (1+OVERLAP)/2] and [(1-OVERLAP)/2, 1] of the width
SPLIT_BLOCK = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
SPLIT_NOTE = ("Note: each original image is shown as two overlapping parts, its left part and "
              "its right part.\n")
pct = lambda k, n: 100.0 * k / n if n else float("nan")


def _encode(im, quality=90):
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def jpeg_noise(b):
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    im = im.resize((max(1, round(im.width * .97)), max(1, round(im.height * .97))))
    return _encode(im, quality=80)


def split_image(b, overlap=OVERLAP):
    """Return (left_crop, right_crop) as JPEG bytes; they share `overlap` of the width."""
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w = im.width
    cut_l, cut_r = round(w * (1 + overlap) / 2), round(w * (1 - overlap) / 2)
    return _encode(im.crop((0, 0, cut_l, im.height))), _encode(im.crop((cut_r, 0, w, im.height)))


def splittable(row):
    return not SPLIT_BLOCK.search(row["question"]) and row["question_type"] != "Motion (Cam.)"


def probe_messages(rows, max_side):
    for i, r in enumerate(rows):
        n_img = len(r["images"])
        base = dict(question=r["question"], order=list(range(n_img)), flip=False, note="")
        yield i, "orig", build_messages(r, base, max_side)
        yield i, "jpeg", build_messages(dict(r, images=[jpeg_noise(b) for b in r["images"]]),
                                        base, max_side)
        if splittable(r):
            content = []
            for k, b in enumerate(r["images"]):
                for part, crop in zip(["left", "right"], split_image(b)):
                    content.append({"type": "text", "text": f"Image {k + 1} ({part} part):"})
                    content.append({"type": "image_url",
                                    "image_url": {"url": img_to_url(crop, False, max_side)}})
            content.append({"type": "text", "text": SPLIT_NOTE + r["question"] + POST_PROMPT})
            yield i, "split", [{"role": "user", "content": content}]


def run(args):
    rows = load_rows(args.parquet)
    if args.exclude:
        seen = {json.loads(l)["id"] for p in args.exclude for l in open(p)}
        rows = [r for r in rows if int(r["id"]) not in seen]
    rows = stratified_sample(rows, args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump(dict({k: v for k, v in vars(args).items() if k != "analyze"}, overlap=OVERLAP),
                  f, indent=2)
    max_imgs = 2 * max(len(r["images"]) for r in rows)
    backend = MockBackend(args.seed) if args.backend == "mock" else VLLMBackend(
        args.model, max_imgs=max_imgs,
        hf_overrides=json.loads(args.hf_overrides) if args.hf_overrides else None)
    jobs = list(probe_messages(rows, args.max_side))
    print(f"{len(rows)} questions, {len(jobs)} greedy prompts, up to {max_imgs} images",
          file=sys.stderr)
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
    analyze(path, args.bootstrap, args.seed)


def analyze(path, bootstrap=1000, seed=0):
    recs = [json.loads(l) for l in open(path)]
    rng = random.Random(seed)
    pred = lambda r, v: r["variants"][v]["pred"]
    flip = lambda r, v: pred(r, v) != pred(r, "orig")
    lines = [f"# Image probes  (N = {len(recs)} questions, source `{path}`)\n",
             "| variant | n | accuracy | flip rate vs orig | flip 95% CI | unparsable |",
             "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        sub = [r for r in recs if v in r["variants"]]
        fl = [flip(r, v) for r in sub]
        ci = bootstrap_ci(len(sub), lambda ix, x=fl: pct(sum(x[i] for i in ix), len(ix)), bootstrap, rng)
        lines.append(f"| {v} | {len(sub)} | {pct(sum(pred(r, v) == r['answer'] for r in sub), len(sub)):.1f}% | "
                     f"{pct(sum(fl), len(fl)):.1f}% | {ci[0]:.1f}–{ci[1]:.1f} | "
                     f"{sum(pred(r, v) is None for r in sub)} |")

    sub = [r for r in recs if "split" in r["variants"]]
    a, b, p = mcnemar_exact([flip(r, "split") for r in sub], [flip(r, "jpeg") for r in sub])
    lines.append(f"\nOn the {len(sub)} questions with a split variant: split flip rate "
                 f"{pct(sum(flip(r, 'split') for r in sub), len(sub)):.1f}% vs. jpeg noise floor "
                 f"{pct(sum(flip(r, 'jpeg') for r in sub), len(sub)):.1f}% "
                 f"(split-only flips={a}, jpeg-only flips={b}, exact McNemar p={p:.4f}).")
    a, b, p = mcnemar_exact([pred(r, "orig") == r["answer"] for r in sub],
                            [pred(r, "split") == r["answer"] for r in sub])
    lines.append(f"Accuracy orig vs split on the same questions: orig-only correct={a}, "
                 f"split-only correct={b}, exact McNemar p={p:.4f}.")

    lines.append("\n## Flip rate by question type")
    lines.append("| type | n (split) | jpeg flip | split flip |")
    lines.append("|---|---|---|---|")
    for t in sorted({r["question_type"] for r in recs}):
        s = [r for r in sub if r["question_type"] == t]
        if s:
            lines.append(f"| {t} | {len(s)} | {pct(sum(flip(r, 'jpeg') for r in s), len(s)):.1f}% | "
                         f"{pct(sum(flip(r, 'split') for r in s), len(s)):.1f}% |")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(os.path.dirname(path) or ".", "probe_report.md"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet")
    ap.add_argument("--analyze", help="only analyze an existing predictions.jsonl")
    ap.add_argument("--exclude", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_side", type=int, default=768)
    ap.add_argument("--hf_overrides", help="model config overrides as JSON (see mmsi_consistency.py)")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="results_probes")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, a.bootstrap, a.seed)
    else:
        run(a)

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
  grid   : every image cut into 2x2 overlapping tiles (each 60% x 60% of the image), shown in
           reading order as "Image k (top-left part):" etc. Same skip rule as split.
  grid_shuffle : the grid tiles of all images in one random order, labels kept, with a note
           that the parts are not in order. Tests whether the model follows the labels.

--match_density shrinks every piece's long side to max_side * 0.6, so a piece has the same
pixel density as the full image it came from (without it, crops of a large image are
downscaled less than the full image and carry extra detail).

A transform only "confuses" the model if its flip rate is clearly above the jpeg noise floor,
so the report compares the two on the same questions with an exact McNemar test.

Usage:
  python image_probes.py --parquet MMSI_Bench.parquet --n 1000 \
      --exclude dev1/predictions.jsonl dev2/predictions.jsonl --out results_probes
  python image_probes.py ... --variants orig jpeg grid grid_shuffle --match_density
  python image_probes.py --analyze results_probes/predictions.jsonl
"""
import argparse, io, json, os, random, re, sys

from mmsi_consistency import (POST_PROMPT, MockBackend, VLLMBackend, bootstrap_ci, build_messages,
                              extract, img_to_url, load_rows, mcnemar_exact, stratified_sample)

VARIANTS = ["orig", "jpeg", "split", "grid", "grid_shuffle"]
PIECE_VARIANTS = ["split", "grid", "grid_shuffle"]
OVERLAP = 0.2  # crops are [0, (1+OVERLAP)/2] and [(1-OVERLAP)/2, 1] of the width
SPLIT_BLOCK = re.compile(r"\b(how many|count|number of)\b", re.IGNORECASE)
SPLIT_NOTE = ("Note: each original image is shown as two overlapping parts, its left part and "
              "its right part.\n")
GRID_NOTE = ("Note: each original image is shown as four overlapping parts: its top-left, "
             "top-right, bottom-left and bottom-right part.\n")
SHUFFLE_NOTE = ("The parts are not shown in order; refer to each part by its label.\n")
GRID_PARTS = ["top-left", "top-right", "bottom-left", "bottom-right"]
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


def grid_image(b, overlap=OVERLAP):
    """Return the 2x2 overlapping tiles (top-left, top-right, bottom-left, bottom-right) as JPEG bytes."""
    from PIL import Image
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    xs = [(0, round(w * (1 + overlap) / 2)), (round(w * (1 - overlap) / 2), w)]
    ys = [(0, round(h * (1 + overlap) / 2)), (round(h * (1 - overlap) / 2), h)]
    return [_encode(im.crop((x0, y0, x1, y1))) for y0, y1 in ys for x0, x1 in xs]


def splittable(row):
    return not SPLIT_BLOCK.search(row["question"]) and row["question_type"] != "Motion (Cam.)"


def probe_messages(rows, max_side, variants=VARIANTS, match_density=False, seed=0):
    rng = random.Random(seed)
    piece_side = round(max_side * (1 + OVERLAP) / 2) if match_density else max_side
    for i, r in enumerate(rows):
        n_img = len(r["images"])
        base = dict(question=r["question"], order=list(range(n_img)), flip=False, note="")
        if "orig" in variants:
            yield i, "orig", build_messages(r, base, max_side)
        if "jpeg" in variants:
            yield i, "jpeg", build_messages(dict(r, images=[jpeg_noise(b) for b in r["images"]]),
                                            base, max_side)
        if not splittable(r):
            continue
        pieces = {}
        if "split" in variants:
            pieces["split"] = ([(f"Image {k + 1} ({part} part)", crop)
                                for k, b in enumerate(r["images"])
                                for part, crop in zip(["left", "right"], split_image(b))], SPLIT_NOTE)
        if "grid" in variants or "grid_shuffle" in variants:
            tiles = [(f"Image {k + 1} ({part} part)", crop) for k, b in enumerate(r["images"])
                     for part, crop in zip(GRID_PARTS, grid_image(b))]
            if "grid" in variants:
                pieces["grid"] = (tiles, GRID_NOTE)
            if "grid_shuffle" in variants:
                order = list(range(len(tiles)))
                while order == sorted(order):
                    rng.shuffle(order)
                pieces["grid_shuffle"] = ([tiles[j] for j in order], GRID_NOTE + SHUFFLE_NOTE)
        for v, (ps, note) in pieces.items():
            content = []
            for label, crop in ps:
                content.append({"type": "text", "text": f"{label}:"})
                content.append({"type": "image_url",
                                "image_url": {"url": img_to_url(crop, False, piece_side)}})
            content.append({"type": "text", "text": note + r["question"] + POST_PROMPT})
            yield i, v, [{"role": "user", "content": content}]


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
    pieces_per_img = 4 if {"grid", "grid_shuffle"} & set(args.variants) else 2
    max_imgs = pieces_per_img * max(len(r["images"]) for r in rows)
    backend = MockBackend(args.seed) if args.backend == "mock" else VLLMBackend(
        args.model, max_imgs=max_imgs,
        hf_overrides=json.loads(args.hf_overrides) if args.hf_overrides else None)
    jobs = list(probe_messages(rows, args.max_side, args.variants, args.match_density, args.seed))
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
        if not sub:
            continue
        fl = [flip(r, v) for r in sub]
        ci = bootstrap_ci(len(sub), lambda ix, x=fl: pct(sum(x[i] for i in ix), len(ix)), bootstrap, rng)
        lines.append(f"| {v} | {len(sub)} | {pct(sum(pred(r, v) == r['answer'] for r in sub), len(sub)):.1f}% | "
                     f"{pct(sum(fl), len(fl)):.1f}% | {ci[0]:.1f}–{ci[1]:.1f} | "
                     f"{sum(pred(r, v) is None for r in sub)} |")

    run_pieces = [v for v in PIECE_VARIANTS if any(v in r["variants"] for r in recs)]
    for v in run_pieces:
        sub = [r for r in recs if v in r["variants"]]
        a, b, p = mcnemar_exact([flip(r, v) for r in sub], [flip(r, "jpeg") for r in sub])
        lines.append(f"\nOn the {len(sub)} questions with a {v} variant: {v} flip rate "
                     f"{pct(sum(flip(r, v) for r in sub), len(sub)):.1f}% vs. jpeg noise floor "
                     f"{pct(sum(flip(r, 'jpeg') for r in sub), len(sub)):.1f}% "
                     f"({v}-only flips={a}, jpeg-only flips={b}, exact McNemar p={p:.4f}).")
        a, b, p = mcnemar_exact([pred(r, "orig") == r["answer"] for r in sub],
                                [pred(r, v) == r["answer"] for r in sub])
        lines.append(f"Accuracy orig vs {v} on the same questions: orig-only correct={a}, "
                     f"{v}-only correct={b}, exact McNemar p={p:.4f}.")
    if "grid" in run_pieces and "grid_shuffle" in run_pieces:
        sub = [r for r in recs if "grid" in r["variants"]]
        a, b, p = mcnemar_exact([flip(r, "grid_shuffle") for r in sub], [flip(r, "grid") for r in sub])
        lines.append(f"\nTile order (grid_shuffle vs grid, n={len(sub)}): shuffle-only flips={a}, "
                     f"grid-only flips={b}, exact McNemar p={p:.4f}.")

    lines.append("\n## Flip rate by question type")
    lines.append("| type | n (pieces) | jpeg flip | " + " | ".join(f"{v} flip" for v in run_pieces) + " |")
    lines.append("|---|---|---|" + "---|" * len(run_pieces))
    for t in sorted({r["question_type"] for r in recs}):
        s = [r for r in recs if r["question_type"] == t and run_pieces and run_pieces[0] in r["variants"]]
        if s:
            lines.append(f"| {t} | {len(s)} | {pct(sum(flip(r, 'jpeg') for r in s), len(s)):.1f}% | " +
                         " | ".join(f"{pct(sum(flip(r, v) for r in s), len(s)):.1f}%" for v in run_pieces)
                         + " |")
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
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    ap.add_argument("--match_density", action="store_true",
                    help="shrink pieces to max_side * 0.6 so they keep the full image's pixel density")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default="results_probes")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, a.bootstrap, a.seed)
    else:
        run(a)

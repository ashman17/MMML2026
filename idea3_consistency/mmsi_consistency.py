"""
Research Idea 3 pilot: transformation-consistency checking on MMSI-Bench.

For each sampled question we build answer-preserving variants:
  orig     : images in order, each labeled "Image k:"
  mirror   : every image horizontally flipped; left/right and clockwise/counterclockwise
             swapped in question + options (correct letter unchanged)
  reorder  : images presented in reversed order but keep their original labels
             ("Image 1:" stays attached to the same picture)
  shuffle  : answer options permuted (letter remapped)
Then we run a VLM (greedy) on all variants, plus a self-consistency baseline on the
original input with the same number of calls, and report:
  accuracy, flip rate per transform, consistency rate, AUROC of disagreement for
  detecting wrong answers, and accuracy of transformation voting vs self-consistency.

Usage:
  python mmsi_consistency.py --parquet MMSI_Bench.parquet --n 100 --backend vllm \
      --model Qwen/Qwen2.5-VL-7B-Instruct --out results/
  python mmsi_consistency.py --analyze results/predictions.jsonl
  python mmsi_consistency.py --parquet fake.parquet --backend mock   # dry run
  LITELLM_API_KEY=... python mmsi_consistency.py --parquet MMSI_Bench.parquet --n 100 \
      --backend openai --model gpt-4o --out results_gpt4o/      # CMU LiteLLM gateway
"""
import argparse, base64, csv, io, json, math, os, random, re, sys
from collections import Counter, defaultdict

POST_PROMPT = ("\nAnswer with the option's letter from the given choices directly. "
               "Enclose the option's letter within ``.")
LETTERS = "ABCD"

# ---------------------------------------------------------------- parsing
OPT_SPLIT = re.compile(r"(?:^|,\s*|\s+)([A-D])\s*[:.]\s+")

def parse_question(q):
    """Split MMSI question into (stem, [opt_A..opt_D]). Returns None if unparsable."""
    m = re.search(r"Options\s*:\s*", q)
    if not m:
        return None
    stem, opt_str = q[:m.start()], q[m.end():]
    parts = OPT_SPLIT.split(opt_str)
    # parts = [prefix, 'A', textA, 'B', textB, ...]
    if parts[0].strip() or len(parts) < 3:
        return None
    labels, texts = parts[1::2], [t.strip().rstrip(",").strip() for t in parts[2::2]]
    if "".join(labels) != LETTERS[:len(labels)] or len(labels) < 2:
        return None
    return stem, texts

def build_question(stem, opts):
    return stem + "Options: " + ", ".join(f"{LETTERS[i]}: {o}" for i, o in enumerate(opts))

# ---------------------------------------------------------------- mirror
MIRROR_WORDS = re.compile(
    r"\b(counter-?clockwise|anti-?clockwise|clockwise|leftmost|rightmost|"
    r"leftwards?|rightwards?|left|right)\b", re.IGNORECASE)
MIRROR_MAP = {"left": "right", "right": "left", "leftmost": "rightmost",
              "rightmost": "leftmost", "leftward": "rightward", "rightward": "leftward",
              "leftwards": "rightwards", "rightwards": "leftwards",
              "clockwise": "counterclockwise", "counterclockwise": "clockwise",
              "counter-clockwise": "clockwise", "anticlockwise": "clockwise",
              "anti-clockwise": "clockwise"}
# mirroring is not answer-preserving for these (text in image, compass directions, idioms)
MIRROR_BLOCK = re.compile(
    r"\b(north(?:east|west)?|south(?:east|west)?|east|west|"
    r"text|written|sign|letters?|words?|reads?|logo|"
    r"right angle|right-angle|all right|right away)\b", re.IGNORECASE)

def _case_like(src, dst):
    if src.isupper(): return dst.upper()
    if src[0].isupper(): return dst[0].upper() + dst[1:]
    return dst

def mirror_text(s):
    return MIRROR_WORDS.sub(lambda m: _case_like(m.group(0), MIRROR_MAP[m.group(0).lower()]), s)

def has_direction(s):
    return bool(MIRROR_WORDS.search(s))

# ---------------------------------------------------------------- variants
def make_variants(row, rng):
    """Return list of variant dicts: name, question, image_order, flip, perm.
    perm[i] = original option index shown at position i."""
    q, n_img = row["question"], len(row["images"])
    base = dict(question=q, order=list(range(n_img)), flip=False, perm=None, note="")
    vs = [dict(base, name="orig")]
    if not MIRROR_BLOCK.search(q):
        vs.append(dict(base, name="mirror", question=mirror_text(q), flip=True))
    if n_img >= 2:
        vs.append(dict(base, name="reorder", order=list(reversed(range(n_img))),
                       note="Note: the images are not shown in numerical order; "
                            "refer to each image by its label.\n"))
    parsed = parse_question(q)
    if parsed:
        stem, opts = parsed
        perm = list(range(len(opts)))
        while perm == list(range(len(opts))):
            rng.shuffle(perm)
        vs.append(dict(base, name="shuffle",
                       question=build_question(stem, [opts[p] for p in perm]), perm=perm))
    return vs

def map_back(pred, perm):
    """Map a letter predicted on a shuffled variant back to the original letter."""
    if pred is None or perm is None:
        return pred
    i = LETTERS.find(pred)
    return LETTERS[perm[i]] if 0 <= i < len(perm) else None

# ---------------------------------------------------------------- answer extraction (official)
def extract(text):
    for pat in [r"``([A-D])``", r"`([A-D])`", r"\{([A-D])\}", r"\b([A-D])\b(?!\s[a-zA-Z])"]:
        m = re.search(pat, text or "")
        if m:
            return m.group(1)
    return None

# ---------------------------------------------------------------- data
def load_rows(path):
    import pandas as pd
    df = pd.read_parquet(path)
    rows = []
    for r in df.to_dict("records"):
        imgs = []
        for im in r["images"]:
            b = im["bytes"] if isinstance(im, dict) else im
            imgs.append(bytes(b))
        r["images"] = imgs
        rows.append(r)
    return rows

def stratified_sample(rows, n, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["question_type"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    out, types = [], sorted(by)
    while len(out) < min(n, len(rows)):
        for t in types:
            if by[t] and len(out) < n:
                out.append(by[t].pop())
    return out

def img_to_url(b, flip, max_side):
    from PIL import Image, ImageOps
    im = Image.open(io.BytesIO(b)).convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    if flip:
        im = ImageOps.mirror(im)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def build_messages(row, v, max_side):
    content = []
    for k in v["order"]:
        content.append({"type": "text", "text": f"Image {k + 1}:"})
        content.append({"type": "image_url",
                        "image_url": {"url": img_to_url(row["images"][k], v["flip"], max_side)}})
    content.append({"type": "text", "text": v["note"] + v["question"] + POST_PROMPT})
    return [{"role": "user", "content": content}]

# ---------------------------------------------------------------- backends
class MockBackend:
    """Random answers with a left/right bias, for dry runs without a GPU."""
    def __init__(self, seed=0): self.rng = random.Random(seed)
    def generate(self, convs, n=1, temperature=0.0):
        outs = []
        for c in convs:
            txt = c[0]["content"][-1]["text"]
            outs.append([f"``{self.rng.choice('AB' if 'left' in txt.lower() else LETTERS)}``"
                         for _ in range(n)])
        return outs

class VLLMBackend:
    def __init__(self, model, max_imgs=10, max_len=32768, tp=1):
        from vllm import LLM
        self.llm = LLM(model=model, limit_mm_per_prompt={"image": max_imgs},
                       max_model_len=max_len, tensor_parallel_size=tp,
                       gpu_memory_utilization=0.9, trust_remote_code=True)
    def generate(self, convs, n=1, temperature=0.0):
        from vllm import SamplingParams
        sp = SamplingParams(n=n, temperature=temperature, top_p=1.0 if temperature == 0 else 0.95,
                            max_tokens=64, seed=0)
        res = self.llm.chat(convs, sp, use_tqdm=True)
        return [[o.text for o in r.outputs] for r in res]

class OpenAIBackend:
    """OpenAI-compatible API backend (e.g. the CMU LiteLLM gateway).
    Key is read from the LITELLM_API_KEY (or OPENAI_API_KEY) environment variable, never hard-coded.
    Every response is cached in <out>/api_cache.jsonl so a crashed or interrupted run resumes
    without paying for the same calls again."""
    def __init__(self, model, base_url, cache_path, workers=8, max_tokens=64, retries=5):
        import hashlib, threading
        from openai import OpenAI
        key = os.environ.get("LITELLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("Set LITELLM_API_KEY (export LITELLM_API_KEY=...) before using --backend openai")
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model, self.workers, self.max_tokens, self.retries = model, workers, max_tokens, retries
        self._hash, self._lock, self.cache_path = hashlib.sha1, threading.Lock(), cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                try:
                    d = json.loads(line); self.cache[d["k"]] = d["text"]
                except Exception:
                    pass
        print(f"[openai] model={model} cached={len(self.cache)}", file=sys.stderr)

    def _key(self, conv, temperature, j):
        h = self._hash(json.dumps(conv, sort_keys=True).encode())
        h.update(f"|{self.model}|{temperature}|{j}".encode())
        return h.hexdigest()

    def _call(self, conv, temperature):
        import time
        kw = dict(model=self.model, messages=conv, max_tokens=self.max_tokens)
        if temperature is not None:
            kw["temperature"] = temperature
            if temperature > 0:
                kw["top_p"] = 0.95
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(**kw)
                return r.choices[0].message.content or ""
            except Exception as e:
                msg = str(e).lower()
                if "temperature" in msg or "top_p" in msg:   # some models only allow default sampling
                    kw.pop("temperature", None); kw.pop("top_p", None)
                    continue
                if attempt == self.retries - 1:
                    print(f"[openai] giving up after {self.retries} tries: {e}", file=sys.stderr)
                    return ""
                time.sleep(min(60, 2 ** attempt))
        return ""

    def _one(self, conv, temperature, j):
        k = self._key(conv, temperature, j)
        if k in self.cache:
            return self.cache[k]
        text = self._call(conv, temperature)
        if text:  # do not cache failures, so a rerun retries them
            with self._lock:
                self.cache[k] = text
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"k": k, "text": text}) + "\n")
        return text

    def generate(self, convs, n=1, temperature=0.0):
        from concurrent.futures import ThreadPoolExecutor
        tasks = [(ci, j) for ci in range(len(convs)) for j in range(n)]
        out = [[None] * n for _ in convs]
        done = 0
        with ThreadPoolExecutor(self.workers) as ex:
            futs = {ex.submit(self._one, convs[ci], temperature, j): (ci, j) for ci, j in tasks}
            for f in futs:
                ci, j = futs[f]
                out[ci][j] = f.result()
                done += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"[openai] {done}/{len(tasks)}", file=sys.stderr)
        return out

# ---------------------------------------------------------------- run
def run(args):
    rows = stratified_sample(load_rows(args.parquet), args.n, args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "analyze"}, f, indent=2)
    if args.backend == "mock":
        backend = MockBackend(args.seed)
    elif args.backend == "openai":
        backend = OpenAIBackend(args.model, args.base_url, os.path.join(args.out, "api_cache.jsonl"),
                                workers=args.workers, max_tokens=args.max_tokens)
    else:
        backend = VLLMBackend(args.model, tp=args.tp)

    jobs = []  # (row_idx, variant)
    for i, r in enumerate(rows):
        for v in make_variants(r, rng):
            jobs.append((i, v))
    print(f"{len(rows)} questions, {len(jobs)} greedy prompts", file=sys.stderr)

    convs = [build_messages(rows[i], v, args.max_side) for i, v in jobs]
    greedy = backend.generate(convs, n=1, temperature=0.0)

    # self-consistency baseline: K samples on the original input, K = #variants of that question
    k_per = Counter(i for i, _ in jobs)
    sc_convs = [build_messages(r, dict(question=r["question"], order=list(range(len(r["images"]))),
                                       flip=False, note=""), args.max_side) for r in rows]
    by_k = defaultdict(list)
    for i in range(len(rows)):
        by_k[k_per[i]].append(i)
    sc_raw = {}
    for k, idxs in by_k.items():
        outs = backend.generate([sc_convs[i] for i in idxs], n=k, temperature=args.sc_temp)
        for i, o in zip(idxs, outs):
            sc_raw[i] = o

    recs = defaultdict(lambda: {"variants": {}})
    for (i, v), out in zip(jobs, greedy):
        r = rows[i]
        rec = recs[i]
        rec.update(id=int(r["id"]), question_type=r["question_type"], answer=r["answer"],
                   n_images=len(r["images"]), has_direction=has_direction(r["question"]),
                   sc_preds=[extract(t) for t in sc_raw[i]])
        raw_pred = extract(out[0])
        rec["variants"][v["name"]] = dict(raw=out[0], pred_raw=raw_pred,
                                          pred=map_back(raw_pred, v["perm"]),
                                          question=v["question"], note=v["note"],
                                          image_order=[k + 1 for k in v["order"]],
                                          images_mirrored=v["flip"], perm=v["perm"])
    path = os.path.join(args.out, "predictions.jsonl")
    with open(path, "w") as f:
        for i in sorted(recs):
            f.write(json.dumps(recs[i]) + "\n")
    print(f"wrote {path}", file=sys.stderr)
    analyze(path, bootstrap=args.bootstrap, seed=args.analysis_seed,
            make_figures=not args.no_figures)

# ---------------------------------------------------------------- analysis
def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))

def vote(preds, tiebreak):
    c = Counter(p for p in preds if p)
    if not c:
        return tiebreak
    top = max(c.values())
    winners = [p for p, v in c.items() if v == top]
    return tiebreak if tiebreak in winners else winners[0]

def percentile(xs, q):
    """Linear-interpolated percentile without a NumPy dependency."""
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)

def bootstrap_ci(n, stat, reps, rng):
    if not n or reps <= 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(reps):
        ix = [rng.randrange(n) for _ in range(n)]
        vals.append(stat(ix))
    return percentile(vals, .025), percentile(vals, .975)

def mcnemar_exact(a, b):
    """Exact two-sided McNemar p-value for paired Boolean outcomes."""
    better_a = sum(x and not y for x, y in zip(a, b))
    better_b = sum(y and not x for x, y in zip(a, b))
    n = better_a + better_b
    if n == 0:
        return better_a, better_b, 1.0
    k = min(better_a, better_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return better_a, better_b, min(1.0, 2 * tail)

def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def _make_figures(outdir, recs, types, orig_ok, tv, sc, flip_ci):
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(outdir, ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(outdir, ".cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[analysis] figures skipped: {e}", file=sys.stderr)
        return []

    made = []
    # Figure 1: transform flip rates, including the mirror direction-word split.
    bars = []
    for key, label in [("mirror", "Mirror (all)"), ("mirror_direction", "Mirror\n(direction)"),
                       ("mirror_no_direction", "Mirror\n(no direction)"),
                       ("reorder", "Reorder"), ("shuffle", "Shuffle")]:
        if key.startswith("mirror_"):
            flag = key == "mirror_direction"
            sub = [r for r in recs if "mirror" in r["variants"] and r["has_direction"] == flag]
            t = "mirror"
        else:
            sub = [r for r in recs if key in r["variants"]]
            t = key
        if sub:
            val = 100 * sum(r["variants"][t]["pred"] != r["variants"]["orig"]["pred"]
                            for r in sub) / len(sub)
            bars.append((key, label, val, len(sub)))
    if bars:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        keys, labels, vals, ns = zip(*bars)
        ax.bar(labels, vals, color=["#4C78A8", "#72B7B2", "#A0CBE8", "#F58518", "#E45756"][:len(vals)])
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(i, v + 1, f"{v:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Flip rate vs. original (%)")
        ax.set_ylim(0, max(vals) * 1.28 + 3)
        ax.set_title("Flip rate by answer-preserving transformation")
        fig.tight_layout()
        p = os.path.join(outdir, "fig_flip_rate_by_transform.png")
        fig.savefig(p, dpi=180); plt.close(fig); made.append(p)

    # Figure 2: original accuracy by how many available variants agree with orig.
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        orig = r["variants"]["orig"]["pred"]
        agree = sum(v["pred"] == orig for v in r["variants"].values())
        groups[(agree, len(r["variants"]))].append(orig_ok[i])
    if groups:
        ordered = sorted(groups, key=lambda z: (z[0] / z[1], z[1], z[0]))
        vals = [100 * sum(groups[k]) / len(groups[k]) for k in ordered]
        labels = [f"{a}/{k}" for a, k in ordered]
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.bar(labels, vals, color="#59A14F")
        for i, (v, k) in enumerate(zip(vals, ordered)):
            ax.text(i, v + 1, f"n={len(groups[k])}", ha="center", fontsize=8)
        ax.set_xlabel("Variants agreeing with original / variants available")
        ax.set_ylabel("Original accuracy (%)")
        ax.set_ylim(0, 108)
        ax.set_title("Original accuracy by transformation agreement")
        fig.tight_layout()
        p = os.path.join(outdir, "fig_accuracy_vs_agreement.png")
        fig.savefig(p, dpi=180); plt.close(fig); made.append(p)

    # Figure 3: greedy, self-consistency, and transform voting by question type.
    if types:
        vals = []
        for t in types:
            ix = [i for i, r in enumerate(recs) if r["question_type"] == t]
            vals.append((100 * sum(orig_ok[i] for i in ix) / len(ix),
                         100 * sum(sc[i] for i in ix) / len(ix),
                         100 * sum(tv[i] for i in ix) / len(ix)))
        fig, ax = plt.subplots(figsize=(max(8, .85 * len(types)), 4.8))
        x = list(range(len(types))); width = .25
        for off, j, label, color in [(-width, 0, "Greedy", "#4C78A8"),
                                     (0, 1, "Self-consistency", "#F58518"),
                                     (width, 2, "Transform vote", "#54A24B")]:
            ax.bar([z + off for z in x], [v[j] for v in vals], width, label=label, color=color)
        ax.set_xticks(x, types, rotation=35, ha="right")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105); ax.legend(frameon=False)
        ax.set_title("Voting comparison by MMSI-Bench question type")
        fig.tight_layout()
        p = os.path.join(outdir, "fig_vote_by_question_type.png")
        fig.savefig(p, dpi=180); plt.close(fig); made.append(p)
    return made

def analyze(path, bootstrap=1000, seed=0, make_figures=True):
    recs = [json.loads(l) for l in open(path)]
    N = len(recs)
    if not N:
        raise ValueError(f"no prediction rows in {path}")
    outdir = os.path.dirname(path) or "."
    config = {}
    config_path = os.path.join(outdir, "run_config.json")
    if os.path.exists(config_path):
        try:
            config = json.load(open(config_path))
        except Exception as e:
            print(f"[analysis] could not read {config_path}: {e}", file=sys.stderr)
    acc = lambda xs: 100.0 * sum(xs) / len(xs) if xs else float("nan")
    lines = [f"# Idea 3 transformation-consistency results  (N = {N} questions)\n"]
    if config:
        lines.append(f"Model: `{config.get('model', 'unknown')}` | backend: "
                     f"`{config.get('backend', 'unknown')}` | sample seed: "
                     f"{config.get('seed', 'unknown')} | max image side: "
                     f"{config.get('max_side', 'unknown')} px\n")

    # If the required human transformation review has been completed, keep its
    # pass rate in the same report instead of making readers reconcile files.
    validity = None
    validity_path = os.path.join(outdir, "validity_review", "validity_check.csv")
    if not os.path.exists(validity_path):
        validity_path = os.path.join(outdir, "validity_check.csv")
    if os.path.exists(validity_path):
        rows = list(csv.DictReader(open(validity_path, newline="")))
        reviewed = [r for r in rows if r.get("valid", "").strip()]
        passed = [r for r in reviewed
                  if r["valid"].strip().lower() in {"y", "yes", "true", "1"}]
        validity = dict(reviewed=len(reviewed), total=len(rows), passed=len(passed),
                        pass_rate=(100.0 * len(passed) / len(reviewed) if reviewed else None))
        if reviewed:
            lines.append(f"AI-assisted visual transformation review: {len(passed)}/{len(reviewed)} "
                         f"valid ({validity['pass_rate']:.1f}%); "
                         f"review completion {len(reviewed)}/{len(rows)}. "
                         "Human sign-off is still recommended before submission.\n")
        else:
            lines.append(f"Transformation validity: 0/{len(rows)} rows reviewed.\n")

    orig_ok = [r["variants"]["orig"]["pred"] == r["answer"] for r in recs]
    rng = random.Random(seed)
    orig_ci = bootstrap_ci(N, lambda ix: acc([orig_ok[i] for i in ix]), bootstrap, rng)
    lines.append(f"Original accuracy: {acc(orig_ok):.1f}% "
                 f"(bootstrap 95% CI {orig_ci[0]:.1f}–{orig_ci[1]:.1f})")
    lines.append(f"Unparsable original answers: {sum(r['variants']['orig']['pred'] is None for r in recs)}\n")

    lines.append("## Per transformation")
    lines.append("| transform | n | variant acc | flip rate vs orig | flip 95% CI | orig acc on subset | unparsable |")
    lines.append("|---|---|---|---|---|---|---|")
    flip_ci = {}
    flip_est = {}
    for t in ["mirror", "reorder", "shuffle"]:
        sub = [r for r in recs if t in r["variants"]]
        if not sub: continue
        va = [r["variants"][t]["pred"] == r["answer"] for r in sub]
        fl = [r["variants"][t]["pred"] != r["variants"]["orig"]["pred"] for r in sub]
        oa = [r["variants"]["orig"]["pred"] == r["answer"] for r in sub]
        ci = bootstrap_ci(len(sub), lambda ix, x=fl: acc([x[i] for i in ix]), bootstrap, rng)
        flip_ci[t] = ci
        flip_est[t] = acc(fl)
        unparsable = sum(r["variants"][t]["pred"] is None for r in sub)
        lines.append(f"| {t} | {len(sub)} | {acc(va):.1f}% | {acc(fl):.1f}% | "
                     f"{ci[0]:.1f}–{ci[1]:.1f} | {acc(oa):.1f}% | {unparsable} |")
    msub = [r for r in recs if "mirror" in r["variants"]]
    mirror_direction_rates = {}
    for flag in [True, False]:
        s = [r for r in msub if r["has_direction"] == flag]
        if s:
            fl = [r["variants"]["mirror"]["pred"] != r["variants"]["orig"]["pred"] for r in s]
            mirror_direction_rates[flag] = acc(fl)
            lines.append(f"\nMirror flip rate, questions {'with' if flag else 'without'} "
                         f"left/right words: {acc(fl):.1f}% (n={len(s)})")

    cons = [len({v["pred"] for v in r["variants"].values()}) == 1 for r in recs]
    acc_consistent = acc([o for o, c in zip(orig_ok, cons) if c])
    acc_inconsistent = acc([o for o, c in zip(orig_ok, cons) if not c])
    show_pct = lambda x: "N/A" if math.isnan(x) else f"{x:.1f}%"
    lines.append(f"\n## Consistency\nConsistency Rate (all variants agree): {acc(cons):.1f}%")
    lines.append(f"Accuracy when consistent: {show_pct(acc_consistent)} "
                 f"(n={sum(cons)}) | when inconsistent: "
                 f"{show_pct(acc_inconsistent)} (n={N - sum(cons)})")
    dis = [sum(v["pred"] != r["variants"]["orig"]["pred"] for k, v in r["variants"].items() if k != "orig")
           / max(1, len(r["variants"]) - 1) for r in recs]
    wrong = [not o for o in orig_ok]
    auc = auroc(dis, wrong)
    auc_ci = bootstrap_ci(N, lambda ix: auroc([dis[i] for i in ix], [wrong[i] for i in ix]),
                          bootstrap, rng)
    lines.append(f"AUROC of disagreement score for detecting wrong original answers: {auc:.3f} "
                 f"(bootstrap 95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}; 0.5 = chance)")

    tv = [vote([v["pred"] for v in r["variants"].values()], r["variants"]["orig"]["pred"]) == r["answer"]
          for r in recs]
    sc = [vote(r["sc_preds"], r["variants"]["orig"]["pred"]) == r["answer"] for r in recs]
    lines.append("\n## Voting (same number of model calls per question)")
    lines.append(f"Original greedy: {acc(orig_ok):.1f}% | Self-consistency: {acc(sc):.1f}% | "
                 f"Transformation vote: {acc(tv):.1f}%")
    gap = acc(tv) - acc(sc)
    gap_ci = bootstrap_ci(N, lambda ix: acc([tv[i] for i in ix]) - acc([sc[i] for i in ix]),
                          bootstrap, rng)
    better_tv, better_sc, p_mc = mcnemar_exact(tv, sc)
    lines.append(f"Transform-vote minus self-consistency: {gap:+.1f} percentage points "
                 f"(bootstrap 95% CI {gap_ci[0]:+.1f} to {gap_ci[1]:+.1f})")
    lines.append(f"Exact McNemar test: transform-only correct={better_tv}, "
                 f"self-consistency-only correct={better_sc}, p={p_mc:.4f}")

    lines.append("\n## Per question type")
    lines.append("| type | n | orig acc | consistency | transform vote | self-consistency |")
    lines.append("|---|---|---|---|---|---|")
    types = sorted({r["question_type"] for r in recs})
    for t in types:
        ix = [i for i, r in enumerate(recs) if r["question_type"] == t]
        lines.append(f"| {t} | {len(ix)} | {acc([orig_ok[i] for i in ix]):.1f}% | "
                     f"{acc([cons[i] for i in ix]):.1f}% | {acc([tv[i] for i in ix]):.1f}% | "
                     f"{acc([sc[i] for i in ix]):.1f}% |")

    # Extra analysis 1: question type x transformation flip table.
    type_flip_rows = []
    lines.append("\n## Flip rate by question type and transformation")
    lines.append("| type | transform | n | flip rate |")
    lines.append("|---|---|---|---|")
    for qt in types:
        for t in ["mirror", "reorder", "shuffle"]:
            sub = [r for r in recs if r["question_type"] == qt and t in r["variants"]]
            if not sub:
                continue
            rate = acc([r["variants"][t]["pred"] != r["variants"]["orig"]["pred"] for r in sub])
            lines.append(f"| {qt} | {t} | {len(sub)} | {rate:.1f}% |")
            type_flip_rows.append(dict(question_type=qt, transform=t, n=len(sub), flip_rate=rate))
    _write_csv(os.path.join(outdir, "per_type_flip_rate.csv"),
               ["question_type", "transform", "n", "flip_rate"], type_flip_rows)

    # Extra analysis 2: displayed answer position, before mapping shuffle back.
    position_rows = []
    lines.append("\n## Predicted option position (with images)")
    lines.append("| variant | position | count | share |")
    lines.append("|---|---|---|---|")
    for t in ["orig", "shuffle"]:
        preds = [r["variants"][t].get("pred_raw") for r in recs if t in r["variants"]]
        for letter in LETTERS:
            count = sum(p == letter for p in preds)
            share = 100 * count / len(preds) if preds else float("nan")
            lines.append(f"| {t} | {letter} | {count} | {share:.1f}% |")
            position_rows.append(dict(variant=t, position=letter, count=count,
                                      share_percent=share, n=len(preds)))
        missing = sum(p is None for p in preds)
        if missing:
            share = 100 * missing / len(preds)
            lines.append(f"| {t} | unparsable | {missing} | {share:.1f}% |")
            position_rows.append(dict(variant=t, position="unparsable", count=missing,
                                      share_percent=share, n=len(preds)))
    _write_csv(os.path.join(outdir, "option_position_bias.csv"),
               ["variant", "position", "count", "share_percent", "n"], position_rows)

    # Apply the pre-registered decision rules without hiding uncertainty.  The
    # wording deliberately distinguishes descriptive direction from statistical
    # evidence so a positive point estimate is not reported as conclusive.
    all_flips_nonzero = bool(flip_est) and all(x > 0 for x in flip_est.values())
    h2_point = (not math.isnan(acc_consistent) and not math.isnan(acc_inconsistent)
                and auc >= .6 and acc_consistent > acc_inconsistent)
    h2_robust = h2_point and auc_ci[0] >= .5
    h3_point = gap > 0
    h3_robust = h3_point and gap_ci[0] > 0 and p_mc < .05
    h4_point = (True in mirror_direction_rates and False in mirror_direction_rates
                and mirror_direction_rates[True] > mirror_direction_rates[False])
    pos_by_variant = {}
    for row in position_rows:
        if row["position"] in LETTERS:
            pos_by_variant.setdefault(row["variant"], []).append(row["share_percent"])
    shuffle_peak = max(pos_by_variant.get("shuffle", [float("nan")]))
    h5_point = flip_est.get("shuffle", 0) > 0 and shuffle_peak > 25

    lines.append("\n## Hypothesis evaluation and discussion")
    lines.append(f"- **H1 — {'supported' if all_flips_nonzero else 'not supported'}:** "
                 + ("every available answer-preserving transformation produced a non-zero flip rate."
                    if all_flips_nonzero else
                    "at least one available transformation produced no answer flips."))
    if h2_robust:
        h2_status = "supported"
    elif h2_point:
        h2_status = "directionally supported, but uncertain"
    else:
        h2_status = "not supported"
    lines.append(f"- **H2 — {h2_status}:** disagreement AUROC was {auc:.3f} "
                 f"(95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f}); original accuracy was "
                 f"{show_pct(acc_consistent)} when consistent versus "
                 f"{show_pct(acc_inconsistent)} when inconsistent.")
    lines.append(f"- **H3 — {'supported' if h3_robust else ('directionally supported, but not statistically conclusive' if h3_point else 'not supported')}:** "
                 f"transformation voting changed accuracy by {gap:+.1f} points relative to "
                 f"self-consistency (95% CI {gap_ci[0]:+.1f} to {gap_ci[1]:+.1f}; "
                 f"McNemar p={p_mc:.4f}).")
    if True in mirror_direction_rates and False in mirror_direction_rates:
        lines.append(f"- **H4 — {'descriptively supported' if h4_point else 'not supported'}:** mirror flip "
                     f"rate was {mirror_direction_rates[True]:.1f}% for direction-word questions "
                     f"versus {mirror_direction_rates[False]:.1f}% otherwise.")
    else:
        lines.append("- **H4 — not testable in this sample:** one mirror subgroup was empty.")
    lines.append(f"- **H5 — {'descriptively supported' if h5_point else 'not supported'}:** "
                 f"shuffle flip rate was {flip_est.get('shuffle', float('nan')):.1f}% and the "
                 f"most frequent predicted shuffled-option position accounted for {shuffle_peak:.1f}% "
                 "of parseable-position trials (25% would be uniform across four positions).")

    if auc >= .6:
        lines.append("\nImprovement direction: use transformation disagreement as a gate for extra "
                     "geometry/reasoning calls. Because the AUROC interval should guide confidence, "
                     "validate the gate on another model or held-out sample before deployment.")
    if h3_point:
        lines.append("Improvement direction: transformation voting is a promising training-free "
                     "baseline, but retain the paired confidence interval and McNemar result when "
                     "describing whether it reliably beats sampling diversity.")
    if h4_point:
        lines.append("Improvement direction: add mirrored training pairs or a consistency loss for "
                     "left/right reasoning.")
    if flip_est.get("reorder", 0) > 0:
        lines.append("Improvement direction: make image viewpoint labels explicit and evaluate whether "
                     "the model follows labels rather than presentation order.")
    if h5_point:
        lines.append("Improvement direction: evaluate every model over multiple option orders so that "
                     "visual benchmarks do not silently reward option-position shortcuts.")

    examples_path = os.path.join(outdir, "examples", "summary.json")
    if os.path.exists(examples_path):
        try:
            examples = json.load(open(examples_path))
        except Exception as e:
            print(f"[analysis] could not read {examples_path}: {e}", file=sys.stderr)
            examples = []
        if examples:
            lines.append("\n## Qualitative examples")
            lines.append("The exported cases below all have a wrong original prediction and "
                         "disagreement across answer-preserving variants.")
            for rank, ex in enumerate(examples[:3], 1):
                preds = ", ".join(f"{k}={v}" for k, v in ex["preds"].items())
                q = " ".join(ex["question"].split())
                if len(q) > 180:
                    q = q[:177] + "..."
                folder = f"examples/example{rank}_id{ex['id']}"
                lines.append(f"{rank}. **ID {ex['id']} ({ex['question_type']})** — gold={ex['gold']}; "
                             f"{preds}. {q} "
                             f"[original/mirror image]({folder}/image1_original_vs_mirrored.png)")

    lines.append("\n## AI usage and contribution disclosure (submission draft)")
    lines.append("Automated tools assisted with implementation, unit tests, Babel job operation, "
                 "figure generation, visual transformation preflight, and drafting the statistical "
                 "interpretation. Wen-Chi Tsai supplied the research specification and remains "
                 "responsible for checking the results, completing the required human validity "
                 "sign-off, and approving the submitted text.")

    ci_rows = [
        dict(metric="original_accuracy_percent", estimate=acc(orig_ok), ci_low=orig_ci[0], ci_high=orig_ci[1]),
        dict(metric="disagreement_auroc", estimate=auc, ci_low=auc_ci[0], ci_high=auc_ci[1]),
        dict(metric="transform_vote_minus_self_consistency_pp", estimate=gap,
             ci_low=gap_ci[0], ci_high=gap_ci[1]),
    ]
    for t, ci in flip_ci.items():
        sub = [r for r in recs if t in r["variants"]]
        est = acc([r["variants"][t]["pred"] != r["variants"]["orig"]["pred"] for r in sub])
        ci_rows.append(dict(metric=f"{t}_flip_rate_percent", estimate=est,
                            ci_low=ci[0], ci_high=ci[1]))
    _write_csv(os.path.join(outdir, "bootstrap_ci.csv"),
               ["metric", "estimate", "ci_low", "ci_high"], ci_rows)

    made = _make_figures(outdir, recs, types, orig_ok, tv, sc, flip_ci) if make_figures else []
    lines.append(f"\nGenerated figures: {len(made)}; bootstrap resamples: {bootstrap}; analysis seed: {seed}.")
    if N <= 100:
        lines.append("\nNote: with N <= 100, differences of a few points are noisy; "
                     "treat this as a feasibility signal, not a definitive result.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(outdir, "analysis_summary.json"), "w") as f:
        clean = lambda x: None if isinstance(x, float) and math.isnan(x) else x
        json.dump(dict(n=N, original_accuracy=acc(orig_ok), original_accuracy_ci=orig_ci,
                       consistency_rate=acc(cons), accuracy_consistent=clean(acc_consistent),
                       accuracy_inconsistent=clean(acc_inconsistent),
                       disagreement_auroc=auc, disagreement_auroc_ci=auc_ci,
                       self_consistency_accuracy=acc(sc), transformation_vote_accuracy=acc(tv),
                       vote_gap_pp=gap, vote_gap_ci=gap_ci,
                       hypotheses=dict(
                           H1="supported" if all_flips_nonzero else "not_supported",
                           H2=("supported" if h2_robust else
                               "directional_uncertain" if h2_point else "not_supported"),
                           H3=("supported" if h3_robust else
                               "directional_inconclusive" if h3_point else "not_supported"),
                           H4="descriptive_support" if h4_point else "not_supported",
                           H5="descriptive_support" if h5_point else "not_supported"),
                       mcnemar=dict(transform_only_correct=better_tv,
                                     self_consistency_only_correct=better_sc, p_exact=p_mc),
                       human_validity=validity,
                       figures=[os.path.basename(p) for p in made]), f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet")
    ap.add_argument("--analyze", help="only analyze an existing predictions.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["vllm", "openai", "mock"], default="vllm")
    ap.add_argument("--base_url", default="https://ai-gateway.andrew.cmu.edu",
                    help="OpenAI-compatible endpoint for --backend openai (CMU LiteLLM gateway)")
    ap.add_argument("--workers", type=int, default=8, help="parallel API requests (openai backend)")
    ap.add_argument("--max_tokens", type=int, default=64,
                    help="openai backend; raise (e.g. 2048) for reasoning models that think before answering")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max_side", type=int, default=768, help="resize long image side (0 = keep)")
    ap.add_argument("--sc_temp", type=float, default=0.7)
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples used by analysis (0 disables intervals)")
    ap.add_argument("--analysis_seed", type=int, default=0)
    ap.add_argument("--no_figures", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, bootstrap=a.bootstrap, seed=a.analysis_seed,
                make_figures=not a.no_figures)
    else:
        run(a)

import json, random, sys
from math import comb

def load(p): return {r["id"]: r for r in map(json.loads, open(p))}

def ci(ok, reps=1000, seed=0):
    rng = random.Random(seed); n = len(ok)
    vals = sorted(sum(ok[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return 100 * vals[25], 100 * vals[974]

def mcnemar(a, b):
    x = sum(i and not j for i, j in zip(a, b)); y = sum(j and not i for i, j in zip(a, b))
    n = x + y
    if n == 0: return x, y, 1.0
    k = min(x, y); p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p

base_path, subset_path = sys.argv[1], sys.argv[2]
models = [("base", base_path)] + [tuple(a.split("=", 1)) for a in sys.argv[3:]]
data = {name: load(p) for name, p in models}
ids_all = sorted(data["base"])
assert all(sorted(d) == ids_all for d in data.values()), "id sets differ"
conf = set(json.load(open(subset_path))["results_23689474/conf"])
subsets = {"all": ids_all, "conf (no-image max prob < 0.7, from base)": [i for i in ids_all if i in conf]}

print("| subset | model | n | orig acc | 95% CI | McNemar vs base (base-only / model-only / p) | no-image acc | orig = no-image |")
print("|---|---|---:|---:|---:|---:|---:|---:|")
for sname, ids in subsets.items():
    base_ok = [data["base"][i]["variants"]["orig"]["pred"] == data["base"][i]["answer"] for i in ids]
    for name, _ in models:
        d = data[name]
        ok = [d[i]["variants"]["orig"]["pred"] == d[i]["answer"] for i in ids]
        lo, hi = ci(ok)
        x, y, p = mcnemar(base_ok, ok)
        ni = sum(d[i]["noimage"]["pred"] == d[i]["answer"] for i in ids) / len(ids)
        agree = sum(d[i]["noimage"]["pred"] == d[i]["variants"]["orig"]["pred"] for i in ids) / len(ids)
        mc = "-" if name == "base" else f"{x} / {y} / {p:.3f}"
        print(f"| {sname} | {name} | {len(ids)} | {100*sum(ok)/len(ids):.1f}% | {lo:.1f}-{hi:.1f}% | {mc} | {100*ni:.1f}% | {100*agree:.1f}% |")

print()
print("| model | transform | n | raw flip rate | 95% CI | flips that match the no-image answer |")
print("|---|---|---:|---:|---:|---:|")
for name, _ in models:
    d = data[name]
    for t in ("mirror", "reorder", "shuffle"):
        rows = [d[i] for i in ids_all if t in d[i]["variants"]]
        flip = [r["variants"][t]["pred"] != r["variants"]["orig"]["pred"] for r in rows]
        lo, hi = ci(flip)
        flipped = [r for r, f in zip(rows, flip) if f]
        follow = sum(r["variants"][t]["pred"] == r["noimage"]["pred"] for r in flipped)
        print(f"| {name} | {t} | {len(rows)} | {100*sum(flip)/len(rows):.1f}% | {lo:.1f}-{hi:.1f}% | {follow}/{len(flipped)} ({100*follow/max(1,len(flipped)):.1f}%) |")

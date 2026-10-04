"""Export real MMSI-Bench examples (images + text + this run's predictions) for the slide.

Run on a Babel compute node:
    python export_example.py --results /data/user_data/vickytsa/idea3/results_<jobid> \
        --parquet /data/user_data/vickytsa/idea3/MMSI_Bench.parquet --out ~/idea3_assets
Then copy ~/idea3_assets back to your Mac.
"""
import argparse, io, json, os, glob
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", default=os.path.expanduser("~/idea3_assets"))
ap.add_argument("--k", type=int, default=3, help="how many candidate examples to export")
a = ap.parse_args()

recs = [json.loads(l) for l in open(os.path.join(a.results, "predictions.jsonl"))]

def score(r):
    """Prefer: mirror changed the answer, question has left/right, 2 images, short question."""
    v = r["variants"]
    if "mirror" not in v or len(r["variants"]["orig"]["question"]) > 400:
        return -1
    s = 0
    if v["mirror"]["pred"] != v["orig"]["pred"]: s += 3
    if r["has_direction"]: s += 2
    if r["n_images"] == 2: s += 2
    if v["orig"]["pred"] != r["answer"]: s += 1   # a wrong original answer makes a better story
    return s

cands = sorted([r for r in recs if score(r) > 0], key=score, reverse=True)[:a.k]
if not cands:
    raise SystemExit("no candidate found")

df = pd.read_parquet(a.parquet).set_index("id")
os.makedirs(a.out, exist_ok=True)
meta = []
for rank, r in enumerate(cands, 1):
    row = df.loc[r["id"]]
    d = os.path.join(a.out, f"example{rank}_id{r['id']}")
    os.makedirs(d, exist_ok=True)
    for i, im in enumerate(row["images"], 1):
        b = im["bytes"] if isinstance(im, dict) else im
        img = Image.open(io.BytesIO(b)).convert("RGB")
        img.thumbnail((900, 900))
        img.save(os.path.join(d, f"image{i}.png"))
        mirrored = ImageOps.mirror(img)
        header = 42
        pair = Image.new("RGB", (img.width * 2, img.height + header), "white")
        pair.paste(img, (0, header))
        pair.paste(mirrored, (img.width, header))
        draw = ImageDraw.Draw(pair)
        draw.text((12, 12), "Original", fill="black")
        draw.text((img.width + 12, 12), "Horizontally mirrored", fill="black")
        pair.save(os.path.join(d, f"image{i}_original_vs_mirrored.png"))
    info = dict(id=int(r["id"]), question_type=r["question_type"], gold=r["answer"],
                n_images=r["n_images"], question=r["variants"]["orig"]["question"],
                mirrored_question=r["variants"].get("mirror", {}).get("question"),
                shuffled_question=r["variants"].get("shuffle", {}).get("question"),
                preds={k: v["pred"] for k, v in r["variants"].items()},
                sc_preds=r["sc_preds"])
    json.dump(info, open(os.path.join(d, "meta.json"), "w"), indent=2)
    meta.append(info)
    print(f"[{rank}] id={r['id']} type={r['question_type']} gold={r['gold'] if 'gold' in r else r['answer']} "
          f"preds={info['preds']}\n    {info['question'][:200]}")
json.dump(meta, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
print("\nwrote", a.out)

"""Build the human-review packet required by EXPERIMENT_SPEC.md Section 8.

Example:
  python prepare_validity_check.py \
      --predictions results_123/predictions.jsonl \
      --parquet MMSI_Bench.parquet \
      --out results_123/validity_review
"""
import argparse
import csv
import io
import json
import os

import pandas as pd
from PIL import Image, ImageOps


def save_image(raw, path, mirror=False):
    if isinstance(raw, dict):
        raw = raw["bytes"]
    im = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    if mirror:
        im = ImageOps.mirror(im)
    im.thumbnail((900, 900))
    im.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mirror_n", type=int, default=20)
    ap.add_argument("--reorder_n", type=int, default=10)
    a = ap.parse_args()

    recs = [json.loads(line) for line in open(a.predictions)]
    by_id = pd.read_parquet(a.parquet).set_index("id")
    selected = []
    for variant, limit in [("mirror", a.mirror_n), ("reorder", a.reorder_n)]:
        candidates = [r for r in recs if variant in r["variants"]]
        selected.extend((r, variant) for r in candidates[:limit])

    os.makedirs(a.out, exist_ok=True)
    csv_rows, md = [], ["# Transformation validity review", ""]
    for rec, variant in selected:
        row = by_id.loc[rec["id"]]
        images = list(row["images"])
        saved_order = rec["variants"][variant].get("image_order")
        order = [i - 1 for i in saved_order] if saved_order else list(range(len(images)))
        if not saved_order and variant == "reorder":
            order.reverse()
        dname = f"id_{rec['id']}_{variant}"
        dpath = os.path.join(a.out, dname)
        os.makedirs(dpath, exist_ok=True)
        rels = []
        for shown_pos, original_idx in enumerate(order, 1):
            name = f"shown_{shown_pos}_label_Image_{original_idx + 1}.jpg"
            save_image(images[original_idx], os.path.join(dpath, name), mirror=variant == "mirror")
            rels.append(f"![Image {original_idx + 1}]({dname}/{name})")

        original_q = rec["variants"]["orig"]["question"].replace("\n", " ")
        transformed_q = rec["variants"][variant]["question"].replace("\n", " ")
        note = rec["variants"][variant].get("note", "").strip()
        md.extend([
            f"## ID {rec['id']} — {variant}", "",
            f"- Type: {rec['question_type']}",
            f"- Gold letter: {rec['answer']}",
            f"- Original: {original_q}",
            f"- Prompt note: {note or '(none saved in this run)'}",
            f"- Transformed: {transformed_q}", "",
            " ".join(rels), "",
            "Review: valid = ___; issue = ___", "",
        ])
        csv_rows.append(dict(id=rec["id"], variant=variant, valid="", issue=""))

    with open(os.path.join(a.out, "validity_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "variant", "valid", "issue"])
        w.writeheader(); w.writerows(csv_rows)
    with open(os.path.join(a.out, "REVIEW.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {len(selected)} review rows to {a.out}")


if __name__ == "__main__":
    main()

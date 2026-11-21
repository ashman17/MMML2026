"""Convert SAT (array/SAT-v2) parquet shards to the MMSI row schema for training.

Each output row has id, question ("<stem>\\nOptions: A: ..., B: ..."), answer letter,
images, question_type, plus mirror_answer: the answer letter after flipping every image
horizontally while keeping the text unchanged. It is set only when the stem has no
direction words and the left/right-swapped correct option is exactly one other option
(e.g. "rotated left" <-> "rotated right"); otherwise it is None. Options are shuffled
with a fixed seed because SAT lists the correct answer first in many rows. Rows with
more than four options are skipped.

    python sat_to_parquet.py --src /projects/bjjb/$USER/data/SAT-v2/data --split train \\
        --n 20000 --out /projects/bjjb/$USER/data/SAT_train20k.parquet
"""
import argparse
import glob
import os
import random

from mmsi_consistency import LETTERS, MIRROR_BLOCK, has_direction, mirror_text, stratified_sample

ID_OFFSET = {"train": 20_000_000, "val": 21_000_000}


def mirror_answer(stem, options, answer_index):
    if has_direction(stem) or MIRROR_BLOCK.search(stem):
        return None
    correct = options[answer_index]
    flipped = mirror_text(correct)
    if flipped == correct:
        return None
    matches = [i for i, option in enumerate(options) if option == flipped]
    return LETTERS[matches[0]] if len(matches) == 1 else None


def convert_sat_row(row, row_id, rng):
    options = [str(option).strip() for option in row["answers"]]
    if not 2 <= len(options) <= len(LETTERS):
        return None
    order = list(range(len(options)))
    rng.shuffle(order)
    options = [options[i] for i in order]
    answer_index = options.index(str(row["correct_answer"]).strip())
    stem = str(row["question"]).strip()
    images = [image["bytes"] if isinstance(image, dict) else image for image in row["images"]]
    return {
        "id": int(row_id),
        "question": stem + "\nOptions: " + ", ".join(
            f"{LETTERS[i]}: {option}" for i, option in enumerate(options)),
        "answer": LETTERS[answer_index],
        "images": [bytes(image) for image in images],
        "question_type": str(row["question_type"]),
        "mirror_answer": mirror_answer(stem, options, answer_index),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="folder with SAT-v2 <split>-*.parquet shards")
    parser.add_argument("--split", choices=sorted(ID_OFFSET), default="train")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pandas as pd
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(args.src, f"{args.split}-*.parquet")))
    meta = pd.concat([pq.read_table(shard, columns=["question_type"]).to_pandas()
                      .assign(shard=shard, row=lambda d: range(len(d))) for shard in shards],
                     ignore_index=True)
    meta["id"] = ID_OFFSET[args.split] + meta.index
    picked = stratified_sample(meta.to_dict("records"), args.n, args.seed)

    rng = random.Random(args.seed)
    rows, skipped = [], 0
    by_shard = {}
    for item in picked:
        by_shard.setdefault(item["shard"], []).append(item)
    for shard, items in sorted(by_shard.items()):
        table = pq.read_table(shard).take([item["row"] for item in items]).to_pylist()
        for item, source in zip(items, table):
            row = convert_sat_row(source, item["id"], rng)
            if row is None:
                skipped += 1
            else:
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    counts = pd.Series([row["question_type"] for row in rows]).value_counts()
    mirrorable = sum(row["mirror_answer"] is not None for row in rows)
    print(counts.to_string())
    print(f"wrote {len(rows)} rows to {args.out} (skipped {skipped} with >4 options; "
          f"{mirrorable} with a mirror answer)")


if __name__ == "__main__":
    main()

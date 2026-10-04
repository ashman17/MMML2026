"""Step 0: build the text-only cache, the explore/confirm split, and a dataset manifest.

Run: conda run -n 11777-project python 00_build_dataset.py
"""

from __future__ import annotations

import hashlib
from collections import Counter

from sklearn.model_selection import train_test_split

from common import (CACHE, CATEGORY_ORDER, OUTPUTS, PARQUET_PATH, SPLIT_PATH, TEXT_CACHE,
                    TSV_PATH, build_text_table, read_parquet_text, read_tsv_text, save_json)

SPLIT_SEED = 0

# Counts documented in "Agentic Project Notes/README.md" section 6.2 for the local TSV.
EXPECTED_CATEGORY_COUNTS = {
    "Multi-step": 198, "Pos: Obj-Obj": 94, "Pos: Cam-Cam": 93, "Pos: Cam-Obj": 86,
    "Pos: Obj-Reg": 85, "Pos: Cam-Reg": 83, "Pos: Reg-Reg": 81, "Motion: Obj": 76,
    "Motion: Cam": 74, "Attr: Appearance": 66, "Attr: Measure": 64,
}
EXPECTED_IMAGE_COUNTS = {2: 810, 3: 58, 4: 40, 5: 19, 6: 38, 7: 15, 8: 15, 9: 2, 10: 3}
EXPECTED_ANSWERS = {"A": 265, "B": 250, "C": 255, "D": 230}


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def main() -> None:
    tsv = read_tsv_text()
    parquet = read_parquet_text() if PARQUET_PATH.exists() else None
    if parquet is None:
        print(f"[WARN] {PARQUET_PATH} not found; traces/difficulty will be missing")
    df = build_text_table(tsv, parquet)

    results = [
        check("1000 rows with unique ids", len(df) == 1000 and df["id"].is_unique, f"rows={len(df)}"),
        check("every question parses into stem + A-D", df[[f"opt_{k}" for k in "ABCD"]].notna().all().all()),
        check("category counts match team notes",
              df["category_en"].value_counts().to_dict() == EXPECTED_CATEGORY_COUNTS),
        check("image-count distribution matches team notes",
              df["n_images"].value_counts().to_dict() == EXPECTED_IMAGE_COUNTS,
              str(dict(sorted(df["n_images"].value_counts().items())))),
        check("answer-letter counts match team notes",
              df["answer"].value_counts().to_dict() == EXPECTED_ANSWERS),
    ]
    if parquet is not None:
        results += [
            check("Parquet question text equals TSV text for every id",
                  df["question_matches_pq"].all(), f"mismatches={int((~df['question_matches_pq']).sum())}"),
            check("Parquet answer equals TSV answer for every id",
                  df["answer_matches_pq"].all(), f"mismatches={int((~df['answer_matches_pq']).sum())}"),
            check("every row has a non-empty reasoning trace",
                  df["thought"].fillna("").str.strip().ne("").all(),
                  f"missing={int(df['thought'].fillna('').str.strip().eq('').sum())}"),
        ]

    explore, confirm = train_test_split(
        df["id"].tolist(), test_size=0.5, random_state=SPLIT_SEED, stratify=df["category_en"])
    cat_of = dict(zip(df["id"], df["category_en"]))
    exp_counts, con_counts = Counter(cat_of[i] for i in explore), Counter(cat_of[i] for i in confirm)
    results.append(check(
        "split is 500/500 and each category differs by at most 1",
        len(explore) == len(confirm) == 500
        and all(abs(exp_counts[c] - con_counts[c]) <= 1 for c in CATEGORY_ORDER)))

    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TEXT_CACHE, index=False)
    save_json({"seed": SPLIT_SEED, "stratify": "category_en",
               "explore": sorted(explore), "confirm": sorted(confirm),
               "explore_counts": dict(exp_counts), "confirm_counts": dict(con_counts)}, SPLIT_PATH)

    manifest = {
        "dataset_source": "RunsenXu/MMSI-Bench",
        "tsv": {"path": str(TSV_PATH), "sha256": sha256(TSV_PATH)},
        "parquet": ({"path": str(PARQUET_PATH), "sha256": sha256(PARQUET_PATH),
                     "columns_used": [c for c in parquet.columns]} if parquet is not None else None),
        "rows": len(df),
        "category_counts": df["category_en"].value_counts().to_dict(),
        "image_count_distribution": {int(k): int(v) for k, v in sorted(df["n_images"].value_counts().items())},
        "answer_counts": df["answer"].value_counts().to_dict(),
        "difficulty_counts": (df["difficulty"].value_counts().to_dict() if "difficulty" in df else None),
        "split_path": str(SPLIT_PATH.relative_to(SPLIT_PATH.parents[1])),
        "split_seed": SPLIT_SEED,
        "checks_passed": all(results),
    }
    save_json(manifest, OUTPUTS / "00_manifest.json")
    print(f"\nwrote {TEXT_CACHE.name}, {SPLIT_PATH.name}, 00_manifest.json; all checks passed: {all(results)}")
    if "difficulty" in df:
        print("difficulty counts:", manifest["difficulty_counts"])


if __name__ == "__main__":
    main()

"""Shared paths, category names, question parsing, and data loaders for the
language-only MMSI-Bench analysis."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
# Local layout is <project>/MMML2026/analyses/language. A Colab bundle is unpacked
# directly at /content/mmsi_language, which has fewer parents.
PROJECT_ROOT = HERE.parents[2] if len(HERE.parents) > 2 else HERE
DATA_DIR = Path(os.environ.get("MMSI_DATA_DIR", PROJECT_ROOT / "MMSI-Bench" / ".analysis"))
TSV_PATH = DATA_DIR / "MMSI_bench.tsv"
PARQUET_PATH = DATA_DIR / "MMSI_Bench.parquet"

OUTPUTS = HERE / "outputs"
CACHE = HERE / "cache"
SPLIT_PATH = HERE / "splits" / "language_explore_confirm.json"
TEXT_CACHE = CACHE / "mmsi_text.parquet"

CATEGORY_EN = {
    "位置关系 cam-cam": "Pos: Cam-Cam",
    "位置关系 cam-object": "Pos: Cam-Obj",
    "位置关系 cam-scene": "Pos: Cam-Reg",
    "位置关系 object-object": "Pos: Obj-Obj",
    "位置关系 object-scene": "Pos: Obj-Reg",
    "位置关系 scene-scene": "Pos: Reg-Reg",
    "几何属性 measurement": "Attr: Measure",
    "几何属性 shape": "Attr: Appearance",
    "运动感知 camera": "Motion: Cam",
    "运动感知 object": "Motion: Obj",
    "多步推理": "Multi-step",
}
CATEGORY_ORDER = list(CATEGORY_EN.values())
POSITION_CATEGORIES = CATEGORY_ORDER[:6]
LETTERS = ("A", "B", "C", "D")

_OPTIONS_SPLIT = re.compile(r"\s*Options:\s*")
_OPTION_LABEL = re.compile(r"(?:^|(?<=[\s,.;]))([A-D])[:：]\s*")


def parse_question(question: str) -> tuple[str, dict[str, str]]:
    """Split an MMSI question string into its stem and an {A..D: text} dict.

    Raises ValueError unless exactly the labels A, B, C, D are found in order,
    so silent mis-parses cannot leak into option-level statistics.
    """
    parts = _OPTIONS_SPLIT.split(question, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("no 'Options:' marker")
    stem, tail = parts[0].strip(), parts[1]
    pieces = _OPTION_LABEL.split(tail)
    labels = pieces[1::2]
    if tuple(labels) != LETTERS:
        raise ValueError(f"option labels {labels} != A-D")
    texts = [t.strip().rstrip(",;.").strip() for t in pieces[2::2]]
    if any(not t for t in texts):
        raise ValueError("empty option text")
    return stem, dict(zip(LETTERS, texts))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_tsv_text(path: Path = TSV_PATH) -> pd.DataFrame:
    """Read the legacy TSV, keeping only text fields plus an image count."""
    df = pd.read_csv(path, sep="\t", usecols=["index", "image", "question", "answer", "category"])
    df["n_images"] = df["image"].str.count(r"',\s*'") + 1
    return df.drop(columns="image").rename(columns={"index": "id"})


def read_parquet_text(path: Path = PARQUET_PATH) -> pd.DataFrame:
    """Read the Hugging Face Parquet release without decoding the image column."""
    import pyarrow.parquet as pq

    wanted = ["id", "question_type", "question", "answer", "thought",
              "mean_normed_duration_seconds", "difficulty"]
    available = set(pq.read_schema(path).names)
    return pq.read_table(path, columns=[c for c in wanted if c in available]).to_pandas()


def build_text_table(tsv: pd.DataFrame, parquet: pd.DataFrame | None) -> pd.DataFrame:
    """Join the TSV and Parquet releases on id and add parsed stem/options."""
    df = tsv.copy()
    df["category_en"] = df["category"].map(CATEGORY_EN)
    if df["category_en"].isna().any():
        raise ValueError(f"unmapped categories: {df.loc[df.category_en.isna(), 'category'].unique()}")

    parsed = df["question"].map(parse_question)
    df["stem"] = parsed.str[0]
    for letter in LETTERS:
        df[f"opt_{letter}"] = parsed.map(lambda p, k=letter: p[1][k])

    if parquet is not None:
        pq_df = parquet.rename(columns={"question": "question_pq", "answer": "answer_pq"})
        df = df.merge(pq_df, on="id", how="left", validate="one_to_one")
        df["question_matches_pq"] = (
            df["question"].map(_normalize_ws) == df["question_pq"].fillna("").map(_normalize_ws)
        )
        df["answer_matches_pq"] = df["answer"] == df["answer_pq"]
    return df


def load_text(require_traces: bool = False) -> pd.DataFrame:
    """Load the cached text table built by 00_build_dataset.py."""
    if not TEXT_CACHE.exists():
        raise FileNotFoundError(f"{TEXT_CACHE} missing; run 00_build_dataset.py first")
    df = pd.read_parquet(TEXT_CACHE)
    if require_traces and ("thought" not in df or df["thought"].isna().all()):
        raise RuntimeError("text cache has no reasoning traces; rebuild with the Parquet release")
    return df


def options(row) -> dict[str, str]:
    """Options of a DataFrame row given as a Series, dict, or itertuples() namedtuple."""
    if hasattr(row, "_fields"):
        return {k: getattr(row, f"opt_{k}") for k in LETTERS}
    return {k: row[f"opt_{k}"] for k in LETTERS}


def load_split() -> dict[str, list[int]]:
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    return {"explore": split["explore"], "confirm": split["confirm"]}


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

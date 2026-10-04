"""Draw the 44-item language audit from the confirm half.

Run, from this directory:
  conda run -n 11777-project python 08_make_audit.py

Four questions from each official category. The draw prefers stems that are
not cosine >= 0.95 near-duplicates of each other, so one template is not
audited twice. Automatic tags are written to audit/key.csv for the later
comparison. The packets used for labeling do not contain those tags.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text
from plotting import expect

SEED = 1
PER_CATEGORY = 4
NEAR_DUP = 0.95
AUDIT = OUTPUTS.parent / "audit"


def component_ids(emb: np.ndarray, threshold: float) -> np.ndarray:
    similarity = emb @ emb.T
    n = similarity.shape[0]
    parent = np.arange(n)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    rows, cols = np.triu_indices(n, k=1)
    keep = similarity[rows, cols] >= threshold
    for left, right in zip(rows[keep], cols[keep]):
        union(int(left), int(right))
    return np.array([find(node) for node in range(n)])


def _stable_category_order(frame: pd.DataFrame) -> pd.DataFrame:
    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    frame = frame.copy()
    frame["_category_order"] = frame["category_en"].map(order)
    frame = frame.sort_values(["_category_order", "id"]).drop(columns="_category_order")
    frame["audit_item"] = np.arange(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def write_questions(frame: pd.DataFrame, path) -> None:
    blocks = ["# Audit packet, part 1: the question",
              "",
              "Label these 44 items before you open `packet_traces.md`.",
              "Write on `response_sheet.csv`. Leave `key.csv` closed.",
              "You are judging the wording. The photographs are not part of this pass.",
              ""]
    for row in frame.itertuples():
        blocks.append(f"## {row.audit_item:02d}. id {row.id} · {row.category_en} · {row.n_images} images")
        blocks.append("")
        blocks.append(str(row.stem).strip())
        blocks.append("")
        blocks.append("Options:")
        for letter in LETTERS:
            blocks.append(f"- {letter}. {getattr(row, f'opt_{letter}')}")
        blocks.append("")
    path.write_text("\n".join(blocks) + "\n")


def write_traces(frame: pd.DataFrame, path) -> None:
    blocks = ["# Audit packet, part 2: the reference trace",
              "",
              "Open this only after part 1 is filled in.",
              "The trace is a reference solution, not something the question stated.",
              ""]
    for row in frame.itertuples():
        blocks.append(f"## {row.audit_item:02d}. id {row.id} · {row.category_en}")
        blocks.append("")
        blocks.append(str(row.thought).strip())
        blocks.append("")
    path.write_text("\n".join(blocks) + "\n")


def response_sheet(frame: pd.DataFrame) -> pd.DataFrame:
    sheet = frame[["audit_item", "id", "category_en"]].copy()
    for column in ("frame", "anchor_image", "mirror", "reorder", "mentions",
                   "bridge_notes", "added_spatial", "added_viewpoint", "added_comparison",
                   "added_reasoning", "notes"):
        sheet[column] = ""
    return sheet


def main() -> None:
    df = load_text(require_traces=True)
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    bridge = pd.read_csv(OUTPUTS / "06_trace_questions.csv")
    additions = pd.read_csv(OUTPUTS / "07_addition_questions.csv")
    mentions = pd.read_csv(OUTPUTS / "02_mentions.csv")
    meta = json.loads((OUTPUTS / "03_stem_embeddings_meta.json").read_text())
    ids = df["id"].astype(int).tolist()
    assert tags["id"].astype(int).tolist() == ids
    assert [int(item) for item in meta["ids"]] == ids
    df = df.assign(split=tags["split"].to_numpy())
    confirm = df[df["split"] == "confirm"].copy()
    positions = np.flatnonzero(df["split"].to_numpy() == "confirm")
    components = component_ids(np.load(OUTPUTS / "03_stem_embeddings.npy"), NEAR_DUP)
    rng = np.random.default_rng(SEED)
    # Select inside the full-index component array, then restrict to confirm rows.
    selected_positions = []
    collision_ids = []
    for category in CATEGORY_ORDER:
        pool = positions[confirm["category_en"].to_numpy() == category]
        order = rng.permutation(pool)
        picked = []
        used = set()
        deferred = []
        for position in order:
            component = int(components[int(position)])
            if component in used:
                deferred.append(int(position))
                continue
            picked.append(int(position))
            used.add(component)
            if len(picked) == PER_CATEGORY:
                break
        if len(picked) < PER_CATEGORY:
            for position in deferred:
                picked.append(position)
                collision_ids.append(int(df.iloc[position]["id"]))
                if len(picked) == PER_CATEGORY:
                    break
        if len(picked) != PER_CATEGORY:
            raise RuntimeError(f"{category} contributed {len(picked)} items")
        selected_positions.extend(picked)

    frame = _stable_category_order(df.iloc[selected_positions])
    AUDIT.mkdir(parents=True, exist_ok=True)
    write_questions(frame, AUDIT / "packet_questions.md")
    write_traces(frame, AUDIT / "packet_traces.md")
    response_sheet(frame).to_csv(AUDIT / "response_sheet.csv", index=False)

    key = tags.merge(frame[["id", "audit_item"]], on="id", how="inner")
    key = key.merge(bridge[["id", "n_bridging", "bridging_heads"]], on="id", how="left")
    add_cols = ["id", "add_spatial", "add_viewpoint", "add_comparison", "add_reasoning",
                "spatial_pairs", "viewpoint_images", "viewpoint_complements", "comparison_words",
                "reasoning_cues"]
    key = key.merge(additions[add_cols], on="id", how="left")
    key_cols = ["audit_item", "id", "category_en", "answer_space", "frame", "anchor_image",
                "mirror_level", "mirror_reasons", "reorder_level", "reorder_reasons",
                "n_bridging", "bridging_heads", *add_cols[1:]]
    key[key_cols].sort_values("audit_item").to_csv(AUDIT / "key.csv", index=False)
    mentions.merge(frame[["id", "audit_item"]], on="id", how="inner").sort_values(
        ["audit_item", "source", "text"]).to_csv(AUDIT / "key_mentions.csv", index=False)

    manifest = {
        "seed": SEED,
        "per_category": PER_CATEGORY,
        "split": "confirm",
        "near_duplicate_threshold": NEAR_DUP,
        "n": int(len(frame)),
        "ids": frame["id"].astype(int).tolist(),
        "template_collisions": collision_ids,
    }
    (AUDIT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    counts = frame["category_en"].value_counts().reindex(CATEGORY_ORDER)
    id_to_pos = {int(i): pos for pos, i in enumerate(ids)}
    frame_components = [int(components[id_to_pos[int(i)]]) for i in frame["id"]]
    shared = len(frame_components) - len(set(frame_components))
    print(f"Audit items: {len(frame)}")
    print(counts.to_string())
    print("Template collisions:", collision_ids or "none")
    print("Items:")
    for row in frame.itertuples():
        print(f"  {row.audit_item:02d} {row.id:4d} {row.category_en:20} {str(row.stem).strip()[:90]}")
    print()
    expect("44 items, four from each category, all on the confirm half",
           len(frame) == 44 and int((counts == PER_CATEGORY).all()) == 1
           and set(frame["split"]) == {"confirm"},
           f"n={len(frame)}")
    expect("No two items share a cosine >= 0.95 template", shared == 0 and not collision_ids,
           f"shared={shared}, fallback={collision_ids or 'none'}")


if __name__ == "__main__":
    main()

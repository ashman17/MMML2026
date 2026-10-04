"""Step G: addition profiles, category baseline, and wording-group summary.

Run, from this directory:
  conda run -n 11777-project python 07_additions.py | tee outputs/07_run.log

Traces are never used as shortcut features. Bridging nouns are copied from
Step F. The MiniLM matrix is the one saved in Step C; nothing is re-embedded.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import spacy

from additions import addition_record
from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text
from plotting import expect, setup

FLAGS = ("add_spatial", "add_viewpoint", "add_comparison", "add_reasoning")
FLAG_LABELS = {
    "add_spatial": "Spatial relation",
    "add_viewpoint": "Viewpoint change",
    "add_comparison": "Comparison",
    "add_reasoning": "Reasoning step",
}
NEAR_DUP = 0.95
K_NEIGHBORS = 5
N_PERMUTATIONS = 1000
SEED = 0
STEP_F_BRIDGING_QUESTIONS = 842


def question_text(row) -> str:
    options = " ".join(str(getattr(row, f"opt_{letter}")) for letter in LETTERS)
    return f"{row.stem}\n{options}"


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _bridging_lemmas(value) -> set[str]:
    text = _clean(value)
    return set(text.split()) if text and text != "nan" else set()


def _join(items, sep: str = " | ") -> str:
    return sep.join(str(item) for item in items)


def _anchor(value):
    if pd.isna(value):
        return None
    return value


def dominant(rates: dict[str, float]) -> tuple[str, float]:
    """Highest rate among the four addition types. Ties and all-zero stay labeled."""
    best = max(rates.values()) if rates else 0.0
    leaders = [name for name, rate in rates.items() if rate == best]
    if best == 0:
        return "none", 0.0
    if len(leaders) > 1:
        return "tie", float(best)
    return leaders[0], float(best)


def question_dominant(flags: dict[str, bool]) -> str:
    on = [name.removeprefix("add_") for name in FLAGS if flags[name]]
    if not on:
        return "none"
    if len(on) == 1:
        return on[0]
    return "multiple"


def analyze(df: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    questions = [question_text(row) for row in df.itertuples()]
    traces = [str(text) for text in df["thought"].tolist()]
    docs = nlp.pipe(questions + traces, batch_size=64)
    question_docs = [next(docs) for _ in questions]
    rows = []
    for row, question_doc, trace_doc, n_bridging, bridging_heads in zip(
            df.itertuples(), question_docs, docs, bridge["n_bridging"], bridge["bridging_heads"]):
        record = addition_record(trace_doc, question_doc, anchor=_anchor(row.anchor_image),
                                 n_images=int(row.n_images))
        lemmas = _bridging_lemmas(bridging_heads)
        landmarks = [lemma for _, lemma in record["spatial_pairs"]]
        bridging_landmarks = [lemma for lemma in landmarks if lemma in lemmas]
        flags = {name: record[name] for name in FLAGS}
        rows.append({
            "id": int(row.id),
            "category_en": row.category_en,
            "split": row.split,
            "frame": row.frame,
            "anchor_image": _anchor(row.anchor_image),
            "n_bridging": int(n_bridging),
            "n_spatial_landmarks": len(landmarks),
            "n_spatial_bridging_landmarks": len(bridging_landmarks),
            "dominant_addition": question_dominant(flags),
            **flags,
            "spatial_pairs": _join(f"{rel}={lemma}" for rel, lemma in record["spatial_pairs"]),
            "viewpoint_images": _join(record["viewpoint_images"], " "),
            "viewpoint_complements": _join(record["viewpoint_complements"]),
            "comparison_words": _join(record["comparison_words"]),
            "reasoning_cues": _join(record["reasoning_cues"]),
        })
    return pd.DataFrame(rows)


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("category_en")
    out = grouped[list(FLAGS)].mean()
    out["n"] = grouped.size()
    totals = grouped["n_spatial_landmarks"].sum()
    bridged = grouped["n_spatial_bridging_landmarks"].sum()
    out["bridging_landmark_share"] = bridged / totals.replace(0, np.nan)
    labels = []
    purity = []
    for category in out.index:
        rates = {name.removeprefix("add_"): float(out.loc[category, name]) for name in FLAGS}
        label, rate = dominant(rates)
        labels.append(label)
        purity.append(rate)
    out["dominant"] = labels
    out["purity"] = purity
    return out.reindex(CATEGORY_ORDER)


def near_duplicate_components(emb: np.ndarray, threshold: float) -> tuple[list[list[int]], int]:
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
    groups: dict[int, list[int]] = {}
    for node in range(n):
        groups.setdefault(find(node), []).append(node)
    components = [members for members in groups.values() if len(members) > 1]
    components.sort(key=lambda members: (-len(members), min(members)))
    return components, int(keep.sum())


def group_table(frame: pd.DataFrame, components: list[list[int]]) -> pd.DataFrame:
    rows = []
    for index, members in enumerate(components):
        part = frame.iloc[members]
        rates = {name.removeprefix("add_"): float(part[name].mean()) for name in FLAGS}
        label, purity = dominant(rates)
        present = [category for category in CATEGORY_ORDER if (part["category_en"] == category).any()]
        rows.append({
            "component_id": index,
            "n": len(members),
            "n_categories": len(present),
            "categories": " | ".join(present),
            "n_explore": int((part["split"] == "explore").sum()),
            "n_confirm": int((part["split"] == "confirm").sum()),
            **{name: rates[name.removeprefix("add_")] for name in FLAGS},
            "dominant": label,
            "purity": purity,
        })
    return pd.DataFrame(rows)


def neighbor_table(emb: np.ndarray, frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    similarity = emb @ emb.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :K_NEIGHBORS]
    labels = frame["category_en"].to_numpy()
    splits = frame["split"].to_numpy()
    same_category = labels[neighbors] == labels[:, None]
    rows = []
    for split in ("explore", "confirm"):
        source = (splits == split)[:, None]
        for flag in FLAGS:
            values = frame[flag].to_numpy(dtype=bool)
            agree = values[neighbors] == values[:, None]
            same_rate, n_same = _masked_mean(agree, source & same_category)
            cross_rate, n_cross = _masked_mean(agree, source & ~same_category)
            null = np.empty(N_PERMUTATIONS)
            cross_mask = source & ~same_category
            for draw in range(N_PERMUTATIONS):
                shuffled = rng.permutation(values)
                shuffled_agree = shuffled[neighbors] == shuffled[:, None]
                null[draw] = _masked_mean(shuffled_agree, cross_mask)[0]
            observed = cross_rate
            p_value = (1 + np.sum(null >= observed)) / (1 + N_PERMUTATIONS)
            rows.append({
                "flag": flag.removeprefix("add_"),
                "split": split,
                "same_category_agreement": same_rate,
                "cross_category_agreement": cross_rate,
                "cross_null_mean": float(null.mean()),
                "cross_null_p95": float(np.quantile(null, 0.95)),
                "cross_p_value": float(p_value),
                "n_same": n_same,
                "n_cross": n_cross,
            })
    return pd.DataFrame(rows)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    chosen = values[mask]
    if chosen.size == 0:
        return float("nan"), 0
    return float(chosen.mean()), int(chosen.size)


def plot(summary: pd.DataFrame, neighbors: pd.DataFrame, path) -> None:
    plt, sns = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.35, 1]})
    heat = 100 * summary[list(FLAGS)]
    heat.columns = ["spatial", "viewpoint", "comparison", "reasoning"]
    short = [category.replace("Pos: ", "").replace("Attr: ", "").replace("Motion: ", "Mot-")
             for category in CATEGORY_ORDER]
    sns.heatmap(heat, ax=axes[0], annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                cbar=False, annot_kws={"fontsize": 6}, yticklabels=short)
    axes[0].set(xlabel="", ylabel="", title="Added claims by category (%)")
    axes[0].tick_params(labelsize=6)
    confirm = neighbors[neighbors["split"] == "confirm"].set_index("flag").loc[
        [name.removeprefix("add_") for name in FLAGS]]
    x = np.arange(len(FLAGS))
    width = 0.36
    axes[1].bar(x - width / 2, confirm["same_category_agreement"], width, color="#4c72b0", label="same category")
    axes[1].bar(x + width / 2, confirm["cross_category_agreement"], width, color="#dd8452", label="different category")
    axes[1].set(xticks=x, xticklabels=["spatial", "viewpoint", "comparison", "reasoning"],
                ylim=(0, 1), ylabel="5-NN agreement", title="Confirm-half neighbor agreement")
    axes[1].tick_params(labelsize=6)
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df = load_text(require_traces=True)
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    bridge = pd.read_csv(OUTPUTS / "06_trace_questions.csv")
    meta = json.loads((OUTPUTS / "03_stem_embeddings_meta.json").read_text())
    ids = df["id"].astype(int).tolist()
    assert tags["id"].astype(int).tolist() == ids, "01_tags.csv must follow the cache order"
    assert bridge["id"].astype(int).tolist() == ids, "06_trace_questions.csv must follow the cache order"
    assert [int(item) for item in meta["ids"]] == ids, "03_stem_embeddings.npy must follow the cache order"
    df = df.assign(split=tags["split"].to_numpy(), frame=tags["frame"].to_numpy(),
                   anchor_image=tags["anchor_image"].to_numpy())

    frame = analyze(df, bridge)
    frame.to_csv(OUTPUTS / "07_addition_questions.csv", index=False)
    summary = category_summary(frame)
    summary.round(3).to_csv(OUTPUTS / "07_addition_by_category.csv")

    embeddings = np.load(OUTPUTS / "03_stem_embeddings.npy")
    components, n_pairs = near_duplicate_components(embeddings, NEAR_DUP)
    groups = group_table(frame, components)
    groups.round(3).to_csv(OUTPUTS / "07_addition_groups.csv", index=False)
    neighbors = neighbor_table(embeddings, frame, np.random.default_rng(SEED))
    neighbors.round(4).to_csv(OUTPUTS / "07_addition_neighbors.csv", index=False)
    plot(summary, neighbors, OUTPUTS / "07_additions.png")

    rates = {name: float(frame[name].mean()) for name in FLAGS}
    by_split = frame.groupby("split")[list(FLAGS)].mean()
    landmark_total = int(frame["n_spatial_landmarks"].sum())
    landmark_bridge = int(frame["n_spatial_bridging_landmarks"].sum())
    spatial_questions = int(frame["add_spatial"].sum())
    spatial_with_bridge = int(((frame["add_spatial"]) & (frame["n_spatial_bridging_landmarks"] > 0)).sum())
    mixed = groups[groups["n_categories"] > 1]
    confirm = neighbors[neighbors["split"] == "confirm"]
    print(f"Questions: {len(frame)}")
    print("Addition rates:", ", ".join(f"{FLAG_LABELS[name]} {rates[name]:.1%}" for name in FLAGS))
    print("By split:")
    print(by_split.rename(columns=lambda name: name.removeprefix("add_")).round(3).to_string())
    print("\nBy category:")
    print(summary[["n", *FLAGS, "dominant", "purity"]].rename(
        columns=lambda name: name.removeprefix("add_") if name in FLAGS else name).round(3).to_string())
    print(f"\nSpatial landmarks also in the copied bridging list: {landmark_bridge}/{landmark_total} landmarks; "
          f"{spatial_with_bridge}/{spatial_questions} spatial questions")
    print(f"Near-duplicate components (cosine >= {NEAR_DUP}): {n_pairs} pairs, {len(components)} components, "
          f"{len(mixed)} mixed-category")
    for row in mixed.itertuples():
        print(f"  mixed n={row.n} [{row.categories}] dominant={row.dominant} purity={row.purity:.2f}")
    print("\nNeighbor agreement:")
    print(neighbors.round(3).to_string(index=False))
    print()

    bridging_questions = int((frame["n_bridging"] > 0).sum())
    expect("Copied bridging rate matches Step F (842/1000)", bridging_questions == STEP_F_BRIDGING_QUESTIONS,
           f"{bridging_questions}/1000")
    expect("Cosine >= 0.95 rebuilds 727 pairs, 64 components, and 4 mixed components",
           n_pairs == 727 and len(components) == 64 and len(mixed) == 4,
           f"{n_pairs} pairs, {len(components)} components, {len(mixed)} mixed")
    for name in FLAGS:
        gap = abs(float(by_split.loc["explore", name]) - float(by_split.loc["confirm", name]))
        expect(f"{FLAG_LABELS[name]} rates differ by at most 8 points across the split", gap <= 0.08,
               f"{by_split.loc['explore', name]:.1%} vs {by_split.loc['confirm', name]:.1%}")
    passing = confirm.loc[confirm["cross_p_value"] < 0.05, "flag"].tolist()
    expect("Confirm cross-category neighbors beat the label permutation for at least one addition type",
           len(passing) > 0, "passing: " + (", ".join(passing) if passing else "none"))


if __name__ == "__main__":
    main()

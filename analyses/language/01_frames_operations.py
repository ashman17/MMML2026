"""Step A: reference frames, answer spaces, operations, and transformation validity.

Run: conda run -n 11777-project python 01_frames_operations.py
"""

from __future__ import annotations

import pandas as pd

from common import CATEGORY_ORDER, OUTPUTS, POSITION_CATEGORIES, load_split, load_text, options
from lexicon import ANSWER_SPACES, tag_question
from plotting import expect, setup

FRAMES = ("camera", "agent", "object_front", "compass", "axes", "object_unspecified",
          "viewer_implicit", "none")
CONDITION_TAGS = ("has_premise", "cap_overlap", "cap_same_place", "img_sequence", "img_index",
                  "img_ordinal", "hedge", "abstain_option")
OPERATIONS = ("op_direction", "op_metric", "op_count", "op_perspective_taking", "op_absolute_frame",
              "op_cross_view_explicit", "op_premise_chain", "op_camera_motion", "op_object_motion",
              "op_temporal_order", "op_route", "op_statement_check")
MIRROR_LEVELS = ("no_change", "swap_options", "rewrite_stem", "unsafe")
REORDER_LEVELS = ("safe", "label_dependent", "unsafe")
LEVEL_COLORS = {"no_change": "#2c7bb6", "safe": "#2c7bb6", "swap_options": "#abd9e9",
                "label_dependent": "#abd9e9", "rewrite_stem": "#fdae61", "unsafe": "#d7191c"}


def tag_all(df: pd.DataFrame) -> pd.DataFrame:
    tags = pd.DataFrame([tag_question(r.stem, options(r), r.n_images) for r in df.itertuples()])
    return pd.concat([df[["id", "category_en", "answer", "n_images"]].reset_index(drop=True), tags], axis=1)


def shares(tags: pd.DataFrame, column: str, levels) -> pd.DataFrame:
    table = pd.crosstab(tags["category_en"], tags[column], normalize="index")
    return table.reindex(index=CATEGORY_ORDER, columns=list(levels), fill_value=0.0)


def rates(tags: pd.DataFrame, columns) -> pd.DataFrame:
    return tags.groupby("category_en")[list(columns)].mean().reindex(CATEGORY_ORDER)


def heatmap(tags: pd.DataFrame, path) -> None:
    plt, sns = setup()
    blocks = [("Answer space (options)", shares(tags, "answer_space", ANSWER_SPACES)),
              ("Frame anchor (stem)", shares(tags, "frame", FRAMES)),
              ("Stated conditions", rates(tags, CONDITION_TAGS)),
              ("Operations", rates(tags, OPERATIONS))]
    widths = [b[1].shape[1] for b in blocks]
    fig, axes = plt.subplots(1, 4, figsize=(0.36 * sum(widths) + 2.2, 3.6),
                             gridspec_kw={"width_ratios": widths, "wspace": 0.06})
    for i, (ax, (title, table)) in enumerate(zip(axes, blocks)):
        labels = [c.replace("op_", "").replace("cap_", "").replace("_", " ") for c in table.columns]
        sns.heatmap(100 * table, ax=ax, vmin=0, vmax=100, cmap="Blues", cbar=i == 3, annot=True,
                    fmt=".0f", annot_kws={"size": 5.5}, xticklabels=labels, yticklabels=i == 0,
                    cbar_kws={"label": "% of questions"} if i == 3 else None, linewidths=0.3,
                    linecolor="white")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=90, labelsize=6.5)
        ax.tick_params(axis="y", labelsize=7)
    fig.savefig(path)
    plt.close(fig)


def eligibility_plot(tags: pd.DataFrame, path) -> None:
    plt, _ = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, (column, levels, title) in zip(axes, [("mirror_level", MIRROR_LEVELS, "Horizontal mirror"),
                                                  ("reorder_level", REORDER_LEVELS, "Image reordering")]):
        table = 100 * shares(tags, column, levels)
        table.iloc[::-1].plot.barh(stacked=True, ax=ax, width=0.8,
                                   color=[LEVEL_COLORS[level] for level in levels])
        ax.set_title(title)
        ax.set_xlabel("% of questions")
        ax.set_ylabel("")
        ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=len(levels), frameon=False)
        ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def operation_profiles(tags: pd.DataFrame) -> pd.DataFrame:
    """Operation signatures shared by several official categories."""
    sig = tags[list(OPERATIONS)].apply(
        lambda r: "+".join(c.replace("op_", "") for c in OPERATIONS if r[c]) or "(none)", axis=1)
    table = pd.crosstab(sig, tags["category_en"]).reindex(columns=CATEGORY_ORDER, fill_value=0)
    table["n"] = table.sum(axis=1)
    table["n_categories"] = (table[CATEGORY_ORDER] > 0).sum(axis=1)
    return table.sort_values("n", ascending=False)


def main() -> None:
    df = load_text()
    split = load_split()
    tags = tag_all(df)
    tags["split"] = tags["id"].map({**{i: "explore" for i in split["explore"]},
                                    **{i: "confirm" for i in split["confirm"]}})
    tags.to_csv(OUTPUTS / "01_tags.csv", index=False)

    space = shares(tags, "answer_space", ANSWER_SPACES)
    frame = shares(tags, "frame", FRAMES)
    summary = pd.concat([space.add_prefix("space_"), frame.add_prefix("frame_"),
                         rates(tags, CONDITION_TAGS), rates(tags, OPERATIONS),
                         shares(tags, "mirror_level", MIRROR_LEVELS).add_prefix("mirror_"),
                         shares(tags, "reorder_level", REORDER_LEVELS).add_prefix("reorder_")], axis=1)
    summary.insert(0, "n", tags.groupby("category_en").size().reindex(CATEGORY_ORDER))
    summary.round(3).to_csv(OUTPUTS / "01_tag_rates_by_category.csv")

    anchored = tags[tags["anchor_image"].notna()].copy()
    anchored["anchor"] = anchored["anchor_image"].astype(int).clip(upper=3).map({1: "1", 2: "2", 3: "3+"})
    anchor_table = pd.crosstab(anchored["category_en"], anchored["anchor"], margins=True)
    anchor_table.to_csv(OUTPUTS / "01_anchor_image_by_category.csv")

    profiles = operation_profiles(tags)
    profiles.to_csv(OUTPUTS / "01_operation_signatures.csv")

    heatmap(tags, OUTPUTS / "01_heatmap.png")
    eligibility_plot(tags, OUTPUTS / "01_transform_eligibility.png")

    pd.set_option("display.width", 200)
    print("Answer space (% of category):\n", (100 * space).round(0).astype(int).to_string(), "\n")
    print("Frame anchor (% of category):\n", (100 * frame).round(0).astype(int).to_string(), "\n")
    print("Anchor image among anchored questions:\n", anchor_table.to_string(), "\n")
    print("Top operation signatures:\n", profiles.head(12)[["n", "n_categories"]].to_string(), "\n")

    pos = tags[tags["category_en"].isin(POSITION_CATEGORIES)]
    cam_targets = anchored[anchored["category_en"].isin(["Pos: Cam-Obj", "Pos: Cam-Reg"])]
    non_first = 1 - cam_targets["anchor_is_first"].astype(bool).mean()
    mirror_any_change = (tags["mirror_level"].isin(["rewrite_stem", "unsafe"])).mean()
    motion = tags[tags["category_en"].str.startswith("Motion")]
    print("Headline numbers")
    print(f"  position questions with an explicit or camera frame: "
          f"{(~pos['frame'].isin(['viewer_implicit', 'none', 'object_unspecified'])).mean():.1%}")
    print(f"  position questions whose relative frame is unstated (object_unspecified): "
          f"{(pos['frame'] == 'object_unspecified').mean():.1%}")
    print(f"  Cam-Obj/Cam-Reg anchored questions NOT anchored at image 1: {non_first:.1%} "
          f"(n={len(cam_targets)})")
    print(f"  all anchored questions NOT anchored at image 1: "
          f"{1 - anchored['anchor_is_first'].astype(bool).mean():.1%} (n={len(anchored)})")
    print(f"  questions needing a stem rewrite or unsafe to mirror: {mirror_any_change:.1%}")
    print(f"  questions unsafe to reorder: {(tags['reorder_level'] == 'unsafe').mean():.1%}")
    print(f"  questions with any stated premise: {tags['has_premise'].mean():.1%}")
    print(f"  questions with an abstention option: {tags['abstain_option'].mean():.1%}")
    print(f"  operation signatures spanning >= 3 official categories: "
          f"{(profiles['n_categories'] >= 3).sum()} covering {profiles.loc[profiles['n_categories'] >= 3, 'n'].sum()} questions\n")

    expect("Measurement is mostly a metric answer space (>= 80%)", space.loc["Attr: Measure", "metric"] >= 0.8,
           f"{space.loc['Attr: Measure', 'metric']:.0%}")
    expect("Cam-Obj and Cam-Reg are mostly egocentric directions (>= 70%)",
           space.loc[["Pos: Cam-Obj", "Pos: Cam-Reg"], "ego_direction"].min() >= 0.7,
           f"{space.loc[['Pos: Cam-Obj', 'Pos: Cam-Reg'], 'ego_direction'].round(2).tolist()}")
    expect("Cam-Cam has an axis/sign block of 15-35%", 0.15 <= space.loc["Pos: Cam-Cam", "axis_sign"] <= 0.35,
           f"{space.loc['Pos: Cam-Cam', 'axis_sign']:.0%}")
    expect("Cam-Obj and Cam-Reg are camera-anchored (>= 70%)",
           frame.loc[["Pos: Cam-Obj", "Pos: Cam-Reg"], "camera"].min() >= 0.7,
           f"{frame.loc[['Pos: Cam-Obj', 'Pos: Cam-Reg'], 'camera'].round(2).tolist()}")
    expect("Obj-Obj, Obj-Reg, Reg-Reg use compass frames in >= 40%",
           frame.loc[["Pos: Obj-Obj", "Pos: Obj-Reg", "Pos: Reg-Reg"], "compass"].min() >= 0.4,
           f"{frame.loc[['Pos: Obj-Obj', 'Pos: Obj-Reg', 'Pos: Reg-Reg'], 'compass'].round(2).tolist()}")
    expect("Most Cam-Obj/Cam-Reg anchors are not image 1 (>= 50%)", non_first >= 0.5, f"{non_first:.0%}")
    expect("Mirroring needs a rewrite or is unsafe for >= 40% of questions", mirror_any_change >= 0.4,
           f"{mirror_any_change:.0%}")
    expect("Reordering is unsafe for >= 70% of motion questions",
           (motion["reorder_level"] == "unsafe").mean() >= 0.7,
           f"{(motion['reorder_level'] == 'unsafe').mean():.0%}")
    uncovered = pos[pos["frame"].isin(["viewer_implicit", "none"])]
    expect("Lexicon leaves < 15% of position questions without a frame", len(uncovered) / len(pos) < 0.15,
           f"{len(uncovered)}/{len(pos)}")

    explore_uncovered = uncovered[uncovered["split"] == "explore"].merge(df[["id", "stem"]], on="id")
    print("\nExplore-half position questions without a frame (lexicon coverage gaps):")
    for r in explore_uncovered.head(15).itertuples():
        print(f"  [{r.id}] {r.category_en} | {r.answer_space} | {r.stem[:150]}")


if __name__ == "__main__":
    main()

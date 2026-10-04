"""Step B: how questions describe the entities a model must ground.

Run: conda run -n 11777-project python 02_entities.py
Needs outputs/01_tags.csv (answer spaces) and the LVIS list (see README).
Stems are always parsed; options only when they name entities (answer space
"entity" or "statement"), since direction options contain no referents.
"""

from __future__ import annotations

import pandas as pd
import spacy

from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text
from entities import ENTITY_TYPES, MODES, head_words, load_lvis_names, mentions_from_doc
from plotting import expect, setup

DISAMBIGUATING = ["relational", "image_anchored", "in_image_position"]
VOCABS = ["in_coco", "in_lvis", "in_ade20k", "in_lvis_or_ade20k"]
REGION_CATEGORIES = ["Pos: Cam-Reg", "Pos: Obj-Reg", "Pos: Reg-Reg"]


def extract(df: pd.DataFrame, tags: pd.DataFrame, lvis: set[str]) -> pd.DataFrame:
    space = dict(zip(tags["id"], tags["answer_space"]))
    jobs = [(r.id, r.category_en, "stem", r.stem) for r in df.itertuples()]
    jobs += [(r.id, r.category_en, f"option_{k}", getattr(r, f"opt_{k}")) for r in df.itertuples()
             if space[r.id] in ("entity", "statement") for k in LETTERS]
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    lvis_heads = head_words(lvis)
    rows = []
    for (qid, cat, source, _), doc in zip(jobs, nlp.pipe([j[3] for j in jobs], batch_size=64)):
        rows += [{"id": qid, "category_en": cat, "source": source, **m}
                 for m in mentions_from_doc(doc, lvis, lvis_heads)]
    mentions = pd.DataFrame(rows)
    mentions["disambiguated"] = mentions[DISAMBIGUATING].any(axis=1)
    is_object = mentions["type"] == "object"
    mentions["in_lvis_or_ade20k"] = None
    mentions.loc[is_object, "in_lvis_or_ade20k"] = (mentions.loc[is_object, "in_lvis"].astype(bool)
                                                   | mentions.loc[is_object, "in_ade20k"].astype(bool))
    mentions = mentions.merge(tags[["id", "split"]], on="id", validate="many_to_one")
    return mentions


def summarize(mentions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    ent = mentions[mentions["type"].isin(ENTITY_TYPES)]
    grounded = ent[ent["type"] != "camera"]
    objects = ent[ent["type"] == "object"]
    out = pd.DataFrame({"n_questions": df.groupby("category_en").size(),
                        "entity_mentions": ent.groupby("category_en").size(),
                        "quantity_mentions": mentions[mentions["type"] == "quantity"].groupby("category_en").size()})
    out["mentions_per_question"] = out["entity_mentions"] / out["n_questions"]
    out = out.join(pd.crosstab(ent["category_en"], ent["type"], normalize="index")
                   .reindex(columns=ENTITY_TYPES, fill_value=0).add_prefix("type_"))
    out = out.join(grounded.groupby("category_en")[MODES + ["disambiguated", "bare"]].mean().add_prefix("mode_"))
    out = out.join(objects.groupby("category_en")[VOCABS]
                   .agg(lambda s: s.astype(float).mean()).add_prefix("object_"))
    return out.reindex(CATEGORY_ORDER).fillna(0)


def plot(summary: pd.DataFrame, mentions: pd.DataFrame, path) -> None:
    plt, sns = setup()
    ent = mentions[mentions["type"].isin(ENTITY_TYPES)]
    n_grounded = ent[ent["type"] != "camera"].groupby("category_en").size().reindex(CATEGORY_ORDER, fill_value=0)
    n_objects = ent[ent["type"] == "object"].groupby("category_en").size().reindex(CATEGORY_ORDER, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})

    types = 100 * summary[[f"type_{t}" for t in ENTITY_TYPES]].iloc[::-1]
    types.columns = [t.replace("region_", "region: ") for t in ENTITY_TYPES]
    types.plot.barh(stacked=True, ax=axes[0], width=0.8, colormap="tab10")
    axes[0].set(title="Entity mention type", xlabel="% of entity mentions", ylabel="", xlim=(0, 100))
    axes[0].legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)

    cells = pd.concat([100 * summary[[f"mode_{m}" for m in MODES + ["disambiguated"]]],
                       100 * summary[[f"object_{v}" for v in VOCABS]]], axis=1)
    cells.columns = ["attri-\nbute", "rela-\ntional", "image\nanchor", "in-image\nposition", "any\nof 3",
                     "COCO\n80", "LVIS\n1203", "ADE20K\n150", "LVIS or\nADE20K"]
    sns.heatmap(cells, ax=axes[1], annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100, cbar=False,
                annot_kws={"fontsize": 5.5}, linewidths=0.4, linecolor="white")
    axes[1].axvline(len(MODES) + 1, color="black", lw=1)
    axes[1].set_yticks([i + 0.5 for i in range(len(cells))],
                       [f"{g} / {o}" for g, o in zip(n_grounded, n_objects)], fontsize=5.5, rotation=0)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel("grounded / object mentions", fontsize=5.5)
    axes[1].tick_params(axis="x", labelsize=5.5, rotation=0)
    axes[1].set_title("Descriptor mode (% grounded) | object head in vocabulary (% objects)", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def clean(texts: pd.Series) -> pd.Series:
    return texts.str.lower().str.replace(r"^(the|a|an|this|that) ", "", regex=True)


def main() -> None:
    df = load_text()
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    lvis = load_lvis_names()
    mentions = extract(df, tags, lvis)
    mentions.to_csv(OUTPUTS / "02_mentions.csv", index=False)
    summary = summarize(mentions, df)
    summary.round(3).to_csv(OUTPUTS / "02_entity_summary_by_category.csv")
    plot(summary, mentions, OUTPUTS / "02_entity_types.png")

    ent = mentions[mentions["type"].isin(ENTITY_TYPES)]
    grounded = ent[ent["type"] != "camera"]
    regions = ent[ent["type"].str.startswith("region")]
    functional = regions[regions["type"] == "region_functional"]
    objects = ent[ent["type"] == "object"]
    in_lvis = objects["in_lvis"].astype(float)
    in_coco = objects["in_coco"].astype(float)
    in_ade = objects["in_ade20k"].astype(float)
    in_either = objects["in_lvis_or_ade20k"].astype(float)

    pd.set_option("display.width", 260)
    cols = ["mentions_per_question", "type_object", "type_region_room", "type_region_functional", "type_part",
            "mode_attribute", "mode_relational", "mode_image_anchored", "mode_in_image_position",
            "mode_bare", "object_in_coco", "object_in_lvis", "object_in_ade20k", "object_in_lvis_or_ade20k"]
    print(summary[cols].round(2).to_string(), "\n")
    print(f"LVIS names loaded: {len(lvis)}; mentions: {len(mentions)} ({len(ent)} entity, "
          f"{int((mentions['type'] == 'quantity').sum())} quantity, "
          f"{int((mentions['source'] != 'stem').sum())} from options)")
    print(f"  questions with >= 1 grounded mention: {grounded['id'].nunique()} / {len(df)}")
    print(f"  region share of entity mentions: {ent['type'].str.startswith('region').mean():.1%}")
    print(f"  functional 'X area' share of region mentions: {len(functional) / max(len(regions), 1):.1%}")
    print(f"  grounded mentions needing more than a name (relational, image-anchored or in-image "
          f"position): {grounded['disambiguated'].mean():.1%}; bare: {grounded['bare'].mean():.1%}")
    print(f"  object heads in COCO-80: {in_coco.mean():.1%}; LVIS (lenient): {in_lvis.mean():.1%}; "
          f"ADE20K-150: {in_ade.mean():.1%}; LVIS or ADE20K: {in_either.mean():.1%}")
    outside = objects[in_lvis == 0]["head"].value_counts().head(20)
    print("  most frequent object heads outside LVIS:", ", ".join(f"{h} ({n})" for h, n in outside.items()))
    rescued = objects[(in_lvis == 0) & (in_ade == 1)]["head"].value_counts().head(12)
    print("  ...of which ADE20K covers:", ", ".join(f"{h} ({n})" for h, n in rescued.items()))
    neither = objects[in_either == 0]["head"].value_counts().head(20)
    print("  most frequent object heads in neither:", ", ".join(f"{h} ({n})" for h, n in neither.items()))
    print("  most frequent functional areas:", ", ".join(clean(functional["text"]).value_counts().head(12).index))
    print("  most frequent named rooms:",
          ", ".join(clean(regions[regions["type"] == "region_room"]["text"]).value_counts().head(10).index))
    print("  sample relational mentions:",
          " | ".join(grounded[grounded["relational"]].sample(6, random_state=0)["text"]))
    print("  sample image-anchored mentions:",
          " | ".join(grounded[grounded["image_anchored"]].sample(6, random_state=0)["text"]), "\n")

    region_share = summary.loc[REGION_CATEGORIES, [f"type_{t}" for t in ENTITY_TYPES if t.startswith("region")]].sum(axis=1)
    by_cat = grounded.groupby("category_en")["disambiguated"].mean()
    expect("Regions are >= 30% of entity mentions in each region category", region_share.min() >= 0.30,
           ", ".join(f"{c[5:]} {v:.0%}" for c, v in region_share.items()))
    expect("Regions are >= 50% of entity mentions in Reg-Reg", region_share["Pos: Reg-Reg"] >= 0.5,
           f"{region_share['Pos: Reg-Reg']:.0%}")
    expect("Functional 'X area' names are >= 40% of region mentions", len(functional) >= 0.4 * len(regions),
           f"{len(functional) / len(regions):.0%}")
    expect("Measurement mentions are disambiguated beyond a name in >= 25%", by_cat["Attr: Measure"] >= 0.25,
           f"{by_cat['Attr: Measure']:.0%}")
    expect("Cam-Obj mentions are disambiguated in <= 15%", by_cat["Pos: Cam-Obj"] <= 0.15,
           f"{by_cat['Pos: Cam-Obj']:.0%}")
    expect(">= 75% of object heads are LVIS categories", in_lvis.mean() >= 0.75, f"{in_lvis.mean():.0%}")
    expect("<= 60% of object heads are COCO-80 categories", in_coco.mean() <= 0.6, f"{in_coco.mean():.0%}")
    expect("Parts/surfaces are >= 10% of Measurement mentions", summary.loc["Attr: Measure", "type_part"] >= 0.10,
           f"{summary.loc['Attr: Measure', 'type_part']:.0%}")


if __name__ == "__main__":
    main()

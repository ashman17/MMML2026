"""Step F: what reference reasoning traces add beyond the question.

Run:
  conda run -n 11777-project python 06_traces.py | tee outputs/06_run.log

Traces are never used as shortcut features. Duration is the released
mean_normed_duration_seconds field, already centered near zero.
"""

from __future__ import annotations

import pandas as pd
import spacy
from scipy.stats import spearmanr

from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text
from plotting import expect, setup
from traces import DIFFICULTY_ORDINAL, STEP_CUES, bridging_lemmas, content_lemmas, cue_flags

DIFFICULTIES = ["easy", "medium", "hard"]


def question_text(row) -> str:
    options = " ".join(getattr(row, f"opt_{letter}") for letter in LETTERS)
    return f"{row.stem}\n{options}"


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    questions = [question_text(row) for row in df.itertuples()]
    traces = df["thought"].tolist()
    rows = []
    docs = nlp.pipe(questions + traces, batch_size=64)
    question_docs = [next(docs) for _ in questions]
    for row, question_doc, trace_doc in zip(df.itertuples(), question_docs, docs):
        flags = cue_flags(row.thought)
        bridging = bridging_lemmas(trace_doc, question_doc)
        rows.append({
            "id": int(row.id),
            "category_en": row.category_en,
            "split": row.split,
            "difficulty": row.difficulty,
            "difficulty_ordinal": DIFFICULTY_ORDINAL[row.difficulty],
            "duration": float(row.mean_normed_duration_seconds),
            "stem_words": len(str(row.stem).split()),
            "trace_words": len(str(row.thought).split()),
            "n_question_nouns": len(content_lemmas(question_doc)),
            "n_trace_nouns": len(content_lemmas(trace_doc)),
            "n_bridging": len(bridging),
            "bridging_heads": " ".join(sorted(bridging)),
            **flags,
        })
    return pd.DataFrame(rows)


def correlations(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("trace_words", "difficulty_ordinal"),
        ("duration", "difficulty_ordinal"),
        ("trace_words", "duration"),
        ("stem_words", "trace_words"),
        ("n_bridging", "difficulty_ordinal"),
    ]
    rows = []
    for left, right in pairs:
        rho, p_value = spearmanr(frame[left], frame[right])
        rows.append({"x": left, "y": right, "spearman_rho": float(rho), "p_value": float(p_value), "n": len(frame)})
    return pd.DataFrame(rows)


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    cues = list(STEP_CUES)
    grouped = frame.groupby("category_en")
    out = grouped[["trace_words", "stem_words", "duration", "n_bridging"]].mean()
    out["bridging_rate"] = grouped["n_bridging"].apply(lambda values: (values > 0).mean())
    out = out.join(grouped[cues].mean().add_prefix("cue_"))
    out["n"] = grouped.size()
    return out.reindex(CATEGORY_ORDER)


def plot(frame: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    plt, sns = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [0.7, 1.5]})
    sns.boxplot(data=frame, x="difficulty", y="trace_words", order=DIFFICULTIES, ax=axes[0],
                color="#4c72b0", fliersize=2)
    axes[0].set(xlabel="", ylabel="Trace length (words)", title="Reference-trace length")
    cues = ["bridging_rate", *[f"cue_{name}" for name in STEP_CUES]]
    heat = 100 * summary[cues]
    heat.columns = ["bridging noun", "numbered", "sequence", "conclusion", "cross-view",
                    "perspective", "comparison", "motion"]
    sns.heatmap(heat, ax=axes[1], annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
                cbar=False, annot_kws={"fontsize": 5.5})
    axes[1].set(xlabel="", ylabel="", title="Trace cues by category (%)")
    axes[1].tick_params(labelsize=5.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    df = load_text(require_traces=True)
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    assert (df["id"].to_numpy() == tags["id"].to_numpy()).all()
    df = df.assign(split=tags["split"].to_numpy())
    frame = analyze(df)
    frame.to_csv(OUTPUTS / "06_trace_questions.csv", index=False)
    summary = category_summary(frame)
    summary.round(3).to_csv(OUTPUTS / "06_trace_summary_by_category.csv")
    corr = correlations(frame)
    corr.round(4).to_csv(OUTPUTS / "06_trace_correlations.csv", index=False)
    heads = (frame["bridging_heads"].str.split().explode().dropna().value_counts().head(30)
             .rename_axis("lemma").reset_index(name="questions"))
    heads.to_csv(OUTPUTS / "06_bridging_heads.csv", index=False)
    plot(frame, summary, OUTPUTS / "06_traces.png")

    by_split = frame.groupby("split")["n_bridging"].apply(lambda values: (values > 0).mean())
    by_difficulty = frame.groupby("difficulty")["trace_words"].median().reindex(DIFFICULTIES)
    rho = dict(zip(corr["x"] + "|" + corr["y"], corr["spearman_rho"]))
    print(f"Questions: {len(frame)}; median trace length by difficulty: "
          f"{by_difficulty.round(1).to_dict()}")
    print(corr.round(3).to_string(index=False))
    print("\nBy category:")
    print(summary[["n", "trace_words", "bridging_rate", "cue_numbered_steps", "cue_conclusion",
                   "cue_cross_view", "cue_perspective"]].round(3).to_string())
    print("\nMost common bridging nouns:", ", ".join(f"{row.lemma} ({row.questions})" for row in heads.head(12).itertuples()))
    print(f"Bridging rate explore/confirm: {by_split.get('explore', float('nan')):.1%} / "
          f"{by_split.get('confirm', float('nan')):.1%}\n")

    overall_bridging = (frame["n_bridging"] > 0).mean()
    multistep = summary.loc["Multi-step", "bridging_rate"]
    numbered_gap = summary.loc["Multi-step", "cue_numbered_steps"] - summary.loc["Pos: Cam-Obj", "cue_numbered_steps"]
    expect("Trace length rises with difficulty (Spearman >= 0.20)", rho["trace_words|difficulty_ordinal"] >= 0.20,
           f"{rho['trace_words|difficulty_ordinal']:.2f}")
    expect("Normalized duration rises with difficulty (Spearman >= 0.15)", rho["duration|difficulty_ordinal"] >= 0.15,
           f"{rho['duration|difficulty_ordinal']:.2f}")
    expect("Stem length and trace length are weakly related (Spearman <= 0.30)",
           rho["stem_words|trace_words"] <= 0.30, f"{rho['stem_words|trace_words']:.2f}")
    expect("At least 40% of traces introduce a noun absent from the question", overall_bridging >= 0.40,
           f"{overall_bridging:.1%}")
    expect("Multi-step bridging rate is at least the overall rate", multistep + 1e-12 >= overall_bridging,
           f"{multistep:.1%} vs {overall_bridging:.1%}")
    expect("Explore and confirm bridging rates differ by at most 8 points",
           abs(by_split["explore"] - by_split["confirm"]) <= 0.08,
           f"{by_split['explore']:.1%} vs {by_split['confirm']:.1%}")
    expect("At least 25% of traces contain a conclusion cue", frame["conclusion"].mean() >= 0.25,
           f"{frame['conclusion'].mean():.1%}")
    expect("At least 40% of traces name a numbered view", frame["cross_view"].mean() >= 0.40,
           f"{frame['cross_view'].mean():.1%}")
    expect("Numbered step lists are at least 10 points more common in Multi-step than Cam-Obj",
           numbered_gap >= 0.10, f"{numbered_gap:.1%}")


if __name__ == "__main__":
    main()

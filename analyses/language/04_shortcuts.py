"""Step D: answer-choice shortcuts and an explore-trained text-only learner.

Run:
  conda run -n 11777-project python 04_shortcuts.py | tee outputs/04_run.log
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text, options
from plotting import expect, setup
from shortcuts import (
    UnionFind,
    candidate_eliminations,
    elimination_policy,
    heuristic_policies,
    holm_adjust,
    ngrams,
    tokens,
)

SEED = 0
BOOTSTRAPS = 10_000
RANDOMIZATIONS = 10_000
CS = (0.01, 0.1, 1.0, 10.0)
ELIMINATION_MIN_N = 10

# Log-odds for a token that was never seen in explore options.
# Beta(1, 3) places the unseen prior at the four-choice chance rate (25%),
# whose log-odds are log(0.25 / 0.75) = log(1/3).
UNSEEN_LOG_ODDS = float(np.log(1 / 3))


def answer_indices(df: pd.DataFrame) -> np.ndarray:
    return np.array([LETTERS.index(letter) for letter in df["answer"]])


def observed_scores(policy: np.ndarray, y: np.ndarray) -> np.ndarray:
    return policy[np.arange(len(y)), y]


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def randomization_p(policy: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """One-sided random-label test; exact binomial equivalent for one-hot policies."""
    random_letters = rng.integers(0, len(LETTERS), size=(RANDOMIZATIONS, len(policy)))
    null = policy[np.arange(len(policy))[None, :], random_letters].mean(axis=1)
    return (1 + np.count_nonzero(null >= observed - 1e-12)) / (RANDOMIZATIONS + 1)


def content_prior_fit(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Smoothed option-token correctness prior learned only on explore.

    Beta(1, 3) centers every unseen token at the four-choice chance odds.
    """
    total, correct = Counter(), Counter()
    for row in df.itertuples():
        for letter, text in options(row).items():
            for token in set(tokens(text)):
                total[token] += 1
                correct[token] += letter == row.answer
    prior = {token: np.log((correct[token] + 1) / (total[token] - correct[token] + 3))
             for token in total}
    table = pd.DataFrame([
        {"token": token, "option_occurrences": total[token], "correct_occurrences": correct[token],
         "correct_rate": correct[token] / total[token], "smoothed_log_odds": prior[token]}
        for token in total
    ]).sort_values(["smoothed_log_odds", "option_occurrences"], ascending=[False, False])
    return prior, table


def content_prior_policy(row, prior: dict[str, float]) -> np.ndarray:
    scores = {}
    for letter, text in options(row).items():
        values = [prior.get(token, UNSEEN_LOG_ODDS) for token in set(tokens(text))]
        scores[letter] = np.mean(values) if values else UNSEEN_LOG_ODDS
    best = max(scores.values())
    mask = np.array([np.isclose(scores[letter], best) for letter in LETTERS])
    return mask / mask.sum()


def feature_dict(row, answer_space: str) -> dict[str, float]:
    """Position-aware question representation for four-way letter prediction."""
    feats: dict[str, float] = {f"answer_space={answer_space}": 1.0}
    stem_words = set(tokens(row.stem))
    for gram in ngrams(row.stem):
        feats[f"stem:{gram}"] = feats.get(f"stem:{gram}", 0.0) + 1.0
    for letter, text in options(row).items():
        option_words = set(tokens(text))
        for gram in ngrams(text):
            key = f"{letter}:option:{gram}"
            feats[key] = feats.get(key, 0.0) + 1.0
        feats[f"{letter}:tokens"] = np.log1p(len(tokens(text)))
        feats[f"{letter}:chars"] = np.log1p(len(text))
        union = stem_words | option_words
        feats[f"{letter}:overlap"] = len(stem_words & option_words) / len(union) if union else 0.0
    return feats


def template_groups(ids: list[int]) -> tuple[np.ndarray, set[int]]:
    path = OUTPUTS / "03_near_duplicates.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run 03_embeddings.py first")
    pairs = pd.read_csv(path)
    uf = UnionFind(ids)
    for row in pairs.itertuples():
        uf.union(int(row.id_a), int(row.id_b))
    return np.array([uf.find(qid) for qid in ids]), {
        int(qid) for qid in pd.concat([pairs["id_a"], pairs["id_b"]]).unique()
    }


def choose_c(X, y: np.ndarray, groups: np.ndarray | None = None) -> tuple[float, dict[float, float]]:
    if groups is None:
        splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
        folds = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        folds = splitter.split(X, y, groups)
    folds = list(folds)
    scores = {}
    for c in CS:
        fold_scores = []
        for train, test in folds:
            model = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
            model.fit(X[train], y[train])
            fold_scores.append((model.predict(X[test]) == y[test]).mean())
        scores[c] = float(np.mean(fold_scores))
    # Prefer stronger regularization on an exact tie.
    return max(CS, key=lambda c: (scores[c], -c)), scores


def fit_text_learner(
    explore: pd.DataFrame,
    confirm: pd.DataFrame,
    tags: pd.DataFrame,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    answer_space = dict(zip(tags["id"], tags["answer_space"]))
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(
        [feature_dict(row, answer_space[row.id]) for row in explore.itertuples()]
    )
    X_confirm = vectorizer.transform(
        [feature_dict(row, answer_space[row.id]) for row in confirm.itertuples()]
    )
    y_train = answer_indices(explore)
    ordinary_c, ordinary_scores = choose_c(X_train, y_train)
    assert len(groups) == len(explore)
    grouped_c, grouped_scores = choose_c(X_train, y_train, groups)
    model = LogisticRegression(C=grouped_c, max_iter=3000, solver="lbfgs")
    model.fit(X_train, y_train)
    prediction = model.predict(X_confirm)
    policy = np.eye(len(LETTERS))[prediction]

    names = vectorizer.get_feature_names_out()
    coefficient_rows = []
    for class_index, letter in enumerate(model.classes_):
        coef = model.coef_[class_index]
        order = np.argsort(np.abs(coef))[-40:][::-1]
        coefficient_rows.extend(
            {"answer": LETTERS[int(letter)], "feature": names[j], "coefficient": coef[j]}
            for j in order
        )
    tuning = {
        "ordinary_c": ordinary_c,
        "grouped_c": grouped_c,
        "ordinary_scores": ordinary_scores,
        "grouped_scores": grouped_scores,
    }
    return policy, tuning, pd.DataFrame(coefficient_rows)


def elimination_selection(explore: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    stats = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in explore.itertuples():
        for name, letters in candidate_eliminations(options(row)).items():
            stats[name]["n"] += len(letters)
            stats[name]["correct"] += row.answer in letters
    raw_p = {
        name: (binomtest(v["correct"], v["n"], 0.25, alternative="less").pvalue if v["n"] else 1.0)
        for name, v in stats.items()
    }
    adjusted = holm_adjust(raw_p)
    rows = []
    selected = set()
    for name, value in stats.items():
        keep = value["n"] >= ELIMINATION_MIN_N and adjusted[name] < 0.05
        if keep:
            selected.add(name)
        rows.append({
            "rule": name,
            "explore_opportunities": value["n"],
            "explore_correct": value["correct"],
            "explore_correct_rate": value["correct"] / value["n"] if value["n"] else np.nan,
            "p_less_than_chance": raw_p[name],
            "p_holm": adjusted[name],
            "selected": keep,
        })
    return selected, pd.DataFrame(rows).sort_values("rule")


def policy_table(
    policies: dict[str, np.ndarray],
    confirm: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y = answer_indices(confirm)
    rows, values = [], {}
    p_values = {}
    for name, policy in policies.items():
        score = observed_scores(policy, y)
        values[name] = score
        accuracy = score.mean()
        low, high = interval(score, rng)
        p_value = randomization_p(policy, accuracy, rng)
        p_values[name] = p_value
        rows.append({"policy": name, "n": len(score), "accuracy": accuracy,
                     "ci_low": low, "ci_high": high, "p_vs_25": p_value})
    adjusted = holm_adjust(p_values)
    table = pd.DataFrame(rows)
    table["p_holm"] = table["policy"].map(adjusted)
    return table, values


def grouped_accuracy(values: dict[str, np.ndarray], groups: pd.Series) -> pd.DataFrame:
    rows = []
    for name, score in values.items():
        frame = pd.DataFrame({"group": groups.to_numpy(), "score": score})
        for group, part in frame.groupby("group", sort=False):
            rows.append({"policy": name, "group": group, "n": len(part), "accuracy": part["score"].mean()})
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, by_category: pd.DataFrame, path) -> None:
    plt, sns = setup()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [0.85, 1.4]})
    ordered = summary.sort_values("accuracy")
    error = np.vstack([ordered["accuracy"] - ordered["ci_low"], ordered["ci_high"] - ordered["accuracy"]])
    axes[0].barh(ordered["policy"], 100 * ordered["accuracy"], xerr=100 * error,
                 color="#4c72b0", height=0.65)
    axes[0].axvline(25, color="grey", ls="--", lw=0.8)
    axes[0].set(xlabel="confirm accuracy (%)", xlim=(0, max(45, 100 * ordered["ci_high"].max() + 3)),
                title="Language-only policies")
    axes[0].tick_params(labelsize=6)
    matrix = by_category.pivot(index="category_en", columns="policy", values="accuracy").reindex(CATEGORY_ORDER)
    sns.heatmap(100 * matrix, ax=axes[1], annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=60,
                cbar=False, annot_kws={"fontsize": 5})
    axes[1].set(xlabel="", ylabel="", title="Confirm accuracy by category (%)")
    axes[1].tick_params(labelsize=5.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_text()
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    assert (df["id"].to_numpy() == tags["id"].to_numpy()).all()
    df = df.assign(split=tags["split"].to_numpy())
    explore = df[df["split"] == "explore"].reset_index(drop=True)
    confirm = df[df["split"] == "confirm"].reset_index(drop=True)

    all_groups, duplicate_ids = template_groups(df["id"].tolist())
    group_by_id = dict(zip(df["id"], all_groups))
    explore_groups = np.array([group_by_id[qid] for qid in explore["id"]])
    explore_roots = set(explore_groups)
    novel_template = np.array([group_by_id[qid] not in explore_roots for qid in confirm["id"]])

    selected_rules, elimination = elimination_selection(explore)
    elimination.to_csv(OUTPUTS / "04_elimination_rules.csv", index=False)
    prior, prior_table = content_prior_fit(explore)
    prior_table.to_csv(OUTPUTS / "04_option_content_priors.csv", index=False)

    majority = explore["answer"].value_counts().reindex(LETTERS, fill_value=0).idxmax()
    policies = {
        "majority_letter": np.tile(np.eye(4)[LETTERS.index(majority)], (len(confirm), 1)),
        "longest_option": np.vstack([
            heuristic_policies(row.stem, options(row))["longest_option"] for row in confirm.itertuples()
        ]),
        "shortest_option": np.vstack([
            heuristic_policies(row.stem, options(row))["shortest_option"] for row in confirm.itertuples()
        ]),
        "greatest_overlap": np.vstack([
            heuristic_policies(row.stem, options(row))["greatest_overlap"] for row in confirm.itertuples()
        ]),
        "option_content_prior": np.vstack([content_prior_policy(row, prior) for row in confirm.itertuples()]),
        "random_after_elimination": np.vstack([
            elimination_policy(options(row), selected_rules) for row in confirm.itertuples()
        ]),
    }
    learner, tuning, coefficients = fit_text_learner(explore, confirm, tags, explore_groups)
    policies["text_learner"] = learner
    coefficients.to_csv(OUTPUTS / "04_text_learner_coefficients.csv", index=False)

    summary, values = policy_table(policies, confirm, rng)
    summary["selected_elimination_rules"] = ",".join(sorted(selected_rules))
    learner_values = values["text_learner"]
    novel_low, novel_high = interval(learner_values[novel_template], rng)
    summary = pd.concat([summary, pd.DataFrame([{
        "policy": "text_learner_novel_templates",
        "n": int(novel_template.sum()),
        "accuracy": learner_values[novel_template].mean(),
        "ci_low": novel_low,
        "ci_high": novel_high,
        "p_vs_25": np.nan,
        "p_holm": np.nan,
        "selected_elimination_rules": ",".join(sorted(selected_rules)),
    }])], ignore_index=True)
    summary.to_csv(OUTPUTS / "04_shortcut_summary.csv", index=False)

    category = grouped_accuracy(values, confirm["category_en"])
    category.rename(columns={"group": "category_en"}).to_csv(
        OUTPUTS / "04_shortcuts_by_category.csv", index=False
    )
    answer_spaces = confirm["id"].map(dict(zip(tags["id"], tags["answer_space"])))
    by_space = grouped_accuracy(values, answer_spaces)
    by_space.rename(columns={"group": "answer_space"}).to_csv(
        OUTPUTS / "04_shortcuts_by_answer_space.csv", index=False
    )
    plot(summary[summary["policy"] != "text_learner_novel_templates"],
         category.rename(columns={"group": "category_en"}), OUTPUTS / "04_shortcuts.png")

    print(f"Explore / confirm: {len(explore)} / {len(confirm)}")
    print(f"Explore majority letter: {majority}; template-paired ids: {len(duplicate_ids)}; "
          f"confirm questions novel at cosine < .95: {novel_template.sum()}")
    print("\nElimination rules (selected on explore only):")
    print(elimination.round(4).to_string(index=False))
    print(f"\nText learner C: ordinary CV {tuning['ordinary_c']}, "
          f"template-grouped CV {tuning['grouped_c']}")
    print("  ordinary:", {c: round(v, 3) for c, v in tuning["ordinary_scores"].items()})
    print("  grouped: ", {c: round(v, 3) for c, v in tuning["grouped_scores"].items()})
    print("\nConfirm results:")
    print(summary[["policy", "n", "accuracy", "ci_low", "ci_high", "p_vs_25", "p_holm"]]
          .round(4).to_string(index=False))
    camcam = category[(category["group"] == "Pos: Cam-Cam") &
                      (category["policy"] == "longest_option")]["accuracy"].iloc[0]
    best_alternative = summary[summary["policy"].isin([
        "majority_letter", "longest_option", "shortest_option", "greatest_overlap",
        "option_content_prior", "random_after_elimination",
    ])]["accuracy"].max()
    result = dict(zip(summary["policy"], summary["accuracy"]))
    adjacent = abs(np.log10(tuning["ordinary_c"]) - np.log10(tuning["grouped_c"])) <= 1
    print()
    expect("Explore-majority letter <= 30% on confirm", result["majority_letter"] <= 0.30,
           f"{result['majority_letter']:.1%}")
    expect("Longest option > 25% overall and > 40% in Cam-Cam",
           result["longest_option"] > 0.25 and camcam > 0.40,
           f"overall {result['longest_option']:.1%}, Cam-Cam {camcam:.1%}")
    expect("Shortest option and overlap <= 30%",
           max(result["shortest_option"], result["greatest_overlap"]) <= 0.30,
           f"{result['shortest_option']:.1%}, {result['greatest_overlap']:.1%}")
    expect("The Sometimes elimination rule is selected", "sometimes" in selected_rules,
           ",".join(sorted(selected_rules)) or "none")
    expect("Random after elimination is between 25% and 35%",
           0.25 < result["random_after_elimination"] < 0.35,
           f"{result['random_after_elimination']:.1%}")
    expect("Text learner beats every simpler alternative but remains below 45%",
           best_alternative < result["text_learner"] < 0.45,
           f"learner {result['text_learner']:.1%}, best alternative {best_alternative:.1%}")
    expect("Ordinary and grouped CV select same or adjacent C", adjacent,
           f"{tuning['ordinary_c']} vs {tuning['grouped_c']}")


if __name__ == "__main__":
    main()

"""Step C: embedding geometry, category probes and template near-duplicates.

Run: conda run -n 11777-project python 03_embeddings.py
The first run downloads all-MiniLM-L6-v2 into cache/hf (needs network);
embeddings are cached in outputs/ and reused while the question ids match.
"""

from __future__ import annotations

import json
import os

from common import CACHE, CATEGORY_ORDER, OUTPUTS, load_text, options, save_json

os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if (CACHE / "hf" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score, f1_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from plotting import CATEGORY_PALETTE, expect, setup  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_PATH = OUTPUTS / "03_stem_embeddings.npy"
EMBED_META = OUTPUTS / "03_stem_embeddings_meta.json"
K_NEIGHBORS = 5
N_PERMUTATIONS = 1000
NEAR_DUP = 0.95
SEED = 0
CATEGORICAL_TAGS = ["answer_space", "frame", "mirror_level", "reorder_level"]
NON_TEXT = {"id", "category_en", "answer", "n_images", "split", "mirror_reasons", "reorder_reasons"}


def write_embeddings(df: pd.DataFrame) -> None:
    from sentence_transformers import SentenceTransformer

    stems = df["stem"].tolist()
    model = SentenceTransformer(MODEL, device="cpu")
    lengths = [len(ids) for ids in model.tokenizer(stems)["input_ids"]]
    emb = model.encode(stems, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(EMBED_PATH, emb.astype(np.float32))
    save_json({"model": MODEL, "ids": df["id"].tolist(), "max_seq_length": model.max_seq_length,
               "max_tokens": max(lengths), "truncated": sum(n > model.max_seq_length for n in lengths)},
              EMBED_META)


def load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    ids = df["id"].tolist()
    if not (EMBED_PATH.exists() and EMBED_META.exists() and json.loads(EMBED_META.read_text())["ids"] == ids):
        write_embeddings(df)
    meta = json.loads(EMBED_META.read_text())
    assert meta["ids"] == ids and meta["model"] == MODEL
    return np.load(EMBED_PATH), meta


def lexicon_features(tags: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in tags.columns if c not in NON_TEXT | set(CATEGORICAL_TAGS)]
    feats = tags[cols].copy()
    feats["anchor_image"] = feats["anchor_image"].fillna(0)
    feats["anchor_is_first"] = feats["anchor_is_first"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)
    feats = pd.concat([feats.astype(float), pd.get_dummies(tags[CATEGORICAL_TAGS]).astype(float)], axis=1)
    return feats


def empath_features(stems: list[str]) -> np.ndarray:
    from empath import Empath

    lex = Empath()
    rows = [lex.analyze(s, normalize=True) or {} for s in stems]
    return pd.DataFrame(rows).fillna(0).to_numpy(dtype=float)


def knn_agreement(emb: np.ndarray, labels: np.ndarray, rng) -> dict:
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argsort(-sim, axis=1)[:, :K_NEIGHBORS]
    observed = (labels[nn] == labels[:, None]).mean(axis=1)
    null = np.empty(N_PERMUTATIONS)
    for b in range(N_PERMUTATIONS):
        perm = rng.permutation(labels)
        null[b] = (perm[nn] == perm[:, None]).mean()
    return {"nn": nn, "per_item": observed, "overall": observed.mean(), "null_mean": null.mean(),
            "null_p95": np.quantile(null, 0.95), "p_value": (1 + (null >= observed.mean()).sum()) / (1 + N_PERMUTATIONS)}


def neighbor_matrix(nn: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = pd.DataFrame({"src": np.repeat(labels, nn.shape[1]), "dst": labels[nn].ravel()})
    return pd.crosstab(rows["src"], rows["dst"], normalize="index").reindex(
        index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)


def probe(name: str, X, y: np.ndarray, folds) -> dict:
    if name == "TF-IDF":
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                              LogisticRegression(max_iter=3000))
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    search = GridSearchCV(model, {"logisticregression__C": [0.01, 0.1, 1, 10]}, cv=3, scoring="f1_macro")
    pred = cross_val_predict(search, X, y, cv=folds)
    fold_f1 = [f1_score(y[test], pred[test], average="macro") for _, test in folds.split(X, y)]
    return {"pred": pred, "macro_f1": f1_score(y, pred, average="macro"), "macro_f1_sd": float(np.std(fold_f1)),
            "accuracy": float((pred == y).mean())}


def near_duplicates(emb: np.ndarray, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sim = emb @ emb.T
    i, j = np.where(np.triu(sim, k=1) >= threshold)
    answer_text = [options(r)[r.answer] for r in df.itertuples()]
    pairs = pd.DataFrame({"id_a": df["id"].to_numpy()[i], "id_b": df["id"].to_numpy()[j], "cosine": sim[i, j]})
    for side, idx in (("a", i), ("b", j)):
        pairs[f"category_{side}"] = df["category_en"].to_numpy()[idx]
        pairs[f"split_{side}"] = df["split"].to_numpy()[idx]
        pairs[f"answer_{side}"] = df["answer"].to_numpy()[idx]
        pairs[f"answer_text_{side}"] = np.array(answer_text, dtype=object)[idx]
        pairs[f"stem_{side}"] = df["stem"].to_numpy()[idx]
    pairs["same_letter"] = pairs["answer_a"] == pairs["answer_b"]
    pairs["same_answer_text"] = pairs["answer_text_a"].str.lower() == pairs["answer_text_b"].str.lower()
    pairs["exact_stem"] = pairs["stem_a"] == pairs["stem_b"]
    return pairs.sort_values("cosine", ascending=False)


def plot(tsne: np.ndarray, labels: np.ndarray, nbr: pd.DataFrame, probes: dict, majority_f1: float, path) -> None:
    plt, sns = setup()
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 0.75], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    for cat in CATEGORY_ORDER:
        mask = labels == cat
        ax.scatter(tsne[mask, 0], tsne[mask, 1], s=3, color=CATEGORY_PALETTE[cat], label=cat, linewidths=0)
    ax.set(xticks=[], yticks=[], title="t-SNE of stem embeddings (MiniLM)")
    ax.legend(fontsize=4.5, markerscale=2.5, loc="upper left", bbox_to_anchor=(-0.02, -0.02), ncol=4,
              frameon=False, handletextpad=0.1, columnspacing=0.6)
    ax = fig.add_subplot(gs[1])
    short = [c.replace("Pos: ", "").replace("Attr: ", "").replace("Motion: ", "Mot-") for c in CATEGORY_ORDER]
    sns.heatmap(100 * nbr.to_numpy(), ax=ax, cmap="Greens", vmin=0, vmax=100, cbar=False, annot=True, fmt=".0f",
                annot_kws={"fontsize": 4}, xticklabels=short, yticklabels=short, square=True)
    ax.tick_params(labelsize=4.5)
    ax.set_title(f"Category of {K_NEIGHBORS} nearest neighbours (%)")
    ax.set(xlabel="neighbour", ylabel="question")
    ax = fig.add_subplot(gs[2])
    names = list(probes)
    ax.barh(names[::-1], [probes[n]["macro_f1"] for n in names[::-1]],
            xerr=[probes[n]["macro_f1_sd"] for n in names[::-1]], color="#4c72b0", height=0.6)
    ax.axvline(majority_f1, color="grey", ls="--", lw=0.8)
    ax.set(xlim=(0, 1), xlabel="macro-F1 (5-fold)", title="Category probes")
    ax.tick_params(labelsize=6)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load_text()
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    assert (tags["id"].to_numpy() == df["id"].to_numpy()).all(), "01_tags.csv must follow the cache order"
    df = df.assign(split=tags["split"].to_numpy())
    stems = df["stem"].tolist()
    labels = df["category_en"].to_numpy()

    emb, meta = load_embeddings(df)
    n_truncated, max_tokens = meta["truncated"], meta["max_tokens"]
    knn = knn_agreement(emb, labels, rng)
    per_cat = pd.Series(knn["per_item"]).groupby(labels).mean().reindex(CATEGORY_ORDER)
    counts = pd.Series(labels).value_counts()
    per_cat_null = ((counts - 1) / (len(labels) - 1)).reindex(CATEGORY_ORDER)
    nbr = neighbor_matrix(knn["nn"], labels)
    nbr.round(3).to_csv(OUTPUTS / "03_neighbor_categories.csv")

    tsne = TSNE(n_components=2, perplexity=30, init="pca", metric="cosine", random_state=SEED).fit_transform(emb)
    pca = PCA(n_components=10, random_state=SEED).fit(emb)
    pd.DataFrame({"id": df["id"], "category_en": labels, "tsne_x": tsne[:, 0], "tsne_y": tsne[:, 1],
                  **{f"pc{k + 1}": v for k, v in enumerate(pca.transform(emb)[:, :2].T)}}).to_csv(
        OUTPUTS / "03_projection.csv", index=False)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    lex = lexicon_features(tags)
    feature_sets = {"TF-IDF": np.array(stems, dtype=object), "MiniLM": emb, "Lexicon tags": lex.to_numpy(),
                    "Empath": empath_features(stems)}
    probes = {name: probe(name, X, labels, folds) for name, X in feature_sets.items()}
    majority = np.full_like(labels, counts.idxmax())
    majority_f1 = f1_score(labels, majority, average="macro")
    best = max(probes, key=lambda n: probes[n]["macro_f1"])
    per_class_f1 = {n: pd.Series(f1_score(labels, p["pred"], average=None, labels=CATEGORY_ORDER),
                                 index=CATEGORY_ORDER) for n, p in probes.items()}
    pd.DataFrame(per_class_f1).round(3).to_csv(OUTPUTS / "03_probe_f1_by_category.csv")

    kmeans = KMeans(n_clusters=len(CATEGORY_ORDER), n_init=10, random_state=SEED).fit_predict(emb)
    ami = {target: adjusted_mutual_info_score(tags[target] if target != "category_en" else labels, kmeans)
           for target in ("category_en", "answer_space", "frame")}

    pairs = near_duplicates(emb, df, NEAR_DUP)
    pairs.drop(columns=["stem_a", "stem_b"]).round(4).to_csv(OUTPUTS / "03_near_duplicates.csv", index=False)
    dup_ids = set(pairs["id_a"]) | set(pairs["id_b"])
    letter_chance = float((df["answer"].value_counts(normalize=True) ** 2).sum())
    upper = np.triu(emb @ emb.T, k=1)
    sensitivity = {t: len(set(np.nonzero(upper >= t)[0]) | set(np.nonzero(upper >= t)[1]))
                   for t in (0.90, 0.95, 0.98)}

    plot(tsne, labels, nbr, probes, majority_f1, OUTPUTS / "03_embeddings.png")
    summary = {
        "truncated_stems": n_truncated, "max_tokens": max_tokens,
        "knn_agreement": round(float(knn["overall"]), 4), "knn_null_mean": round(float(knn["null_mean"]), 4),
        "knn_null_p95": round(float(knn["null_p95"]), 4), "knn_p_value": knn["p_value"],
        "knn_by_category": per_cat.round(3).to_dict(), "knn_null_by_category": per_cat_null.round(3).to_dict(),
        "pca_variance_top10": pca.explained_variance_ratio_.round(4).tolist(),
        "probes": {n: {k: round(float(v), 4) for k, v in p.items() if k != "pred"} for n, p in probes.items()},
        "majority_macro_f1": round(float(majority_f1), 4), "n_lexicon_features": lex.shape[1],
        "kmeans_ami": {k: round(float(v), 4) for k, v in ami.items()},
        "near_duplicates": {"threshold": NEAR_DUP, "pairs": len(pairs), "questions": len(dup_ids),
                            "questions_by_threshold": sensitivity,
                            "same_category": round(float((pairs["category_a"] == pairs["category_b"]).mean()), 4),
                            "cross_split": int((pairs["split_a"] != pairs["split_b"]).sum()),
                            "exact_stem": int(pairs["exact_stem"].sum()),
                            "same_letter": round(float(pairs["same_letter"].mean()), 4),
                            "same_answer_text": round(float(pairs["same_answer_text"].mean()), 4),
                            "letter_chance": round(letter_chance, 4)},
    }
    save_json(summary, OUTPUTS / "03_summary.json")

    print(f"Stems: {len(stems)}; max {max_tokens} tokens; truncated at 256: {n_truncated}")
    print(f"{K_NEIGHBORS}-NN category agreement {knn['overall']:.3f} vs permutation null {knn['null_mean']:.3f} "
          f"(95th pct {knn['null_p95']:.3f}, p = {knn['p_value']:.4f})")
    print(pd.DataFrame({"agreement": per_cat, "null": per_cat_null, "n": counts.reindex(CATEGORY_ORDER)})
          .round(3).to_string(), "\n")
    print(f"Probes (5-fold, macro-F1 +- fold sd; majority baseline {majority_f1:.3f}; "
          f"lexicon has {lex.shape[1]} features):")
    for n, p in probes.items():
        print(f"  {n:13} macro-F1 {p['macro_f1']:.3f} +- {p['macro_f1_sd']:.3f}   accuracy {p['accuracy']:.3f}")
    print(f"Per-category F1:\n{pd.DataFrame(per_class_f1).round(2).to_string()}\n")
    print("k-means (k=11) AMI:", ", ".join(f"{k} {v:.3f}" for k, v in ami.items()))
    nd = summary["near_duplicates"]
    print(f"Near-duplicates (cosine >= {NEAR_DUP}): {nd['pairs']} pairs over {nd['questions']} questions "
          f"(by threshold {sensitivity}); same category {nd['same_category']:.0%}; cross-split pairs "
          f"{nd['cross_split']}; exact stems {nd['exact_stem']}; same letter {nd['same_letter']:.0%} "
          f"(chance {letter_chance:.0%}); same answer text {nd['same_answer_text']:.0%}")
    for r in pairs.head(3).itertuples():
        print(f"  {r.cosine:.3f} [{r.id_a}] {r.stem_a[:110]!r}\n        [{r.id_b}] {r.stem_b[:110]!r}  "
              f"answers {r.answer_text_a!r} / {r.answer_text_b!r}")
    print()

    position = ["Pos: Obj-Obj", "Pos: Obj-Reg", "Pos: Reg-Reg", "Multi-step"]
    expect("No stem exceeds 256 tokens", n_truncated == 0, f"max {max_tokens}")
    expect("5-NN agreement >= 0.5 against a null near 0.10", knn["overall"] >= 0.5 and knn["p_value"] < 0.01,
           f"{knn['overall']:.2f} vs {knn['null_mean']:.2f}")
    expect("Motion: Cam and Measurement agreement >= 0.7",
           min(per_cat["Motion: Cam"], per_cat["Attr: Measure"]) >= 0.7,
           f"{per_cat['Motion: Cam']:.2f}, {per_cat['Attr: Measure']:.2f}")
    expect("At least two of Obj-Obj/Obj-Reg/Reg-Reg/Multi-step have agreement <= 0.45",
           (per_cat[position] <= 0.45).sum() >= 2, per_cat[position].round(2).to_dict())
    expect("TF-IDF macro-F1 >= MiniLM", probes["TF-IDF"]["macro_f1"] >= probes["MiniLM"]["macro_f1"],
           f"{probes['TF-IDF']['macro_f1']:.2f} vs {probes['MiniLM']['macro_f1']:.2f}")
    expect("Lexicon tags within 0.15 macro-F1 of the best set",
           probes[best]["macro_f1"] - probes["Lexicon tags"]["macro_f1"] <= 0.15,
           f"best {best} {probes[best]['macro_f1']:.2f}, lexicon {probes['Lexicon tags']['macro_f1']:.2f}")
    expect("Empath macro-F1 <= 0.35", probes["Empath"]["macro_f1"] <= 0.35, f"{probes['Empath']['macro_f1']:.2f}")
    expect("Multi-step has the lowest F1 in the best probe",
           per_class_f1[best].idxmin() == "Multi-step", f"lowest: {per_class_f1[best].idxmin()}")
    expect("k-means AMI with answer space >= AMI with category", ami["answer_space"] >= ami["category_en"],
           f"{ami['answer_space']:.2f} vs {ami['category_en']:.2f}")
    expect(">= 5% of questions have a near-duplicate stem", len(dup_ids) >= 50, f"{len(dup_ids)}")
    expect("Near-duplicate pairs agree on the letter at most 35% of the time",
           len(pairs) == 0 or nd["same_letter"] <= 0.35, f"{nd['same_letter']:.0%}")


if __name__ == "__main__":
    main()

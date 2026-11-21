"""Transformation-Guided Contrastive Decoding for MMSI-Bench.

Score collection and analysis are separate so aggregation and alpha selection stay
offline. Analyze a dev score file without --selection to choose a configuration;
analyze held-out scores with the resulting tgcd_selection.json.
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

from mmsi_consistency import (LETTERS, MockBackend, VLLMBackend,
                              bootstrap_ci, build_messages, build_score_messages, extract,
                              letter_score_prompt, load_rows, make_variants, mcnemar_exact,
                              option_letters, stratified_sample, vote)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
AGGREGATIONS = ("logprob_mean", "prob_mean")


def argmax_score(scores):
    return max(scores, key=scores.get)


def normalize_logprobs(scores):
    peak = max(scores.values())
    total = sum(math.exp(score - peak) for score in scores.values())
    denom = peak + math.log(total)
    return {letter: score - denom for letter, score in scores.items()}


def map_back_scores(scores, perm):
    if perm is None:
        return dict(scores)
    mapped = {}
    for shown_index, original_index in enumerate(perm):
        mapped[LETTERS[original_index]] = scores[LETTERS[shown_index]]
    return mapped


def aggregate_scores(score_sets, method):
    if not score_sets:
        raise ValueError("cannot aggregate an empty set of views")
    normalized = [normalize_logprobs(scores) for scores in score_sets]
    letters = tuple(normalized[0])
    if any(set(scores) != set(letters) for scores in normalized[1:]):
        raise ValueError("all score sets must contain the same option letters")
    if method == "logprob_mean":
        return {letter: sum(scores[letter] for scores in normalized) / len(normalized)
                for letter in letters}
    if method == "prob_mean":
        return {letter: math.log(sum(math.exp(scores[letter]) for scores in normalized)
                                 / len(normalized)) for letter in letters}
    raise ValueError(f"unknown aggregation: {method}")


def contrast_scores(aggregate, noimage, alpha):
    noimage = normalize_logprobs(noimage)
    if set(aggregate) != set(noimage):
        raise ValueError("aggregate and no-image scores must contain the same option letters")
    return {letter: score - alpha * noimage[letter]
            for letter, score in aggregate.items()}


def noimage_messages(row):
    return [{"role": "user", "content": [
        {"type": "text", "text": row["question"]
         + letter_score_prompt(option_letters(row["question"]))}]}]


def _ids_from(paths):
    return {int(json.loads(line)["id"]) for path in paths for line in open(path)}


def run(args):
    rows = load_rows(args.parquet)
    if args.include:
        included = _ids_from(args.include)
        rows = [row for row in rows if int(row["id"]) in included]
        print(f"included {len(included)} unique dev ids, found {len(rows)} rows", file=sys.stderr)
    if args.exclude:
        excluded = _ids_from(args.exclude)
        rows = [row for row in rows if int(row["id"]) not in excluded]
        print(f"excluded {len(excluded)} unique dev ids, {len(rows)} rows remain", file=sys.stderr)
    rows = stratified_sample(rows, args.n, args.seed)
    if not rows:
        raise ValueError("no questions selected")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as handle:
        json.dump({key: value for key, value in vars(args).items() if key != "analyze"},
                  handle, indent=2)
    backend = MockBackend(args.seed) if args.backend == "mock" else VLLMBackend(
        args.model, tp=args.tp,
        hf_overrides=json.loads(args.hf_overrides) if args.hf_overrides else None)

    rng = random.Random(args.seed)
    variants = [make_variants(row, rng) for row in rows]
    jobs = [(i, variant) for i, row_variants in enumerate(variants) for variant in row_variants]
    score_convs = [build_score_messages(rows[i], variant, args.max_side)
                   for i, variant in jobs]
    noimage_convs = [noimage_messages(row) for row in rows]
    variant_letter_sets = [option_letters(variant["question"]) for _, variant in jobs]
    row_letter_sets = [option_letters(row["question"]) for row in rows]
    print(f"{len(rows)} questions, {len(jobs)} transformed scores, "
          f"{len(noimage_convs)} no-image scores", file=sys.stderr)
    scored = backend.score_letters(score_convs + noimage_convs,
                                    variant_letter_sets + row_letter_sets)
    variant_scores, noimage_scores = scored[:len(jobs)], scored[len(jobs):]

    original_variants = [row_variants[0] for row_variants in variants]
    greedy_convs = [build_messages(row, variant, args.max_side)
                    for row, variant in zip(rows, original_variants)]
    greedy_raw = backend.generate(greedy_convs, n=1, temperature=0.0)

    k_per = [len(row_variants) for row_variants in variants]
    sc_raw = {}
    by_k = defaultdict(list)
    for i, count in enumerate(k_per):
        by_k[count].append(i)
    for count, indices in by_k.items():
        outputs = backend.generate([greedy_convs[i] for i in indices], n=count,
                                   temperature=args.sc_temp)
        for i, output in zip(indices, outputs):
            sc_raw[i] = output

    records = {i: {"id": int(row["id"]), "question_type": row["question_type"],
                   "answer": row["answer"], "n_images": len(row["images"]),
                   "n_options": len(row_letter_sets[i]), "variants": {}}
               for i, row in enumerate(rows)}
    for (i, variant), (raw_scores, missing_letters) in zip(jobs, variant_scores):
        mapped = map_back_scores(raw_scores, variant["perm"])
        records[i]["variants"][variant["name"]] = {
            "scores_raw": raw_scores,
            "scores": mapped,
            "pred_raw": argmax_score(raw_scores),
            "pred": argmax_score(mapped),
            "perm": variant["perm"],
            "missing_letters": missing_letters,
        }
    for i, (scores, missing_letters) in enumerate(noimage_scores):
        greedy_pred = extract(greedy_raw[i][0])
        records[i].update(
            noimage={"scores": scores, "pred": argmax_score(scores),
                     "missing_letters": missing_letters},
            greedy_pred=greedy_pred,
            score_pred=records[i]["variants"]["orig"]["pred"],
            sc_preds=[extract(text) for text in sc_raw[i]],
        )

    path = os.path.join(args.out, "scores.jsonl")
    with open(path, "w") as handle:
        for i in sorted(records):
            handle.write(json.dumps(records[i]) + "\n")
    comparable = [record for record in records.values() if record["greedy_pred"] is not None]
    matches = sum(record["score_pred"] == record["greedy_pred"] for record in comparable)
    rate = 100.0 * matches / len(comparable) if comparable else float("nan")
    missing_variants = sum(variant["missing_letters"] for record in records.values()
                           for variant in record["variants"].values())
    missing_noimage = sum(record["noimage"]["missing_letters"] for record in records.values())
    fallback_label = ("A-D" if all(letters == LETTERS for letters in row_letter_sets)
                      else "valid-option-letter")
    print(f"original score argmax vs greedy: {matches}/{len(comparable)} ({rate:.1f}%)",
          file=sys.stderr)
    print(f"missing {fallback_label} fallbacks: {missing_variants + missing_noimage} "
          f"({missing_variants} variant, {missing_noimage} no-image)", file=sys.stderr)
    print(f"wrote {path}", file=sys.stderr)


def _ensemble(record, method, alpha):
    aggregate = aggregate_scores([variant["scores"] for variant in record["variants"].values()],
                                 method)
    return argmax_score(contrast_scores(aggregate, record["noimage"]["scores"], alpha))


def _vcd(record, alpha):
    original = normalize_logprobs(record["variants"]["orig"]["scores"])
    return argmax_score(contrast_scores(original, record["noimage"]["scores"], alpha))


def _single_contrast(record, variant, alpha):
    scores = normalize_logprobs(record["variants"][variant]["scores"])
    return argmax_score(contrast_scores(scores, record["noimage"]["scores"], alpha))


def _accuracy(predictions, records):
    return 100.0 * sum(pred == record["answer"] for pred, record in zip(predictions, records)) \
        / len(records)


def _option_count(record):
    return int(record.get("n_options", len(record["variants"]["orig"]["scores"])))


def _selection(records):
    best = None
    grid = []
    for method in AGGREGATIONS:
        for alpha in ALPHAS:
            predictions = [_ensemble(record, method, alpha) for record in records]
            accuracy = _accuracy(predictions, records)
            candidate = {"aggregation": method, "alpha": alpha, "dev_accuracy": accuracy}
            grid.append(candidate)
            if best is None or accuracy > best["dev_accuracy"]:
                best = candidate
    return dict(best), grid


def _metric_row(name, predictions, records, original_ok, bootstrap, rng):
    correct = [prediction == record["answer"] for prediction, record in zip(predictions, records)]
    accuracy = 100.0 * sum(correct) / len(correct)
    interval = bootstrap_ci(len(records),
                            lambda indices: 100.0 * sum(correct[i] for i in indices) / len(indices),
                            bootstrap, rng)
    orig_only, method_only, p_value = mcnemar_exact(original_ok, correct)
    return {"name": name, "accuracy": accuracy, "ci": interval, "orig_only": orig_only,
            "method_only": method_only, "p": p_value, "correct": correct,
            "predictions": predictions}


def _figure(outdir, grid, original_accuracy):
    try:
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(outdir, ".mplconfig"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(outdir, ".cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        print(f"[analysis] figure skipped: {error}", file=sys.stderr)
        return None
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    for method, label in [("logprob_mean", "Mean log-probability"),
                          ("prob_mean", "Mean probability")]:
        rows = [row for row in grid if row["aggregation"] == method]
        axis.plot([row["alpha"] for row in rows], [row["dev_accuracy"] for row in rows],
                  marker="o", label=label)
    axis.axhline(original_accuracy, color="black", linestyle="--", label="Original")
    axis.set_xlabel("Contrast weight alpha")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(ALPHAS)
    axis.legend(frameon=False)
    axis.set_title("TGCD development accuracy")
    fig.tight_layout()
    path = os.path.join(outdir, "fig_accuracy_vs_alpha.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def analyze(path, selection_path=None, bootstrap=1000, seed=0, make_figure=True):
    with open(path) as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        raise ValueError(f"no score rows in {path}")
    outdir = os.path.dirname(path) or "."
    original_predictions = [record["variants"]["orig"]["pred"] for record in records]
    original_ok = [prediction == record["answer"]
                   for prediction, record in zip(original_predictions, records)]

    if selection_path:
        with open(selection_path) as handle:
            selected = json.load(handle)
        grid = []
        selection_source = f"fixed from `{selection_path}`"
    else:
        selected, grid = _selection(records)
        selected.update(n_dev=len(records), source=os.path.abspath(path))
        selection_file = os.path.join(outdir, "tgcd_selection.json")
        with open(selection_file, "w") as handle:
            json.dump(selected, handle, indent=2)
        selection_source = f"selected on these {len(records)} development questions"

    method, alpha = selected["aggregation"], float(selected["alpha"])
    tgcd_predictions = [_ensemble(record, method, alpha) for record in records]
    vcd_predictions = [_vcd(record, alpha) for record in records]
    sc_predictions = [vote(record["sc_preds"], record["variants"]["orig"]["pred"])
                      for record in records]
    noimage_predictions = [record["noimage"]["pred"] for record in records]
    missing_variants = sum(variant.get("missing_letters", False) for record in records
                           for variant in record["variants"].values())
    variant_score_count = sum(len(record["variants"]) for record in records)
    missing_noimage = sum(record["noimage"].get("missing_letters", False)
                          for record in records)
    variable_options = any(_option_count(record) != len(LETTERS) for record in records)

    metrics = [("Original letter score", original_predictions)]
    metrics.extend([
        ("No image", noimage_predictions),
        ("Self-consistency majority", sc_predictions),
        ("Ensemble mean log-probability", [_ensemble(record, "logprob_mean", 0.0)
                                            for record in records]),
        ("Ensemble mean probability", [_ensemble(record, "prob_mean", 0.0)
                                        for record in records]),
        (f"VCD-style (alpha={alpha:g})", vcd_predictions),
        (f"TGCD ({method}, alpha={alpha:g})", tgcd_predictions),
    ])
    rng = random.Random(seed)
    metric_rows = [_metric_row(name, predictions, records, original_ok, bootstrap, rng)
                   for name, predictions in metrics]

    fallback_label = "valid-option-letter" if variable_options else "A-D"
    lines = [f"# TGCD results (N = {len(records)})\n",
             f"Configuration: `{method}`, alpha={alpha:g}; {selection_source}.",
             f"Missing {fallback_label} fallbacks: {missing_variants + missing_noimage}/"
             f"{variant_score_count + len(records)} score sets "
             f"({missing_variants}/{variant_score_count} variant; "
             f"{missing_noimage}/{len(records)} no-image).",
             "\n## Accuracy and paired inference",
             "| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        lines.append(f"| {row['name']} | {row['accuracy']:.1f}% | "
                     f"{row['ci'][0]:.1f}-{row['ci'][1]:.1f}% | {row['orig_only']} | "
                     f"{row['method_only']} | {row['p']:.4f} |")

    lines.extend(["\n## Single-view accuracy",
                  "| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for variant in ("orig", "mirror", "reorder", "shuffle"):
        subset = [record for record in records if variant in record["variants"]]
        if not subset:
            continue
        predictions = [record["variants"][variant]["pred"] for record in subset]
        originals = [record["variants"]["orig"]["pred"] for record in subset]
        correct = [prediction == record["answer"]
                   for prediction, record in zip(predictions, subset)]
        original_correct = [prediction == record["answer"]
                            for prediction, record in zip(originals, subset)]
        interval = bootstrap_ci(
            len(subset),
            lambda indices, values=correct: 100.0 * sum(values[i] for i in indices)
            / len(indices), bootstrap, rng)
        orig_only, view_only, p_value = mcnemar_exact(original_correct, correct)
        lines.append(f"| {variant} | {len(subset)} | {_accuracy(predictions, subset):.1f}% | "
                     f"{interval[0]:.1f}-{interval[1]:.1f}% | "
                     f"{_accuracy(originals, subset):.1f}% | {orig_only} | {view_only} | "
                     f"{p_value:.4f} |")

    if not selection_path:
        lines.extend(["\n## Offline alpha sweep",
                      "| aggregation | alpha | accuracy | VCD-style accuracy |",
                      "|---|---:|---:|---:|"])
        for aggregation in AGGREGATIONS:
            for grid_alpha in ALPHAS:
                ensemble_predictions = [_ensemble(record, aggregation, grid_alpha)
                                        for record in records]
                vcd_grid = [_vcd(record, grid_alpha) for record in records]
                lines.append(f"| {aggregation} | {grid_alpha:g} | "
                             f"{_accuracy(ensemble_predictions, records):.1f}% | "
                             f"{_accuracy(vcd_grid, records):.1f}% |")

    lines.extend(["\n## Flip rate after contrast decoding",
                  "Each transformed view is contrast-decoded with the selected alpha and compared "
                  "with the contrast-decoded original view. The earlier midterm range was 24-35%.",
                  "| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |",
                  "|---|---:|---:|---:|---:|"])
    for variant in ("mirror", "reorder", "shuffle"):
        subset = [record for record in records if variant in record["variants"]]
        if not subset:
            continue
        raw_flips = [record["variants"][variant]["pred"] != record["variants"]["orig"]["pred"]
                     for record in subset]
        contrast_flips = [_single_contrast(record, variant, alpha) !=
                          _single_contrast(record, "orig", alpha) for record in subset]
        interval = bootstrap_ci(
            len(subset),
            lambda indices, values=contrast_flips: 100.0 * sum(values[i] for i in indices)
            / len(indices), bootstrap, rng)
        lines.append(f"| {variant} | {len(subset)} | "
                     f"{100.0 * sum(raw_flips) / len(subset):.1f}% | "
                     f"{100.0 * sum(contrast_flips) / len(subset):.1f}% | "
                     f"{interval[0]:.1f}-{interval[1]:.1f}% |")

    lines.append("\n## Per question type")
    if variable_options:
        lines.extend([
            "| type | n | chance | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
    else:
        lines.extend([
            "| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
    ensemble_zero = [_ensemble(record, method, 0.0) for record in records]
    for question_type in sorted({record["question_type"] for record in records}):
        indices = [i for i, record in enumerate(records) if record["question_type"] == question_type]
        def subset_accuracy(predictions):
            return 100.0 * sum(predictions[i] == records[i]["answer"] for i in indices) / len(indices)
        chance = 100.0 * sum(1.0 / _option_count(records[i]) for i in indices) / len(indices)
        chance_cell = f"{chance:.1f}% | " if variable_options else ""
        lines.append(f"| {question_type} | {len(indices)} | {chance_cell}"
                     f"{subset_accuracy(original_predictions):.1f}% | "
                     f"{subset_accuracy(noimage_predictions):.1f}% | "
                     f"{subset_accuracy(sc_predictions):.1f}% | "
                     f"{subset_accuracy(ensemble_zero):.1f}% | "
                     f"{subset_accuracy(vcd_predictions):.1f}% | "
                     f"{subset_accuracy(tgcd_predictions):.1f}% |")

    if not selection_path and make_figure:
        figure = _figure(outdir, grid, 100.0 * sum(original_ok) / len(records))
        if figure:
            lines.append(f"\nFigure: `{os.path.basename(figure)}`.")
    lines.append(f"\nBootstrap resamples: {bootstrap}; analysis seed: {seed}.")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(outdir, "tgcd_report.md"), "w") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet")
    parser.add_argument("--analyze", help="analyze an existing scores.jsonl")
    parser.add_argument("--selection",
                        help="dev tgcd_selection.json; required to keep test configuration fixed")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include", nargs="+", default=[],
                        help="prediction files whose question ids define the dev split")
    parser.add_argument("--exclude", nargs="+", default=[],
                        help="prediction files whose question ids are excluded from held-out test")
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--hf_overrides", help="model config overrides as JSON")
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--sc_temp", type=float, default=0.7)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--analysis_seed", type=int, default=0)
    parser.add_argument("--no_figures", action="store_true")
    parser.add_argument("--out", default="results_tgcd")
    arguments = parser.parse_args()
    if arguments.analyze:
        analyze(arguments.analyze, selection_path=arguments.selection,
                bootstrap=arguments.bootstrap, seed=arguments.analysis_seed,
                make_figure=not arguments.no_figures)
    else:
        if not arguments.parquet:
            parser.error("--parquet is required unless --analyze is used")
        run(arguments)

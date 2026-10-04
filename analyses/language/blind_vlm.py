"""Text-only Qwen scoring and analysis for Step E.

The model receives no image tensors. It scores the next-token logits for
the four capital letters A-D after a chat prompt that requests one letter.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import CATEGORY_ORDER, LETTERS, OUTPUTS, load_text, options

PRIMARY_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct"
RAW_PATH = OUTPUTS / "05_blind_vlm_raw.jsonl"
SEED = 0
BOOTSTRAPS = 10_000


def rotate_options(opts: dict[str, str], shift: int) -> tuple[dict[str, str], dict[str, str]]:
    """Left-rotate option contents; return displayed options and label mapping."""
    displayed, mapping = {}, {}
    for displayed_index, displayed_letter in enumerate(LETTERS):
        original_letter = LETTERS[(displayed_index + shift) % len(LETTERS)]
        displayed[displayed_letter] = opts[original_letter]
        mapping[displayed_letter] = original_letter
    return displayed, mapping


def map_logits_to_original(logits: list[float], shift: int) -> np.ndarray:
    """Reorder displayed A-D logits into original-option order."""
    mapped = np.empty(len(LETTERS), dtype=float)
    for displayed_index, value in enumerate(logits):
        original_index = (displayed_index + shift) % len(LETTERS)
        mapped[original_index] = value
    return mapped


def make_prompt(stem: str, opts: dict[str, str], options_only: bool = False) -> str:
    lines = [
        "Select the best answer. Respond with exactly one capital letter: A, B, C, or D.",
    ]
    if options_only:
        lines.append("The question and images are intentionally hidden. Use only the answer choices.")
    else:
        lines.extend(["Question:", stem])
    lines.append("Options:")
    lines.extend(f"{letter}. {opts[letter]}" for letter in LETTERS)
    lines.append("Answer:")
    return "\n".join(lines)


def build_tasks(df: pd.DataFrame) -> list[dict]:
    """Five scored prompts per question.

    The original prompt doubles as circular shift 0 during analysis, avoiding
    a duplicate forward pass. Shifts 1-3 plus original form all four rotations.
    """
    tasks = []
    for row in df.itertuples():
        opts = options(row)
        base = {
            "id": int(row.id),
            "category_en": row.category_en,
            "split": row.split,
            "gold_original": row.answer,
        }
        tasks.append({**base, "condition": "original", "shift": 0,
                      "prompt": make_prompt(row.stem, opts)})
        tasks.append({**base, "condition": "options_only", "shift": 0,
                      "prompt": make_prompt(row.stem, opts, options_only=True)})
        for shift in (1, 2, 3):
            rotated, _ = rotate_options(opts, shift)
            tasks.append({**base, "condition": "circular", "shift": shift,
                          "prompt": make_prompt(row.stem, rotated)})
    return tasks


def read_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[WARN] Ignoring incomplete JSONL line {line_number} in {path}")
    return rows


def load_model(model_name: str, quantization: str, trust_remote_code: bool):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if quantization == "auto":
        try:
            import bitsandbytes  # noqa: F401
            quantization = "4bit" if torch.cuda.is_available() else "none"
        except ImportError:
            quantization = "none"
    kwargs = {"device_map": "auto", "torch_dtype": "auto",
              "low_cpu_mem_usage": True, "trust_remote_code": trust_remote_code}
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    is_vl = "vl" in config.model_type.lower()
    if config.model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(model_name, **kwargs).eval()
    if is_vl:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        tokenizer = processor.tokenizer
        template_owner = processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        template_owner = tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    letter_ids = []
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{letter!r} is not one token for {model_name}: {ids}")
        letter_ids.append(ids[0])
    if len(set(letter_ids)) != 4:
        raise ValueError(f"A-D token ids are not unique: {letter_ids}")
    return model, tokenizer, template_owner, is_vl, letter_ids, quantization


def render_chat(template_owner, prompt: str, is_vl: bool) -> str:
    content = [{"type": "text", "text": prompt}] if is_vl else prompt
    messages = [{"role": "user", "content": content}]
    return template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_tasks(
    tasks: list[dict],
    model_name: str,
    fallback_model: str | None,
    quantization: str,
    batch_size: int,
    raw_path: Path,
    trust_remote_code: bool,
) -> str:
    import torch

    try:
        loaded_name = model_name
        loaded = load_model(model_name, quantization, trust_remote_code)
    except Exception as error:
        if not fallback_model or fallback_model == model_name:
            raise
        print(f"[WARN] Could not load {model_name}: {type(error).__name__}: {error}")
        print(f"[WARN] Falling back to {fallback_model}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        loaded_name = fallback_model
        loaded = load_model(fallback_model, quantization, trust_remote_code)
    model, tokenizer, template_owner, is_vl, letter_ids, actual_quantization = loaded
    print(f"Loaded {loaded_name} ({actual_quantization}); A-D token ids: {letter_ids}")

    existing = read_raw(raw_path)
    completed = {
        (row["model"], int(row["id"]), row["condition"], int(row["shift"]))
        for row in existing
    }
    pending = [
        task for task in tasks
        if (loaded_name, task["id"], task["condition"], task["shift"]) not in completed
    ]
    print(f"Tasks: {len(tasks)} total, {len(tasks) - len(pending)} resumed, {len(pending)} pending")
    max_tokens = max((int(row.get("input_tokens", 0)) for row in existing
                      if row.get("model") == loaded_name), default=0)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    with raw_path.open("a") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            rendered = [render_chat(template_owner, task["prompt"], is_vl) for task in batch]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True,
                                max_length=2048)
            input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            max_tokens = max(max_tokens, max(input_lengths))
            if max(input_lengths) == 2048:
                print("[WARN] At least one prompt reached the 2,048-token truncation limit")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded, use_cache=False).logits
            last = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(batch), device=device)
            scores = logits[rows, last][:, letter_ids].float().cpu().numpy()
            probabilities = torch.softmax(torch.from_numpy(scores), dim=1).numpy()
            for task, score, probability, input_tokens in zip(
                batch, scores, probabilities, input_lengths
            ):
                record = {key: value for key, value in task.items() if key != "prompt"}
                record.update({
                    "model": loaded_name,
                    "quantization": actual_quantization,
                    "input_tokens": int(input_tokens),
                    "pred_display": LETTERS[int(np.argmax(score))],
                    **{f"logit_{letter}": float(score[i]) for i, letter in enumerate(LETTERS)},
                    **{f"prob_{letter}": float(probability[i]) for i, letter in enumerate(LETTERS)},
                })
                record["pred_original"] = LETTERS[
                    (LETTERS.index(record["pred_display"]) + record["shift"]) % 4
                ]
                record["correct"] = record["pred_original"] == record["gold_original"]
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            done = min(start + len(batch), len(pending))
            if done % 100 < batch_size or done == len(pending):
                print(f"  scored {done}/{len(pending)} pending tasks")
    print(f"Maximum rendered prompt length: {max_tokens} tokens")
    return loaded_name


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def circular_questions(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per question using all four mapped cyclic predictions.

    Rotation accuracy, consistency, and all-four-correct are derived from the
    recorded ``pred_original`` column when it is present.  Mean-logit consensus
    is always computed from mapped logits and is kept as a separate metric so
    the two tie-breaking conventions are not conflated.
    """
    original = rows[rows["condition"] == "original"].copy()
    original["condition"] = "circular"
    cyclic = pd.concat([original, rows[rows["condition"] == "circular"]], ignore_index=True)
    use_recorded = "pred_original" in cyclic.columns
    output = []
    for qid, group in cyclic.groupby("id"):
        if set(group["shift"]) != {0, 1, 2, 3}:
            continue
        sorted_group = group.sort_values("shift")
        mapped_logits = np.vstack([
            map_logits_to_original(
                [getattr(row, f"logit_{letter}") for letter in LETTERS], int(row.shift)
            )
            for row in sorted_group.itertuples()
        ])
        consensus = int(np.argmax(mapped_logits.mean(axis=0)))
        gold = LETTERS.index(group["gold_original"].iloc[0])
        if use_recorded:
            predictions = np.array(
                [LETTERS.index(p) for p in sorted_group["pred_original"]]
            )
        else:
            predictions = np.argmax(mapped_logits, axis=1)
        output.append({
            "id": int(qid),
            "category_en": group["category_en"].iloc[0],
            "split": group["split"].iloc[0],
            "gold_original": LETTERS[gold],
            "consensus_prediction": LETTERS[consensus],
            "consensus_correct": consensus == gold,
            "all_correct": bool(np.all(predictions == gold)),
            "consistent": len(set(predictions.tolist())) == 1,
            "rotation_accuracy": float(np.mean(predictions == gold)),
        })
    return pd.DataFrame(output)


def metric_row(scope: str, metric: str, values: np.ndarray, rng) -> dict:
    low, high = bootstrap_ci(values.astype(float), rng)
    return {"scope": scope, "metric": metric, "n": len(values),
            "value": values.mean(), "ci_low": low, "ci_high": high}


def report_expectation(name: str, condition: bool, detail: str) -> None:
    print(f"[{'EXPECTED' if condition else 'SURPRISE'}] {name} -- {detail}")


def analyze(
    raw_path: Path,
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    out = output_dir if output_dir is not None else OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    records = read_raw(raw_path)
    if not records:
        raise ValueError(f"No predictions in {raw_path}")
    available = list(dict.fromkeys(row["model"] for row in records))
    model_name = model_name or available[-1]
    rows = pd.DataFrame([row for row in records if row["model"] == model_name])
    rows = rows.drop_duplicates(["id", "condition", "shift"], keep="last")
    circular = circular_questions(rows)
    rows.to_csv(out / "05_blind_vlm_predictions.csv", index=False)
    circular.to_csv(out / "05_circular_questions.csv", index=False)

    rng = np.random.default_rng(SEED)
    summaries, categories = [], []
    for scope, split in (("all", None), ("confirm", "confirm")):
        direct = rows if split is None else rows[rows["split"] == split]
        circ = circular if split is None else circular[circular["split"] == split]
        for condition in ("original", "options_only"):
            part = direct[direct["condition"] == condition]
            summaries.append(metric_row(scope, f"{condition}_accuracy",
                                        part["correct"].to_numpy(bool), rng))
            letter_share = part["pred_display"].value_counts(normalize=True).reindex(LETTERS, fill_value=0)
            for letter, value in letter_share.items():
                summaries.append({"scope": scope, "metric": f"{condition}_pred_{letter}",
                                  "n": len(part), "value": value, "ci_low": np.nan, "ci_high": np.nan})
        for metric in ("rotation_accuracy", "consensus_correct", "all_correct", "consistent"):
            summaries.append(metric_row(scope, f"circular_{metric}",
                                        circ[metric].to_numpy(float), rng))
        for condition in ("original", "options_only"):
            part = direct[direct["condition"] == condition]
            for category in CATEGORY_ORDER:
                cat = part[part["category_en"] == category]
                categories.append({"scope": scope, "condition": condition, "category_en": category,
                                   "n": len(cat), "accuracy": cat["correct"].mean()})
        for category in CATEGORY_ORDER:
            cat = circ[circ["category_en"] == category]
            categories.append({"scope": scope, "condition": "circular_consensus",
                               "category_en": category, "n": len(cat),
                               "accuracy": cat["consensus_correct"].mean()})
    summary = pd.DataFrame(summaries)
    summary.insert(0, "model", model_name)
    summary = pd.concat([summary, pd.DataFrame([{
        "model": model_name, "scope": "all", "metric": "max_input_tokens",
        "n": len(rows), "value": rows["input_tokens"].max(),
        "ci_low": np.nan, "ci_high": np.nan,
    }])], ignore_index=True)
    summary.to_csv(out / "05_blind_vlm_summary.csv", index=False)
    pd.DataFrame(categories).to_csv(out / "05_blind_vlm_by_category.csv", index=False)
    print(f"Analyzed model: {model_name}; {len(rows)} scored prompts; "
          f"{len(circular)} complete circular question sets")
    print(summary[summary["metric"].str.contains("accuracy|correct|consistent")]
          .round(4).to_string(index=False))
    if len(rows[rows["condition"] == "original"]) >= 1000 and len(circular) >= 1000:
        values = summary.set_index(["scope", "metric"])["value"]
        original = values["all", "original_accuracy"]
        options_only = values["all", "options_only_accuracy"]
        consensus = values["all", "circular_consensus_correct"]
        all_correct = values["all", "circular_all_correct"]
        consistency = values["all", "circular_consistent"]
        max_letter = max(values["all", f"original_pred_{letter}"] for letter in LETTERS)
        by_category = pd.read_csv(out / "05_blind_vlm_by_category.csv")
        original_categories = by_category[
            (by_category["scope"] == "all") & (by_category["condition"] == "original")
        ]["accuracy"]
        category_span = original_categories.max() - original_categories.min()
        print()
        report_expectation("No prompt reaches 2,048 tokens", rows["input_tokens"].max() < 2048,
                           f"max {rows['input_tokens'].max()}")
        report_expectation("Original blind accuracy is 25-35%", 0.25 <= original <= 0.35,
                           f"{original:.1%}")
        report_expectation("Options-only is within 5 points of original",
                           abs(options_only - original) <= 0.05,
                           f"{options_only:.1%} vs {original:.1%}")
        report_expectation("Circular consensus is within 5 points of original",
                           abs(consensus - original) <= 0.05,
                           f"{consensus:.1%} vs {original:.1%}")
        report_expectation("All-four-correct is at least 5 points below original",
                           all_correct <= original - 0.05,
                           f"{all_correct:.1%} vs {original:.1%}")
        report_expectation("Circular content consistency is below 80%", consistency < 0.80,
                           f"{consistency:.1%}")
        report_expectation("No original displayed letter exceeds 45%", max_letter <= 0.45,
                           f"max {max_letter:.1%}")
        report_expectation("Original category accuracy spans at least 15 points",
                           category_span >= 0.15, f"{category_span:.1%}")
    else:
        print("[PARTIAL] Full-data expectations deferred until all 1,000 questions are scored.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL)
    parser.add_argument("--quantization", choices=("auto", "4bit", "none"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Score only the first N questions (smoke test)")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--analysis-model", help="Model name to select from a multi-model JSONL")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for CSV outputs (default: outputs/)")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        analyze(args.raw_path, args.analysis_model, output_dir=args.output_dir)
        return
    df = load_text()
    tags = pd.read_csv(OUTPUTS / "01_tags.csv")
    assert (df["id"].to_numpy() == tags["id"].to_numpy()).all()
    df = df.assign(split=tags["split"].to_numpy())
    if args.limit:
        df = df.iloc[:args.limit]
    tasks = build_tasks(df)
    loaded_name = score_tasks(
        tasks, args.model, args.fallback_model, args.quantization,
        args.batch_size, args.raw_path, args.trust_remote_code,
    )
    analyze(args.raw_path, loaded_name, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

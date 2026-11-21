"""Collect option-letter scores with and without images for training-data filtering.

For every row the model scores the original question with its images and the same
question with no images. The no-image distribution is the text prior used as the bias
model during training; questions it already answers confidently are filtered out
offline (multimodal_subset.py uses the same rule on MMSI).

    python prior_scores.py --parquet SAT_train20k.parquet --out results_x
    python prior_scores.py --parquet some.parquet --n 20 --backend mock --out results_mock

Output: <out>/prior_scores.jsonl with id, question_type, answer, n_options, mirror_answer,
orig{scores,pred,missing_letters}, noimage{scores,pred,missing_letters}.
"""
import argparse
import json
import os
import sys

from mmsi_consistency import (MockBackend, VLLMBackend, build_score_messages, load_rows,
                              option_letters, stratified_sample)
from tgcd import argmax_score, noimage_messages


def original_view(row):
    return dict(name="orig", question=row["question"], order=list(range(len(row["images"]))),
                flip=False, perm=None, note="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--n", type=int, default=0, help="0 = all rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = load_rows(args.parquet)
    if args.n:
        rows = stratified_sample(rows, args.n, args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)

    backend = MockBackend(args.seed) if args.backend == "mock" else VLLMBackend(
        args.model, tp=args.tp)
    letters = [option_letters(row["question"]) for row in rows]
    convs = ([build_score_messages(row, original_view(row), args.max_side) for row in rows]
             + [noimage_messages(row) for row in rows])
    print(f"{len(rows)} questions, {len(convs)} scoring prompts", file=sys.stderr)
    scored = backend.score_letters(convs, letters + letters)

    path = os.path.join(args.out, "prior_scores.jsonl")
    with open(path, "w") as handle:
        for i, row in enumerate(rows):
            record = {"id": int(row["id"]), "question_type": row["question_type"],
                      "answer": row["answer"], "n_options": len(letters[i]),
                      "mirror_answer": row.get("mirror_answer")}
            for name, (scores, missing) in (("orig", scored[i]),
                                            ("noimage", scored[len(rows) + i])):
                record[name] = {"scores": scores, "pred": argmax_score(scores),
                                "missing_letters": missing}
            handle.write(json.dumps(record) + "\n")
    for name in ("orig", "noimage"):
        records = [json.loads(line) for line in open(path)]
        accuracy = 100.0 * sum(r[name]["pred"] == r["answer"] for r in records) / len(records)
        missing = sum(r[name]["missing_letters"] for r in records)
        print(f"{name}: accuracy {accuracy:.1f}%, missing-letter fallbacks {missing}",
              file=sys.stderr)
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()

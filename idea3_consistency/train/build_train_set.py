import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multimodal_subset import probs


def normalized_logprobs(scores):
    probabilities = probs(scores)
    return {letter: math.log(probability) for letter, probability in probabilities.items()}


def filter_rows(rows, priors, max_conf):
    prior_by_id = {int(record["id"]): record for record in priors}
    counts = collections.defaultdict(lambda: {"kept": 0, "dropped": 0})
    kept = []
    for source in rows:
        row = dict(source)
        prior = prior_by_id[int(row["id"])]
        noimage = prior["noimage"]
        drop = (bool(noimage["missing_letters"])
                or max(probs(noimage["scores"]).values()) >= max_conf)
        outcome = "dropped" if drop else "kept"
        counts[str(row["question_type"])][outcome] += 1
        if not drop:
            row["noimage_logprobs"] = json.dumps(
                normalized_logprobs(noimage["scores"]), sort_keys=True)
            kept.append(row)
    return kept, counts


def build_train_set(parquet_path, priors_path, max_conf, out_path):
    import pandas as pd

    rows = pd.read_parquet(parquet_path).to_dict("records")
    with open(priors_path) as handle:
        priors = [json.loads(line) for line in handle]
    kept, counts = filter_rows(rows, priors, max_conf)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pd.DataFrame(kept, columns=list(rows[0]) + ["noimage_logprobs"]).to_parquet(
        out_path, index=False)
    for question_type in sorted(counts):
        values = counts[question_type]
        print(f"{question_type}: kept {values['kept']}, dropped {values['dropped']}")
    mirrorable = sum(
        value is not None and not (isinstance(value, float) and math.isnan(value))
        for value in (row.get("mirror_answer") for row in kept))
    print(f"wrote {len(kept)} rows to {out_path}; {mirrorable} kept rows have mirror_answer")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--priors", required=True)
    parser.add_argument("--max_conf", type=float, default=0.7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_train_set(args.parquet, args.priors, args.max_conf, args.out)


if __name__ == "__main__":
    main()

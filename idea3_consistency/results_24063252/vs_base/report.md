# Model A (ce LoRA, train_24047101) vs base Qwen2.5-VL-7B on MMSI test 700

Base: results_23689474/scores.jsonl. Model A: results_24063252/scores.jsonl. Same 700 ids. Subset ids: multimodal_subset/subset_ids.json (results_23689474/conf, rule from the base no-image scores, no labels). Bootstrap 1000, seed 0; exact McNemar.

| subset | model | n | orig acc | 95% CI | McNemar vs base (base-only / model-only / p) | no-image acc | orig = no-image |
|---|---|---:|---:|---:|---:|---:|---:|
| all | base | 700 | 28.7% | 25.3-32.0% | - | 26.3% | 52.7% |
| all | A_ce | 700 | 29.0% | 25.6-32.6% | 87 / 89 / 0.940 | 25.1% | 38.3% |
| conf (no-image max prob < 0.7, from base) | base | 597 | 26.1% | 22.8-29.3% | - | 22.9% | 49.4% |
| conf (no-image max prob < 0.7, from base) | A_ce | 597 | 29.8% | 26.1-33.7% | 57 / 79 / 0.071 | 22.3% | 39.0% |

| model | transform | n | raw flip rate | 95% CI | flips that match the no-image answer |
|---|---|---:|---:|---:|---:|
| base | mirror | 534 | 32.2% | 28.7-36.3% | 51/172 (29.7%) |
| base | reorder | 700 | 28.4% | 24.9-31.7% | 73/199 (36.7%) |
| base | shuffle | 700 | 41.4% | 38.0-45.1% | 54/290 (18.6%) |
| A_ce | mirror | 534 | 34.6% | 30.9-38.6% | 54/185 (29.2%) |
| A_ce | reorder | 700 | 34.0% | 30.4-37.6% | 85/238 (35.7%) |
| A_ce | shuffle | 700 | 26.4% | 23.4-29.7% | 44/185 (23.8%) |

Agreement of A orig with the base no-image prediction: 35.9%. A no-image = base no-image: 69.1%.
Predicted letter counts (A/B/C/D): base orig 60/168/214/258, A orig 120/159/186/235, gold 172/182/187/159.

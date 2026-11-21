# Base vs model A (ce, train_24047101) vs model B (lmh + mirror, train_24047102) on MMSI test 700

Base: results_23689474. A: results_24063252 (selection results_24063250). B: results_24197218 (selection results_24197216, prob_mean alpha=0.25). Subset ids: multimodal_subset/subset_ids.json (results_23689474/conf). Bootstrap 1000, seed 0; exact McNemar.

    python compare_models.py results_23689474/scores.jsonl multimodal_subset/subset_ids.json A_ce=results_24063252/scores.jsonl B_lmh_mirror=results_24197218/scores.jsonl

| subset | model | n | orig acc | 95% CI | McNemar vs base (base-only / model-only / p) | no-image acc | orig = no-image |
|---|---|---:|---:|---:|---:|---:|---:|
| all | base | 700 | 28.7% | 25.3-32.0% | - | 26.3% | 52.7% |
| all | A_ce | 700 | 29.0% | 25.6-32.6% | 87 / 89 / 0.940 | 25.1% | 38.3% |
| all | B_lmh_mirror | 700 | 27.9% | 24.6-31.4% | 95 / 89 / 0.713 | 26.0% | 40.7% |
| conf (no-image max prob < 0.7, from base) | base | 597 | 26.1% | 22.8-29.3% | - | 22.9% | 49.4% |
| conf (no-image max prob < 0.7, from base) | A_ce | 597 | 29.8% | 26.1-33.7% | 57 / 79 / 0.071 | 22.3% | 39.0% |
| conf (no-image max prob < 0.7, from base) | B_lmh_mirror | 597 | 27.5% | 24.0-31.0% | 69 / 77 / 0.563 | 23.3% | 41.5% |

| model | transform | n | raw flip rate | 95% CI | flips that match the no-image answer |
|---|---|---:|---:|---:|---:|
| base | mirror | 534 | 32.2% | 28.7-36.3% | 51/172 (29.7%) |
| base | reorder | 700 | 28.4% | 24.9-31.7% | 73/199 (36.7%) |
| base | shuffle | 700 | 41.4% | 38.0-45.1% | 54/290 (18.6%) |
| A_ce | mirror | 534 | 34.6% | 30.9-38.6% | 54/185 (29.2%) |
| A_ce | reorder | 700 | 34.0% | 30.4-37.6% | 85/238 (35.7%) |
| A_ce | shuffle | 700 | 26.4% | 23.4-29.7% | 44/185 (23.8%) |
| B_lmh_mirror | mirror | 534 | 37.8% | 34.1-41.9% | 53/202 (26.2%) |
| B_lmh_mirror | reorder | 700 | 33.3% | 30.0-37.0% | 86/233 (36.9%) |
| B_lmh_mirror | shuffle | 700 | 27.7% | 24.6-31.1% | 47/194 (24.2%) |

A vs B paired: all 700 A-only 52 / B-only 44 (p=0.475); subset 597 A-only 49 / B-only 35 (p=0.156).
TGCD on B (alpha=0.25 from dev): 28.7% (p=0.58 vs B orig).
Predicted letter counts (A/B/C/D): B orig 178/121/158/243.

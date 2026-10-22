# Idea 3 failure analysis  (N = 700 questions, source `results_10892041/predictions.jsonl`)

## 1. Transform-subset ablation (equal-compute self-consistency)
| transforms | calls/q | transform vote | self-consistency | McNemar p | AUROC | consistency |
|---|---|---|---|---|---|---|
| mirror | 1.76 | 29.4% | 29.0% | 0.824 | 0.537 | 75.9% |
| reorder | 2.00 | 29.4% | 28.9% | 0.704 | 0.505 | 75.6% |
| shuffle | 2.00 | 29.4% | 28.9% | 0.704 | 0.553 | 65.3% |
| mirror+reorder | 2.76 | 30.0% | 28.9% | 0.466 | 0.527 | 60.0% |
| mirror+shuffle | 2.76 | 30.7% | 28.9% | 0.187 | 0.565 | 50.1% |
| reorder+shuffle | 3.00 | 29.6% | 29.3% | 0.919 | 0.550 | 50.3% |
| mirror+reorder+shuffle | 3.76 | 30.1% | 28.7% | 0.302 | 0.556 | 41.0% |

## 2. Error taxonomy
| question type | n | Correct, consistent | Correct, inconsistent | Wrong, inconsistent (detectable) | Wrong, consistent (confident error) |
|---|---|---|---|---|---|
| All | 700 | 13.9% | 15.6% | 43.4% | 27.1% |
| Attribute (Appr.) | 38 | 5.3% | 10.5% | 42.1% | 42.1% |
| Attribute (Meas.) | 36 | 8.3% | 11.1% | 30.6% | 50.0% |
| MSR | 170 | 11.8% | 17.6% | 44.1% | 26.5% |
| Motion (Cam.) | 47 | 8.5% | 4.3% | 68.1% | 19.1% |
| Motion (Obj.) | 49 | 18.4% | 18.4% | 40.8% | 22.4% |
| Positional Relationship (Cam.–Cam.) | 66 | 22.7% | 10.6% | 45.5% | 21.2% |
| Positional Relationship (Cam.–Obj.) | 59 | 13.6% | 20.3% | 50.8% | 15.3% |
| Positional Relationship (Cam.–Reg.) | 56 | 10.7% | 26.8% | 44.6% | 17.9% |
| Positional Relationship (Obj.–Obj.) | 67 | 17.9% | 11.9% | 35.8% | 34.3% |
| Positional Relationship (Obj.–Reg.) | 58 | 19.0% | 13.8% | 36.2% | 31.0% |
| Positional Relationship (Reg.–Reg.) | 54 | 13.0% | 18.5% | 37.0% | 31.5% |

Of 494 wrong original answers, 190 (38.5%) are confident errors that no transformation exposes.

## 3. Flip outcomes per transform
| transform | n | flips | correct->wrong | wrong->correct | wrong->other wrong |
|---|---|---|---|---|---|
| mirror | 534 | 169 | 39 | 52 | 78 |
| reorder | 700 | 171 | 49 | 52 | 70 |
| shuffle | 700 | 243 | 56 | 80 | 107 |

## 4. Mirror: image-following vs lexical anchoring
Defined on questions whose original answer text has a left/right-swapped twin among the mirrored options (n=237).
- same letter (follows the flipped image): 130 (54.9%)
- swapped twin (keeps the original word, ignores the flip): 60 (25.3%)
- other option: 47 (19.8%)
- among changed answers, share going to the swapped twin: 56.1% (about 33% if random)

## 5. Shuffle: content vs position anchoring
Defined when the permutation moved the original answer (n=527).
- same content (consistent): 335 (63.6%)
- same displayed letter, different content: 70 (13.3%)
- other: 122 (23.1%)
- among changed answers, share keeping the displayed letter: 36.5% (about 33% if random)

## 6. Reorder flip rate by image count and explicit image references
| subset | n | flip rate |
|---|---|---|
| 2 images | 536 | 23.7% |
| 3+ images | 164 | 26.8% |
| mentions 'Image/Figure k' | 151 | 25.8% |
| no explicit image index | 549 | 24.0% |

Figure: fig_error_taxonomy.png. Per-question tags for picking qualitative examples: `cases.csv`.

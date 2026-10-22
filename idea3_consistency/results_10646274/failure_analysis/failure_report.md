# Idea 3 failure analysis  (N = 100 questions, source `results_10646274/predictions.jsonl`)

## 1. Transform-subset ablation (equal-compute self-consistency)
| transforms | calls/q | transform vote | self-consistency | McNemar p | AUROC | consistency |
|---|---|---|---|---|---|---|
| mirror | 1.74 | 19.0% | 17.0% | 0.804 | 0.480 | 77.0% |
| reorder | 2.00 | 19.0% | 18.0% | 1.000 | 0.594 | 69.0% |
| shuffle | 2.00 | 19.0% | 18.0% | 1.000 | 0.499 | 58.0% |
| mirror+reorder | 2.74 | 21.0% | 20.0% | 1.000 | 0.588 | 54.0% |
| mirror+shuffle | 2.74 | 19.0% | 20.0% | 1.000 | 0.514 | 45.0% |
| reorder+shuffle | 3.00 | 23.0% | 21.0% | 0.791 | 0.566 | 41.0% |
| mirror+reorder+shuffle | 3.74 | 22.0% | 19.0% | 0.581 | 0.576 | 32.0% |

## 2. Error taxonomy
| question type | n | Correct, consistent | Correct, inconsistent | Wrong, inconsistent (detectable) | Wrong, consistent (confident error) |
|---|---|---|---|---|---|
| All | 100 | 7.0% | 12.0% | 56.0% | 25.0% |
| Attribute (Appr.) | 10 | 0.0% | 20.0% | 50.0% | 30.0% |
| Attribute (Meas.) | 9 | 0.0% | 0.0% | 66.7% | 33.3% |
| MSR | 9 | 11.1% | 11.1% | 44.4% | 33.3% |
| Motion (Cam.) | 9 | 0.0% | 11.1% | 66.7% | 22.2% |
| Motion (Obj.) | 9 | 0.0% | 44.4% | 44.4% | 11.1% |
| Positional Relationship (Cam.–Cam.) | 9 | 11.1% | 11.1% | 44.4% | 33.3% |
| Positional Relationship (Cam.–Obj.) | 9 | 0.0% | 11.1% | 66.7% | 22.2% |
| Positional Relationship (Cam.–Reg.) | 9 | 11.1% | 22.2% | 55.6% | 11.1% |
| Positional Relationship (Obj.–Obj.) | 9 | 22.2% | 0.0% | 44.4% | 33.3% |
| Positional Relationship (Obj.–Reg.) | 9 | 11.1% | 0.0% | 77.8% | 11.1% |
| Positional Relationship (Reg.–Reg.) | 9 | 11.1% | 0.0% | 55.6% | 33.3% |

Of 81 wrong original answers, 25 (30.9%) are confident errors that no transformation exposes.

## 3. Flip outcomes per transform
| transform | n | flips | correct->wrong | wrong->correct | wrong->other wrong |
|---|---|---|---|---|---|
| mirror | 74 | 23 | 5 | 8 | 10 |
| reorder | 100 | 31 | 3 | 16 | 12 |
| shuffle | 100 | 42 | 8 | 20 | 14 |

## 4. Mirror: image-following vs lexical anchoring
Defined on questions whose original answer text has a left/right-swapped twin among the mirrored options (n=32).
- same letter (follows the flipped image): 15 (46.9%)
- swapped twin (keeps the original word, ignores the flip): 7 (21.9%)
- other option: 10 (31.2%)
- among changed answers, share going to the swapped twin: 41.2% (about 33% if random)

## 5. Shuffle: content vs position anchoring
Defined when the permutation moved the original answer (n=77).
- same content (consistent): 38 (49.4%)
- same displayed letter, different content: 24 (31.2%)
- other: 15 (19.5%)
- among changed answers, share keeping the displayed letter: 61.5% (about 33% if random)

## 6. Reorder flip rate by image count and explicit image references
| subset | n | flip rate |
|---|---|---|
| 2 images | 91 | 31.9% |
| 3+ images | 9 | 22.2% |
| mentions 'Image/Figure k' | 27 | 40.7% |
| no explicit image index | 73 | 27.4% |

Figure: fig_error_taxonomy.png. Per-question tags for picking qualitative examples: `cases.csv`.

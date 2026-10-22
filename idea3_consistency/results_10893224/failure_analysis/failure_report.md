# Idea 3 failure analysis  (N = 700 questions, source `results_10893224/predictions.jsonl`)

## 1. Transform-subset ablation (equal-compute self-consistency)
| transforms | calls/q | transform vote | self-consistency | McNemar p | AUROC | consistency |
|---|---|---|---|---|---|---|
| mirror | 1.76 | 26.9% | 27.9% | 0.489 | 0.540 | 73.4% |
| reorder | 2.00 | 26.9% | 27.4% | 0.689 | 0.518 | 69.7% |
| shuffle | 2.00 | 26.9% | 27.4% | 0.689 | 0.543 | 61.3% |
| mirror+reorder | 2.76 | 27.4% | 27.7% | 0.910 | 0.537 | 53.7% |
| mirror+shuffle | 2.76 | 27.7% | 27.7% | 1.000 | 0.563 | 45.1% |
| reorder+shuffle | 3.00 | 27.3% | 28.3% | 0.505 | 0.547 | 44.4% |
| mirror+reorder+shuffle | 3.76 | 27.0% | 26.4% | 0.724 | 0.562 | 34.6% |

## 2. Error taxonomy
| question type | n | Correct, consistent | Correct, inconsistent | Wrong, inconsistent (detectable) | Wrong, consistent (confident error) |
|---|---|---|---|---|---|
| All | 700 | 11.1% | 15.7% | 49.7% | 23.4% |
| Attribute (Appr.) | 38 | 5.3% | 15.8% | 50.0% | 28.9% |
| Attribute (Meas.) | 36 | 16.7% | 5.6% | 30.6% | 47.2% |
| MSR | 170 | 9.4% | 14.7% | 51.8% | 24.1% |
| Motion (Cam.) | 47 | 8.5% | 12.8% | 55.3% | 23.4% |
| Motion (Obj.) | 49 | 10.2% | 20.4% | 46.9% | 22.4% |
| Positional Relationship (Cam.–Cam.) | 66 | 9.1% | 10.6% | 63.6% | 16.7% |
| Positional Relationship (Cam.–Obj.) | 59 | 13.6% | 25.4% | 47.5% | 13.6% |
| Positional Relationship (Cam.–Reg.) | 56 | 16.1% | 12.5% | 55.4% | 16.1% |
| Positional Relationship (Obj.–Obj.) | 67 | 7.5% | 25.4% | 46.3% | 20.9% |
| Positional Relationship (Obj.–Reg.) | 58 | 24.1% | 8.6% | 41.4% | 25.9% |
| Positional Relationship (Reg.–Reg.) | 54 | 5.6% | 18.5% | 46.3% | 29.6% |

Of 512 wrong original answers, 164 (32.0%) are confident errors that no transformation exposes.

## 3. Flip outcomes per transform
| transform | n | flips | correct->wrong | wrong->correct | wrong->other wrong |
|---|---|---|---|---|---|
| mirror | 534 | 186 | 39 | 50 | 97 |
| reorder | 700 | 212 | 52 | 59 | 101 |
| shuffle | 700 | 271 | 61 | 86 | 124 |

## 4. Mirror: image-following vs lexical anchoring
Defined on questions whose original answer text has a left/right-swapped twin among the mirrored options (n=285).
- same letter (follows the flipped image): 150 (52.6%)
- swapped twin (keeps the original word, ignores the flip): 102 (35.8%)
- other option: 33 (11.6%)
- among changed answers, share going to the swapped twin: 75.6% (about 33% if random)

## 5. Shuffle: content vs position anchoring
Defined when the permutation moved the original answer (n=526).
- same content (consistent): 290 (55.1%)
- same displayed letter, different content: 96 (18.3%)
- other: 140 (26.6%)
- among changed answers, share keeping the displayed letter: 40.7% (about 33% if random)

## 6. Reorder flip rate by image count and explicit image references
| subset | n | flip rate |
|---|---|---|
| 2 images | 536 | 29.1% |
| 3+ images | 164 | 34.1% |
| mentions 'Image/Figure k' | 151 | 30.5% |
| no explicit image index | 549 | 30.2% |

Figure: fig_error_taxonomy.png. Per-question tags for picking qualitative examples: `cases.csv`.

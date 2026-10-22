# Idea 3 failure analysis  (N = 300 questions, source `results_10648545/predictions.jsonl`)

## 1. Transform-subset ablation (equal-compute self-consistency)
| transforms | calls/q | transform vote | self-consistency | McNemar p | AUROC | consistency |
|---|---|---|---|---|---|---|
| mirror | 1.75 | 27.3% | 27.0% | 1.000 | 0.533 | 75.7% |
| reorder | 2.00 | 27.3% | 26.7% | 0.871 | 0.523 | 65.0% |
| shuffle | 2.00 | 27.3% | 26.7% | 0.871 | 0.513 | 60.3% |
| mirror+reorder | 2.75 | 28.7% | 27.7% | 0.784 | 0.533 | 51.7% |
| mirror+shuffle | 2.75 | 28.0% | 27.7% | 1.000 | 0.524 | 47.0% |
| reorder+shuffle | 3.00 | 28.0% | 27.7% | 1.000 | 0.532 | 40.0% |
| mirror+reorder+shuffle | 3.75 | 28.0% | 27.3% | 0.880 | 0.543 | 32.7% |

## 2. Error taxonomy
| question type | n | Correct, consistent | Correct, inconsistent | Wrong, inconsistent (detectable) | Wrong, consistent (confident error) |
|---|---|---|---|---|---|
| All | 300 | 10.7% | 16.7% | 50.7% | 22.0% |
| Attribute (Appr.) | 28 | 0.0% | 25.0% | 35.7% | 39.3% |
| Attribute (Meas.) | 28 | 10.7% | 14.3% | 50.0% | 25.0% |
| MSR | 28 | 10.7% | 25.0% | 46.4% | 17.9% |
| Motion (Cam.) | 27 | 0.0% | 7.4% | 74.1% | 18.5% |
| Motion (Obj.) | 27 | 11.1% | 25.9% | 48.1% | 14.8% |
| Positional Relationship (Cam.–Cam.) | 27 | 14.8% | 11.1% | 55.6% | 18.5% |
| Positional Relationship (Cam.–Obj.) | 27 | 3.7% | 18.5% | 59.3% | 18.5% |
| Positional Relationship (Cam.–Reg.) | 27 | 14.8% | 14.8% | 51.9% | 18.5% |
| Positional Relationship (Obj.–Obj.) | 27 | 14.8% | 11.1% | 51.9% | 22.2% |
| Positional Relationship (Obj.–Reg.) | 27 | 22.2% | 14.8% | 40.7% | 22.2% |
| Positional Relationship (Reg.–Reg.) | 27 | 14.8% | 14.8% | 44.4% | 25.9% |

Of 218 wrong original answers, 66 (30.3%) are confident errors that no transformation exposes.

## 3. Flip outcomes per transform
| transform | n | flips | correct->wrong | wrong->correct | wrong->other wrong |
|---|---|---|---|---|---|
| mirror | 225 | 73 | 16 | 23 | 34 |
| reorder | 300 | 105 | 26 | 40 | 39 |
| shuffle | 300 | 119 | 31 | 42 | 46 |

## 4. Mirror: image-following vs lexical anchoring
Defined on questions whose original answer text has a left/right-swapped twin among the mirrored options (n=83).
- same letter (follows the flipped image): 42 (50.6%)
- swapped twin (keeps the original word, ignores the flip): 25 (30.1%)
- other option: 16 (19.3%)
- among changed answers, share going to the swapped twin: 61.0% (about 33% if random)

## 5. Shuffle: content vs position anchoring
Defined when the permutation moved the original answer (n=237).
- same content (consistent): 133 (56.1%)
- same displayed letter, different content: 48 (20.3%)
- other: 56 (23.6%)
- among changed answers, share keeping the displayed letter: 46.2% (about 33% if random)

## 6. Reorder flip rate by image count and explicit image references
| subset | n | flip rate |
|---|---|---|
| 2 images | 274 | 35.4% |
| 3+ images | 26 | 30.8% |
| mentions 'Image/Figure k' | 76 | 43.4% |
| no explicit image index | 224 | 32.1% |

Figure: fig_error_taxonomy.png. Per-question tags for picking qualitative examples: `cases.csv`.

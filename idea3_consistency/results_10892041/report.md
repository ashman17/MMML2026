# Idea 3 transformation-consistency results  (N = 700 questions)

Model: `Qwen/Qwen2.5-VL-7B-Instruct` | backend: `vllm` | sample seed: 0 | max image side: 768 px

Original accuracy: 29.4% (bootstrap 95% CI 26.0–32.7)
Unparsable original answers: 2

## Per transformation
| transform | n | variant acc | flip rate vs orig | flip 95% CI | orig acc on subset | unparsable |
|---|---|---|---|---|---|---|
| mirror | 534 | 31.8% | 31.6% | 27.7–35.4 | 29.4% | 1 |
| reorder | 700 | 29.9% | 24.4% | 21.4–27.6 | 29.4% | 2 |
| shuffle | 700 | 32.9% | 34.7% | 31.0–38.3 | 29.4% | 2 |

Mirror flip rate, questions with left/right words: 35.0% (n=429)

Mirror flip rate, questions without left/right words: 18.1% (n=105)

## Consistency
Consistency Rate (all variants agree): 41.0%
Accuracy when consistent: 33.8% (n=287) | when inconsistent: 26.4% (n=413)
AUROC of disagreement score for detecting wrong original answers: 0.556 (bootstrap 95% CI 0.510–0.599; 0.5 = chance)

## Voting (same number of model calls per question)
Original greedy: 29.4% | Self-consistency: 28.7% | Transformation vote: 30.1%
Transform-vote minus self-consistency: +1.4 percentage points (bootstrap 95% CI -1.0 to +3.9)
Exact McNemar test: transform-only correct=43, self-consistency-only correct=33, p=0.3019

## Per question type
| type | n | orig acc | consistency | transform vote | self-consistency |
|---|---|---|---|---|---|
| Attribute (Appr.) | 38 | 15.8% | 47.4% | 21.1% | 18.4% |
| Attribute (Meas.) | 36 | 19.4% | 58.3% | 19.4% | 16.7% |
| MSR | 170 | 29.4% | 38.2% | 28.2% | 24.1% |
| Motion (Cam.) | 47 | 12.8% | 27.7% | 14.9% | 21.3% |
| Motion (Obj.) | 49 | 36.7% | 40.8% | 36.7% | 38.8% |
| Positional Relationship (Cam.–Cam.) | 66 | 33.3% | 43.9% | 34.8% | 30.3% |
| Positional Relationship (Cam.–Obj.) | 59 | 33.9% | 28.8% | 33.9% | 35.6% |
| Positional Relationship (Cam.–Reg.) | 56 | 37.5% | 28.6% | 41.1% | 35.7% |
| Positional Relationship (Obj.–Obj.) | 67 | 29.9% | 52.2% | 31.3% | 31.3% |
| Positional Relationship (Obj.–Reg.) | 58 | 32.8% | 50.0% | 32.8% | 34.5% |
| Positional Relationship (Reg.–Reg.) | 54 | 31.5% | 44.4% | 31.5% | 29.6% |

## Flip rate by question type and transformation
| type | transform | n | flip rate |
|---|---|---|---|
| Attribute (Appr.) | mirror | 24 | 33.3% |
| Attribute (Appr.) | reorder | 38 | 23.7% |
| Attribute (Appr.) | shuffle | 38 | 36.8% |
| Attribute (Meas.) | mirror | 36 | 2.8% |
| Attribute (Meas.) | reorder | 36 | 11.1% |
| Attribute (Meas.) | shuffle | 36 | 30.6% |
| MSR | mirror | 119 | 30.3% |
| MSR | reorder | 170 | 27.6% |
| MSR | shuffle | 170 | 36.5% |
| Motion (Cam.) | mirror | 47 | 44.7% |
| Motion (Cam.) | reorder | 47 | 17.0% |
| Motion (Cam.) | shuffle | 47 | 42.6% |
| Motion (Obj.) | mirror | 48 | 25.0% |
| Motion (Obj.) | reorder | 49 | 18.4% |
| Motion (Obj.) | shuffle | 49 | 32.7% |
| Positional Relationship (Cam.–Cam.) | mirror | 58 | 41.4% |
| Positional Relationship (Cam.–Cam.) | reorder | 66 | 25.8% |
| Positional Relationship (Cam.–Cam.) | shuffle | 66 | 25.8% |
| Positional Relationship (Cam.–Obj.) | mirror | 59 | 27.1% |
| Positional Relationship (Cam.–Obj.) | reorder | 59 | 40.7% |
| Positional Relationship (Cam.–Obj.) | shuffle | 59 | 35.6% |
| Positional Relationship (Cam.–Reg.) | mirror | 53 | 37.7% |
| Positional Relationship (Cam.–Reg.) | reorder | 56 | 32.1% |
| Positional Relationship (Cam.–Reg.) | shuffle | 56 | 35.7% |
| Positional Relationship (Obj.–Obj.) | mirror | 36 | 27.8% |
| Positional Relationship (Obj.–Obj.) | reorder | 67 | 23.9% |
| Positional Relationship (Obj.–Obj.) | shuffle | 67 | 32.8% |
| Positional Relationship (Obj.–Reg.) | mirror | 29 | 31.0% |
| Positional Relationship (Obj.–Reg.) | reorder | 58 | 19.0% |
| Positional Relationship (Obj.–Reg.) | shuffle | 58 | 31.0% |
| Positional Relationship (Reg.–Reg.) | mirror | 25 | 48.0% |
| Positional Relationship (Reg.–Reg.) | reorder | 54 | 14.8% |
| Positional Relationship (Reg.–Reg.) | shuffle | 54 | 40.7% |

## Predicted option position (with images)
| variant | position | count | share |
|---|---|---|---|
| orig | A | 74 | 10.6% |
| orig | B | 201 | 28.7% |
| orig | C | 219 | 31.3% |
| orig | D | 204 | 29.1% |
| orig | unparsable | 2 | 0.3% |
| shuffle | A | 83 | 11.9% |
| shuffle | B | 199 | 28.4% |
| shuffle | C | 223 | 31.9% |
| shuffle | D | 193 | 27.6% |
| shuffle | unparsable | 2 | 0.3% |

## Hypothesis evaluation and discussion
- **H1 — supported:** every available answer-preserving transformation produced a non-zero flip rate.
- **H2 — not supported:** disagreement AUROC was 0.556 (95% CI 0.510–0.599); original accuracy was 33.8% when consistent versus 26.4% when inconsistent.
- **H3 — directionally supported, but not statistically conclusive:** transformation voting changed accuracy by +1.4 points relative to self-consistency (95% CI -1.0 to +3.9; McNemar p=0.3019).
- **H4 — descriptively supported:** mirror flip rate was 35.0% for direction-word questions versus 18.1% otherwise.
- **H5 — descriptively supported:** shuffle flip rate was 34.7% and the most frequent predicted shuffled-option position accounted for 31.9% of parseable-position trials (25% would be uniform across four positions).
Improvement direction: transformation voting is a promising training-free baseline, but retain the paired confidence interval and McNemar result when describing whether it reliably beats sampling diversity.
Improvement direction: add mirrored training pairs or a consistency loss for left/right reasoning.
Improvement direction: make image viewpoint labels explicit and evaluate whether the model follows labels rather than presentation order.
Improvement direction: evaluate every model over multiple option orders so that visual benchmarks do not silently reward option-position shortcuts.


Generated figures: 3; bootstrap resamples: 1000; analysis seed: 0.

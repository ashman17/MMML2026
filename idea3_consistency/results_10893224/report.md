# Idea 3 transformation-consistency results  (N = 700 questions)

Model: `OpenGVLab/InternVL3-8B` | backend: `vllm` | sample seed: 0 | max image side: 768 px

Original accuracy: 26.9% (bootstrap 95% CI 24.0–30.3)
Unparsable original answers: 0

## Per transformation
| transform | n | variant acc | flip rate vs orig | flip 95% CI | orig acc on subset | unparsable |
|---|---|---|---|---|---|---|
| mirror | 534 | 29.6% | 34.8% | 30.7–39.0 | 27.5% | 0 |
| reorder | 700 | 27.9% | 30.3% | 26.9–33.7 | 26.9% | 0 |
| shuffle | 700 | 30.4% | 38.7% | 35.1–42.1 | 26.9% | 0 |

Mirror flip rate, questions with left/right words: 41.3% (n=429)

Mirror flip rate, questions without left/right words: 8.6% (n=105)

## Consistency
Consistency Rate (all variants agree): 34.6%
Accuracy when consistent: 32.2% (n=242) | when inconsistent: 24.0% (n=458)
AUROC of disagreement score for detecting wrong original answers: 0.562 (bootstrap 95% CI 0.519–0.606; 0.5 = chance)

## Voting (same number of model calls per question)
Original greedy: 26.9% | Self-consistency: 26.4% | Transformation vote: 27.0%
Transform-vote minus self-consistency: +0.6 percentage points (bootstrap 95% CI -1.7 to +2.7)
Exact McNemar test: transform-only correct=38, self-consistency-only correct=34, p=0.7239

## Per question type
| type | n | orig acc | consistency | transform vote | self-consistency |
|---|---|---|---|---|---|
| Attribute (Appr.) | 38 | 21.1% | 34.2% | 23.7% | 26.3% |
| Attribute (Meas.) | 36 | 22.2% | 63.9% | 22.2% | 22.2% |
| MSR | 170 | 24.1% | 33.5% | 23.5% | 22.4% |
| Motion (Cam.) | 47 | 21.3% | 31.9% | 21.3% | 21.3% |
| Motion (Obj.) | 49 | 30.6% | 32.7% | 30.6% | 32.7% |
| Positional Relationship (Cam.–Cam.) | 66 | 19.7% | 25.8% | 21.2% | 18.2% |
| Positional Relationship (Cam.–Obj.) | 59 | 39.0% | 27.1% | 37.3% | 30.5% |
| Positional Relationship (Cam.–Reg.) | 56 | 28.6% | 32.1% | 28.6% | 26.8% |
| Positional Relationship (Obj.–Obj.) | 67 | 32.8% | 28.4% | 32.8% | 32.8% |
| Positional Relationship (Obj.–Reg.) | 58 | 32.8% | 50.0% | 32.8% | 34.5% |
| Positional Relationship (Reg.–Reg.) | 54 | 24.1% | 35.2% | 25.9% | 29.6% |

## Flip rate by question type and transformation
| type | transform | n | flip rate |
|---|---|---|---|
| Attribute (Appr.) | mirror | 24 | 25.0% |
| Attribute (Appr.) | reorder | 38 | 34.2% |
| Attribute (Appr.) | shuffle | 38 | 42.1% |
| Attribute (Meas.) | mirror | 36 | 2.8% |
| Attribute (Meas.) | reorder | 36 | 19.4% |
| Attribute (Meas.) | shuffle | 36 | 25.0% |
| MSR | mirror | 119 | 37.8% |
| MSR | reorder | 170 | 34.1% |
| MSR | shuffle | 170 | 40.6% |
| Motion (Cam.) | mirror | 47 | 34.0% |
| Motion (Cam.) | reorder | 47 | 12.8% |
| Motion (Cam.) | shuffle | 47 | 44.7% |
| Motion (Obj.) | mirror | 48 | 27.1% |
| Motion (Obj.) | reorder | 49 | 22.4% |
| Motion (Obj.) | shuffle | 49 | 38.8% |
| Positional Relationship (Cam.–Cam.) | mirror | 58 | 46.6% |
| Positional Relationship (Cam.–Cam.) | reorder | 66 | 33.3% |
| Positional Relationship (Cam.–Cam.) | shuffle | 66 | 42.4% |
| Positional Relationship (Cam.–Obj.) | mirror | 59 | 35.6% |
| Positional Relationship (Cam.–Obj.) | reorder | 59 | 40.7% |
| Positional Relationship (Cam.–Obj.) | shuffle | 59 | 32.2% |
| Positional Relationship (Cam.–Reg.) | mirror | 53 | 39.6% |
| Positional Relationship (Cam.–Reg.) | reorder | 56 | 30.4% |
| Positional Relationship (Cam.–Reg.) | shuffle | 56 | 33.9% |
| Positional Relationship (Obj.–Obj.) | mirror | 36 | 36.1% |
| Positional Relationship (Obj.–Obj.) | reorder | 67 | 38.8% |
| Positional Relationship (Obj.–Obj.) | shuffle | 67 | 43.3% |
| Positional Relationship (Obj.–Reg.) | mirror | 29 | 41.4% |
| Positional Relationship (Obj.–Reg.) | reorder | 58 | 15.5% |
| Positional Relationship (Obj.–Reg.) | shuffle | 58 | 34.5% |
| Positional Relationship (Reg.–Reg.) | mirror | 25 | 44.0% |
| Positional Relationship (Reg.–Reg.) | reorder | 54 | 35.2% |
| Positional Relationship (Reg.–Reg.) | shuffle | 54 | 40.7% |

## Predicted option position (with images)
| variant | position | count | share |
|---|---|---|---|
| orig | A | 100 | 14.3% |
| orig | B | 221 | 31.6% |
| orig | C | 202 | 28.9% |
| orig | D | 177 | 25.3% |
| shuffle | A | 102 | 14.6% |
| shuffle | B | 202 | 28.9% |
| shuffle | C | 209 | 29.9% |
| shuffle | D | 187 | 26.7% |

## Hypothesis evaluation and discussion
- **H1 — supported:** every available answer-preserving transformation produced a non-zero flip rate.
- **H2 — not supported:** disagreement AUROC was 0.562 (95% CI 0.519–0.606); original accuracy was 32.2% when consistent versus 24.0% when inconsistent.
- **H3 — directionally supported, but not statistically conclusive:** transformation voting changed accuracy by +0.6 points relative to self-consistency (95% CI -1.7 to +2.7; McNemar p=0.7239).
- **H4 — descriptively supported:** mirror flip rate was 41.3% for direction-word questions versus 8.6% otherwise.
- **H5 — descriptively supported:** shuffle flip rate was 38.7% and the most frequent predicted shuffled-option position accounted for 29.9% of parseable-position trials (25% would be uniform across four positions).
Improvement direction: transformation voting is a promising training-free baseline, but retain the paired confidence interval and McNemar result when describing whether it reliably beats sampling diversity.
Improvement direction: add mirrored training pairs or a consistency loss for left/right reasoning.
Improvement direction: make image viewpoint labels explicit and evaluate whether the model follows labels rather than presentation order.
Improvement direction: evaluate every model over multiple option orders so that visual benchmarks do not silently reward option-position shortcuts.

## AI usage and contribution disclosure (submission draft)
Automated tools assisted with implementation, unit tests, Babel job operation, figure generation, visual transformation preflight, and drafting the statistical interpretation. Wen-Chi Tsai supplied the research specification and remains responsible for checking the results, completing the required human validity sign-off, and approving the submitted text.

Generated figures: 3; bootstrap resamples: 1000; analysis seed: 0.

# Idea 3 transformation-consistency results  (N = 100 questions)

Model: `Qwen/Qwen2.5-VL-7B-Instruct` | backend: `vllm` | sample seed: 0 | max image side: 768 px

Human transformation validity: 30/30 valid (100.0%); review completion 30/30.

Original accuracy: 19.0% (bootstrap 95% CI 12.0–27.0)
Unparsable original answers: 0

## Per transformation
| transform | n | variant acc | flip rate vs orig | flip 95% CI | orig acc on subset | unparsable |
|---|---|---|---|---|---|---|
| mirror | 74 | 23.0% | 31.1% | 21.6–40.5 | 18.9% | 0 |
| reorder | 100 | 32.0% | 31.0% | 22.0–40.0 | 19.0% | 0 |
| shuffle | 100 | 31.0% | 42.0% | 33.0–52.0 | 19.0% | 0 |

Mirror flip rate, questions with left/right words: 39.0% (n=59)

Mirror flip rate, questions without left/right words: 0.0% (n=15)

## Consistency
Consistency Rate (all variants agree): 32.0%
Accuracy when consistent: 21.9% (n=32) | when inconsistent: 17.6% (n=68)
AUROC of disagreement score for detecting wrong original answers: 0.576 (bootstrap 95% CI 0.430–0.692; 0.5 = chance)

## Voting (same number of model calls per question)
Original greedy: 19.0% | Self-consistency: 19.0% | Transformation vote: 22.0%
Transform-vote minus self-consistency: +3.0 percentage points (bootstrap 95% CI -5.0 to +11.0)
Exact McNemar test: transform-only correct=8, self-consistency-only correct=5, p=0.5811

## Per question type
| type | n | orig acc | consistency | transform vote | self-consistency |
|---|---|---|---|---|---|
| Attribute (Appr.) | 10 | 20.0% | 30.0% | 20.0% | 10.0% |
| Attribute (Meas.) | 9 | 0.0% | 33.3% | 0.0% | 22.2% |
| MSR | 9 | 22.2% | 44.4% | 22.2% | 22.2% |
| Motion (Cam.) | 9 | 11.1% | 22.2% | 22.2% | 11.1% |
| Motion (Obj.) | 9 | 44.4% | 11.1% | 44.4% | 44.4% |
| Positional Relationship (Cam.–Cam.) | 9 | 22.2% | 44.4% | 33.3% | 0.0% |
| Positional Relationship (Cam.–Obj.) | 9 | 11.1% | 22.2% | 0.0% | 11.1% |
| Positional Relationship (Cam.–Reg.) | 9 | 33.3% | 22.2% | 33.3% | 33.3% |
| Positional Relationship (Obj.–Obj.) | 9 | 22.2% | 55.6% | 33.3% | 33.3% |
| Positional Relationship (Obj.–Reg.) | 9 | 11.1% | 22.2% | 22.2% | 11.1% |
| Positional Relationship (Reg.–Reg.) | 9 | 11.1% | 44.4% | 11.1% | 11.1% |

## Flip rate by question type and transformation
| type | transform | n | flip rate |
|---|---|---|---|
| Attribute (Appr.) | mirror | 4 | 0.0% |
| Attribute (Appr.) | reorder | 10 | 20.0% |
| Attribute (Appr.) | shuffle | 10 | 60.0% |
| Attribute (Meas.) | mirror | 9 | 11.1% |
| Attribute (Meas.) | reorder | 9 | 44.4% |
| Attribute (Meas.) | shuffle | 9 | 44.4% |
| MSR | mirror | 6 | 16.7% |
| MSR | reorder | 9 | 11.1% |
| MSR | shuffle | 9 | 44.4% |
| Motion (Cam.) | mirror | 9 | 44.4% |
| Motion (Cam.) | reorder | 9 | 11.1% |
| Motion (Cam.) | shuffle | 9 | 55.6% |
| Motion (Obj.) | mirror | 9 | 33.3% |
| Motion (Obj.) | reorder | 9 | 33.3% |
| Motion (Obj.) | shuffle | 9 | 55.6% |
| Positional Relationship (Cam.–Cam.) | mirror | 8 | 25.0% |
| Positional Relationship (Cam.–Cam.) | reorder | 9 | 44.4% |
| Positional Relationship (Cam.–Cam.) | shuffle | 9 | 22.2% |
| Positional Relationship (Cam.–Obj.) | mirror | 9 | 55.6% |
| Positional Relationship (Cam.–Obj.) | reorder | 9 | 44.4% |
| Positional Relationship (Cam.–Obj.) | shuffle | 9 | 44.4% |
| Positional Relationship (Cam.–Reg.) | mirror | 8 | 25.0% |
| Positional Relationship (Cam.–Reg.) | reorder | 9 | 66.7% |
| Positional Relationship (Cam.–Reg.) | shuffle | 9 | 33.3% |
| Positional Relationship (Obj.–Obj.) | mirror | 3 | 33.3% |
| Positional Relationship (Obj.–Obj.) | reorder | 9 | 22.2% |
| Positional Relationship (Obj.–Obj.) | shuffle | 9 | 33.3% |
| Positional Relationship (Obj.–Reg.) | mirror | 4 | 50.0% |
| Positional Relationship (Obj.–Reg.) | reorder | 9 | 22.2% |
| Positional Relationship (Obj.–Reg.) | shuffle | 9 | 44.4% |
| Positional Relationship (Reg.–Reg.) | mirror | 5 | 40.0% |
| Positional Relationship (Reg.–Reg.) | reorder | 9 | 22.2% |
| Positional Relationship (Reg.–Reg.) | shuffle | 9 | 22.2% |

## Predicted option position (with images)
| variant | position | count | share |
|---|---|---|---|
| orig | A | 11 | 11.0% |
| orig | B | 32 | 32.0% |
| orig | C | 23 | 23.0% |
| orig | D | 34 | 34.0% |
| shuffle | A | 13 | 13.0% |
| shuffle | B | 34 | 34.0% |
| shuffle | C | 23 | 23.0% |
| shuffle | D | 30 | 30.0% |

## Hypothesis evaluation and discussion
- **H1 — supported:** every available answer-preserving transformation produced a non-zero flip rate.
- **H2 — not supported:** disagreement AUROC was 0.576 (95% CI 0.430–0.692); original accuracy was 21.9% when consistent versus 17.6% when inconsistent.
- **H3 — directionally supported, but not statistically conclusive:** transformation voting changed accuracy by +3.0 points relative to self-consistency (95% CI -5.0 to +11.0; McNemar p=0.5811).
- **H4 — descriptively supported:** mirror flip rate was 39.0% for direction-word questions versus 0.0% otherwise.
- **H5 — descriptively supported:** shuffle flip rate was 42.0% and the most frequent predicted shuffled-option position accounted for 34.0% of parseable-position trials (25% would be uniform across four positions).
Improvement direction: transformation voting is a promising training-free baseline, but retain the paired confidence interval and McNemar result when describing whether it reliably beats sampling diversity.
Improvement direction: add mirrored training pairs or a consistency loss for left/right reasoning.
Improvement direction: make image viewpoint labels explicit and evaluate whether the model follows labels rather than presentation order.
Improvement direction: evaluate every model over multiple option orders so that visual benchmarks do not silently reward option-position shortcuts.

Generated figures: 3; bootstrap resamples: 200; analysis seed: 0.

Note: with N <= 100, differences of a few points are noisy; treat this as a feasibility signal, not a definitive result.

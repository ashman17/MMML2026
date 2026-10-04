# Idea 3 transformation-consistency results  (N = 300 questions)

Model: `Qwen/Qwen2.5-VL-7B-Instruct` | backend: `vllm` | sample seed: 0 | max image side: 768 px

AI-assisted visual transformation review: 30/30 valid (100.0%); review completion 30/30. Human sign-off is still recommended before submission.

Original accuracy: 27.3% (bootstrap 95% CI 22.7–32.7)
Unparsable original answers: 0

## Per transformation
| transform | n | variant acc | flip rate vs orig | flip 95% CI | orig acc on subset | unparsable |
|---|---|---|---|---|---|---|
| mirror | 225 | 28.4% | 32.4% | 26.2–38.7 | 25.3% | 0 |
| reorder | 300 | 32.0% | 35.0% | 30.0–40.7 | 27.3% | 0 |
| shuffle | 300 | 31.0% | 39.7% | 34.7–44.7 | 27.3% | 0 |

Mirror flip rate, questions with left/right words: 38.1% (n=168)

Mirror flip rate, questions without left/right words: 15.8% (n=57)

## Consistency
Consistency Rate (all variants agree): 32.7%
Accuracy when consistent: 32.7% (n=98) | when inconsistent: 24.8% (n=202)
AUROC of disagreement score for detecting wrong original answers: 0.543 (bootstrap 95% CI 0.477–0.610; 0.5 = chance)

## Voting (same number of model calls per question)
Original greedy: 27.3% | Self-consistency: 27.3% | Transformation vote: 28.0%
Transform-vote minus self-consistency: +0.7 percentage points (bootstrap 95% CI -4.0 to +5.0)
Exact McNemar test: transform-only correct=23, self-consistency-only correct=21, p=0.8804

## Per question type
| type | n | orig acc | consistency | transform vote | self-consistency |
|---|---|---|---|---|---|
| Attribute (Appr.) | 28 | 25.0% | 39.3% | 25.0% | 10.7% |
| Attribute (Meas.) | 28 | 25.0% | 35.7% | 25.0% | 39.3% |
| MSR | 28 | 35.7% | 28.6% | 35.7% | 35.7% |
| Motion (Cam.) | 27 | 7.4% | 18.5% | 11.1% | 11.1% |
| Motion (Obj.) | 27 | 37.0% | 25.9% | 33.3% | 25.9% |
| Positional Relationship (Cam.–Cam.) | 27 | 25.9% | 33.3% | 29.6% | 25.9% |
| Positional Relationship (Cam.–Obj.) | 27 | 22.2% | 22.2% | 18.5% | 25.9% |
| Positional Relationship (Cam.–Reg.) | 27 | 29.6% | 33.3% | 33.3% | 25.9% |
| Positional Relationship (Obj.–Obj.) | 27 | 25.9% | 37.0% | 29.6% | 33.3% |
| Positional Relationship (Obj.–Reg.) | 27 | 37.0% | 44.4% | 37.0% | 33.3% |
| Positional Relationship (Reg.–Reg.) | 27 | 29.6% | 40.7% | 29.6% | 33.3% |

## Flip rate by question type and transformation
| type | transform | n | flip rate |
|---|---|---|---|
| Attribute (Appr.) | mirror | 14 | 21.4% |
| Attribute (Appr.) | reorder | 28 | 32.1% |
| Attribute (Appr.) | shuffle | 28 | 42.9% |
| Attribute (Meas.) | mirror | 28 | 17.9% |
| Attribute (Meas.) | reorder | 28 | 46.4% |
| Attribute (Meas.) | shuffle | 28 | 42.9% |
| MSR | mirror | 23 | 30.4% |
| MSR | reorder | 28 | 25.0% |
| MSR | shuffle | 28 | 42.9% |
| Motion (Cam.) | mirror | 27 | 59.3% |
| Motion (Cam.) | reorder | 27 | 11.1% |
| Motion (Cam.) | shuffle | 27 | 44.4% |
| Motion (Obj.) | mirror | 27 | 29.6% |
| Motion (Obj.) | reorder | 27 | 33.3% |
| Motion (Obj.) | shuffle | 27 | 48.1% |
| Positional Relationship (Cam.–Cam.) | mirror | 25 | 28.0% |
| Positional Relationship (Cam.–Cam.) | reorder | 27 | 40.7% |
| Positional Relationship (Cam.–Cam.) | shuffle | 27 | 40.7% |
| Positional Relationship (Cam.–Obj.) | mirror | 27 | 33.3% |
| Positional Relationship (Cam.–Obj.) | reorder | 27 | 55.6% |
| Positional Relationship (Cam.–Obj.) | shuffle | 27 | 29.6% |
| Positional Relationship (Cam.–Reg.) | mirror | 26 | 30.8% |
| Positional Relationship (Cam.–Reg.) | reorder | 27 | 44.4% |
| Positional Relationship (Cam.–Reg.) | shuffle | 27 | 40.7% |
| Positional Relationship (Obj.–Obj.) | mirror | 9 | 22.2% |
| Positional Relationship (Obj.–Obj.) | reorder | 27 | 25.9% |
| Positional Relationship (Obj.–Obj.) | shuffle | 27 | 48.1% |
| Positional Relationship (Obj.–Reg.) | mirror | 10 | 30.0% |
| Positional Relationship (Obj.–Reg.) | reorder | 27 | 33.3% |
| Positional Relationship (Obj.–Reg.) | shuffle | 27 | 25.9% |
| Positional Relationship (Reg.–Reg.) | mirror | 9 | 55.6% |
| Positional Relationship (Reg.–Reg.) | reorder | 27 | 37.0% |
| Positional Relationship (Reg.–Reg.) | shuffle | 27 | 29.6% |

## Predicted option position (with images)
| variant | position | count | share |
|---|---|---|---|
| orig | A | 36 | 12.0% |
| orig | B | 107 | 35.7% |
| orig | C | 70 | 23.3% |
| orig | D | 87 | 29.0% |
| shuffle | A | 43 | 14.3% |
| shuffle | B | 93 | 31.0% |
| shuffle | C | 83 | 27.7% |
| shuffle | D | 81 | 27.0% |

## Hypothesis evaluation and discussion
- **H1 — supported:** every available answer-preserving transformation produced a non-zero flip rate.
- **H2 — not supported:** disagreement AUROC was 0.543 (95% CI 0.477–0.610); original accuracy was 32.7% when consistent versus 24.8% when inconsistent.
- **H3 — directionally supported, but not statistically conclusive:** transformation voting changed accuracy by +0.7 points relative to self-consistency (95% CI -4.0 to +5.0; McNemar p=0.8804).
- **H4 — descriptively supported:** mirror flip rate was 38.1% for direction-word questions versus 15.8% otherwise.
- **H5 — descriptively supported:** shuffle flip rate was 39.7% and the most frequent predicted shuffled-option position accounted for 31.0% of parseable-position trials (25% would be uniform across four positions).
Improvement direction: transformation voting is a promising training-free baseline, but retain the paired confidence interval and McNemar result when describing whether it reliably beats sampling diversity.
Improvement direction: add mirrored training pairs or a consistency loss for left/right reasoning.
Improvement direction: make image viewpoint labels explicit and evaluate whether the model follows labels rather than presentation order.
Improvement direction: evaluate every model over multiple option orders so that visual benchmarks do not silently reward option-position shortcuts.

## Qualitative examples
The exported cases below all have a wrong original prediction and disagreement across answer-preserving variants.
1. **ID 376 (Motion (Obj.))** — gold=A; orig=D, mirror=B, reorder=D, shuffle=A. The images are taken in a continuous first-person perspective. Is the pan being rotated? If so, in which direction is it being rotated? Options: A: Rotated clockwise, B: Rotated... [original/mirror image](examples/example1_id376/image1_original_vs_mirrored.png)
2. **ID 562 (Positional Relationship (Obj.–Obj.))** — gold=C; orig=B, mirror=D, reorder=B, shuffle=B. What is the direction of the desk lamp by the window relative to the desk lamp next to the portrait of a person (given that in two consecutively taken frames, the portrait of th... [original/mirror image](examples/example2_id562/image1_original_vs_mirrored.png)
3. **ID 155 (Attribute (Meas.))** — gold=C; orig=D, mirror=C, reorder=C, shuffle=D. Comparing twice the height of the yellow cup with a toothbrush inserted, placed on the table at the far left in Figure 1, to the height of the electric toothbrush (excluding the... [original/mirror image](examples/example3_id155/image1_original_vs_mirrored.png)

## AI usage and contribution disclosure (submission draft)
Automated tools assisted with implementation, unit tests, Babel job operation, figure generation, visual transformation preflight, and drafting the statistical interpretation. Wen-Chi Tsai supplied the research specification and remains responsible for checking the results, completing the required human validity sign-off, and approving the submitted text.

Generated figures: 3; bootstrap resamples: 1000; analysis seed: 0.

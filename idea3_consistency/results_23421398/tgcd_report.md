# TGCD results (N = 300)

Configuration: `prob_mean`, alpha=1; selected on these 300 development questions.
Missing A-D fallbacks: 0/1425 score sets (0/1125 variant; 0/300 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 29.7% | 24.3-35.0% | 0 | 0 | 1.0000 |
| No image | 26.7% | 21.7-31.7% | 39 | 30 | 0.3356 |
| Self-consistency majority | 28.3% | 23.3-33.3% | 25 | 21 | 0.6587 |
| Ensemble mean log-probability | 29.0% | 24.7-34.0% | 17 | 15 | 0.8601 |
| Ensemble mean probability | 29.0% | 23.7-34.3% | 15 | 13 | 0.8506 |
| VCD-style (alpha=1) | 27.7% | 23.0-32.7% | 57 | 51 | 0.6306 |
| TGCD (prob_mean, alpha=1) | 29.3% | 24.7-34.3% | 65 | 64 | 1.0000 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 300 | 29.7% | 24.3-35.0% | 29.7% | 0 | 0 | 1.0000 |
| mirror | 225 | 28.0% | 22.2-33.8% | 26.7% | 18 | 21 | 0.7493 |
| reorder | 300 | 29.7% | 24.7-34.7% | 29.7% | 25 | 25 | 1.0000 |
| shuffle | 300 | 30.0% | 25.3-35.3% | 29.7% | 40 | 41 | 1.0000 |

## Offline alpha sweep
| aggregation | alpha | accuracy | VCD-style accuracy |
|---|---:|---:|---:|
| logprob_mean | 0 | 29.0% | 29.7% |
| logprob_mean | 0.25 | 25.0% | 26.7% |
| logprob_mean | 0.5 | 26.0% | 27.0% |
| logprob_mean | 0.75 | 27.7% | 26.0% |
| logprob_mean | 1 | 28.7% | 27.7% |
| prob_mean | 0 | 29.0% | 29.7% |
| prob_mean | 0.25 | 26.0% | 26.7% |
| prob_mean | 0.5 | 26.3% | 27.0% |
| prob_mean | 0.75 | 28.3% | 26.0% |
| prob_mean | 1 | 29.3% | 27.7% |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 225 | 31.6% | 29.3% | 23.6-35.1% |
| reorder | 300 | 29.0% | 31.3% | 26.0-36.7% |
| shuffle | 300 | 42.0% | 36.7% | 31.3-42.3% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 28 | 17.9% | 28.6% | 10.7% | 10.7% | 32.1% | 32.1% |
| Attribute (Meas.) | 28 | 28.6% | 14.3% | 39.3% | 28.6% | 39.3% | 28.6% |
| MSR | 28 | 39.3% | 21.4% | 35.7% | 42.9% | 32.1% | 32.1% |
| Motion (Cam.) | 27 | 7.4% | 22.2% | 11.1% | 14.8% | 18.5% | 25.9% |
| Motion (Obj.) | 27 | 29.6% | 37.0% | 22.2% | 25.9% | 14.8% | 22.2% |
| Positional Relationship (Cam.–Cam.) | 27 | 37.0% | 33.3% | 25.9% | 33.3% | 25.9% | 22.2% |
| Positional Relationship (Cam.–Obj.) | 27 | 22.2% | 18.5% | 29.6% | 25.9% | 29.6% | 33.3% |
| Positional Relationship (Cam.–Reg.) | 27 | 33.3% | 18.5% | 33.3% | 37.0% | 33.3% | 40.7% |
| Positional Relationship (Obj.–Obj.) | 27 | 33.3% | 33.3% | 33.3% | 37.0% | 22.2% | 22.2% |
| Positional Relationship (Obj.–Reg.) | 27 | 37.0% | 33.3% | 29.6% | 33.3% | 29.6% | 37.0% |
| Positional Relationship (Reg.–Reg.) | 27 | 40.7% | 33.3% | 40.7% | 29.6% | 25.9% | 25.9% |

Figure: `fig_accuracy_vs_alpha.png`.

Bootstrap resamples: 1000; analysis seed: 0.

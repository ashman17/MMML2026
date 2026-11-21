# TGCD results (N = 300)

Configuration: `prob_mean`, alpha=0.75; selected on these 300 development questions.
Missing A-D fallbacks: 0/1425 score sets (0/1125 variant; 0/300 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 24.3% | 19.3-29.3% | 0 | 0 | 1.0000 |
| No image | 24.3% | 19.7-29.0% | 46 | 46 | 1.0000 |
| Self-consistency majority | 27.7% | 22.7-32.7% | 13 | 23 | 0.1325 |
| Ensemble mean log-probability | 26.7% | 22.0-31.7% | 11 | 18 | 0.2649 |
| Ensemble mean probability | 26.0% | 21.0-31.3% | 11 | 16 | 0.4421 |
| VCD-style (alpha=0.75) | 25.7% | 20.7-30.7% | 25 | 29 | 0.6835 |
| TGCD (prob_mean, alpha=0.75) | 30.3% | 25.3-35.7% | 23 | 41 | 0.0328 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 300 | 24.3% | 19.7-29.3% | 24.3% | 0 | 0 | 1.0000 |
| mirror | 225 | 27.1% | 21.8-32.9% | 23.6% | 16 | 24 | 0.2682 |
| reorder | 300 | 26.3% | 21.7-31.0% | 24.3% | 24 | 30 | 0.4966 |
| shuffle | 300 | 26.7% | 21.3-32.0% | 24.3% | 25 | 32 | 0.4270 |

## Offline alpha sweep
| aggregation | alpha | accuracy | VCD-style accuracy |
|---|---:|---:|---:|
| logprob_mean | 0 | 26.7% | 24.3% |
| logprob_mean | 0.25 | 28.0% | 24.7% |
| logprob_mean | 0.5 | 28.0% | 25.0% |
| logprob_mean | 0.75 | 29.7% | 25.7% |
| logprob_mean | 1 | 29.0% | 27.3% |
| prob_mean | 0 | 26.0% | 24.3% |
| prob_mean | 0.25 | 27.3% | 24.7% |
| prob_mean | 0.5 | 28.0% | 25.0% |
| prob_mean | 0.75 | 30.3% | 25.7% |
| prob_mean | 1 | 29.7% | 27.3% |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 225 | 36.0% | 34.7% | 28.4-40.9% |
| reorder | 300 | 34.0% | 42.7% | 37.0-48.7% |
| shuffle | 300 | 33.0% | 33.0% | 28.0-38.0% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 28 | 14.3% | 21.4% | 7.1% | 14.3% | 25.0% | 25.0% |
| Attribute (Meas.) | 28 | 28.6% | 35.7% | 39.3% | 32.1% | 35.7% | 32.1% |
| MSR | 28 | 28.6% | 14.3% | 32.1% | 32.1% | 25.0% | 28.6% |
| Motion (Cam.) | 27 | 22.2% | 7.4% | 29.6% | 25.9% | 29.6% | 44.4% |
| Motion (Obj.) | 27 | 25.9% | 25.9% | 29.6% | 25.9% | 25.9% | 33.3% |
| Positional Relationship (Cam.–Cam.) | 27 | 22.2% | 37.0% | 14.8% | 22.2% | 11.1% | 37.0% |
| Positional Relationship (Cam.–Obj.) | 27 | 22.2% | 18.5% | 33.3% | 29.6% | 25.9% | 33.3% |
| Positional Relationship (Cam.–Reg.) | 27 | 25.9% | 18.5% | 37.0% | 29.6% | 44.4% | 44.4% |
| Positional Relationship (Obj.–Obj.) | 27 | 7.4% | 37.0% | 18.5% | 11.1% | 3.7% | 3.7% |
| Positional Relationship (Obj.–Reg.) | 27 | 44.4% | 25.9% | 40.7% | 37.0% | 25.9% | 25.9% |
| Positional Relationship (Reg.–Reg.) | 27 | 25.9% | 25.9% | 22.2% | 25.9% | 29.6% | 25.9% |

Figure: `fig_accuracy_vs_alpha.png`.

Bootstrap resamples: 1000; analysis seed: 0.

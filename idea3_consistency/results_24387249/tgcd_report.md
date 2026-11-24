# TGCD results (N = 300)

Configuration: `prob_mean`, alpha=0.75; selected on these 300 development questions.
Missing A-D fallbacks: 0/1425 score sets (0/1125 variant; 0/300 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 28.7% | 23.7-33.7% | 0 | 0 | 1.0000 |
| No image | 24.7% | 20.0-29.7% | 59 | 47 | 0.2853 |
| Self-consistency majority | 30.3% | 24.7-35.7% | 17 | 22 | 0.5224 |
| Ensemble mean log-probability | 25.3% | 20.7-30.3% | 24 | 14 | 0.1433 |
| Ensemble mean probability | 26.3% | 21.3-31.3% | 22 | 15 | 0.3240 |
| VCD-style (alpha=0.75) | 27.3% | 22.3-32.3% | 28 | 24 | 0.6778 |
| TGCD (prob_mean, alpha=0.75) | 31.0% | 25.7-36.3% | 34 | 41 | 0.4887 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 300 | 28.7% | 23.7-33.7% | 28.7% | 0 | 0 | 1.0000 |
| mirror | 225 | 24.9% | 19.1-30.2% | 26.7% | 23 | 19 | 0.6440 |
| reorder | 300 | 26.7% | 22.0-31.7% | 28.7% | 34 | 28 | 0.5258 |
| shuffle | 300 | 27.3% | 22.3-32.3% | 28.7% | 33 | 29 | 0.7035 |

## Offline alpha sweep
| aggregation | alpha | accuracy | VCD-style accuracy |
|---|---:|---:|---:|
| logprob_mean | 0 | 25.3% | 28.7% |
| logprob_mean | 0.25 | 25.3% | 26.3% |
| logprob_mean | 0.5 | 27.7% | 27.3% |
| logprob_mean | 0.75 | 29.7% | 27.3% |
| logprob_mean | 1 | 30.7% | 25.7% |
| prob_mean | 0 | 26.3% | 28.7% |
| prob_mean | 0.25 | 25.7% | 26.3% |
| prob_mean | 0.5 | 28.7% | 27.3% |
| prob_mean | 0.75 | 31.0% | 27.3% |
| prob_mean | 1 | 30.0% | 25.7% |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 225 | 36.4% | 28.4% | 23.1-34.7% |
| reorder | 300 | 33.3% | 37.7% | 32.3-43.3% |
| shuffle | 300 | 34.7% | 39.3% | 33.7-44.7% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 28 | 17.9% | 28.6% | 21.4% | 14.3% | 21.4% | 28.6% |
| Attribute (Meas.) | 28 | 35.7% | 28.6% | 28.6% | 32.1% | 32.1% | 39.3% |
| MSR | 28 | 28.6% | 14.3% | 25.0% | 21.4% | 32.1% | 35.7% |
| Motion (Cam.) | 27 | 33.3% | 7.4% | 40.7% | 25.9% | 44.4% | 48.1% |
| Motion (Obj.) | 27 | 22.2% | 33.3% | 25.9% | 25.9% | 22.2% | 14.8% |
| Positional Relationship (Cam.–Cam.) | 27 | 37.0% | 33.3% | 25.9% | 29.6% | 22.2% | 37.0% |
| Positional Relationship (Cam.–Obj.) | 27 | 29.6% | 22.2% | 37.0% | 37.0% | 25.9% | 33.3% |
| Positional Relationship (Cam.–Reg.) | 27 | 29.6% | 7.4% | 33.3% | 22.2% | 29.6% | 37.0% |
| Positional Relationship (Obj.–Obj.) | 27 | 22.2% | 37.0% | 33.3% | 29.6% | 18.5% | 18.5% |
| Positional Relationship (Obj.–Reg.) | 27 | 37.0% | 25.9% | 44.4% | 33.3% | 33.3% | 29.6% |
| Positional Relationship (Reg.–Reg.) | 27 | 22.2% | 33.3% | 18.5% | 18.5% | 18.5% | 18.5% |

Figure: `fig_accuracy_vs_alpha.png`.

Bootstrap resamples: 1000; analysis seed: 0.

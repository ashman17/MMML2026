# TGCD results (N = 300)

Configuration: `prob_mean`, alpha=0.25; selected on these 300 development questions.
Missing A-D fallbacks: 0/1425 score sets (0/1125 variant; 0/300 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 25.7% | 20.0-30.3% | 0 | 0 | 1.0000 |
| No image | 29.7% | 24.3-35.0% | 41 | 53 | 0.2564 |
| Self-consistency majority | 27.3% | 22.3-32.7% | 22 | 27 | 0.5682 |
| Ensemble mean log-probability | 29.0% | 24.7-34.3% | 12 | 22 | 0.1214 |
| Ensemble mean probability | 28.7% | 23.7-34.0% | 12 | 21 | 0.1628 |
| VCD-style (alpha=0.25) | 27.3% | 22.7-32.7% | 8 | 13 | 0.3833 |
| TGCD (prob_mean, alpha=0.25) | 30.0% | 24.7-35.0% | 14 | 27 | 0.0596 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 300 | 25.7% | 21.0-30.7% | 25.7% | 0 | 0 | 1.0000 |
| mirror | 225 | 28.9% | 22.7-34.2% | 22.7% | 17 | 31 | 0.0595 |
| reorder | 300 | 27.7% | 23.0-32.3% | 25.7% | 27 | 33 | 0.5190 |
| shuffle | 300 | 29.3% | 24.3-34.7% | 25.7% | 25 | 36 | 0.2000 |

## Offline alpha sweep
| aggregation | alpha | accuracy | VCD-style accuracy |
|---|---:|---:|---:|
| logprob_mean | 0 | 29.0% | 25.7% |
| logprob_mean | 0.25 | 29.3% | 27.3% |
| logprob_mean | 0.5 | 29.3% | 28.0% |
| logprob_mean | 0.75 | 29.0% | 27.3% |
| logprob_mean | 1 | 29.0% | 28.0% |
| prob_mean | 0 | 28.7% | 25.7% |
| prob_mean | 0.25 | 30.0% | 27.3% |
| prob_mean | 0.5 | 28.7% | 28.0% |
| prob_mean | 0.75 | 29.0% | 27.3% |
| prob_mean | 1 | 29.3% | 28.0% |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 225 | 35.6% | 37.8% | 31.6-44.4% |
| reorder | 300 | 32.7% | 36.0% | 30.7-41.7% |
| shuffle | 300 | 34.7% | 33.0% | 28.0-38.3% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 28 | 17.9% | 17.9% | 14.3% | 17.9% | 21.4% | 17.9% |
| Attribute (Meas.) | 28 | 25.0% | 42.9% | 32.1% | 32.1% | 25.0% | 28.6% |
| MSR | 28 | 28.6% | 25.0% | 32.1% | 28.6% | 28.6% | 32.1% |
| Motion (Cam.) | 27 | 18.5% | 14.8% | 29.6% | 22.2% | 29.6% | 29.6% |
| Motion (Obj.) | 27 | 25.9% | 33.3% | 25.9% | 25.9% | 25.9% | 29.6% |
| Positional Relationship (Cam.–Cam.) | 27 | 22.2% | 37.0% | 22.2% | 33.3% | 18.5% | 29.6% |
| Positional Relationship (Cam.–Obj.) | 27 | 29.6% | 18.5% | 29.6% | 29.6% | 29.6% | 33.3% |
| Positional Relationship (Cam.–Reg.) | 27 | 25.9% | 25.9% | 25.9% | 33.3% | 37.0% | 37.0% |
| Positional Relationship (Obj.–Obj.) | 27 | 14.8% | 48.1% | 22.2% | 14.8% | 14.8% | 14.8% |
| Positional Relationship (Obj.–Reg.) | 27 | 44.4% | 29.6% | 37.0% | 40.7% | 40.7% | 40.7% |
| Positional Relationship (Reg.–Reg.) | 27 | 29.6% | 33.3% | 29.6% | 37.0% | 29.6% | 37.0% |

Figure: `fig_accuracy_vs_alpha.png`.

Bootstrap resamples: 1000; analysis seed: 0.

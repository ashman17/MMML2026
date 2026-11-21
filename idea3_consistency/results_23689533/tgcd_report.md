# TGCD results (N = 522)

Configuration: `logprob_mean`, alpha=0.25; selected on these 522 development questions.
Missing valid-option-letter fallbacks: 0/2220 score sets (0/1698 variant; 0/522 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 68.6% | 64.4-72.4% | 0 | 0 | 1.0000 |
| No image | 53.6% | 49.2-57.9% | 100 | 22 | 0.0000 |
| Self-consistency majority | 65.9% | 62.3-69.9% | 34 | 20 | 0.0759 |
| Ensemble mean log-probability | 68.4% | 64.8-71.8% | 21 | 20 | 1.0000 |
| Ensemble mean probability | 68.4% | 64.4-72.2% | 20 | 19 | 1.0000 |
| VCD-style (alpha=0.25) | 68.8% | 64.9-72.6% | 11 | 12 | 1.0000 |
| TGCD (logprob_mean, alpha=0.25) | 68.6% | 64.8-72.6% | 32 | 32 | 1.0000 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 522 | 68.6% | 64.6-72.6% | 68.6% | 0 | 0 | 1.0000 |
| mirror | 521 | 65.1% | 61.0-69.3% | 68.7% | 70 | 51 | 0.1014 |
| reorder | 133 | 55.6% | 48.1-63.9% | 55.6% | 0 | 0 | 1.0000 |
| shuffle | 522 | 61.1% | 56.5-65.3% | 68.6% | 156 | 117 | 0.0213 |

## Offline alpha sweep
| aggregation | alpha | accuracy | VCD-style accuracy |
|---|---:|---:|---:|
| logprob_mean | 0 | 68.4% | 68.6% |
| logprob_mean | 0.25 | 68.6% | 68.8% |
| logprob_mean | 0.5 | 66.7% | 69.7% |
| logprob_mean | 0.75 | 65.3% | 68.2% |
| logprob_mean | 1 | 61.7% | 65.7% |
| prob_mean | 0 | 68.4% | 68.6% |
| prob_mean | 0.25 | 68.0% | 68.8% |
| prob_mean | 0.5 | 66.5% | 69.7% |
| prob_mean | 0.75 | 63.0% | 68.2% |
| prob_mean | 1 | 61.1% | 65.7% |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 521 | 23.2% | 24.8% | 20.9-28.4% |
| reorder | 133 | 0.0% | 0.0% | 0.0-0.0% |
| shuffle | 522 | 52.3% | 55.7% | 51.5-59.8% |

## Per question type
| type | n | chance | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multi-view_Reasoning | 133 | 50.0% | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% |
| Object_Localization | 122 | 50.0% | 54.9% | 56.6% | 53.3% | 50.0% | 52.5% | 50.0% |
| Relative_Depth | 124 | 50.0% | 76.6% | 51.6% | 75.0% | 79.0% | 79.0% | 80.6% |
| Spatial_Relation | 143 | 50.0% | 85.3% | 51.0% | 78.3% | 86.7% | 86.0% | 86.0% |

Figure: `fig_accuracy_vs_alpha.png`.

Bootstrap resamples: 1000; analysis seed: 0.

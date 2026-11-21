# TGCD results (N = 522)

Configuration: `prob_mean`, alpha=1; fixed from `/projects/bjjb/vtsai/idea3_results/results_23421398/tgcd_selection.json`.
Missing valid-option-letter fallbacks: 0/2220 score sets (0/1698 variant; 0/522 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 68.6% | 64.4-72.4% | 0 | 0 | 1.0000 |
| No image | 53.6% | 49.2-57.9% | 100 | 22 | 0.0000 |
| Self-consistency majority | 65.9% | 62.3-69.9% | 34 | 20 | 0.0759 |
| Ensemble mean log-probability | 68.4% | 64.8-71.8% | 21 | 20 | 1.0000 |
| Ensemble mean probability | 68.4% | 64.4-72.2% | 20 | 19 | 1.0000 |
| VCD-style (alpha=1) | 65.7% | 61.5-69.7% | 59 | 44 | 0.1674 |
| TGCD (prob_mean, alpha=1) | 61.1% | 56.9-65.3% | 113 | 74 | 0.0053 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 522 | 68.6% | 64.6-72.6% | 68.6% | 0 | 0 | 1.0000 |
| mirror | 521 | 65.1% | 61.0-69.3% | 68.7% | 70 | 51 | 0.1014 |
| reorder | 133 | 55.6% | 48.1-63.9% | 55.6% | 0 | 0 | 1.0000 |
| shuffle | 522 | 61.1% | 56.5-65.3% | 68.6% | 156 | 117 | 0.0213 |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 521 | 23.2% | 33.4% | 29.0-37.4% |
| reorder | 133 | 0.0% | 0.0% | 0.0-0.0% |
| shuffle | 522 | 52.3% | 49.8% | 45.6-54.2% |

## Per question type
| type | n | chance | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multi-view_Reasoning | 133 | 50.0% | 55.6% | 55.6% | 55.6% | 55.6% | 55.6% | 56.4% |
| Object_Localization | 122 | 50.0% | 54.9% | 56.6% | 53.3% | 49.2% | 50.8% | 44.3% |
| Relative_Depth | 124 | 50.0% | 76.6% | 51.6% | 75.0% | 79.8% | 76.6% | 72.6% |
| Spatial_Relation | 143 | 50.0% | 85.3% | 51.0% | 78.3% | 86.7% | 78.3% | 69.9% |

Bootstrap resamples: 1000; analysis seed: 0.

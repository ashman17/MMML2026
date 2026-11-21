# TGCD results (N = 700)

Configuration: `prob_mean`, alpha=0.75; fixed from `results_24063250/tgcd_selection.json`.
Missing A-D fallbacks: 0/3334 score sets (0/2634 variant; 0/700 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 29.0% | 25.9-32.3% | 0 | 0 | 1.0000 |
| No image | 25.1% | 21.7-28.3% | 129 | 102 | 0.0869 |
| Self-consistency majority | 28.0% | 24.7-31.1% | 43 | 36 | 0.4999 |
| Ensemble mean log-probability | 28.4% | 25.0-32.1% | 31 | 27 | 0.6940 |
| Ensemble mean probability | 29.1% | 26.0-32.6% | 23 | 24 | 1.0000 |
| VCD-style (alpha=0.75) | 30.0% | 26.6-33.3% | 52 | 59 | 0.5692 |
| TGCD (prob_mean, alpha=0.75) | 30.0% | 26.7-33.4% | 73 | 80 | 0.6278 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 700 | 29.0% | 25.4-32.4% | 29.0% | 0 | 0 | 1.0000 |
| mirror | 534 | 28.7% | 24.7-32.6% | 27.3% | 42 | 49 | 0.5296 |
| reorder | 700 | 26.6% | 23.1-29.7% | 29.0% | 79 | 62 | 0.1776 |
| shuffle | 700 | 28.1% | 24.7-31.4% | 29.0% | 59 | 53 | 0.6368 |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 534 | 34.6% | 36.7% | 32.4-41.0% |
| reorder | 700 | 34.0% | 40.6% | 37.3-43.9% |
| shuffle | 700 | 26.4% | 31.9% | 28.4-35.1% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 38 | 26.3% | 21.1% | 26.3% | 26.3% | 28.9% | 21.1% |
| Attribute (Meas.) | 36 | 8.3% | 13.9% | 5.6% | 11.1% | 22.2% | 25.0% |
| MSR | 170 | 26.5% | 22.4% | 27.1% | 25.3% | 25.3% | 24.1% |
| Motion (Cam.) | 47 | 31.9% | 19.1% | 27.7% | 27.7% | 42.6% | 38.3% |
| Motion (Obj.) | 49 | 36.7% | 38.8% | 24.5% | 32.7% | 28.6% | 30.6% |
| Positional Relationship (Cam.–Cam.) | 66 | 24.2% | 28.8% | 25.8% | 19.7% | 25.8% | 22.7% |
| Positional Relationship (Cam.–Obj.) | 59 | 27.1% | 20.3% | 28.8% | 32.2% | 35.6% | 39.0% |
| Positional Relationship (Cam.–Reg.) | 56 | 39.3% | 32.1% | 33.9% | 46.4% | 39.3% | 42.9% |
| Positional Relationship (Obj.–Obj.) | 67 | 34.3% | 25.4% | 38.8% | 31.3% | 35.8% | 28.4% |
| Positional Relationship (Obj.–Reg.) | 58 | 32.8% | 29.3% | 29.3% | 37.9% | 25.9% | 37.9% |
| Positional Relationship (Reg.–Reg.) | 54 | 29.6% | 25.9% | 31.5% | 31.5% | 27.8% | 29.6% |

Bootstrap resamples: 1000; analysis seed: 0.

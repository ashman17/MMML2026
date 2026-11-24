# TGCD results (N = 700)

Configuration: `prob_mean`, alpha=0.75; fixed from `results_24387249/tgcd_selection.json`.
Missing A-D fallbacks: 0/3334 score sets (0/2634 variant; 0/700 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 29.1% | 25.9-32.3% | 0 | 0 | 1.0000 |
| No image | 26.1% | 23.0-29.4% | 130 | 109 | 0.1957 |
| Self-consistency majority | 28.3% | 24.9-31.6% | 37 | 31 | 0.5446 |
| Ensemble mean log-probability | 30.7% | 27.3-34.6% | 27 | 38 | 0.2145 |
| Ensemble mean probability | 30.7% | 27.3-34.3% | 25 | 36 | 0.2000 |
| VCD-style (alpha=0.75) | 29.1% | 25.6-32.6% | 58 | 58 | 1.0000 |
| TGCD (prob_mean, alpha=0.75) | 30.6% | 27.3-34.0% | 69 | 79 | 0.4595 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 700 | 29.1% | 25.9-32.7% | 29.1% | 0 | 0 | 1.0000 |
| mirror | 534 | 29.2% | 25.7-33.0% | 28.3% | 46 | 51 | 0.6849 |
| reorder | 700 | 28.1% | 24.7-31.6% | 29.1% | 67 | 60 | 0.5946 |
| shuffle | 700 | 32.0% | 28.6-35.1% | 29.1% | 59 | 79 | 0.1055 |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 534 | 31.8% | 38.8% | 34.5-42.9% |
| reorder | 700 | 31.0% | 36.0% | 32.3-39.4% |
| shuffle | 700 | 31.9% | 36.0% | 32.6-39.6% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 38 | 23.7% | 26.3% | 23.7% | 23.7% | 28.9% | 28.9% |
| Attribute (Meas.) | 36 | 11.1% | 25.0% | 13.9% | 16.7% | 22.2% | 27.8% |
| MSR | 170 | 23.5% | 22.9% | 24.1% | 28.8% | 24.1% | 24.7% |
| Motion (Cam.) | 47 | 31.9% | 12.8% | 25.5% | 29.8% | 44.7% | 42.6% |
| Motion (Obj.) | 49 | 34.7% | 28.6% | 32.7% | 32.7% | 28.6% | 32.7% |
| Positional Relationship (Cam.–Cam.) | 66 | 28.8% | 31.8% | 25.8% | 30.3% | 19.7% | 25.8% |
| Positional Relationship (Cam.–Obj.) | 59 | 39.0% | 28.8% | 37.3% | 39.0% | 37.3% | 35.6% |
| Positional Relationship (Cam.–Reg.) | 56 | 37.5% | 28.6% | 35.7% | 44.6% | 39.3% | 44.6% |
| Positional Relationship (Obj.–Obj.) | 67 | 31.3% | 26.9% | 32.8% | 25.4% | 32.8% | 25.4% |
| Positional Relationship (Obj.–Reg.) | 58 | 31.0% | 34.5% | 29.3% | 29.3% | 24.1% | 31.0% |
| Positional Relationship (Reg.–Reg.) | 54 | 31.5% | 24.1% | 31.5% | 35.2% | 29.6% | 31.5% |

Bootstrap resamples: 1000; analysis seed: 0.

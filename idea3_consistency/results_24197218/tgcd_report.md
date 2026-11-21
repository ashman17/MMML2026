# TGCD results (N = 700)

Configuration: `prob_mean`, alpha=0.25; fixed from `results_24197216/tgcd_selection.json`.
Missing A-D fallbacks: 0/3334 score sets (0/2634 variant; 0/700 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 27.9% | 24.9-31.1% | 0 | 0 | 1.0000 |
| No image | 26.0% | 22.7-29.1% | 127 | 114 | 0.4396 |
| Self-consistency majority | 27.6% | 24.1-30.7% | 39 | 37 | 0.9088 |
| Ensemble mean log-probability | 28.0% | 24.7-31.6% | 29 | 30 | 1.0000 |
| Ensemble mean probability | 27.9% | 24.7-31.4% | 31 | 31 | 1.0000 |
| VCD-style (alpha=0.25) | 28.1% | 24.9-31.6% | 20 | 22 | 0.8776 |
| TGCD (prob_mean, alpha=0.25) | 28.7% | 25.4-31.9% | 37 | 43 | 0.5764 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 700 | 27.9% | 24.9-31.0% | 27.9% | 0 | 0 | 1.0000 |
| mirror | 534 | 27.9% | 24.3-31.8% | 26.2% | 52 | 61 | 0.4519 |
| reorder | 700 | 27.7% | 24.6-31.1% | 27.9% | 67 | 66 | 1.0000 |
| shuffle | 700 | 29.1% | 25.9-32.4% | 27.9% | 53 | 62 | 0.4558 |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 534 | 37.8% | 39.7% | 35.4-44.0% |
| reorder | 700 | 33.3% | 37.6% | 33.9-41.1% |
| shuffle | 700 | 27.7% | 30.3% | 26.9-33.6% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 38 | 26.3% | 26.3% | 21.1% | 18.4% | 31.6% | 15.8% |
| Attribute (Meas.) | 36 | 13.9% | 11.1% | 16.7% | 13.9% | 19.4% | 16.7% |
| MSR | 170 | 25.9% | 24.1% | 24.7% | 27.1% | 26.5% | 28.8% |
| Motion (Cam.) | 47 | 27.7% | 25.5% | 23.4% | 31.9% | 29.8% | 31.9% |
| Motion (Obj.) | 49 | 32.7% | 28.6% | 28.6% | 36.7% | 30.6% | 32.7% |
| Positional Relationship (Cam.–Cam.) | 66 | 28.8% | 28.8% | 28.8% | 22.7% | 25.8% | 22.7% |
| Positional Relationship (Cam.–Obj.) | 59 | 28.8% | 27.1% | 27.1% | 35.6% | 30.5% | 39.0% |
| Positional Relationship (Cam.–Reg.) | 56 | 39.3% | 35.7% | 37.5% | 35.7% | 33.9% | 39.3% |
| Positional Relationship (Obj.–Obj.) | 67 | 22.4% | 23.9% | 32.8% | 22.4% | 25.4% | 22.4% |
| Positional Relationship (Obj.–Reg.) | 58 | 31.0% | 29.3% | 31.0% | 29.3% | 31.0% | 32.8% |
| Positional Relationship (Reg.–Reg.) | 54 | 29.6% | 24.1% | 29.6% | 29.6% | 27.8% | 27.8% |

Bootstrap resamples: 1000; analysis seed: 0.

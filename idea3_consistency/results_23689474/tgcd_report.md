# TGCD results (N = 700)

Configuration: `prob_mean`, alpha=1; fixed from `/projects/bjjb/vtsai/idea3_results/results_23421398/tgcd_selection.json`.
Missing A-D fallbacks: 0/3334 score sets (0/2634 variant; 0/700 no-image).

## Accuracy and paired inference
| method | accuracy | bootstrap 95% CI | orig-only correct | method-only correct | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Original letter score | 28.7% | 25.3-32.1% | 0 | 0 | 1.0000 |
| No image | 26.3% | 23.1-29.6% | 97 | 80 | 0.2290 |
| Self-consistency majority | 29.0% | 25.7-32.3% | 37 | 39 | 0.9088 |
| Ensemble mean log-probability | 30.3% | 26.9-33.7% | 26 | 37 | 0.2074 |
| Ensemble mean probability | 30.7% | 27.3-34.4% | 25 | 39 | 0.1034 |
| VCD-style (alpha=1) | 28.6% | 25.0-31.7% | 108 | 107 | 1.0000 |
| TGCD (prob_mean, alpha=1) | 29.4% | 26.0-32.9% | 131 | 136 | 0.8067 |

## Single-view accuracy
| view | n | accuracy | bootstrap 95% CI | original on subset | orig-only | view-only | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| orig | 700 | 28.7% | 25.6-32.3% | 28.7% | 0 | 0 | 1.0000 |
| mirror | 534 | 30.1% | 26.0-34.3% | 27.9% | 39 | 51 | 0.2461 |
| reorder | 700 | 28.0% | 24.6-31.4% | 28.7% | 55 | 50 | 0.6965 |
| shuffle | 700 | 32.1% | 28.7-35.4% | 28.7% | 67 | 91 | 0.0669 |

## Flip rate after contrast decoding
Each transformed view is contrast-decoded with the selected alpha and compared with the contrast-decoded original view. The earlier midterm range was 24-35%.
| transform | n | raw score flip | contrast-decoded flip | bootstrap 95% CI |
|---|---:|---:|---:|---:|
| mirror | 534 | 32.2% | 34.8% | 30.9-39.0% |
| reorder | 700 | 28.4% | 31.1% | 27.7-34.4% |
| shuffle | 700 | 41.4% | 41.6% | 38.0-45.1% |

## Per question type
| type | n | orig | no image | self-consistency | ensemble alpha=0 | VCD-style | TGCD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attribute (Appr.) | 38 | 23.7% | 31.6% | 21.1% | 28.9% | 21.1% | 21.1% |
| Attribute (Meas.) | 36 | 19.4% | 11.1% | 16.7% | 19.4% | 33.3% | 36.1% |
| MSR | 170 | 28.2% | 25.9% | 22.9% | 30.0% | 30.6% | 29.4% |
| Motion (Cam.) | 47 | 21.3% | 21.3% | 25.5% | 21.3% | 34.0% | 34.0% |
| Motion (Obj.) | 49 | 36.7% | 36.7% | 38.8% | 38.8% | 22.4% | 26.5% |
| Positional Relationship (Cam.–Cam.) | 66 | 28.8% | 28.8% | 31.8% | 31.8% | 27.3% | 30.3% |
| Positional Relationship (Cam.–Obj.) | 59 | 33.9% | 25.4% | 33.9% | 28.8% | 25.4% | 30.5% |
| Positional Relationship (Cam.–Reg.) | 56 | 30.4% | 25.0% | 37.5% | 42.9% | 32.1% | 35.7% |
| Positional Relationship (Obj.–Obj.) | 67 | 31.3% | 26.9% | 32.8% | 26.9% | 31.3% | 22.4% |
| Positional Relationship (Obj.–Reg.) | 58 | 32.8% | 24.1% | 32.8% | 36.2% | 31.0% | 36.2% |
| Positional Relationship (Reg.–Reg.) | 54 | 24.1% | 29.6% | 29.6% | 29.6% | 20.4% | 22.2% |

Bootstrap resamples: 1000; analysis seed: 0.

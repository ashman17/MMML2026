# Multimodal subset of MMSI-Bench

Rule: keep a question when the no-image max probability < 0.7.

| split | rule | n | orig acc | no-image acc | orig = no-image |
|---|---|---:|---:|---:|---:|
| results_23421398 | all | 300 | 29.7% | 26.7% | 54.3% |
| results_23421398 | conf | 258 | 27.9% | 23.3% | 51.9% |
| results_23421398 | noimage_wrong | 220 | 17.7% | 0.0% | 51.4% |

### results_23421398: kept per question type
| type | n | kept |
|---|---:|---:|
| Attribute (Appr.) | 28 | 28 |
| Attribute (Meas.) | 28 | 22 |
| MSR | 28 | 27 |
| Motion (Cam.) | 27 | 20 |
| Motion (Obj.) | 27 | 20 |
| Positional Relationship (Cam.–Cam.) | 27 | 24 |
| Positional Relationship (Cam.–Obj.) | 27 | 27 |
| Positional Relationship (Cam.–Reg.) | 27 | 24 |
| Positional Relationship (Obj.–Obj.) | 27 | 23 |
| Positional Relationship (Obj.–Reg.) | 27 | 22 |
| Positional Relationship (Reg.–Reg.) | 27 | 21 |

| results_23689474 | all | 700 | 28.7% | 26.3% | 52.7% |
| results_23689474 | conf | 597 | 26.1% | 22.9% | 49.4% |
| results_23689474 | noimage_wrong | 516 | 18.8% | 0.0% | 51.4% |

### results_23689474: kept per question type
| type | n | kept |
|---|---:|---:|
| Attribute (Appr.) | 38 | 34 |
| Attribute (Meas.) | 36 | 25 |
| MSR | 170 | 152 |
| Motion (Cam.) | 47 | 35 |
| Motion (Obj.) | 49 | 39 |
| Positional Relationship (Cam.–Cam.) | 66 | 57 |
| Positional Relationship (Cam.–Obj.) | 59 | 49 |
| Positional Relationship (Cam.–Reg.) | 56 | 43 |
| Positional Relationship (Obj.–Obj.) | 67 | 61 |
| Positional Relationship (Obj.–Reg.) | 58 | 52 |
| Positional Relationship (Reg.–Reg.) | 54 | 50 |


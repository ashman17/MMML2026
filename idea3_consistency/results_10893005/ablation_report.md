# Modality ablation  (N = 700 questions, source `results_10893005/predictions.jsonl`)

| variant | accuracy | same answer as orig | unparsable |
|---|---|---|---|
| orig | 29.0% | 100.0% | 2 |
| noimage | 26.3% | 51.4% | 8 |
| randimage | 24.9% | 50.6% | 29 |

Chance accuracy with four options is about 25%.
- orig vs noimage: orig-only correct=99, noimage-only correct=80, exact McNemar p=0.1783
- orig vs randimage: orig-only correct=120, randimage-only correct=91, exact McNemar p=0.0536

## By question type
| type | n | orig | noimage | randimage | noimage same as orig |
|---|---|---|---|---|---|
| Attribute (Appr.) | 38 | 15.8% | 34.2% | 13.2% | 47.4% |
| Attribute (Meas.) | 36 | 19.4% | 19.4% | 25.0% | 58.3% |
| MSR | 170 | 28.8% | 26.5% | 24.1% | 44.7% |
| Motion (Cam.) | 47 | 17.0% | 21.3% | 10.6% | 70.2% |
| Motion (Obj.) | 49 | 38.8% | 32.7% | 32.7% | 44.9% |
| Positional Relationship (Cam.–Cam.) | 66 | 31.8% | 30.3% | 31.8% | 65.2% |
| Positional Relationship (Cam.–Obj.) | 59 | 33.9% | 20.3% | 18.6% | 52.5% |
| Positional Relationship (Cam.–Reg.) | 56 | 35.7% | 26.8% | 32.1% | 42.9% |
| Positional Relationship (Obj.–Obj.) | 67 | 29.9% | 25.4% | 32.8% | 52.2% |
| Positional Relationship (Obj.–Reg.) | 58 | 32.8% | 24.1% | 27.6% | 55.2% |
| Positional Relationship (Reg.–Reg.) | 54 | 25.9% | 27.8% | 18.5% | 46.3% |

## Joined with baseline run (`results_10892041/predictions.jsonl`, 700 shared questions)
Share of questions whose no-image answer equals the baseline original answer:
| group | n | no-image answer = original answer |
|---|---|---|
| all | 700 | 50.9% |
| mirror: lexical | 60 | 43.3% |
| mirror: other | 47 | 34.0% |
| mirror: same | 130 | 48.5% |
| taxonomy: correct_consistent | 97 | 62.9% |
| taxonomy: correct_inconsistent | 109 | 38.5% |
| taxonomy: wrong_consistent | 190 | 56.8% |
| taxonomy: wrong_inconsistent | 304 | 47.7% |

If 'mirror: lexical' is well above 'mirror: same', the questions where the model ignored the flip are the ones it can answer from text alone.

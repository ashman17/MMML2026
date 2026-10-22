# Image probes  (N = 700 questions, source `results_10893268/predictions.jsonl`)

| variant | n | accuracy | flip rate vs orig | flip 95% CI | unparsable |
|---|---|---|---|---|---|
| orig | 700 | 29.7% | 0.0% | 0.0–0.0 | 2 |
| jpeg | 700 | 30.0% | 8.3% | 6.3–10.4 | 2 |
| split | 632 | 27.5% | 29.7% | 26.3–33.2 | 0 |

On the 632 questions with a split variant: split flip rate 29.7% vs. jpeg noise floor 8.5% (split-only flips=153, jpeg-only flips=19, exact McNemar p=0.0000).
Accuracy orig vs split on the same questions: orig-only correct=69, split-only correct=48, exact McNemar p=0.0640.

## Flip rate by question type
| type | n (split) | jpeg flip | split flip |
|---|---|---|---|
| Attribute (Appr.) | 23 | 13.0% | 17.4% |
| Attribute (Meas.) | 36 | 0.0% | 8.3% |
| MSR | 166 | 11.4% | 31.9% |
| Motion (Obj.) | 47 | 10.6% | 21.3% |
| Positional Relationship (Cam.–Cam.) | 66 | 6.1% | 33.3% |
| Positional Relationship (Cam.–Obj.) | 59 | 8.5% | 42.4% |
| Positional Relationship (Cam.–Reg.) | 56 | 8.9% | 48.2% |
| Positional Relationship (Obj.–Obj.) | 67 | 7.5% | 23.9% |
| Positional Relationship (Obj.–Reg.) | 58 | 10.3% | 31.0% |
| Positional Relationship (Reg.–Reg.) | 54 | 3.7% | 18.5% |

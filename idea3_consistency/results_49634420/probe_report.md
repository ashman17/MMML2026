# Image probes  (N = 700 questions, source `/ocean/projects/cis260328p/vtsai/idea3_results/results_49634420/predictions.jsonl`)

| variant | n | accuracy | flip rate vs orig | flip 95% CI | unparsable |
|---|---|---|---|---|---|
| orig | 700 | 30.1% | 0.0% | 0.0–0.0 | 2 |
| jpeg | 700 | 30.0% | 7.4% | 5.6–9.4 | 2 |
| split | 632 | 28.0% | 30.9% | 27.1–34.3 | 1 |
| grid | 632 | 28.8% | 32.0% | 28.5–35.6 | 1 |
| grid_shuffle | 632 | 30.1% | 35.9% | 32.1–39.7 | 1 |

On the 632 questions with a split variant: split flip rate 30.9% vs. jpeg noise floor 7.4% (split-only flips=170, jpeg-only flips=22, exact McNemar p=0.0000).
Accuracy orig vs split on the same questions: orig-only correct=70, split-only correct=49, exact McNemar p=0.0663.

On the 632 questions with a grid variant: grid flip rate 32.0% vs. jpeg noise floor 7.4% (grid-only flips=175, jpeg-only flips=20, exact McNemar p=0.0000).
Accuracy orig vs grid on the same questions: orig-only correct=73, grid-only correct=57, exact McNemar p=0.1881.

On the 632 questions with a grid_shuffle variant: grid_shuffle flip rate 35.9% vs. jpeg noise floor 7.4% (grid_shuffle-only flips=200, jpeg-only flips=20, exact McNemar p=0.0000).
Accuracy orig vs grid_shuffle on the same questions: orig-only correct=74, grid_shuffle-only correct=66, exact McNemar p=0.5543.

Tile order (grid_shuffle vs grid, n=632): shuffle-only flips=87, grid-only flips=62, exact McNemar p=0.0489.

## Flip rate by question type
| type | n (pieces) | jpeg flip | split flip | grid flip | grid_shuffle flip |
|---|---|---|---|---|---|
| Attribute (Appr.) | 23 | 8.7% | 17.4% | 26.1% | 30.4% |
| Attribute (Meas.) | 36 | 0.0% | 16.7% | 19.4% | 16.7% |
| MSR | 166 | 9.6% | 30.7% | 30.7% | 37.3% |
| Motion (Obj.) | 47 | 6.4% | 25.5% | 27.7% | 42.6% |
| Positional Relationship (Cam.–Cam.) | 66 | 6.1% | 28.8% | 28.8% | 39.4% |
| Positional Relationship (Cam.–Obj.) | 59 | 8.5% | 39.0% | 49.2% | 40.7% |
| Positional Relationship (Cam.–Reg.) | 56 | 7.1% | 44.6% | 42.9% | 44.6% |
| Positional Relationship (Obj.–Obj.) | 67 | 9.0% | 31.3% | 28.4% | 25.4% |
| Positional Relationship (Obj.–Reg.) | 58 | 10.3% | 29.3% | 32.8% | 34.5% |
| Positional Relationship (Reg.–Reg.) | 54 | 1.9% | 31.5% | 27.8% | 37.0% |

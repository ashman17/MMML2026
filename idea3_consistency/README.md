# Research Idea 3: transformation-consistency on MMSI-Bench

Goal: a quick feasibility check. Does a VLM change its answer when we apply
transformations that should NOT change the correct answer, and does that
disagreement point at wrong answers?

## Variants per question
| name | what changes | why the correct letter stays the same |
|---|---|---|
| orig | nothing (images labeled "Image 1:", "Image 2:", ...) | baseline |
| mirror | every image flipped horizontally; left/right, leftmost/rightmost, clockwise/counterclockwise swapped in question and options | the whole scene is mirrored consistently. Skipped when the question mentions north/south/east/west, text, signs, "right angle", etc. |
| reorder | images shown in reversed order, each keeps its original label | the question refers to labels, not positions |
| shuffle | answer options permuted, prediction mapped back | pure label change |

Baseline with the same compute: self-consistency (K samples at temperature 0.7 on
the original input, where K = number of variants of that question).

## Setup on Babel (once)
```bash
# on a login node
conda create -n idea3 python=3.11 -y && conda activate idea3
pip install "vllm>=0.8" pandas pyarrow pillow matplotlib "huggingface_hub[cli]"
```
Copy this folder to Babel (e.g. `scp -r idea3_consistency vickytsa@<babel-login-host>:~/`).

## Run
```bash
cd ~/idea3_consistency
sbatch run_babel.sbatch                      # N=100 pilot, Qwen2.5-VL-7B
N=300 sbatch run_babel.sbatch                # N=300 main run
MODEL=Qwen/Qwen2.5-VL-3B-Instruct sbatch run_babel.sbatch
# held-out test split: every question not used in the pilot / N=300 dev runs
N=1000 EXCLUDE="$HOME/idea3_results/results_10646274/predictions.jsonl $HOME/idea3_results/results_10648545/predictions.jsonl" \
    sbatch run_babel.sbatch
```
Expect roughly 15 to 30 minutes for 100 questions on one GPU, including model loading.

During execution, outputs are written to node-local storage. The batch script's
exit trap then persists them to `~/idea3_results/results_<jobid>/`, including
when the Python process exits with an error. Copy that directory back to the
laptop before analysis.

Outputs:
* `predictions.jsonl`: every variant's rewritten question, raw output, and mapped prediction
* `report.md`: core metrics, bootstrap intervals, McNemar test, and extra-analysis tables
* `analysis_summary.json`: machine-readable headline metrics
* `per_type_flip_rate.csv`, `option_position_bias.csv`, `bootstrap_ci.csv`
* `fig_*.png`: the three report figures
* `examples/`: three qualitative cases and original-vs-mirrored image panels

Create the required 20-mirror/10-reorder human-review packet with:
```bash
python prepare_validity_check.py --predictions results_x/predictions.jsonl \
  --parquet MMSI_Bench.parquet --out results_x/validity_review
```

Re-analyze without rerunning: `python mmsi_consistency.py --analyze results_x/predictions.jsonl`

Dry run without a GPU (fake data): `python mmsi_consistency.py --parquet some.parquet --backend mock`

## Run with an API model (CMU LiteLLM gateway, no GPU)
```bash
pip install openai
export LITELLM_API_KEY=...                  # do not put the key in code or git
python check_gateway.py --model gpt-4o      # list models + test an image request
python mmsi_consistency.py --parquet MMSI_Bench.parquet --n 20 \
    --backend openai --model gpt-4o --out results_gpt4o
```
Responses are cached in `<out>/api_cache.jsonl`, so rerunning resumes for free.
Use `--max_tokens 2048` for reasoning models.

## How to read the report (what counts as "it works")
1. **Flip rate** per transform: well above 0 means the model is not invariant, so the idea has something to measure. Mirror flips concentrated on questions with left/right words point to a left/right bias.
2. **Accuracy when consistent vs inconsistent** and **AUROC**: AUROC clearly above 0.5 (say 0.6 or more) means disagreement is a usable error signal.
3. **Transformation vote vs self-consistency**: if the vote is higher at the same number of calls, that supports the test-time use of the idea.

Before trusting mirror/reorder numbers, complete and sign off the generated
20-mirror/10-reorder validity packet.
With 100 questions, a difference of a few points is noise; this is a feasibility signal only.

## TGCD letter-score experiment

`tgcd.py` collects valid option-letter first-token log-probabilities for the original and transformed
views plus a no-image view. It also records greedy and self-consistency answers. Score
collection does not choose hyperparameters; all aggregation and contrast calculations
are performed later by `--analyze`.

Local mock check:
```bash
python tgcd.py --backend mock --parquet ../data/MMSI_Bench.parquet --n 20 \
  --out results_tgcd_smoke
python tgcd.py --analyze results_tgcd_smoke/scores.jsonl
```

On PSC, run the 300 unique development IDs represented by the 300-question run and
the nested 100-question pilot with this exact submission command:
```bash
sbatch --export="ALL,N=300,SCRIPT=tgcd.py,ARGS=--include $PROJECT/idea3_results/results_10648545/predictions.jsonl $PROJECT/idea3_results/results_10646274/predictions.jsonl" run_psc.sbatch
```

After that job finishes, select aggregation and alpha on development data only:
```bash
python tgcd.py --analyze "$PROJECT/idea3_results/results_<dev_jobid>/scores.jsonl"
```

Run the 700-question held-out test split with this exact submission command:
```bash
sbatch --export="ALL,N=700,SCRIPT=tgcd.py,EXCLUDE=$PROJECT/idea3_results/results_10648545/predictions.jsonl $PROJECT/idea3_results/results_10646274/predictions.jsonl" run_psc.sbatch
```

Analyze test scores with the development selection file. Supplying `--selection` skips
the alpha sweep, so test labels cannot change the chosen configuration:
```bash
python tgcd.py --analyze "$PROJECT/idea3_results/results_<test_jobid>/scores.jsonl" \
  --selection "$PROJECT/idea3_results/results_<dev_jobid>/tgcd_selection.json"
```

On Babel (results land in `~/idea3_results/results_<jobid>/`):
```bash
cd ~/idea3_consistency
N=300 SCRIPT=tgcd.py ARGS="--include $HOME/idea3_results/results_10648545/predictions.jsonl $HOME/idea3_results/results_10646274/predictions.jsonl" \
    sbatch run_babel.sbatch
python tgcd.py --analyze ~/idea3_results/results_<dev_jobid>/scores.jsonl
N=700 SCRIPT=tgcd.py EXCLUDE="$HOME/idea3_results/results_10648545/predictions.jsonl $HOME/idea3_results/results_10646274/predictions.jsonl" \
    sbatch run_babel.sbatch
python tgcd.py --analyze ~/idea3_results/results_<test_jobid>/scores.jsonl \
  --selection ~/idea3_results/results_<dev_jobid>/tgcd_selection.json
```

On NCSA Delta (one-time `bash delta_setup.sh`; results land in
`/projects/bjjb/$USER/idea3_results/results_<jobid>/`):
```bash
cd /projects/bjjb/$USER/idea3_consistency
D=/projects/bjjb/$USER/idea3_dev
sbatch --export="ALL,N=300,SCRIPT=tgcd.py,ARGS=--include $D/results_10648545/predictions.jsonl $D/results_10646274/predictions.jsonl" run_delta.sbatch
sbatch --export="ALL,N=700,SCRIPT=tgcd.py,EXCLUDE=$D/results_10648545/predictions.jsonl $D/results_10646274/predictions.jsonl" run_delta.sbatch
```

To run TGCD on the four selected BLINK validation subsets, first convert them on a
login node with network access. This produces 522 rows (143 Spatial Relation, 124
Relative Depth, 133 Multi-view Reasoning, and 122 Object Localization):
```bash
conda activate /projects/bjjb/$USER/envs/idea3
pip install datasets
cd /projects/bjjb/$USER/idea3_consistency
python blink_to_parquet.py --out /projects/bjjb/$USER/data/BLINK_val.parquet
sbatch --export="ALL,N=522,SCRIPT=tgcd.py,PARQUET=/projects/bjjb/$USER/data/BLINK_val.parquet" run_delta.sbatch
```

`PARQUET` defaults to `/projects/bjjb/$USER/data/MMSI_Bench.parquet`, so existing
MMSI Delta commands do not need to set it.

TGCD outputs are `scores.jsonl`, `tgcd_report.md`, and, for development analysis,
`tgcd_selection.json` and `fig_accuracy_vs_alpha.png`. The score-collection log also
reports how often original score argmax agrees with the greedy text answer.

## Debiased LoRA fine-tuning

Build filtered train and validation parquets by joining SAT rows with their precomputed
no-image priors. Rows with missing letter scores or a no-image maximum probability of at
least 0.7 are removed.

```bash
python train/build_train_set.py --parquet SAT_train20k.parquet \
  --priors results_train/prior_scores.jsonl --out SAT_train_debiased.parquet
python train/train_lora.py --smoke
TRAIN=/projects/bjjb/$USER/data/SAT_train_debiased.parquet VAL=/projects/bjjb/$USER/data/SAT_val_debiased.parquet LOSS=ce \
  sbatch train/run_train_delta.sbatch
TRAIN=/projects/bjjb/$USER/data/SAT_train_debiased.parquet VAL=/projects/bjjb/$USER/data/SAT_val_debiased.parquet LOSS=lmh MIRROR=1 \
  sbatch train/run_train_delta.sbatch
N=700 SCRIPT=tgcd.py MODEL=/projects/bjjb/$USER/idea3_results/train_<jobid>/merged \
  EXCLUDE="/projects/bjjb/$USER/idea3_dev/results_10648545/predictions.jsonl /projects/bjjb/$USER/idea3_dev/results_10646274/predictions.jsonl" \
  sbatch run_delta.sbatch
```

Each training directory contains `train_log.jsonl`, the LoRA weights and
`mixin_head.pt` under `adapter/`, and a processor-equipped safetensors model under
`merged/` for direct vLLM evaluation. Validation accuracy in the log always uses the
plain image-model letter logits without the no-image bias or mixin gate.

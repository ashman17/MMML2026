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
* `fig_*.png`: the three report figures specified in `EXPERIMENT_SPEC.md`
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
20-mirror/10-reorder validity packet. An AI-assisted review is useful for
preflight, but the specification still requires a human validity check.
With 100 questions, a difference of a few points is noise; this is a feasibility signal only.

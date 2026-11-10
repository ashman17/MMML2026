# World-state graph experiment

Three conditions on the pinned MMSI-Bench parquet (`data/MMSI_Bench.parquet`):

- `raw`: the original images and the full question
- `flat` and `graph`: one shared extraction call, then the same question with a flat evidence list or a world-state graph rendered from that exact inventory

`analyze` is the only command that reads the `answer` column. Mock runs set `synthetic: true` and the analysis report starts with `MOCK`. Those outputs are not model findings.

The live extraction instruction is `prompts/extract_v3.txt`, the sparse entity/observation baseline. A six-question check of `extract_v4.txt` produced no claims, so prompt iteration stops at revision 3 and v4 is not sent. An unknown string may be 200 characters. A shared entity id with null `cross_view` is filled as `same` with uncertainty `medium`; that uncertainty is a parser default and is listed in `repairs/<id>.json`. Earlier run directories stay as they are.

Run every command from the repository root (`11777Project/MMML2026`) with the `11777-project` environment. A new PSC server can resume an existing run when the only manifest changes are `created_at`, `host`, and `slurm_job_id`. A different model revision, tokenizer revision, serving setting, dependency fingerprint, GPU identity, client setting, prompt, schema, or serializer still requires a new output directory. The first manifest stays in `run_config.json`. Later sessions are appended to `server_sessions.jsonl`. The interpreter used for development is:

```bash
/Users/zhengfeili/opt/anaconda3/envs/11777-project/bin/python
```

Below, `python` means that interpreter. A paused run is resumed by repeating the same command with `--resume`. A transport failure exits 75 and writes `status.json` with `"state": "paused"`. A contract or configuration error exits 2. A finished run exits 0. Reusing an output directory with different settings exits 2; use a new directory.

## 1. Prepare

```bash
python -m idea4_world_state prepare \
  --dataset data/MMSI_Bench.parquet \
  --split analyses/language/splits/language_explore_confirm.json \
  --out idea4_world_state/artifacts/prepare_v1
```

The committed artifact is `idea4_world_state/artifacts/prepare_v1/selection.json`. `prepare` refuses a non-empty output directory. There is no `--dataset` flag on `run`: the runner reads `contracts.DEFAULT_DATASET` and checks its SHA-256 against `selection.json`.

## 2. E1: server, then synthetic10

Bring up vLLM with `idea4_world_state/psc/README.md` (`env_setup.sh`, `serve_interactive.sh` or `serve.sbatch`, `tunnel.sh`, `fetch_manifest.sh`). Then, on the laptop, with the tunnel up and the token in the environment:

```bash
export IDEA4_API_KEY="$(cat "$TOKEN_FILE")"
export MANIFEST="idea4_world_state/artifacts/server_manifest.json"   # path printed by fetch_manifest.sh

python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile synthetic10 \
  --conditions raw \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/synthetic10 \
  --cache runs/idea4_world_state/_cache
```

`synthetic10` is ten generated 768×768 images and one fixed four-option question, raw only. `synthetic: true` on that run means the images are generated, not that the backend is mocked. The run config records `backend` as `vllm` or `mock`. The gate is a raw `ok` record whose prompt tokens are at most 30,000. A real response with missing usage fails. The mock backend is the only missing-usage exemption. The run is not part of analysis.

## 3. E2: development, timing, optional concurrency, freeze

```bash
python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile dev \
  --conditions raw,flat,graph \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/dev \
  --cache runs/idea4_world_state/_cache \
  --concurrency 1

python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile timing \
  --conditions raw \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/timing \
  --cache runs/idea4_world_state/_cache \
  --concurrency 1
```

Optional concurrency probe, only after the sequential dev run is complete. Do not point this at the sequential directory.

```bash
python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile dev \
  --conditions raw,flat,graph \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/dev-concurrency2 \
  --cache runs/idea4_world_state/_cache \
  --concurrency 2
```

Freeze. Replace the allocation, GPU, wall time, and deadline. `freeze` refuses to overwrite `--out`. A failed development criterion writes nothing and exits 2. A budget shortfall writes `decision: "shortfall"` and exits 2.

```bash
python -m idea4_world_state freeze \
  --dev-run runs/idea4_world_state/dev \
  --timing-run runs/idea4_world_state/timing \
  --synthetic-run runs/idea4_world_state/synthetic10 \
  --concurrency-run runs/idea4_world_state/dev-concurrency2 \
  --allocation-su "$ALLOCATION_SU" \
  --gpu l40s \
  --walltime-hours 4 \
  --deadline "$DEADLINE" \
  --out idea4_world_state/artifacts/protocol_freeze.json
```

Omit `--concurrency-run` if that probe was not run. The dev run used for freeze must contain no `transport_error` line in `attempts.jsonl`, including attempts that were later resumed. Pass thresholds (20/22 schema-valid, at most 1/22 extraction truncations, 100% answer parsing, serializer fact-atom equivalence, concurrency speedup 1.15, 30% allowance, 0.5 h startup per job) are constants in `runner.py` and are copied into the freeze file. Section 11 of the plan still marks the 20/22 and 1.15 figures as awaiting confirmation. Selecting concurrency 2 also needs a human check that the server log has no out-of-memory or preemption messages; `freeze` does not read that log.

The dev, timing, and synthetic runs must be `complete`, share the model identity, concurrency 1 (except the optional concurrency run), and the same recorded prompt, inventory-schema, and serializer fingerprints. Those fingerprints must also match the current files. Freeze does not fill in a missing fingerprint and does not accept an artifact edited after those runs. The error names the profiles to rerun. A rerun of a directory that lacks the fingerprint needs a new output directory; cached calls are reused when the request is unchanged.

Job count uses usable inference time, `walltime - 0.5 h`. A wall time of 0.5 h or less is rejected. Modeled hours are padded inference plus 0.5 h per job. Queue delay is not part of that total. With a 4 h wall, 3.9 h of padded inference is two jobs and 4.9 modeled hours.

## 4. Full raw run, then the active evidence profile

`raw_all`, `eval200`, and `eval100` require `--freeze` unless `--mock` is set. The settings, prompt hashes, inventory schema, `serialize.py`, and model identity must match the freeze. `eval200` / `eval100` must also be the freeze's `active_profile`. Use the concurrency recorded in the freeze (1 or 2). Repeat the same command with `--resume` after any exit 75.

```bash
python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile raw_all \
  --conditions raw \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/raw_all \
  --cache runs/idea4_world_state/_cache \
  --concurrency 1 \
  --freeze idea4_world_state/artifacts/protocol_freeze.json

python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile eval200 \
  --conditions flat,graph \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$MANIFEST" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/eval200 \
  --cache runs/idea4_world_state/_cache \
  --concurrency 1 \
  --freeze idea4_world_state/artifacts/protocol_freeze.json
```

If the freeze decision is `eval100`, change `--profile` and `--out` to `eval100`. Do not edit the flags that the freeze locks (token limits, `--max-side`, `--timeout`, `--concurrency`).

## 5. Analyze and audit

```bash
python -m idea4_world_state analyze \
  --run runs/idea4_world_state/raw_all \
  --out runs/idea4_world_state/raw_all/analysis

python -m idea4_world_state analyze \
  --run runs/idea4_world_state/eval200 \
  --raw-run runs/idea4_world_state/raw_all \
  --out runs/idea4_world_state/eval200/analysis

python -m idea4_world_state audit make \
  --run runs/idea4_world_state/eval200 \
  --n 20 \
  --out runs/idea4_world_state/eval200/audit
```

After the packet is labeled, import it and list outcomes:

```bash
python -m idea4_world_state audit import \
  --packet runs/idea4_world_state/eval200/audit \
  --labels path/to/filled-labels

python -m idea4_world_state audit outcomes \
  --run runs/idea4_world_state/eval200 \
  --analysis runs/idea4_world_state/eval200/analysis \
  --out runs/idea4_world_state/eval200/audit_outcomes
```

`predictions.csv` has exactly the columns in `contracts.PREDICTION_COLUMNS` and does not contain gold answers.

## Mock walkthrough

This uses the committed prepare artifact and the mock backend. It does not start vLLM. Outputs are synthetic. The freeze file it writes is not the E2 protocol freeze.

```bash
set -euo pipefail
cd /Users/zhengfeili/2026_Fall/11777/11777Project/MMML2026
PY=/Users/zhengfeili/opt/anaconda3/envs/11777-project/bin/python
TMP="$(mktemp -d)"
PREP=idea4_world_state/artifacts/prepare_v1
MANIFEST="$TMP/manifest-not-read.json"

"$PY" -m idea4_world_state run \
  --prepare-dir "$PREP" --profile synthetic10 --conditions raw \
  --server-manifest "$MANIFEST" \
  --out "$TMP/synthetic10" --cache "$TMP/cache" --mock

"$PY" -m idea4_world_state run \
  --prepare-dir "$PREP" --profile dev --conditions raw,flat,graph \
  --server-manifest "$MANIFEST" \
  --out "$TMP/dev" --cache "$TMP/cache" --concurrency 1 --mock

"$PY" -m idea4_world_state run \
  --prepare-dir "$PREP" --profile timing --conditions raw \
  --server-manifest "$MANIFEST" \
  --out "$TMP/timing" --cache "$TMP/cache" --concurrency 1 --mock

"$PY" -m idea4_world_state freeze \
  --dev-run "$TMP/dev" --timing-run "$TMP/timing" --synthetic-run "$TMP/synthetic10" \
  --allocation-su 100 --gpu l40s --walltime-hours 4 --deadline 2027-12-31 \
  --out "$TMP/protocol_freeze.json"

"$PY" -m idea4_world_state analyze \
  --run "$TMP/dev" --out "$TMP/dev/analysis"

"$PY" -m idea4_world_state audit make \
  --run "$TMP/dev" --n 20 --out "$TMP/dev/audit"

echo "TMP=$TMP"
echo "----- report.md head -----"
head -n 8 "$TMP/dev/analysis/report.md"
```

Mock latencies are zero, so this freeze's projection is 0 hours and the decision is `eval200` for any positive allocation. A real freeze uses the measured latencies from E1 and E2.

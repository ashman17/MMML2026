# PSC bring-up (E1)

These scripts start `Qwen/Qwen3-VL-8B-Instruct` on Bridges-2 and write the
flat server manifest from Section 8.5. They do not choose your allocation.
Pass it with `sbatch -A`, and pass your PSC username to the tunnel and
fetch scripts. Laptop `$USER` and the PSC username are often different.

Run the server from a checkout of this repository that compute nodes can
see (`$HOME` or `$PROJECT`). Submit batch scripts from the repository root.

## Check the allocation first

On a Bridges-2 login node:

```bash
projects
sinfo -p GPU,GPU-shared
```

`projects` prints allocation ids, which resource each one includes
("Bridges-2 GPU" is required), the remaining service units, and the Ocean
directory behind `$PROJECT`. `sinfo` shows whether `GPU` and `GPU-shared`
nodes are idle, allocated, or down. Confirm the GRES names below on your
account before relying on them. This is open decision 6 in the plan.

Useful dry run, still on a login node:

```bash
sbatch --test-only -A "$ALLOCATION" -p GPU-shared --gpus=l40s-48:1 -t 00:10:00 idea4_world_state/psc/serve.sbatch
```

## Service units

Rates below are from the Bridges-2 user guide (accounting section), checked
when these scripts were written. They are SU per GPU-hour of allocated
time, including idle time inside the job. They are not dollar or per-token
prices. Re-check with `projects` at launch.

| GPU | Partition for this experiment | Charge |
|---|---|---|
| L40S (`l40s-48`, 48 GB) | `GPU-shared`, 1 GPU | 1 SU per GPU-hour |
| H100 (`h100-80`, 80 GB) | `GPU-shared`, 1 GPU | 2 SU per GPU-hour |
| Either, whole node | `GPU`, all 8 GPUs | 8 or 16 SU per hour |

`GPU` does not share a node and bills all 8 GPUs. `GPU-shared` bills the
GPUs you request (at most 4). One GPU is the experiment setting, so
`GPU-shared` is the script default. `GPU-small` is a V100 node; do not use
it for L40S or H100.

A 4-hour one-GPU job is 4 SU on L40S or 8 SU on H100, plus any time the
server sits idle waiting for the laptop. Queue wait is not charged.

## Storage

| Variable | What it is | Use here |
|---|---|---|
| `$HOME` | `/jet/home/<user>`, 25 GB, backed up | Token file only (`~/.idea4/`, mode 0600) |
| `$PROJECT` | `/ocean/projects/<group>/<user>`, persistent, not backed up | Conda/venv, Hugging Face cache, manifest, logs, copied runs |
| `$LOCAL` | Node-local scratch, deleted when the job ends | Optional staging for `batch_all.sbatch`; copied back to `$PROJECT` |

`echo "$PROJECT"` on the login node. If you have more than one allocation,
it follows the allocation of the current session. `my_quotas` shows Ocean
usage. The conda default environment directory is under `$HOME`;
`env_setup.sh` uses `conda create --prefix` under `$PROJECT` so the vLLM
install does not fill the 25 GB home quota.

## Pinned server

- vLLM **0.32.0** (released 2026-10-18), plus `huggingface_hub` (whatever
  version pip resolves; `pip freeze` records it).
- Qwen's deployment notes require `vllm>=0.11.0` for Qwen3-VL. 0.32.0 is
  the current release and its notes still include Qwen3-VL fixes.
- 0.32.0 dropped Python 3.10. The environment is Python 3.12.
- The default PyPI wheel is CUDA 13.0. CUDA 12.9 wheels are attached to
  the GitHub release. Compare `nvidia-smi` ("CUDA Version") during E1.

Launch flags, from `contracts.py` and the vLLM 0.32 CLI:

```text
vllm serve Qwen/Qwen3-VL-8B-Instruct
  --revision <sha from huggingface_hub.model_info>
  --tokenizer-revision <same sha>
  --served-model-name qwen3-vl-8b
  --dtype bfloat16
  --max-model-len 32768
  --limit-mm-per-prompt.image 10
  --limit-mm-per-prompt.video 0
  --seed 0 --host 0.0.0.0 --port 8000
```

The token is not on the command line, because any user on a shared node can
read argv through `ps`. `serve_common.sh` sets `VLLM_API_KEY` only in the
environment of the `vllm serve` process. vLLM's OpenAI server uses
`--api-key` if given and otherwise falls back to `envs.VLLM_API_KEY`. The
readiness check then requires `/v1/models` to return 200 with the token and
401 without it. If the installed vLLM ignored the variable, startup fails
instead of serving unauthenticated.

`--limit-mm-per-prompt.image 10` is the current dotted form ("JSON keys
passed individually"). The JSON form `--limit-mm-per-prompt '{"image":10}'`
is also accepted by vLLM 0.32; the scripts emit the dotted form, and
`write_manifest.py` records either as a canonical JSON string.
`--limit-mm-per-prompt.video 0` is the Qwen3-VL image-only setting from the
vLLM recipe. The default video limit is 999 and reserves extra memory.
`--tokenizer-revision` is set because the vLLM docs describe an omitted
tokenizer revision as "the default version", not as "follow `--revision`".
The tokenizer files are in this model repo, so both SHAs match.

## 1. Create the environment

Do this on a CPU interact session so the install is not billed as a GPU.

```bash
export PSC_USER="<psc-username>"
export ALLOCATION="<allocation-id from projects>"
interact -p RM-shared -t 02:00:00
cd /path/to/MMML2026
bash idea4_world_state/psc/env_setup.sh
```

`module load anaconda3` is what the user guide shows. If that module name
differs, run `module avail anaconda3` and load the one you see, then rerun
the script (`conda` will already be on `PATH`). A venv instead of conda:

```bash
IDEA4_ENV_KIND=venv bash idea4_world_state/psc/env_setup.sh
```

The venv interpreter must be Python 3.11 or newer. The script writes:

- `$PROJECT/idea4_world_state/env/` — conda prefix or venv
- `$PROJECT/idea4_world_state/hf_home/` — `HF_HOME`
- `$PROJECT/idea4_world_state/pip-freeze.txt`
- `$PROJECT/idea4_world_state/env.sh` — sourced by the serve scripts

## 2a. Interactive server

`interact` allows at most 8 hours. An idle interact session is logged out
after 30 minutes, which kills the server. Use this path for setup and a
short `synthetic10`. Use batch for anything you will leave unattended.

```bash
interact -p GPU-shared --gres=gpu:l40s-48:1 -t 02:00:00
# or: interact -p GPU-shared --gres=gpu:h100-80:1 -t 02:00:00
cd /path/to/MMML2026
bash idea4_world_state/psc/serve_interactive.sh
```

The script resolves `main` to a commit SHA, writes
`~/.idea4/api_key_<jobid>` mode 0600, starts vLLM, waits until
`/v1/models` returns 200, writes the manifest, and prints three lines:

```text
MANIFEST: /ocean/projects/.../idea4_world_state/manifests/<jobid>.json
TOKEN_FILE: /jet/home/<user>/.idea4/api_key_<jobid>
TUNNEL: ssh -N -L 8000:<compute-node>:8000 <user>@bridges2.psc.edu
```

Leave that session running. On the laptop:

```bash
bash idea4_world_state/psc/tunnel.sh <compute-node> "$PSC_USER"
```

That is `ssh -N -L 8000:<compute-node>:8000 "$PSC_USER"@bridges2.psc.edu`.
The laptop's port 8000 then reaches the compute node. It does not open
port 8000 to the internet.

## 2b. Batch server, local client

Edit the `#SBATCH` GPU line in `serve.sbatch` if you want H100
(`--gpus=h100-80:1`) or a different wall time. Then, from the repo root:

```bash
sbatch -A "$ALLOCATION" idea4_world_state/psc/serve.sbatch
squeue -u "$PSC_USER"
```

The job log `idea4-ws-serve-<jobid>.log` contains the same `TUNNEL`,
`MANIFEST`, and `TOKEN_FILE` lines. Open the tunnel from the laptop as
above. Command-line `-p` and `--gpus` override the script.

## 2c. Batch server and client

`batch_all.sbatch` starts the server on the compute node, runs the client
against `http://127.0.0.1:8000/v1` in the same job, and copies the run
directory to `$PROJECT/idea4_world_state/runs/<run-id>/` (with the manifest
and a redacted `vllm.log`). No tunnel.

The run arguments are the block marked `run arguments` near the top of
`batch_all.sbatch`. Defaults are `synthetic10` and `raw`. Override without
editing the file:

```bash
sbatch -A "$ALLOCATION" --export=ALL,IDEA4_PROFILE=dev,IDEA4_CONDITIONS=raw,flat,graph,IDEA4_RUN_ID=dev1 \
  idea4_world_state/psc/batch_all.sbatch
```

Wall time is 4 hours in the file. GPU-shared allows up to 48 hours. A full
evaluation will not fit in 4 hours; change `#SBATCH -t` after the freeze
tells you the projection. Staging uses `$LOCAL` when Slurm sets it, and
the copy to `$PROJECT` runs on exit so the scratch copy is not the only
copy. The shared cache stays at `$PROJECT/idea4_world_state/cache`.

Each job gets a new staging directory. Resume does not point `--resume` at
that empty directory. Set `IDEA4_RESUME=1` and the same `IDEA4_RUN_ID` as
the published run. The job copies
`$PROJECT/idea4_world_state/runs/<run-id>/` into the new stage, runs, and
publishes the stage back only after that copy is complete. It refuses to
resume when no saved run exists, and it refuses to replace an existing saved
run when `IDEA4_RESUME` is not `1`.

```bash
sbatch -A "$ALLOCATION" \
  --export=ALL,IDEA4_PROFILE=dev,IDEA4_CONDITIONS=raw,flat,graph,IDEA4_RUN_ID=dev1,IDEA4_RESUME=1 \
  idea4_world_state/psc/batch_all.sbatch
```

If a publish is interrupted, the completed copy may be at
`runs/<run-id>.partial` and the previous copy at `runs/<run-id>.previous`.
The next resume moves a completed `.partial` back into place before it
touches the saved run. Do not delete those directories until
`runs/<run-id>/run_config.json` is present. The local-client workflow does
not use this script.

## Fetch the manifest

From the laptop, outside the repo if you can:

```bash
mkdir -p "$HOME/idea4_manifests"
bash idea4_world_state/psc/fetch_manifest.sh \
  /ocean/projects/.../idea4_world_state/manifests/<jobid>.json \
  "$HOME/idea4_manifests/server_manifest.json" \
  "$PSC_USER"
```

The remote path is the `MANIFEST:` line. It is on `$PROJECT`, so the login
node can see it. Do not point `--server-manifest` at a copy inside the
repository.

## Set IDEA4_API_KEY locally

The token never goes into the manifest, the fetch script, or a file in
this repo. Read the 0600 file on Bridges-2 into a shell variable:

```bash
export IDEA4_API_KEY="$(ssh "${PSC_USER}@bridges2.psc.edu" 'cat ~/.idea4/api_key_<jobid>')"
# Confirm it is set without printing it:
if [[ -n "${IDEA4_API_KEY}" ]]; then printf '%s\n' "IDEA4_API_KEY is set"; fi
```

Use the `TOKEN_FILE` path from the job if you want to be exact. The
variable lives in that shell. Do not write it into `.env`, a script,
`server_manifest.json`, or the Slurm log. Close the shell when you are
done. Each job creates a new file; an old value is a token mismatch.

The live vLLM log under `$PROJECT/idea4_world_state/logs/` is mode 0600
and can contain the server command line. `batch_all.sbatch` copies a
redacted `vllm.log` into the run directory. Do not copy the live log
into the repository.

Then, from the repository root on the laptop, with the tunnel still open:

```bash
python -m idea4_world_state run \
  --prepare-dir idea4_world_state/artifacts/prepare_v1 \
  --profile synthetic10 \
  --conditions raw \
  --endpoint http://localhost:8000/v1 \
  --served-model qwen3-vl-8b \
  --server-manifest "$HOME/idea4_manifests/server_manifest.json" \
  --api-key-env IDEA4_API_KEY \
  --out runs/idea4_world_state/e1-synthetic10
```

E1 passes when `synthetic10` succeeds and the run's model identity matches
this manifest (`resolved_revision`, `served_model_name` `qwen3-vl-8b`,
dtype, and the vLLM version).

## Manifest shape

`server_manifest.json` has exactly the Section 8.5 keys, in that order.
Missing probes are JSON `null`. Two values that are mappings in vLLM are
stored as strings so the object stays flat:

- `limit_mm_per_prompt`: canonical JSON text, for example
  `{"image":10,"video":0}`, or null
- `mm_processor_kwargs`: canonical JSON text of `--mm-processor-kwargs`,
  or null when the checkpoint defaults are left in place
- `serve_command`: the launched argv as a list of strings. It never contains
  `--api-key` or the token; `write_manifest.py` refuses to write it otherwise
- `gpu_memory_mib`, `max_model_len`, `seed`, `port`: integers or null
- `slurm_job_id`: string or null
- `pip_freeze_sha256`: SHA-256 of the `pip freeze` file bytes

`quantization` is null unless `--quantization` was passed. The BF16
checkpoint's own config is identified by `resolved_revision`.

## Troubleshooting

**Out of memory.** `Qwen3-VL-8B-Instruct` in BF16 is about 16 GB of
weights. L40S is 48 GB and H100 is 80 GB. The launcher already sets
`--limit-mm-per-prompt.video 0` so vLLM does not profile a long video.
If the log still shows a CUDA OOM during startup, stop the job and rerun
with a shorter context, then record that as a protocol change before
freeze:

```bash
IDEA4_MAX_MODEL_LEN=16384 bash idea4_world_state/psc/serve_interactive.sh
```

If it OOMs on L40S only under the 10-image probe, use H100
(`--gpus=h100-80:1` or `--gres=gpu:h100-80:1`). `nvidia-smi` on the node
shows whether another process is on the same GPU. Peak memory for the
calibration notes is in the vLLM log, not in `gpu_memory_mib` (that field
is the GPU's total memory).

**Port 8000 is in use.** `GPU-shared` places other jobs on the same node.
The script binds the port before starting vLLM and exits if it cannot.
Cancel the other job, ask for a new node, or pick a different node. The
tunnel script always forwards 8000, so changing `IDEA4_PORT` also means
editing `tunnel.sh` and the laptop client. Do not point the client at a
server whose manifest says a different port.

**Token mismatch.** A 401 from `/v1/models` or from the client means
`IDEA4_API_KEY` is not the token for this job. Read `TOKEN_FILE` again.
Do not reuse `api_key_<old-jobid>`. The manifest does not contain the
token; a mismatch is not visible there. If the client says the variable
is missing, the export was in another shell.

**vLLM wheel does not match the driver.** `nvidia-smi` prints the maximum
CUDA version the driver supports. vLLM 0.32.0's PyPI wheel is CUDA 13.0.
If the driver is older, install the CUDA 12.9 wheel from the v0.32.0
GitHub release assets instead of the default `pip install vllm==0.32.0`,
rerun `pip freeze`, and keep that freeze file. Do not mix the two.

**`module load anaconda3` fails.** Run `module avail anaconda3` and load
the listed module, then rerun `env_setup.sh`.

**Login node, or `nvidia-smi` fails.** The serve scripts refuse hosts
whose names contain `login` and hosts where `nvidia-smi -L` fails. Start
`interact` or wait for the batch job.

**Interact session dies while you are on the laptop.** That is the 30
minute idle logout. Resubmit with `serve.sbatch` or `batch_all.sbatch`.

**`model_info` fails or the weight download hangs.** The GPU node needs
HTTPS to Hugging Face, or the snapshot must already be in `HF_HOME`.
`env_setup.sh` points `HF_HOME` at `$PROJECT`. A login node can download
into that directory if the compute node cannot. The ready wait is 30
minutes (`IDEA4_READY_SECONDS`).

**`run` exits 2 with "not implemented".** The server scripts are in place,
and the client command is WP5. E1 step 3 needs that command in this
checkout.

**`sbatch` rejects `--mem` or `--gpus`.** The user guide's own GPU-shared
examples disagree: the prose says batch jobs use `--gpus=type:n`, while
one sample command uses `--gres`. The scripts follow `--gpus` for
`sbatch` and `--gres=gpu:type:n` for `interact`. If the submission is
rejected, try the other form once, then keep the one `sbatch --test-only`
accepts. Drop `#SBATCH --mem=64G` if the partition refuses a memory
request. `scontrol show partition GPU-shared` shows the limits your
account actually has.

## Sources

- Bridges-2 user guide (partitions, `interact`, `sbatch`, `$PROJECT`,
  `$LOCAL`, `projects`, `sinfo`):
  <https://www.psc.edu/resources/bridges-2/user-guide/>
- Accounting (L40S 1 SU/GPU-hour, H100 2 SU/GPU-hour):
  <https://www.psc.edu/resources/bridges-2/user-guide/#accounting-for-bridges-2-use>
- GPU type names: <https://www.psc.edu/nodes-added-to-bridges-2/>
- Queue wall times (GPU and GPU-shared max 48 h; GPU-small is V100),
  checked 2026-10-24:
  <https://support.access-ci.org/documentation/resources/bridges-2-gpu>
- vLLM 0.32.0 release (2026-10-18; Python 3.10 removed; CUDA 13.0 wheel;
  Qwen3-VL fixes): <https://github.com/vllm-project/vllm/releases/tag/v0.32.0>
- Qwen3-VL requires `vllm>=0.11.0`:
  <https://github.com/QwenLM/Qwen3-VL>
- `--limit-mm-per-prompt` JSON or dotted keys:
  <https://docs.vllm.ai/en/latest/configuration/engine_args.html>
- Dotted example `--limit-mm-per-prompt.image`:
  <https://github.com/vllm-project/vllm/blob/main/docs/features/multimodal_inputs.md>
- Image-only `--limit-mm-per-prompt.video 0`:
  <https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html>
- `--api-key` or the `VLLM_API_KEY` environment variable authenticates
  `/v1` routes: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- Fallback in the server source (`args.api_key or [envs.VLLM_API_KEY]`):
  <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py>
  and `vllm/envs.py`. This was read from `main`, not the `v0.32.0` tag; the
  401 readiness check verifies the installed version at startup.

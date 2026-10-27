#!/usr/bin/env bash
# Server and client in one GPU job, then copy the run directory to $PROJECT.
# Submit from the repository root:
#   sbatch -A "$ALLOCATION" idea4_world_state/psc/batch_all.sbatch
# Do not hardcode an allocation id.
#
# One GPU on GPU-shared. Verify with sinfo. H100: --gpus=h100-80:1
# Whole node (8 GPUs, full-node SU charge): -p GPU and --gpus=l40s-48:8 or h100-80:8
# Edit -t. GPU-shared maximum is 48 hours; the default if you delete -t is 1 hour.
#SBATCH --job-name=idea4-ws-batch
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o idea4-ws-batch-%j.log

set -euo pipefail

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
  echo "Submit this file with sbatch from the repository root." >&2
  exit 2
fi
cd "${SLURM_SUBMIT_DIR}"
if [[ ! -f idea4_world_state/psc/serve_common.sh ]]; then
  echo "Submit from the repository root. serve_common.sh was not found." >&2
  exit 2
fi
export IDEA4_REPO_ROOT="${SLURM_SUBMIT_DIR}"

# shellcheck source=serve_common.sh
source "${SLURM_SUBMIT_DIR}/idea4_world_state/psc/serve_common.sh"

# ----- run arguments (edit these, or export them before sbatch) -----
# These are the `python -m idea4_world_state run` flags for this job.
IDEA4_PROFILE="${IDEA4_PROFILE:-synthetic10}"
IDEA4_CONDITIONS="${IDEA4_CONDITIONS:-raw}"
IDEA4_RUN_ID="${IDEA4_RUN_ID:-synthetic10-${SLURM_JOB_ID:-local}}"
IDEA4_PREPARE_DIR="${IDEA4_PREPARE_DIR:-idea4_world_state/artifacts/prepare_v1}"
IDEA4_EXTRACT_TOKENS="${IDEA4_EXTRACT_TOKENS:-4096}"
IDEA4_ANSWER_TOKENS="${IDEA4_ANSWER_TOKENS:-64}"
IDEA4_MAX_SIDE="${IDEA4_MAX_SIDE:-768}"
IDEA4_TIMEOUT="${IDEA4_TIMEOUT:-300}"
IDEA4_CONCURRENCY="${IDEA4_CONCURRENCY:-1}"
# Leave empty until a protocol freeze exists.
IDEA4_FREEZE="${IDEA4_FREEZE:-}"
# Set to 1 only to continue $PROJECT/idea4_world_state/runs/$IDEA4_RUN_ID.
# The saved run is copied into this job's new staging directory before --resume.
# A missing saved run is refused. An existing saved run is not replaced when this is unset.
IDEA4_RESUME="${IDEA4_RESUME:-0}"
# ----- end run arguments -----

idea4_copy_run() {
  idea4_publish_batch_run
}

IDEA4_STOP_ON_EXIT=1
idea4_start_server
idea4_prepare_batch_run

IDEA4_CACHE="${IDEA4_CACHE:-${PROJECT}/idea4_world_state/cache}"
mkdir -p "$(dirname "${IDEA4_STAGE_RUN}")" "${IDEA4_CACHE}"

run_cmd=(
  "${IDEA4_ENV_PREFIX}/bin/python" -m idea4_world_state run
  --prepare-dir "${IDEA4_PREPARE_DIR}"
  --profile "${IDEA4_PROFILE}"
  --conditions "${IDEA4_CONDITIONS}"
  --endpoint "http://127.0.0.1:${IDEA4_PORT}/v1"
  --served-model "${IDEA4_SERVED_MODEL}"
  --server-manifest "${IDEA4_MANIFEST}"
  --api-key-env IDEA4_API_KEY
  --out "${IDEA4_STAGE_RUN}"
  --cache "${IDEA4_CACHE}"
  --extract-tokens "${IDEA4_EXTRACT_TOKENS}"
  --answer-tokens "${IDEA4_ANSWER_TOKENS}"
  --max-side "${IDEA4_MAX_SIDE}"
  --timeout "${IDEA4_TIMEOUT}"
  --concurrency "${IDEA4_CONCURRENCY}"
)
if [[ -n "${IDEA4_FREEZE}" ]]; then
  run_cmd+=(--freeze "${IDEA4_FREEZE}")
fi
if [[ "${IDEA4_RESUME}" == "1" ]]; then
  run_cmd+=(--resume)
fi

set +e
env IDEA4_API_KEY="${IDEA4_TOKEN}" "${run_cmd[@]}"
run_status=$?
set -e
idea4_copy_run
exit "${run_status}"

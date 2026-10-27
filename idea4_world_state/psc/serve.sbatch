#!/usr/bin/env bash
# Unattended vLLM server. Submit from the repository root:
#   sbatch -A "$ALLOCATION" idea4_world_state/psc/serve.sbatch
# Do not hardcode an allocation id. Omit -A to use your default allocation.
#
# One GPU on GPU-shared (charged per GPU). Verify the partition with sinfo.
# H100: change --gpus=l40s-48:1 to --gpus=h100-80:1
# Whole node (charges all 8 GPUs): -p GPU and --gpus=l40s-48:8 or --gpus=h100-80:8
# GPU and GPU-shared batch jobs: default 1 hour, maximum 48 hours. Edit -t.
#SBATCH --job-name=idea4-ws-serve
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o idea4-ws-serve-%j.log

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
idea4_serve_foreground

#!/bin/bash
#SBATCH --job-name=mmml-vggt
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
: "${MMML_WORKDIR:?Set MMML_WORKDIR to the copied PSC project directory}"
cd "$MMML_WORKDIR"
IDS="${IDS:-all}"
python -c 'import vggt, torch' 2>/dev/null || { echo 'Install torch and VGGT (pip install git+https://github.com/facebookresearch/vggt) in the PSC environment first' >&2; exit 1; }

python workflow.py vggt --ids "$IDS"

#!/bin/bash
#SBATCH --job-name=idea3-consistency
#SBATCH --partition=neulab-mini
#SBATCH --qos=neulab_mini_qos
#SBATCH --gres=gpu:1                 # 7B VLM fits on one 48GB GPU (A6000 / L40S)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -e
WORK=/data/user_data/vickytsa/idea3
PERSIST=$HOME/idea3_results
OUT=$WORK/results_${SLURM_JOB_ID}
export HF_HOME=$WORK/hf_cache
mkdir -p $WORK $PERSIST logs

# /data is node-local on Babel.  Always copy the small experiment outputs back
# to the shared home filesystem before the allocation ends so they remain
# downloadable from the login node.  The trap also preserves partial outputs
# if the Python process exits non-zero after creating them.
persist_results() {
  if [ -d "$OUT" ]; then
    rm -rf "$PERSIST/results_${SLURM_JOB_ID}.partial"
    cp -a "$OUT" "$PERSIST/results_${SLURM_JOB_ID}.partial"
    rm -rf "$PERSIST/results_${SLURM_JOB_ID}"
    mv "$PERSIST/results_${SLURM_JOB_ID}.partial" "$PERSIST/results_${SLURM_JOB_ID}"
  fi
}
trap persist_results EXIT
source ~/miniforge3/etc/profile.d/conda.sh
conda activate idea3

# 1) download MMSI-Bench once
if [ ! -f $WORK/MMSI_Bench.parquet ]; then
  python - "$WORK" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="RunsenXu/MMSI-Bench", filename="MMSI_Bench.parquet",
                repo_type="dataset", local_dir=sys.argv[1])
PY
fi

# 2) run: defaults to the N=100 pilot; set N=300 for the main experiment
python mmsi_consistency.py \
    --parquet $WORK/MMSI_Bench.parquet \
    --n ${N:-100} \
    --model ${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct} \
    --out "$OUT"

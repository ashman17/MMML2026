#!/bin/bash
#SBATCH --job-name=idea3-consistency
#SBATCH --account=cis260328p
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1             # same GPU class as Babel A6000/L40S; vLLM 0.29 does not run on V100
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# PSC Bridges-2 version of run_babel.sbatch.  One-time setup: bash psc_setup.sh
# $PROJECT (Ocean) is shared storage, so outputs are written there directly.
set -e
OUT=$PROJECT/idea3_results/results_${SLURM_JOB_ID}
export HF_HOME=$PROJECT/hf_cache HF_HUB_OFFLINE=1
mkdir -p logs $PROJECT/idea3_results
module load anaconda3/2024.10-1
source activate $PROJECT/envs/idea3

# N / EXCLUDE / MODEL work as in run_babel.sbatch
python mmsi_consistency.py \
    --parquet $PROJECT/data/MMSI_Bench.parquet \
    --n ${N:-100} \
    ${EXCLUDE:+--exclude $EXCLUDE} \
    --model ${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct} \
    --out "$OUT"

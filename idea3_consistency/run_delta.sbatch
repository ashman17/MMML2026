#!/bin/bash
#SBATCH --job-name=idea3-consistency
#SBATCH --account=bjjb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

# NCSA Delta version of run_psc.sbatch.  One-time setup: bash delta_setup.sh, then mkdir -p logs.
# /projects is shared storage, so outputs are written there directly.
set -e
BASE=/projects/bjjb/$USER
OUT=$BASE/idea3_results/results_${SLURM_JOB_ID}
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
mkdir -p logs $BASE/idea3_results
module load miniforge3-python
source activate $BASE/envs/idea3

# N / PARQUET / EXCLUDE / MODEL / SCRIPT / HF_OVERRIDES / ARGS work as in run_psc.sbatch.
python ${SCRIPT:-mmsi_consistency.py} \
    --parquet "${PARQUET:-$BASE/data/MMSI_Bench.parquet}" \
    --n ${N:-100} \
    ${EXCLUDE:+--exclude $EXCLUDE} \
    ${HF_OVERRIDES:+--hf_overrides "$HF_OVERRIDES"} \
    --model ${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct} \
    --out "$OUT" \
    $ARGS

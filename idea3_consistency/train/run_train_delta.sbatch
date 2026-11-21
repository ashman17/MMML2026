#!/bin/bash
#SBATCH --job-name=idea3-train
#SBATCH --account=bjjb-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -e
BASE=/projects/bjjb/$USER
OUT=$BASE/idea3_results/train_${SLURM_JOB_ID}
export HF_HOME=$BASE/hf_cache HF_HUB_OFFLINE=1
mkdir -p logs "$OUT"
module load miniforge3-python
source activate $BASE/envs/idea3
MIRROR_ARG=
if [ "${MIRROR:-0}" = 1 ]; then
    MIRROR_ARG=--mirror_aug
fi

python train/train_lora.py \
    --train "$TRAIN" \
    --val "$VAL" \
    --loss "${LOSS:-lmh}" \
    --out "$OUT" \
    $MIRROR_ARG \
    $ARGS

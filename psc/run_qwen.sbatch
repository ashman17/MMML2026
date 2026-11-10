#!/bin/bash
#SBATCH --job-name=mmml-qwen
#SBATCH --partition=GPU-shared
#SBATCH --gpus=h100-80:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8

set -euo pipefail
: "${MMML_WORKDIR:?Set MMML_WORKDIR to the copied PSC project directory}"
cd "$MMML_WORKDIR"
mkdir -p runs
CONDITIONS="${CONDITIONS-images}"     # may be set to "" to only run the judge
IDS="${IDS:-sample}"
JUDGE_SOURCES="${JUDGE_SOURCES:-}"   # e.g. "images images_vggt": judge those traces against VGGT
command -v vllm >/dev/null || { echo 'Install vLLM in the PSC environment first' >&2; exit 1; }
if [[ " $CONDITIONS " == *" images_vggt "* ]]; then
  python -c 'import vggt, torch' 2>/dev/null || { echo 'Install torch and VGGT in the PSC environment first' >&2; exit 1; }
  python workflow.py vggt --ids "$IDS"
fi
command -v python >/dev/null || { echo 'Python is required' >&2; exit 1; }

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 32768 --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520}' --enforce-eager \
  > runs/qwen-vllm.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

ready=0
for attempt in $(seq 1 180); do
  if python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2)' >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'vLLM exited before becoming ready; see runs/qwen-vllm.log' >&2
    exit 1
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo 'vLLM did not become ready; see runs/qwen-vllm.log' >&2
  exit 1
fi

for condition in $CONDITIONS; do
  python workflow.py model --provider compatible --model Qwen/Qwen2.5-VL-7B-Instruct \
    --base-url http://127.0.0.1:8000/v1 --condition "$condition" --ids "$IDS" --timeout 600
done
if [ -n "$CONDITIONS" ]; then
  python workflow.py score --provider compatible --model Qwen/Qwen2.5-VL-7B-Instruct \
    --ids "$IDS" --conditions $CONDITIONS
fi
for source in $JUDGE_SOURCES; do
  python workflow.py judge --provider compatible --model Qwen/Qwen2.5-VL-7B-Instruct \
    --base-url http://127.0.0.1:8000/v1 --source-condition "$source" --ids "$IDS" --timeout 600
done

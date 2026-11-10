#!/bin/bash
#SBATCH --job-name=mmml-serve
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8

# Runs VGGT (optional), then serves Qwen through vLLM on this compute node so
# workflow.py can run on your laptop through an SSH tunnel. See README.

set -euo pipefail
: "${MMML_WORKDIR:?Set MMML_WORKDIR to the copied PSC project directory}"
cd "$MMML_WORKDIR"
mkdir -p runs
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
RUN_VGGT="${RUN_VGGT:-1}"
KEY_FILE="$HOME/.mmml_vllm_key"
command -v vllm >/dev/null || { echo 'Install vLLM in the PSC environment first' >&2; exit 1; }

# The server listens on the cluster network, so require a key other users cannot read.
if [ -z "${MMML_API_KEY:-}" ]; then
  if [ ! -s "$KEY_FILE" ]; then
    (umask 077; python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$KEY_FILE")
  fi
  MMML_API_KEY="$(cat "$KEY_FILE")"
fi

if [ "$RUN_VGGT" = "1" ]; then
  python -c 'import vggt, torch' 2>/dev/null || { echo 'Install torch and VGGT in the PSC environment first' >&2; exit 1; }
  python workflow.py vggt --ids all
fi

vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --api-key "$MMML_API_KEY" --dtype bfloat16 \
  --max-model-len 32768 --limit-mm-per-prompt '{"image":10,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":1003520}' --enforce-eager \
  > runs/qwen-vllm-serve.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT

for attempt in $(seq 1 180); do
  if MMML_API_KEY="$MMML_API_KEY" PORT="$PORT" python -c 'import os, urllib.request; urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:" + os.environ["PORT"] + "/v1/models", headers={"Authorization": "Bearer " + os.environ["MMML_API_KEY"]}), timeout=2)' >/dev/null 2>&1; then
    echo "vLLM is ready on $(hostname):$PORT"
    echo "On your laptop, run:"
    echo "  ssh -N -L $PORT:$(hostname):$PORT $USER@bridges2.psc.edu"
    echo "Set MMML_API_KEY in your local .env.local to the key you passed, or to the output of: cat $KEY_FILE"
    wait "$server_pid"
    exit $?
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'vLLM exited before becoming ready; see runs/qwen-vllm-serve.log' >&2
    exit 1
  fi
  sleep 5
done
echo 'vLLM did not become ready; see runs/qwen-vllm-serve.log' >&2
exit 1

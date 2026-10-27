#!/usr/bin/env bash
# Shared by serve_interactive.sh, serve.sbatch, and batch_all.sbatch.
# Source this file. Do not execute it.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Source serve_common.sh from a serve script. Do not execute it." >&2
  exit 2
fi

idea4_require_project() {
  if [[ -z "${PROJECT:-}" || "${PROJECT}" != /* ]]; then
    echo "Set PROJECT to the absolute Bridges-2 Ocean directory." >&2
    exit 1
  fi
}

idea4_source_env() {
  local env_file="${IDEA4_ENV_FILE:-${PROJECT}/idea4_world_state/env.sh}"
  if [[ ! -f "${env_file}" ]]; then
    echo "Missing ${env_file}. Run idea4_world_state/psc/env_setup.sh first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${env_file}"
  if [[ ! -x "${IDEA4_ENV_PREFIX}/bin/python" || ! -x "${IDEA4_ENV_PREFIX}/bin/vllm" ]]; then
    echo "Environment ${IDEA4_ENV_PREFIX} has no python or vllm binary." >&2
    exit 1
  fi
}

idea4_defaults() {
  IDEA4_REPO_ROOT="${IDEA4_REPO_ROOT:-${PWD}}"
  IDEA4_MODEL_ID="${IDEA4_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
  IDEA4_SERVED_MODEL="${IDEA4_SERVED_MODEL:-qwen3-vl-8b}"
  IDEA4_REQUESTED_REVISION="${IDEA4_REQUESTED_REVISION:-main}"
  IDEA4_DTYPE="${IDEA4_DTYPE:-bfloat16}"
  IDEA4_MAX_MODEL_LEN="${IDEA4_MAX_MODEL_LEN:-32768}"
  IDEA4_MAX_IMAGES="${IDEA4_MAX_IMAGES:-10}"
  IDEA4_VIDEO_LIMIT="${IDEA4_VIDEO_LIMIT:-0}"
  IDEA4_SEED="${IDEA4_SEED:-0}"
  IDEA4_PORT="${IDEA4_PORT:-8000}"
  IDEA4_READY_SECONDS="${IDEA4_READY_SECONDS:-1800}"
  IDEA4_PIP_FREEZE="${IDEA4_PIP_FREEZE:-${PROJECT}/idea4_world_state/pip-freeze.txt}"
  local job_tag="${SLURM_JOB_ID:-$$}"
  IDEA4_JOB_TAG="${job_tag}"
  IDEA4_MANIFEST="${IDEA4_MANIFEST:-${PROJECT}/idea4_world_state/manifests/${job_tag}.json}"
  IDEA4_VLLM_LOG="${IDEA4_VLLM_LOG:-${PROJECT}/idea4_world_state/logs/vllm-${job_tag}.log}"
  mkdir -p "$(dirname "${IDEA4_MANIFEST}")" "$(dirname "${IDEA4_VLLM_LOG}")" "${HOME}/.idea4"
  chmod 700 "${HOME}/.idea4"
}

idea4_refuse_login_node() {
  local host
  host=$(hostname -s 2>/dev/null || hostname)
  case "${host}" in
    *login*)
      echo "This host (${host}) looks like a login node. Start an interact or sbatch GPU job first." >&2
      exit 1
      ;;
  esac
  if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "nvidia-smi failed. Request a GPU (GPU-shared, one L40S or H100) before serving." >&2
    exit 1
  fi
}

idea4_compute_node() {
  if [[ -n "${SLURM_NODELIST:-}" && "${SLURM_NODELIST}" != *"["* && "${SLURM_NODELIST}" != *","* ]]; then
    printf '%s\n' "${SLURM_NODELIST}"
    return
  fi
  hostname -s 2>/dev/null || hostname
}

idea4_resolve_sha() {
  local py="${IDEA4_ENV_PREFIX}/bin/python"
  IDEA4_RESOLVED_REVISION=$("${py}" - "${IDEA4_MODEL_ID}" "${IDEA4_REQUESTED_REVISION}" <<'PY'
import sys
from huggingface_hub import model_info

model_id, revision = sys.argv[1], sys.argv[2]
info = model_info(model_id, revision=revision)
sha = getattr(info, "sha", None) or ""
if len(sha) != 40 or any(char not in "0123456789abcdefABCDEF" for char in sha):
    sys.stderr.write("model_info did not return a 40-character commit SHA\n")
    sys.exit(1)
sys.stdout.write(sha.lower())
PY
)
}

idea4_make_token() {
  IDEA4_TOKEN_FILE="${HOME}/.idea4/api_key_${IDEA4_JOB_TAG}"
  local old_umask
  old_umask=$(umask)
  umask 077
  IDEA4_TOKEN=$(openssl rand -hex 24)
  printf '%s\n' "${IDEA4_TOKEN}" > "${IDEA4_TOKEN_FILE}"
  umask "${old_umask}"
  chmod 0600 "${IDEA4_TOKEN_FILE}"
  if [[ ! "${IDEA4_TOKEN}" =~ ^[0-9a-f]{48}$ ]]; then
    echo "openssl rand -hex 24 did not return 48 hex characters." >&2
    exit 1
  fi
}

idea4_check_port() {
  "${IDEA4_ENV_PREFIX}/bin/python" - "${IDEA4_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    sys.stderr.write(f"port {port} is in use ({exc}). See psc/README.md.\n")
    sys.exit(1)
finally:
    sock.close()
PY
}

idea4_build_argv() {
  SERVE_ARGV=(
    "${IDEA4_ENV_PREFIX}/bin/vllm"
    serve
    "${IDEA4_MODEL_ID}"
    --revision "${IDEA4_RESOLVED_REVISION}"
    --tokenizer-revision "${IDEA4_RESOLVED_REVISION}"
    --served-model-name "${IDEA4_SERVED_MODEL}"
    --dtype "${IDEA4_DTYPE}"
    --max-model-len "${IDEA4_MAX_MODEL_LEN}"
    --limit-mm-per-prompt.image "${IDEA4_MAX_IMAGES}"
    --limit-mm-per-prompt.video "${IDEA4_VIDEO_LIMIT}"
    --seed "${IDEA4_SEED}"
    --host 0.0.0.0
    --port "${IDEA4_PORT}"
  )
}

idea4_write_argv_file() {
  IDEA4_ARGV_FILE="${HOME}/.idea4/serve_argv_${IDEA4_JOB_TAG}.json"
  "${IDEA4_ENV_PREFIX}/bin/python" - "${IDEA4_ARGV_FILE}" "${SERVE_ARGV[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

path, *argv = sys.argv[1:]
file = Path(path)
file.write_text(json.dumps(argv), encoding="utf-8")
os.chmod(file, 0o600)
PY
}

idea4_launch() {
  touch "${IDEA4_VLLM_LOG}"
  chmod 600 "${IDEA4_VLLM_LOG}"
  # The token goes only into this process's environment; argv is world-readable via ps.
  VLLM_API_KEY="${IDEA4_TOKEN}" "${SERVE_ARGV[@]}" >> "${IDEA4_VLLM_LOG}" 2>&1 &
  IDEA4_VLLM_PID=$!
  echo "vLLM pid ${IDEA4_VLLM_PID}, log ${IDEA4_VLLM_LOG}"
}

idea4_wait_ready() {
  echo "Waiting for http://127.0.0.1:${IDEA4_PORT}/v1/models"
  "${IDEA4_ENV_PREFIX}/bin/python" - "${IDEA4_TOKEN_FILE}" "${IDEA4_PORT}" "${IDEA4_READY_SECONDS}" "${IDEA4_VLLM_PID}" <<'PY'
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

token_path, port, seconds, pid = sys.argv[1:]
token = Path(token_path).read_text(encoding="utf-8").strip()
url = f"http://127.0.0.1:{port}/v1/models"
deadline = time.time() + int(seconds)
attempt = 0
while time.time() < deadline:
    try:
        os.kill(int(pid), 0)
    except OSError:
        sys.stderr.write("vLLM exited before /v1/models became ready\n")
        sys.exit(1)
    try:
        request = urllib.request.Request(
            url, headers={"Authorization": "Bearer " + token}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            ready = response.status == 200
        if ready:
            try:
                with urllib.request.urlopen(url, timeout=5):
                    pass
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    sys.exit(0)
                sys.stderr.write(f"unauthenticated /v1/models returned {exc.code}, expected 401\n")
                sys.exit(1)
            sys.stderr.write("server accepted an unauthenticated request; VLLM_API_KEY was not applied\n")
            sys.exit(1)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    attempt += 1
    if attempt % 6 == 0:
        sys.stderr.write("still waiting for /v1/models\n")
    time.sleep(10)
sys.stderr.write("timed out waiting for /v1/models\n")
sys.exit(1)
PY
}

idea4_write_manifest() {
  IDEA4_COMPUTE_NODE=$(idea4_compute_node)
  export IDEA4_COMPUTE_NODE
  if [[ ! "${IDEA4_COMPUTE_NODE}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Compute node name is not a single hostname: ${IDEA4_COMPUTE_NODE}" >&2
    exit 1
  fi
  "${IDEA4_ENV_PREFIX}/bin/python" \
    "${IDEA4_REPO_ROOT}/idea4_world_state/psc/write_manifest.py" \
    --out "${IDEA4_MANIFEST}" \
    --serve-argv-file "${IDEA4_ARGV_FILE}" \
    --requested-revision "${IDEA4_REQUESTED_REVISION}" \
    --pip-freeze "${IDEA4_PIP_FREEZE}" \
    --api-key-file "${IDEA4_TOKEN_FILE}"
  rm -f "${IDEA4_ARGV_FILE}"
  IDEA4_ARGV_FILE=""
}

idea4_print_access() {
  local user="${PSC_USER:-${USER}}"
  echo "MANIFEST: ${IDEA4_MANIFEST}"
  echo "TOKEN_FILE: ${IDEA4_TOKEN_FILE}"
  echo "TOKEN_FILE is mode 0600 on this host. It is not printed here."
  if [[ "${IDEA4_PORT}" == "8000" ]]; then
    echo "TUNNEL: ssh -N -L 8000:${IDEA4_COMPUTE_NODE}:8000 ${user}@bridges2.psc.edu"
  else
    echo "TUNNEL: ssh -N -L ${IDEA4_PORT}:${IDEA4_COMPUTE_NODE}:${IDEA4_PORT} ${user}@bridges2.psc.edu"
    echo "warning: psc/tunnel.sh always forwards port 8000." >&2
  fi
  echo "FETCH: idea4_world_state/psc/fetch_manifest.sh ${IDEA4_MANIFEST} LOCAL_PATH ${user}"
  echo "Set IDEA4_API_KEY in the local shell from TOKEN_FILE. Do not write the token into the repo. See psc/README.md."
}

idea4_stop_server() {
  if [[ -n "${IDEA4_VLLM_PID:-}" ]] && kill -0 "${IDEA4_VLLM_PID}" 2>/dev/null; then
    kill "${IDEA4_VLLM_PID}" 2>/dev/null || true
    wait "${IDEA4_VLLM_PID}" 2>/dev/null || true
  fi
  IDEA4_VLLM_PID=""
}

idea4_on_exit() {
  local status=$?
  if [[ -n "${IDEA4_ARGV_FILE:-}" ]]; then
    rm -f "${IDEA4_ARGV_FILE}"
  fi
  if declare -F idea4_copy_run >/dev/null 2>&1; then
    idea4_copy_run || true
  fi
  if [[ "${IDEA4_STOP_ON_EXIT:-0}" == "1" ]]; then
    idea4_stop_server || true
  fi
  if [[ "${status}" -ne 0 && "${IDEA4_SERVER_READY:-0}" != "1" && -n "${IDEA4_VLLM_LOG:-}" && -f "${IDEA4_VLLM_LOG}" ]]; then
    echo "Last lines of ${IDEA4_VLLM_LOG}:" >&2
    tail -n 40 "${IDEA4_VLLM_LOG}" | idea4_redact_stream >&2 || true
  fi
  return "${status}"
}

idea4_install_traps() {
  trap idea4_on_exit EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
}

idea4_start_server() {
  idea4_require_project
  idea4_source_env
  idea4_defaults
  idea4_refuse_login_node
  idea4_install_traps
  idea4_resolve_sha
  idea4_make_token
  idea4_check_port
  idea4_build_argv
  idea4_write_argv_file
  idea4_launch
  if ! idea4_wait_ready; then
    echo "Server did not become ready. See ${IDEA4_VLLM_LOG}." >&2
    exit 1
  fi
  idea4_write_manifest
  IDEA4_SERVER_READY=1
  idea4_print_access
}

idea4_redact_stream() {
  if [[ -n "${IDEA4_TOKEN:-}" ]]; then
    sed "s#${IDEA4_TOKEN}#REDACTED#g"
  else
    cat
  fi
}

idea4_serve_foreground() {
  IDEA4_STOP_ON_EXIT=1
  idea4_start_server
  wait "${IDEA4_VLLM_PID}"
}

# batch_all.sbatch stages each job in a fresh directory. Resume has to copy
# the published run in first, and publish has to leave the previous copy in
# place until the replacement is complete.

idea4_saved_run_paths() {
  IDEA4_SAVED_RUN="${PROJECT}/idea4_world_state/runs/${IDEA4_RUN_ID}"
  IDEA4_SAVED_PARTIAL="${IDEA4_SAVED_RUN}.partial"
  IDEA4_SAVED_PREVIOUS="${IDEA4_SAVED_RUN}.previous"
}

idea4_recover_saved_run() {
  idea4_saved_run_paths
  if [[ -f "${IDEA4_SAVED_RUN}/run_config.json" ]]; then
    return 0
  fi
  if [[ -f "${IDEA4_SAVED_PARTIAL}/run_config.json" ]]; then
    echo "Recovering completed publish copy ${IDEA4_SAVED_PARTIAL} to ${IDEA4_SAVED_RUN}." >&2
    mv "${IDEA4_SAVED_PARTIAL}" "${IDEA4_SAVED_RUN}"
    return 0
  fi
  if [[ -f "${IDEA4_SAVED_PREVIOUS}/run_config.json" ]]; then
    echo "Recovering ${IDEA4_SAVED_PREVIOUS} to ${IDEA4_SAVED_RUN} after an interrupted publish." >&2
    mv "${IDEA4_SAVED_PREVIOUS}" "${IDEA4_SAVED_RUN}"
  fi
}

idea4_prepare_batch_run() {
  if [[ -z "${PROJECT:-}" || -z "${IDEA4_RUN_ID:-}" || -z "${IDEA4_JOB_TAG:-}" ]]; then
    echo "PROJECT, IDEA4_RUN_ID, and IDEA4_JOB_TAG must be set before staging a batch run." >&2
    return 2
  fi
  if [[ "${PROJECT}" != /* ]]; then
    echo "PROJECT must be an absolute directory." >&2
    return 2
  fi
  idea4_recover_saved_run
  if [[ -n "${LOCAL:-}" && -d "${LOCAL}" ]]; then
    IDEA4_STAGE="${LOCAL}/idea4_world_state_${IDEA4_JOB_TAG}"
  else
    IDEA4_STAGE="${PROJECT}/idea4_world_state/staging/${IDEA4_JOB_TAG}"
  fi
  IDEA4_STAGE_RUN="${IDEA4_STAGE}/run"
  if [[ "${IDEA4_STAGE_RUN}" == "${IDEA4_SAVED_RUN}" || "${IDEA4_STAGE_RUN}" == "${IDEA4_SAVED_PARTIAL}" || "${IDEA4_STAGE_RUN}" == "${IDEA4_SAVED_PREVIOUS}" ]]; then
    echo "Refusing to stage a batch run on top of the saved run (${IDEA4_STAGE_RUN})." >&2
    return 2
  fi
  if [[ "${IDEA4_RESUME:-0}" == "1" ]]; then
    if [[ ! -f "${IDEA4_SAVED_RUN}/run_config.json" ]]; then
      echo "IDEA4_RESUME=1 but ${IDEA4_SAVED_RUN} has no saved run. Refusing to resume an empty directory." >&2
      return 2
    fi
    if [[ -e "${IDEA4_STAGE_RUN}" ]]; then
      echo "Staging run already exists at ${IDEA4_STAGE_RUN}. Refusing to overwrite it." >&2
      return 2
    fi
    mkdir -p "${IDEA4_STAGE}"
    local restore_tmp="${IDEA4_STAGE_RUN}.restore"
    rm -rf "${restore_tmp}"
    if ! cp -a "${IDEA4_SAVED_RUN}" "${restore_tmp}"; then
      rm -rf "${restore_tmp}"
      echo "Restore failed. Saved run left unchanged: ${IDEA4_SAVED_RUN}" >&2
      return 2
    fi
    if [[ ! -f "${restore_tmp}/run_config.json" || ! -f "${IDEA4_SAVED_RUN}/run_config.json" ]]; then
      rm -rf "${restore_tmp}"
      echo "Restore did not keep both copies of run_config.json. Saved run left unchanged: ${IDEA4_SAVED_RUN}" >&2
      return 2
    fi
    mv "${restore_tmp}" "${IDEA4_STAGE_RUN}"
    return 0
  fi
  if [[ -e "${IDEA4_SAVED_RUN}" || -e "${IDEA4_SAVED_PARTIAL}" || -e "${IDEA4_SAVED_PREVIOUS}" ]]; then
    echo "Saved run already exists at ${IDEA4_SAVED_RUN}. Set IDEA4_RESUME=1 to continue it, or choose a new IDEA4_RUN_ID. Refusing to overwrite." >&2
    return 2
  fi
  mkdir -p "$(dirname "${IDEA4_STAGE_RUN}")"
}

idea4_publish_batch_run() {
  if [[ -n "${IDEA4_COPIED:-}" ]]; then
    return 0
  fi
  if [[ -z "${PROJECT:-}" || -z "${IDEA4_RUN_ID:-}" ]]; then
    echo "PROJECT and IDEA4_RUN_ID must be set to publish a run." >&2
    return 2
  fi
  idea4_recover_saved_run
  if [[ -z "${IDEA4_STAGE_RUN:-}" || ! -d "${IDEA4_STAGE_RUN}" ]]; then
    echo "No run directory to copy (${IDEA4_STAGE_RUN:-unset})." >&2
    return 0
  fi
  if [[ ! -f "${IDEA4_STAGE_RUN}/run_config.json" ]]; then
    echo "Staging run has no run_config.json. Leaving the saved run unchanged." >&2
    return 0
  fi
  local final partial backup
  final="${IDEA4_SAVED_RUN}"
  partial="${IDEA4_SAVED_PARTIAL}"
  backup="${IDEA4_SAVED_PREVIOUS}"
  if [[ "${IDEA4_STAGE_RUN}" == "${final}" || "${IDEA4_STAGE_RUN}" == "${partial}" || "${IDEA4_STAGE_RUN}" == "${backup}" ]]; then
    echo "Refusing to publish a stage path that is the saved run, the partial copy, or the backup." >&2
    return 2
  fi
  rm -rf "${partial}"
  mkdir -p "$(dirname "${final}")"
  if ! cp -a "${IDEA4_STAGE_RUN}" "${partial}"; then
    rm -rf "${partial}"
    echo "Copy to ${partial} failed. Leaving ${final} unchanged." >&2
    return 1
  fi
  if [[ -n "${IDEA4_MANIFEST:-}" && -f "${IDEA4_MANIFEST}" ]]; then
    cp -a "${IDEA4_MANIFEST}" "${partial}/server_manifest.json"
  fi
  if [[ -n "${IDEA4_VLLM_LOG:-}" && -f "${IDEA4_VLLM_LOG}" ]]; then
    idea4_redact_stream < "${IDEA4_VLLM_LOG}" > "${partial}/vllm.log"
    chmod 600 "${partial}/vllm.log"
  fi
  if [[ ! -f "${partial}/run_config.json" ]]; then
    rm -rf "${partial}"
    echo "Partial copy is incomplete. Leaving ${final} unchanged." >&2
    return 1
  fi
  if [[ -e "${final}" ]]; then
    rm -rf "${backup}"
    if ! mv "${final}" "${backup}"; then
      rm -rf "${partial}"
      echo "Could not move ${final} aside. Leaving it in place." >&2
      return 1
    fi
  fi
  if ! mv "${partial}" "${final}"; then
    echo "Could not move ${partial} to ${final}." >&2
    if [[ ! -e "${final}" && -e "${backup}" ]]; then
      mv "${backup}" "${final}"
    fi
    return 1
  fi
  rm -rf "${backup}"
  IDEA4_COPIED=1
  echo "Run directory: ${final}"
}

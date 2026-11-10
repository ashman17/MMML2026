#!/usr/bin/env bash
# Create the PSC environment for the world-state server.
# Run this on a compute node (RM-shared is enough; a GPU is not required).
# The environment, pip cache, and Hugging Face cache live under $PROJECT.
# $HOME is only 25GB and is not used for those files.
#
# Pinned server: vLLM 0.32.0 (PyPI, 2026-10-18), plus huggingface_hub.
# vLLM 0.32.0 dropped Python 3.10 and its default wheel is CUDA 13.0.
# Confirm both against `nvidia-smi` during E1. See psc/README.md.

set -euo pipefail

VLLM_VERSION=0.32.0
IDEA4_PYTHON="${IDEA4_PYTHON:-3.12}"
IDEA4_ENV_KIND="${IDEA4_ENV_KIND:-conda}"

if [[ -z "${PROJECT:-}" || "${PROJECT}" != /* ]]; then
  echo "Set PROJECT to the absolute Bridges-2 Ocean directory (echo \"\$PROJECT\")." >&2
  exit 1
fi

ROOT="${PROJECT}/idea4_world_state"
ENV_PREFIX="${IDEA4_ENV_PREFIX:-${ROOT}/env}"
FREEZE_PATH="${IDEA4_PIP_FREEZE:-${ROOT}/pip-freeze.txt}"
ENV_FILE="${IDEA4_ENV_FILE:-${ROOT}/env.sh}"
HF_HOME="${HF_HOME:-${ROOT}/hf_home}"
PIP_CACHE_DIR="${ROOT}/pip-cache"
CONDA_PKGS_DIRS="${ROOT}/conda-pkgs"
TMPDIR="${ROOT}/tmp"

mkdir -p "${ROOT}" "${HF_HOME}/hub" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${TMPDIR}" "$(dirname "${ENV_PREFIX}")"

export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PIP_CACHE_DIR
export CONDA_PKGS_DIRS
export TMPDIR
export PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1

python_is_new_enough() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'
}

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi
  if type module >/dev/null 2>&1; then
    module load anaconda3 || true
  fi
  if ! command -v conda >/dev/null 2>&1 && [[ -f /opt/packages/anaconda3/etc/profile.d/conda.sh ]]; then
    # Path quoted in the Bridges-2 user guide. Confirm with `module avail anaconda3`.
    # shellcheck disable=SC1091
    source /opt/packages/anaconda3/etc/profile.d/conda.sh
  fi
  command -v conda >/dev/null 2>&1
}

create_conda() {
  if ! load_conda; then
    echo "conda is not available. Load anaconda3, or rerun with IDEA4_ENV_KIND=venv and Python >= 3.11." >&2
    exit 1
  fi
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create --prefix "${ENV_PREFIX}" "python=${IDEA4_PYTHON}" -y
  fi
}

create_venv() {
  local candidate=""
  local found=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_is_new_enough "${candidate}"; then
      found="${candidate}"
      break
    fi
  done
  if [[ -z "${found}" ]]; then
    echo "venv setup needs Python >= 3.11 on PATH. vLLM ${VLLM_VERSION} dropped Python 3.10." >&2
    exit 1
  fi
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    "${found}" -m venv "${ENV_PREFIX}"
  fi
}

case "${IDEA4_ENV_KIND}" in
  conda) create_conda ;;
  venv) create_venv ;;
  *)
    echo "IDEA4_ENV_KIND must be conda or venv (got ${IDEA4_ENV_KIND})." >&2
    exit 2
    ;;
esac

if ! python_is_new_enough "${ENV_PREFIX}/bin/python"; then
  echo "Environment at ${ENV_PREFIX} is older than Python 3.11. Remove that directory and rerun." >&2
  exit 1
fi

PY="${ENV_PREFIX}/bin/python"
"${PY}" -m pip install --upgrade pip
pip_cmd=("${PY}" -m pip install)
if [[ -n "${IDEA4_PIP_EXTRA_INDEX:-}" ]]; then
  pip_cmd+=(--extra-index-url "${IDEA4_PIP_EXTRA_INDEX}")
fi
pip_cmd+=("vllm==${VLLM_VERSION}" huggingface_hub)
"${pip_cmd[@]}"
"${PY}" -m pip freeze > "${FREEZE_PATH}"
"${PY}" -c 'import huggingface_hub, vllm; print("vllm", vllm.__version__)'

{
  printf 'export HF_HOME=%q\n' "${HF_HOME}"
  printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "${HF_HOME}/hub"
  printf 'export IDEA4_ENV_PREFIX=%q\n' "${ENV_PREFIX}"
  printf 'export IDEA4_PIP_FREEZE=%q\n' "${FREEZE_PATH}"
  printf 'export PIP_CACHE_DIR=%q\n' "${PIP_CACHE_DIR}"
  printf 'export TMPDIR=%q\n' "${TMPDIR}"
  printf 'export PYTHONNOUSERSITE=1\n'
  printf 'export HF_HUB_DISABLE_TELEMETRY=1\n'
  printf 'export PATH=%q:"$PATH"\n' "${ENV_PREFIX}/bin"
} > "${ENV_FILE}"

echo "Environment: ${ENV_PREFIX}"
echo "HF_HOME: ${HF_HOME}"
echo "pip freeze: ${FREEZE_PATH}"
echo "Source on later jobs: source ${ENV_FILE}"

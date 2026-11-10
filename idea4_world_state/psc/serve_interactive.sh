#!/usr/bin/env bash
# Start the Qwen3-VL server in an existing interact session and leave it in the foreground.
# Request the GPU before running this script. From a login node:
#   interact -p GPU-shared --gres=gpu:l40s-48:1 -t 02:00:00
#   interact -p GPU-shared --gres=gpu:h100-80:1 -t 02:00:00
# interact allows at most 8 hours, and idle sessions are logged out after 30 minutes.

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}/../.."
export IDEA4_REPO_ROOT="${PWD}"

# shellcheck source=serve_common.sh
source "${script_dir}/serve_common.sh"
idea4_serve_foreground

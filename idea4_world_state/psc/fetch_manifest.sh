#!/usr/bin/env bash
# Copy server_manifest.json from Bridges-2 to a local path.
# Usage: fetch_manifest.sh <remote-path> <local-path> [user]
# Pass <local-path> to: python -m idea4_world_state run --server-manifest <local-path>
#
# This script does not copy the API token. Set IDEA4_API_KEY in the local
# shell from the 0600 token file. Do not write that value into the repository.
# See "Set IDEA4_API_KEY locally" in psc/README.md.

set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: fetch_manifest.sh <remote-path> <local-path> [user]" >&2
  exit 2
fi

remote=$1
local_path=$2
user=${3:-${PSC_USER:-}}

if [[ -z "${user}" ]]; then
  echo "Pass the PSC username, or set PSC_USER. Laptop \$USER is often different." >&2
  exit 2
fi
if [[ "${remote}" != /* ]]; then
  echo "remote-path must be absolute on Bridges-2 (the MANIFEST line from the server)." >&2
  exit 2
fi
if [[ ! "${user}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "user must be the PSC username." >&2
  exit 2
fi

mkdir -p "$(dirname "${local_path}")"
scp "${user}@bridges2.psc.edu:${remote}" "${local_path}"
echo "Copied manifest to ${local_path}"
echo "Pass it as: python -m idea4_world_state run --server-manifest ${local_path}"
echo "Do not commit that file. The API key is not in it."
echo "Set IDEA4_API_KEY from the token file in a local shell. Instructions: idea4_world_state/psc/README.md"

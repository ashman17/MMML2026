#!/usr/bin/env bash
# Forward local port 8000 to the compute node through the Bridges-2 login host.
# Usage: tunnel.sh <compute-node> [user]
# The user defaults to $PSC_USER, then $USER. PSC and laptop usernames often differ.

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: tunnel.sh <compute-node> [user]" >&2
  exit 2
fi

node=$1
user=${2:-${PSC_USER:-${USER}}}

if [[ ! "${node}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "compute node must be a hostname, not a shell metacharacter." >&2
  exit 2
fi
if [[ ! "${user}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "user must be the PSC username." >&2
  exit 2
fi

echo "ssh -N -L 8000:${node}:8000 ${user}@bridges2.psc.edu"
exec ssh -N -L "8000:${node}:8000" "${user}@bridges2.psc.edu"

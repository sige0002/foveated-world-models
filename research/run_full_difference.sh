#!/usr/bin/env bash
# Historical Lightning flags. The documented command is scripts/difference-stage1.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if (( $# == 0 )); then
  exec "$root/scripts/difference-stage1" train stage1-difference-30k-20261124
fi
exec "$root/scripts/difference-stage1" --legacy-fit "$@"

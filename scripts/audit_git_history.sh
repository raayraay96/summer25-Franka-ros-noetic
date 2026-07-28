#!/usr/bin/env bash
set -euo pipefail
git fetch --prune origin '+refs/heads/*:refs/remotes/origin/*' '+refs/tags/*:refs/tags/*'
git count-objects -vH
size_pack_kib=$(git count-objects -v | awk '/^size-pack:/ {print $2}')
if [[ "${size_pack_kib:-0}" -ge 20480 ]]; then
  echo "Packed repository exceeds 20 MiB: ${size_pack_kib} KiB" >&2
  exit 1
fi
large_objects=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '$1 == "blob" && $3 > 10485760 {print}')
if [[ -n "$large_objects" ]]; then
  echo "Objects larger than 10 MiB found:" >&2
  echo "$large_objects" >&2
  exit 1
fi
forbidden_paths=$(git rev-list --objects --all | awk '{$1=""; sub(/^ /, ""); print}' | grep -E '(^|/)(models/|[^/]+\.(pt|pth|bag)$)' || true)
if [[ -n "$forbidden_paths" ]]; then
  echo "Forbidden historical model/bag paths found:" >&2
  echo "$forbidden_paths" >&2
  exit 1
fi
echo "History audit passed: no blob >10 MiB and packed objects <20 MiB."

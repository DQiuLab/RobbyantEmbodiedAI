#!/usr/bin/env bash
# 实例 → 本地：拉回 HF checkpoint、评测结果与日志（不拉 DCP）。可重复执行（rsync 断点续传）。
# 仅 AMD 路径（/workspace，初版 AMD 方案），AutoDL 未使用。
# 用法：bash scripts/local/pull_outputs.sh <user>@<host> <port> [remote_subdir]
set -euo pipefail
target="${1:?usage: pull_outputs.sh user@host port [remote_subdir]}"
port="${2:?}"
subdir="${3:-}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="${repo}/outputs/${subdir}"
mkdir -p "${dest}"

start=$(date +%s)
rsync -az --partial --info=progress2 -e "ssh -p ${port}" \
  --exclude 'checkpoints/global_step_*/*.distcp' \
  "${target}:/workspace/runtime/outputs/${subdir}" "${dest}/"
echo "pulled in $(( $(date +%s) - start )) s → ${dest}"

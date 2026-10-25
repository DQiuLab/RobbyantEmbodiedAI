#!/usr/bin/env bash
# 新实例（如 H800，只训练）从旧实例拉取训练所需的已处理数据，不重新转换，保证与已有训练用的数据逐字节一致：
#   data/lerobot/（50 个 clean 任务 LeRobot v3）、清单、clean-only 统计量（D-07）、seen 指令池（D-06）。
# 旧实例可用无卡模式开机。迁移专用密钥 /root/.ssh/id_migrate 在新实例上生成，公钥加入旧实例 authorized_keys。
# 可重复执行（rsync 增量）；结束时比对两端校验和并检查清单路径。
# 用法：bash scripts/autodl/pull_runtime.sh <old_host> <old_port>
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
host="${1:?usage: pull_runtime.sh old_host old_port}"; port="${2:?}"
key=/root/.ssh/id_migrate
[[ -f "$key" ]] || { echo "missing $key" >&2; exit 1; }
ssh_cmd="ssh -i ${key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${port}"
files=(robotwin_demo_clean_joint_v30.txt norm_stats_clean.json seen_instruction_pool.json)
log() { echo "[pull_runtime $(date +%H:%M:%S)] $*"; }

mkdir -p "${DATA_ROOT}/lerobot" "$RUNTIME"
log "lerobot data"
rsync -a --partial --info=progress2 -e "$ssh_cmd" "root@${host}:${DATA_ROOT}/lerobot/" "${DATA_ROOT}/lerobot/"
for f in "${files[@]}"; do
  rsync -a -e "$ssh_cmd" "root@${host}:${RUNTIME}/${f}" "${RUNTIME}/${f}"
done

digest='cd "$1" && find lerobot -type f | LC_ALL=C sort | xargs -d "\n" sha256sum | sha256sum | cut -c1-16'
remote=$($ssh_cmd "root@${host}" bash -s -- "${DATA_ROOT}" <<<"$digest")
local_=$(bash -s -- "${DATA_ROOT}" <<<"$digest")
[[ "$remote" == "$local_" ]] || { echo "lerobot checksum mismatch: remote ${remote} local ${local_}" >&2; exit 1; }
for f in "${files[@]}"; do
  r=$($ssh_cmd "root@${host}" "sha256sum ${RUNTIME}/${f}" | cut -c1-64)
  [[ "$r" == "$(sha256sum "${RUNTIME}/${f}" | cut -c1-64)" ]] || { echo "checksum mismatch: ${f}" >&2; exit 1; }
done
while read -r _ path; do [[ -d "$path" ]] || { echo "missing dataset ${path}" >&2; exit 1; }; done \
  < "${RUNTIME}/robotwin_demo_clean_joint_v30.txt"
log "OK lerobot ${local_}; $(wc -l < "${RUNTIME}/robotwin_demo_clean_joint_v30.txt") datasets; $(du -sh "${DATA_ROOT}/lerobot" | cut -f1)"

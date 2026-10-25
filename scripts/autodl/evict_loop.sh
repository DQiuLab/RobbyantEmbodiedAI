#!/usr/bin/env bash
# 后台循环：每 INTERVAL 秒把输出目录（DCP、HF 快照、恢复用快照）的大文件页缓存写回并丢弃（src/evict_cache.py），
# 使 DCP 保存 / 快照合并期间容器 memory.current 不被页缓存推过 120 GiB 配额（D-13；实测一次从 110 降到 87 GiB）。
# 不包括训练数据目录（其缓存有助于读数据）。
# 用法（训练实例）：setsid nohup bash scripts/autodl/evict_loop.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
INTERVAL="${INTERVAL:-15}"
log="${RUNTIME}/outputs/logs/evict_loop.log"
dirs=("${RUNTIME}/outputs" "${RUNTIME}/recover")
echo "[evict_loop $(date '+%m-%d %H:%M:%S')] start interval ${INTERVAL}s: ${dirs[*]}" >> "$log"
i=0
while true; do
  "$PY" "${REPO_ROOT}/src/evict_cache.py" "${dirs[@]}" --min-mb 64 > /dev/null 2>&1
  i=$((i + 1))
  if (( i % 40 == 0 )); then
    echo "[evict_loop $(date '+%m-%d %H:%M:%S')] cur=$(( $(cat /sys/fs/cgroup/memory.current) / 2**30 ))GiB" >> "$log"
  fi
  sleep "$INTERVAL"
done

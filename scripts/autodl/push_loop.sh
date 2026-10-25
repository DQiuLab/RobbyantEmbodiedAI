#!/usr/bin/env bash
# 训练进行中持续把新 HF 快照推到评测实例：发现 hf/global_step_N（合并完成后才从 .partial 改名，可直接认作完整）
# → push_snapshot.sh（并行 rsync + sha256）→ 校验通过后删除本机副本（释放训练实例磁盘）。
# 评测实例不可达时等待重试，不影响训练；本机快照会暂存（snapshot_steps.sh 的磁盘保护兜底）。
# 两端仓库与环境路径相同（configs/platform/autodl.env）；评测实例上的仓库路径取本机 REPO_ROOT。
# 用法（训练实例）：setsid nohup bash scripts/autodl/push_loop.sh <hf_root> <eval_host> <eval_port> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
hf_root="${1:?usage: push_loop.sh hf_root eval_host eval_port}"; host="${2:?}"; port="${3:?}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ssh_cmd=(ssh -i /root/.ssh/id_migrate -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p "$port")   # 同 push_snapshot.sh
log="${RUNTIME}/push_loop.log"
say() { echo "[push_loop $(date '+%m-%d %H:%M:%S')] $*" >> "$log"; }
say "start: ${hf_root} → ${host}:${port}"
while true; do
  for d in $(ls -d "${hf_root}"/global_step_* 2>/dev/null | grep -v '\.partial$' | sort -t_ -k3 -n); do
    [[ -f "${d}/model.safetensors.index.json" ]] || continue
    if ! "${ssh_cmd[@]}" -o ConnectTimeout=15 "root@${host}" true 2>/dev/null; then
      say "eval host unreachable; retry in 5 min"; break
    fi
    say "push $(basename "$d")"
    if bash "${here}/push_snapshot.sh" "$d" "$host" "$port" >> "$log" 2>&1; then
      rm -rf "$d" && say "deleted local $(basename "$d"); free $(df -BG --output=avail "$RUNTIME" | tail -1 | tr -d ' ')"
      # 评测机上修正快照内的绝对路径（src/relocate_snapshot.py），否则 policy server reset() 找不到 norm_stats
      dest="${RUNTIME}/outputs/snapshots/$(basename "$(dirname "$hf_root")")/$(basename "$d")"   # 同 push_snapshot.sh 默认 dest_root
      "${ssh_cmd[@]}" "root@${host}" \
        "cd '${REPO_ROOT}' && '${PY}' src/relocate_snapshot.py '$dest'" >> "$log" 2>&1 \
        || say "WARN relocate failed for ${dest}"
    else
      say "WARN push failed for $(basename "$d"); will retry"
    fi
  done
  sleep 300
done

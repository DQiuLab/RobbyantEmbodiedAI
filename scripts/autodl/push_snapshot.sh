#!/usr/bin/env bash
# 训练实例（西北B）→ 评测实例（北京B）传一个 HF 快照：每个文件一路 rsync 并行（跨区单流约 1 MB/s，
# 并行可叠加），传完两端逐文件比对 sha256。评测实例可用无卡模式开机。密钥同 pull_runtime.sh（/root/.ssh/id_migrate）。
# 用法（在训练实例上）：bash scripts/autodl/push_snapshot.sh <hf_dir> <eval_host> <eval_port> [dest_root]
#   例：bash scripts/autodl/push_snapshot.sh /root/autodl-tmp/runtime/outputs/vlmlora_run/hf/global_step_2000 \
#         <评测实例 SSH 主机> <评测实例 SSH 端口>
set -euo pipefail
src="${1:?usage: push_snapshot.sh hf_dir eval_host eval_port [dest_root]}"; host="${2:?}"; port="${3:?}"
run="$(basename "$(dirname "$(dirname "$src")")")"
dest_root="${4:-/root/autodl-tmp/runtime/outputs/snapshots/${run}}"
dest="${dest_root}/$(basename "$src")"
key=/root/.ssh/id_migrate
ssh_cmd="ssh -i ${key} -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${port}"
log() { echo "[push_snapshot $(date '+%H:%M:%S')] $*"; }

[[ -f "${src}/model.safetensors.index.json" ]] || { echo "not an HF dir: ${src}" >&2; exit 1; }
$ssh_cmd "root@${host}" "mkdir -p '${dest}'"
t0=$(date +%s); size=$(du -sb "$src" | cut -f1)
log "${src} → ${host}:${dest} ($(( size / 2**20 )) MiB, $(find "$src" -type f | wc -l) files)"
pids=()
while IFS= read -r -d '' f; do
  rel="${f#"${src}"/}"
  rsync -a --partial --relative -e "$ssh_cmd" "${src}/./${rel}" "root@${host}:${dest}/" &
  pids+=($!)
done < <(find "$src" -type f -size +100M -print0)
rsync -a --partial -e "$ssh_cmd" --exclude '*.safetensors' "${src}/" "root@${host}:${dest}/" &
pids+=($!)
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
(( fail == 0 )) || { log "FAIL: some rsync failed; re-run to resume"; exit 1; }
dt=$(( $(date +%s) - t0 ))
log "transferred in ${dt}s ($(( size / 2**20 / (dt > 0 ? dt : 1) )) MiB/s aggregate); verifying sha256"
# 链路中断时 rsync 会在接收端留下隐藏临时文件（.<name>.XXXXXX），不清理会导致校验误报（10-17 07:51）
$ssh_cmd "root@${host}" "find '${dest}' -maxdepth 2 -type f -name '.*.safetensors.*' -delete"
local_sums=$(cd "$src" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
remote_sums=$($ssh_cmd "root@${host}" "cd '${dest}' && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum")
[[ "$local_sums" == "$remote_sums" ]] || { log "FAIL: sha256 mismatch"; diff <(echo "$local_sums") <(echo "$remote_sums") | head; exit 1; }
log "OK ${dest} verified ($(echo "$local_sums" | wc -l) files)"

#!/usr/bin/env bash
# 无人值守主训练：失败自动续训（最多 MAX_ATTEMPTS 次，enable_resume 从最近 DCP 继续）；OOM 后改 micro batch 为
# OOM_FALLBACK_MB（gbs 不变，仅梯度累积变化）；在 SNAPSHOT_STEPS 的 DCP 被轮转删除前用 CPU 转 HF 快照；
# 日志超过 STALL_SECONDS 未更新视为挂死并重启。日志：${RUNTIME}/outputs/logs/<run>_supervised.log
# SHUTDOWN=1：结束（成功、STOP、重试耗尽）后执行 /usr/bin/shutdown 关机，避免云实例空转计费；默认不关机。
# 用法：TRAIN_CONFIG=configs/train/autodl_expert.yaml [SNAPSHOT_STEPS="5000 10000"] [SHUTDOWN=1] \
#       setsid nohup bash scripts/autodl/train_supervised.sh [k=v ...] > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ckpt_guard.sh"
cd "$REPO_ROOT"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
OOM_FALLBACK_MB="${OOM_FALLBACK_MB:-8}"
SNAPSHOT_STEPS="${SNAPSHOT_STEPS:-}"
STALL_SECONDS="${STALL_SECONDS:-900}"   # 训练日志这么久没更新视为挂死（如 NCCL 死锁，进程不会自己退出），杀掉后续训
MEM_ALERT_GB="${MEM_ALERT_GB:-105}"     # 容器上限 120 GiB；平台看护进程超限直接 SIGKILL（cgroup 不记 oom_kill）
sets=(); for kv in "$@"; do sets+=(--set "$kv"); done

read -r out max_steps stop_file use_lora < <("$PY" - "$TRAIN_CONFIG" "$@" <<'EOF'
import sys
from pathlib import Path
from src.launch_train import load_config, apply_overrides
cfg = apply_overrides(load_config(Path(sys.argv[1])), sys.argv[2:])
print(cfg["run"]["output_dir"], cfg["run"]["max_steps"], cfg["run"].get("stop_file") or "none", bool(cfg["train"].get("use_lora")))
EOF
)
[[ -n "${out:-}" ]] || { echo "cannot read config ${TRAIN_CONFIG}" >&2; exit 1; }
name="$(basename "$out")"
logs="${RUNTIME}/outputs/logs"; mkdir -p "$logs"
sup="${logs}/${name}_supervised.log"
log() { echo "[supervise $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$sup"; }
finish() {
  log "$1"
  [[ -n "${snap_pid:-}" ]] && kill "$snap_pid" 2>/dev/null
  [[ -n "${mem_pid:-}" ]] && kill "$mem_pid" 2>/dev/null
  if [[ "${SHUTDOWN:-0}" != "1" ]]; then log "SHUTDOWN not set, instance left running"; exit "${2:-0}"; fi
  log "shutdown (SHUTDOWN=1)"; sync; /usr/bin/shutdown; exit "${2:-0}"
}
convert_step() {  # 已转过或 DCP 未写完则跳过；checkpointer 按 model/ → optimizer/ → extra_state/ 顺序写，最后一个文件为完成标志
  local s="$1" ckpt="${out}/checkpoints/global_step_$1"
  [[ -d "${out}/hf/global_step_${s}" || ! -f "${ckpt}/extra_state/extra_state_rank_0.pt" ]] && return 0
  local tool=convert.sh; [[ "$use_lora" == "True" ]] && tool=merge.sh   # LoRA 运行须合并（merge_lora_dcp 从 DCP 取全部训练后权重）
  log "snapshot: ${tool} step ${s} → HF (CPU)"
  CUDA_VISIBLE_DEVICES="" bash "scripts/cloud/${tool}" "$out" "$s" "${out}/hf/global_step_${s}" >> "${logs}/${name}_convert.log" 2>&1 \
    && log "snapshot ${s} ok" || log "WARN snapshot ${s} failed (see ${name}_convert.log)"
}

( while true; do   # 内存采样（EXP-20261015-02：保存 DCP 时两次被 SIGKILL）
    cur=$(( $(cat /sys/fs/cgroup/memory.current) / 2**30 ))
    st=$(awk '$1=="anon"||$1=="shmem"||$1=="file"{printf "%s=%d ", $1, $2/2^30}' /sys/fs/cgroup/memory.stat)
    echo "$(date '+%m-%d %H:%M:%S') cur=${cur} ${st}" >> "${logs}/${name}_mem.log"
    (( cur > MEM_ALERT_GB )) && log "MEM ALERT container ${cur} GiB > ${MEM_ALERT_GB} (${st})"
    sleep 30; done ) &
mem_pid=$!

if [[ -n "$SNAPSHOT_STEPS" ]]; then
  ( while true; do for s in $SNAPSHOT_STEPS; do convert_step "$s"; done; sleep 120; done ) &
  snap_pid=$!
fi

extra=()
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  tlog="${logs}/${name}_attempt${attempt}_$(date +%m%d%H%M).log"   # 带时间戳：重启 supervisor 不覆盖之前的日志
  while read -r q; do log "quarantined incomplete checkpoint ${q} → checkpoints_incomplete/"; done \
    < <(quarantine_incomplete "${out}/checkpoints")
  log "attempt ${attempt}/${MAX_ATTEMPTS}: ${TRAIN_CONFIG} $* ${extra[*]} → ${tlog}"
  "$PY" -m src.launch_train --config "$TRAIN_CONFIG" "${sets[@]}" "${extra[@]}" > "$tlog" 2>&1 &
  tpid=$!
  while kill -0 "$tpid" 2>/dev/null; do
    sleep 60
    idle=$(( $(date +%s) - $(stat -c %Y "$tlog") ))
    if (( idle > STALL_SECONDS )); then
      log "STALL: ${tlog} idle ${idle}s → kill attempt ${attempt}"
      ps -eo pid,args | grep -E "[t]asks\.vla\.train_lingbotvla|[t]orch\.distributed\.run" | awk '{print $1}' | xargs -r kill -9
      break
    fi
  done
  wait "$tpid"; rc=$?
  last=$(ls -d "${out}"/checkpoints/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
  log "attempt ${attempt} rc=${rc}; latest checkpoint step ${last:-none}"
  if [[ $rc -eq 0 ]]; then
    [[ "$stop_file" != "none" && -f "$stop_file" ]] && finish "stopped by STOP file at step ${last:-?}" 0
    convert_step "${last:-$max_steps}"
    finish "training finished (step ${last:-?})" 0
  fi
  if grep -aqE "OutOfMemoryError|CUDA out of memory" "$tlog"; then
    log "OOM detected → micro batch ${OOM_FALLBACK_MB}"
    extra=(--set "run.micro_batch_size=${OOM_FALLBACK_MB}")
  fi
  sleep 30
done
finish "FAIL: ${MAX_ATTEMPTS} attempts exhausted; see ${logs}/${name}_attempt*.log" 1

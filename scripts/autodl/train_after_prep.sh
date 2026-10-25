#!/usr/bin/env bash
# 无人值守：等正在进行的数据转换结束 → finalize（清单校验、指令池合并、统计量）→ 主训练。
# 初版单卡 LoRA 方案（configs/train/autodl_lora.yaml，D-08），未用于最终权重；最终流程见 train.sh。
# SHUTDOWN=1：结束或任一步失败后执行 /usr/bin/shutdown 关机，避免云实例空转计费；默认不关机。
# 日志：runtime/outputs/logs/train_after_prep.log
# 用法：[SHUTDOWN=1] setsid nohup bash scripts/autodl/train_after_prep.sh >> /root/autodl-tmp/runtime/outputs/logs/train_after_prep.log 2>&1 < /dev/null &
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
cd "$REPO_ROOT"
log() { echo "[pipeline $(date '+%m-%d %H:%M:%S')] $*"; }
finish() {
  if [[ "${SHUTDOWN:-0}" != "1" ]]; then log "$1 (SHUTDOWN not set, instance left running)"; exit "${2:-0}"; fi
  log "$1 → shutdown"; sync; /usr/bin/shutdown; exit "${2:-0}"
}

log "waiting for running prepare_data.sh"
while ps -eo args | grep -q "^bash scripts/autodl/[p]repare_data.sh"; do sleep 60; done

log "finalize: remaining tasks + manifest + pool + norm stats"
JOBS=4 bash scripts/autodl/prepare_data.sh ALL --finalize || finish "FAIL prepare_data finalize" 1
for f in "${RUNTIME}/robotwin_demo_clean_joint_v30.txt" "${RUNTIME}/seen_instruction_pool.json" "${RUNTIME}/norm_stats_clean.json"; do
  [[ -s "$f" ]] || finish "FAIL missing $f" 1
done
df -h /root/autodl-tmp | tail -1

rm -f "${RUNTIME}/STOP"
out="$("$PY" -c "import sys; from pathlib import Path; from src.launch_train import load_config; print(load_config(Path(sys.argv[1]))['run']['output_dir'])" "$TRAIN_CONFIG")"
[[ -n "$out" ]] || finish "FAIL cannot read run.output_dir from ${TRAIN_CONFIG}" 1
name="$(basename "$out")"
log "training: $(grep -E '^\s+(max_steps|save_steps|micro_batch_size|global_batch_size):' "$TRAIN_CONFIG" | tr -s ' ' | tr '\n' ';')"
"$PY" -m src.launch_train --config "$TRAIN_CONFIG" >> "${RUNTIME}/outputs/logs/${name}.log" 2>&1
rc=$?
ls "${out}/checkpoints/" 2>/dev/null
[[ $rc -eq 0 ]] && finish "training finished (rc=0)" 0
finish "FAIL training rc=${rc}; see ${RUNTIME}/outputs/logs/${name}.log" 1

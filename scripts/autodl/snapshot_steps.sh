#!/usr/bin/env bash
# 在训练进行中，为额外的步数补 HF 快照（与 train_supervised.sh 的 SNAPSHOT_STEPS 互补，步数不要重叠）。
# LoRA 合并用 STREAM=1（低内存，训练中整模型合并曾导致训练被 SIGKILL）。只在 DCP 写完（extra_state 存在）且磁盘余量 ≥ MIN_FREE_GB 时转换：磁盘写满会让下一次 DCP 保存失败、训练崩溃。
# 用法：TRAIN_CONFIG=configs/train/autodl_vlmlora.yaml MIN_FREE_GB=55 \
#       setsid nohup bash scripts/autodl/snapshot_steps.sh "1000 2000 3000" > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
cd "$REPO_ROOT"
steps="${1:?usage: snapshot_steps.sh \"s1 s2 ...\"}"
MIN_FREE_GB="${MIN_FREE_GB:-55}"   # 一份 DCP（约 40 GB）+ 15 GB 余量
read -r out use_lora < <("$PY" -c "
import sys; from pathlib import Path; from src.launch_train import load_config
c = load_config(Path(sys.argv[1])); print(c['run']['output_dir'], bool(c['train'].get('use_lora')))" "$TRAIN_CONFIG")
name="$(basename "$out")"; logs="${RUNTIME}/outputs/logs"
tool=convert.sh; [[ "$use_lora" == "True" ]] && tool=merge.sh
log() { echo "[snapshot $(date '+%m-%d %H:%M:%S')] $*" >> "${logs}/${name}_supervised.log"; }
log "extra snapshot watcher: ${steps} (${tool}, min free ${MIN_FREE_GB} GB)"
pending="$steps"
while [[ -n "${pending// /}" ]]; do
  next=""
  for s in $pending; do
    ckpt="${out}/checkpoints/global_step_${s}"; hf="${out}/hf/global_step_${s}"
    if [[ -d "$hf" ]]; then continue; fi
    if [[ -f "${ckpt}/extra_state/extra_state_rank_0.pt" ]]; then
      free=$(df -BG --output=avail "$RUNTIME" | tail -1 | tr -dc 0-9)
      if (( free < MIN_FREE_GB )); then log "WARN disk free ${free} GB < ${MIN_FREE_GB}: skip snapshot ${s}"; continue; fi
      log "snapshot: ${tool} step ${s} → HF (CPU)"
      STREAM=1 CUDA_VISIBLE_DEVICES="" bash "scripts/cloud/${tool}" "$out" "$s" "$hf" >> "${logs}/${name}_convert.log" 2>&1 \
        && log "snapshot ${s} ok" || log "WARN snapshot ${s} failed (see ${name}_convert.log)"
      continue
    fi
    latest=$(ls -d "${out}"/checkpoints/global_step_* 2>/dev/null | sed 's/.*_//' | sort -n | tail -1)
    if [[ -n "$latest" ]] && (( latest > s )); then log "WARN snapshot ${s} missed (DCP already rotated, latest ${latest})"; continue; fi
    next="$next $s"
  done
  pending="$next"
  sleep 60
done
log "extra snapshot watcher done"

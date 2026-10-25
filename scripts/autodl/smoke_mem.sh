#!/usr/bin/env bash
# 初版单卡 LoRA 方案的显存决策点（D-08）：只用 1 个任务的数据，测 LoRA 训练在本卡上能否装下、每步多久。
# 依次尝试 fp32 权重布局的 mb2 → mb1 → fsdp1+CPU offload，第一个跑通的即为结论；非 OOM 失败也继续往下。
# 冒烟用的统计量/指令池只来自该任务的 clean 数据（仍合规），产物不用于提交。
# 前置：setup_env.sh、fetch_assets.sh、prepare_data.sh <task>（GPU 开机）。
# 用法：bash scripts/autodl/smoke_mem.sh [task=adjust_bottle] [steps=20]
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
task="${1:-adjust_bottle}"
steps="${2:-20}"
work="${RUNTIME}/smoke_mem"
logs="${RUNTIME}/outputs/logs"
report="${logs}/smoke_mem_$(date +%m%d_%H%M).report"
mkdir -p "$work" "$logs"
say() { echo "[smoke_mem] $*" | tee -a "$report"; }

lerobot_dir="${DATA_ROOT}/lerobot/${task}_joint_v30"
[[ -f "${lerobot_dir}/meta/info.json" && -f "${RUNTIME}/pool_parts/${task}.json" ]] \
  || { echo "run: bash scripts/autodl/prepare_data.sh ${task}" >&2; exit 1; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$report"
"$PY" -c "import torch; print('capability', torch.cuda.get_device_capability())" | tee -a "$report"

# 1 个任务的清单、统计量、robot config、指令池
echo "robotwin ${lerobot_dir}" > "${work}/manifest.txt"
bash "${REPO_ROOT}/scripts/cloud/norm_stats.sh" "${work}/manifest.txt" "${work}/norm.json" | tee -a "$report"
mkdir -p "${work}/robot"
"$PY" - "${REPO_ROOT}/configs/robot_autodl/robotwin.yaml" "${work}/robot/robotwin.yaml" "${work}/norm.json" <<'PY'
import sys, yaml
src, dst, norm = sys.argv[1:4]
cfg = yaml.safe_load(open(src)); cfg["norm_stats"] = norm
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
PY
cp "${RUNTIME}/pool_parts/${task}.json" "${work}/pool.json"

try() {  # try <name> <extra --set kv...>
  local name="$1"; shift
  local out="${work}/run_${name}" log="${logs}/smoke_mem_${name}.log" sets=() kv t0 rc
  for kv in "$@"; do sets+=(--set "$kv"); done
  rm -rf "$out"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "${work}/${name}.mem" &
  local mon=$!
  t0=$(date +%s)
  set +e
  (cd "$REPO_ROOT" && "$PY" -m src.launch_train --config "$TRAIN_CONFIG" \
    --set platform.data_list="${work}/manifest.txt" \
    --set data.norm_stats_file="${work}/norm.json" --set data.robot_config_root="${work}/robot" \
    --set run.seen_instruction_pool="${work}/pool.json" \
    --set run.output_dir="$out" --set run.max_steps="$steps" --set run.save_steps=100000 \
    --set run.require_resume=false "${sets[@]}") > "$log" 2>&1
  rc=$?
  set -e
  kill "$mon" 2>/dev/null || true
  local peak; peak=$(sort -n "${work}/${name}.mem" | tail -1)
  if [[ $rc -eq 0 ]]; then
    say "${name}: OK, ${steps} steps incl. startup in $(( $(date +%s) - t0 )) s, peak GPU mem ${peak} MiB"
    grep -oE "[0-9.]+ ?(s/it|it/s)" "$log" | tail -3 | tee -a "$report" || true
    grep -m 3 "\[seen_instructions\]" "$log" | tee -a "$report" || say "WARN: no [seen_instructions] lines"
    grep -m 1 "LoRA enabled" "$log" | tee -a "$report" || true
    du -sh "${DATA_ROOT}/lerobot/${task}_joint_v30" | tee -a "$report"
    return 0
  fi
  if grep -qiE "out of memory|CUDA error: out of memory|OutOfMemoryError" "$log"; then
    say "${name}: OOM (peak ${peak} MiB)"
  else
    say "${name}: FAILED rc=${rc} (not OOM) — see ${log}"; { grep -iE "error|exception" "$log" || true; } | tail -5 | tee -a "$report"
  fi
  return 1
}

# bf16 基座（enable_mixed_precision=false）已实测不可用：state_proj 处 Float/BFloat16 dtype 不匹配（2026-10-13）。
# 只测 AMD 验证的 fp32 权重 + bf16 计算布局；CPU offload 不支持梯度累积，global batch = micro batch。
try mp_mb2 run.micro_batch_size=2 run.global_batch_size=32 train.enable_mixed_precision=true \
 || try mp_mb1 run.micro_batch_size=1 run.global_batch_size=32 train.enable_mixed_precision=true \
 || try mp_mb1_offload run.micro_batch_size=1 run.global_batch_size=1 train.enable_mixed_precision=true \
      train.data_parallel_mode=fsdp1 train.enable_fsdp_offload=true \
 || { say "all variants failed → read the logs above; if all were OOM, migrate to vGPU-48GB (D-08)"; exit 1; }
say "→ 用跑通的那一档的设置写回 configs/train/autodl_lora.yaml（初版 LoRA 配置；run.micro_batch_size，必要时 train.*）"
say "report: ${report}"

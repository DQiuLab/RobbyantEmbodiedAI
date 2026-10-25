#!/usr/bin/env bash
# 【仅 AMD 多卡，AMD 路径（未实测）】全 50 任务评测（AMD run_clean_benchmark.py，--gpu-count 只接受 4/8，可 --resume）。
# 单卡平台（AutoDL）请用：bash scripts/cloud/eval_subset.sh <ckpt> <run> <config> <episodes> 0 ALL
# 用法：bash scripts/cloud/eval_full.sh <hf_ckpt_dir> <run_name> <episodes> [demo_clean|demo_randomized|both] [gpu_count]
# 注：run_clean_benchmark 通过 launch_official_server.sh 起服务，use_length 固定 25。
set -euo pipefail
model="${1:?usage: eval_full.sh hf_ckpt run_name episodes [task_config] [gpu_count]}"
run_name="${2:?}"; episodes="${3:?}"; task_config="${4:-both}"; gpus="${5:-8}"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
[[ "$PLATFORM" == "amd" ]] || { echo "eval_full.sh is AMD-only; use eval_subset.sh ... ALL" >&2; exit 2; }
[[ -f "${model}/lingbotvla_cli.yaml" ]] || { echo "missing ${model}/lingbotvla_cli.yaml" >&2; exit 1; }
grep -q '^USE_LENGTH=25$' "${REPO_ROOT}/configs/eval/infer.env" || { echo "infer.env USE_LENGTH != 25; full benchmark would not match it" >&2; exit 1; }

export LINGBOTVLA_TRAINING_CONFIG="${model}/lingbotvla_cli.yaml"
cd "${ROBOTWIN_ROOT}"
"$PY" experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count "$gpus" --episodes "$episodes" --task-config "$task_config" \
  --model-path "$model" --expert-check --accept-expert-info-on-failure --no-video \
  --robotwin-root "$ROBOTWIN_ROOT" --python "$PY" \
  --run-name "$run_name" --runtime-dir "$RUNTIME" --resume

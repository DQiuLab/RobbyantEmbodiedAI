#!/usr/bin/env bash
# 全参数 DCP → 可部署 HF checkpoint，并打包训练配置、robot config 与归一化统计量（src/bundle_ckpt.py）。
# LoRA 训练请用 scripts/cloud/merge.sh。
# 用法：[PLATFORM=...] bash scripts/cloud/convert.sh <train_output_dir> <step> [hf_out_dir]
set -euo pipefail
train_out="${1:?usage: convert.sh train_output_dir step [hf_out_dir]}"
step="${2:?}"
hf_out="${3:-${train_out}/hf/global_step_${step}}"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"

ckpt="${train_out}/checkpoints/global_step_${step}"
[[ -d "$ckpt" ]] || { echo "missing $ckpt" >&2; exit 1; }
[[ -f "${train_out}/lingbotvla_cli.yaml" ]] || { echo "missing ${train_out}/lingbotvla_cli.yaml" >&2; exit 1; }

start=$(date +%s)
cd "${ROBOTWIN_ROOT}"
"$PY" experiments/lingbot_vla_v2_6b_robotwin/scripts/convert_full_sft_dcp.py \
  --checkpoint "$ckpt" --training-output "$train_out" --output "$hf_out"
"$PY" "${REPO_ROOT}/src/bundle_ckpt.py" --train-cli "${train_out}/lingbotvla_cli.yaml" --hf-dir "$hf_out"
du -sh "$hf_out"
echo "[convert] ${ckpt} → ${hf_out} in $(( $(date +%s) - start )) s"

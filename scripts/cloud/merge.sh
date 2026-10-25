#!/usr/bin/env bash
# LoRA DCP → 合并回基座的全量 HF 权重（AMD merge_lora_dcp.py），并打包配置与统计量（src/bundle_ckpt.py）。
# rank/alpha 从训练配置读取，避免手填出错。赛事官方《常见 Q&A》（比赛页面，不随仓库分发）第 4 条允许提交 LoRA 合并权重，提交 README（docs/submission/README.md）已披露。
# 用法：[PLATFORM=autodl] [STREAM=1] bash scripts/cloud/merge.sh <train_output_dir> <step> [hf_out_dir]
set -euo pipefail
train_out="${1:?usage: merge.sh train_output_dir step [hf_out_dir]}"
step="${2:?}"
hf_out="${3:-${train_out}/merged/global_step_${step}}"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"

ckpt="${train_out}/checkpoints/global_step_${step}"
cli="${train_out}/lingbotvla_cli.yaml"
[[ -d "$ckpt" && -f "$cli" ]] || { echo "missing $ckpt or $cli" >&2; exit 1; }
read -r use_lora rank alpha < <("$PY" -c "import yaml,sys; t=yaml.safe_load(open(sys.argv[1]))['train']; print(t['use_lora'], t['lora_rank'], t['lora_alpha'])" "$cli")
[[ "$use_lora" == "True" ]] || { echo "$cli is not a LoRA run; use convert.sh" >&2; exit 1; }

start=$(date +%s)
cd "${ROBOTWIN_ROOT}"
if [[ "${STREAM:-0}" == "1" ]]; then
  # 低内存流式合并（src/stream_merge.py，与 AMD 脚本逐张量一致，EXP-20261015-02）：训练进行中使用。
  # prlimit 限制地址空间：超限时合并自己 MemoryError 退出，不会把训练进程挤到被 SIGKILL。
  CUDA_VISIBLE_DEVICES="" nice -n 19 prlimit --as=$((24 * 2**30)) "$PY" "${REPO_ROOT}/src/stream_merge.py" \
    --checkpoint "$ckpt" --base-model "${EXP}/models/robbyant_lingbot-vla-v2-6b" \
    --output "$hf_out" --rank "$rank" --alpha "$alpha" --group-gb 1 --shard-gb 2
else
  "$PY" experiments/lingbot_vla_v2_6b_robotwin/scripts/merge_lora_dcp.py \
    --checkpoint "$ckpt" --training-output "$train_out" \
    --base-model "${EXP}/models/robbyant_lingbot-vla-v2-6b" \
    --output "$hf_out" --rank "$rank" --alpha "$alpha"
fi
"$PY" "${REPO_ROOT}/src/bundle_ckpt.py" --train-cli "$cli" --hf-dir "$hf_out" --lora-merged
du -sh "$hf_out"
echo "[merge] ${ckpt} (rank=${rank}, alpha=${alpha}) → ${hf_out} in $(( $(date +%s) - start )) s"

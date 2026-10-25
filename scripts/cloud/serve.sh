#!/usr/bin/env bash
# 启动一个模型推理服务（前台）。推理参数来自 configs/eval/infer.env；路径来自 configs/platform/${PLATFORM}.env。
# 用法：[PLATFORM=autodl] [INFER_ENV=configs/eval/sweep/ul10.env] bash scripts/cloud/serve.sh <hf_ckpt_dir> <gpu_id> <port>
set -euo pipefail
model="${1:?usage: serve.sh hf_ckpt_dir gpu_id port}"
gpu="${2:?}"
port="${3:?}"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"

[[ -f "${model}/lingbotvla_cli.yaml" ]] || { echo "missing ${model}/lingbotvla_cli.yaml (run convert.sh / merge)" >&2; exit 1; }
infer_env="${INFER_ENV:-configs/eval/infer.env}"   # 推理参数扫描时指向 configs/eval/sweep/*.env（docs/specs/eval.md E6）
set -a; source "${REPO_ROOT}/${infer_env}"; set +a
echo "[serve] infer params from ${infer_env}: USE_LENGTH=${USE_LENGTH} CHUNK_RET=${CHUNK_RET} USE_BF16=${USE_BF16} USE_FP32=${USE_FP32} NUM_STEPS=${NUM_STEPS:-default}" >&2
if [[ -n "${NUM_STEPS:-}" ]]; then export LINGBOT_NUM_STEPS="$NUM_STEPS"; else unset LINGBOT_NUM_STEPS; fi   # 去噪步数（infer-num-steps 补丁）
visible_gpus "$gpu"
export QWEN3VL_PATH="${EXP}/models/Qwen3-VL-4B-Instruct-config-tokenizer"
export LINGBOTVLA_TRAINING_CONFIG="${model}/lingbotvla_cli.yaml"   # 必须是本次训练的配置，见 docs/specs/eval.md
if [[ -n "${VIT_ATTN:-}" ]]; then   # 仅用于自测（提交评测不使用）：新架构卡（如 5090 sm_120）无 flash-attn 内核时用 VIT_ATTN=sdpa；快照本身不改
  tmp_cfg="$(mktemp --suffix=.yaml)"
  "$PY" -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); c['model']['vit_attn_implementation']=sys.argv[3]; yaml.safe_dump(c, open(sys.argv[2],'w'), sort_keys=False)" \
    "${model}/lingbotvla_cli.yaml" "$tmp_cfg" "$VIT_ATTN"
  export LINGBOTVLA_TRAINING_CONFIG="$tmp_cfg"
  echo "[serve] VIT_ATTN=${VIT_ATTN} (self-test eval override, config copy ${tmp_cfg})" >&2
fi

cd "${EXP}/source/lingbot-vla-v2"
exec "$PY" -m deploy.lingbot_vla_v2_policy \
  --model_path "$model" \
  --use_length "$USE_LENGTH" \
  --chunk_ret "$CHUNK_RET" \
  --use_bf16 "$USE_BF16" \
  --use_fp32 "$USE_FP32" \
  --use_compile "$USE_COMPILE" \
  --port "$port"

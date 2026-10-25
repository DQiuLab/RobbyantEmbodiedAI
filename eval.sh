#!/usr/bin/env bash
# 复现评测（docs/specs/submission.md S4）：加载提交的 checkpoint，在 clean / randomized 下评测 50 个任务并生成 results.json。
# 推理参数：configs/eval/infer.env（单一权重，README 披露）。评测协议：RoboTwin scripts/eval_policy.sh，
#   --seed 0 --expert_check true --accept_expert_info_on_failure true（与 AMD Reproduce_Guide 一致）。
# 平台路径：configs/platform/${PLATFORM}.env（默认 autodl，即 scripts/autodl/setup_env.sh 搭出的布局）。
# CLIENTS=N：同一推理服务挂 N 个评测进程并行（仅加速，seed 与 episode 对应不变；RTX 4080 SUPER 32 GB 上限见 configs/eval/clients.env：clean 6、randomized 3，超出自动降到上限）。
# TASKS="a b"：只评测指定任务（默认 ALL = 50 个任务，冒烟或抽查用）。
# 用法：[CLIENTS=1] [TASKS=ALL] bash eval.sh <checkpoint_dir> [clean|randomized|both] [episodes=100] [gpu=0]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ckpt="$(cd "${1:?usage: eval.sh checkpoint_dir [clean|randomized|both] [episodes] [gpu]}" && pwd)"
setting="${2:-both}"; episodes="${3:-100}"; gpu="${4:-0}"
export PLATFORM="${PLATFORM:-autodl}"
source scripts/lib/platform.sh
case "$setting" in
  clean) cfgs=(clean) ;; randomized) cfgs=(randomized) ;; both) cfgs=(clean randomized) ;;
  *) echo "setting must be clean|randomized|both" >&2; exit 2 ;;
esac

"$PY" src/relocate_snapshot.py "$ckpt"          # 权重内配置的绝对路径改为本目录
tag="${EVAL_TAG:-eval_$(basename "$ckpt")}"
run_dirs=()
for cfg in "${cfgs[@]}"; do
  short="${cfg/randomized/rand}"
  # shellcheck disable=SC2086
  CLIENTS="${CLIENTS:-1}" bash scripts/cloud/eval_subset.sh "$ckpt" "${tag}_${short}" "demo_${cfg}" "$episodes" "$gpu" ${TASKS:-ALL}
  run_dirs+=(--run-dir "${RUNTIME}/outputs/${tag}_${short}")
done
partial=(); [[ "$setting" == both && "${TASKS:-ALL}" == ALL ]] || partial=(--allow-partial)
"$PY" tools/results.py "${run_dirs[@]}" --out "${RUNTIME}/outputs/${tag}_results.json" "${partial[@]}"
echo "results: ${RUNTIME}/outputs/${tag}_results.json"

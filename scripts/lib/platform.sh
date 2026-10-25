# shellcheck shell=bash
# 由 scripts/cloud/*.sh 与 scripts/autodl/*.sh source。按 PLATFORM（amd|autodl，默认 amd）加载路径与环境 profile。
# 导出：REPO_ROOT ROBOTWIN_ROOT PY LEROBOT_PY RUNTIME DATA_ROOT ENV_PROFILE TRAIN_CONFIG EXP GPU_VAR，以及 profile 中的变量。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLATFORM="${PLATFORM:-amd}"
_platform_file="${REPO_ROOT}/configs/platform/${PLATFORM}.env"
[[ -f "$_platform_file" ]] || { echo "unknown PLATFORM=${PLATFORM} (no ${_platform_file})" >&2; exit 2; }
_train_config_override="${TRAIN_CONFIG:-}"   # 允许 TRAIN_CONFIG=configs/train/xxx.yaml 覆盖平台默认
set -a
# shellcheck disable=SC1090
source "$_platform_file"
# shellcheck disable=SC1090
source "${REPO_ROOT}/${ENV_PROFILE}"
set +a
if [[ -n "$_train_config_override" ]]; then TRAIN_CONFIG="$_train_config_override"; fi
GPU_VAR="${GPU_VISIBLE_VAR}"
unset GPU_VISIBLE_VAR
EXP="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin"
export PLATFORM REPO_ROOT EXP GPU_VAR
export PATH="$(dirname "$PY"):${PATH}"   # 让 `python` 指向平台环境（venv 或 conda 均可）

# 只让指定 GPU 可见：visible_gpus 0,1
visible_gpus() {
  unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
  export "${GPU_VAR}=$1"
}

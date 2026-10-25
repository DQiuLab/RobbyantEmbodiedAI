#!/usr/bin/env bash
# 逐任务：下载 demo_clean.zip → 解压 → 构建该任务的 seen 指令池分片（只读 HDF5，D-06）→ 转 LeRobot v3 → 删除原始数据。
# 全部任务完成后（finalize）：写训练清单并校验（data_checks.py）→ 合并指令池并检查覆盖率 → clean-only 归一化统计量（D-07，需要 GPU）。
# 可重复执行：已转换的任务跳过。磁盘上同时只保留 JOBS 个任务的原始数据。
# 用法：bash scripts/autodl/prepare_data.sh <task...|ALL> [--finalize]
#       JOBS=4 bash scripts/autodl/prepare_data.sh ALL --finalize
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"

DATA_REV=a967b852afa21a9cbf19a198f7e653109042e87c
JOBS="${JOBS:-4}"
LEROBOT_DIR="${DATA_ROOT}/lerobot"
POOL_DIR="${RUNTIME}/pool_parts"
LOGS="${RUNTIME}/outputs/logs/prepare_data"
ALL_TASKS_YML="${ROBOTWIN_ROOT}/env_cfg/eval/all_tasks.yml"
mkdir -p "$LEROBOT_DIR" "$POOL_DIR" "$LOGS"

finalize=false
args=()
for a in "$@"; do [[ "$a" == "--finalize" ]] && finalize=true || args+=("$a"); done
[[ ${#args[@]} -gt 0 ]] || { echo "usage: prepare_data.sh <task...|ALL> [--finalize]" >&2; exit 2; }
if [[ "${args[0]}" == "ALL" ]]; then
  mapfile -t tasks < <("$PY" -c "import yaml,sys; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['tasks']))" "$ALL_TASKS_YML")
else
  tasks=("${args[@]}")
fi

prepare_task() {
  local task="$1" log="${LOGS}/$1.log" raw="${ROBOTWIN_ROOT}/data/demo_clean/$1" t0
  t0=$(date +%s)
  if [[ -f "${LEROBOT_DIR}/${task}_joint_v30/meta/info.json" && -f "${POOL_DIR}/${task}.json" ]]; then
    echo "[skip] ${task}"; return 0
  fi
  {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60   # 同 fetch_assets.sh
    # 下载与布局规整（RoboTwin 官方脚本；python3 = lerobot 环境，含 huggingface_hub）
    (cd "$ROBOTWIN_ROOT" && PATH="$(dirname "$LEROBOT_PY"):$PATH" \
      HF_REVISION="$DATA_REV" HF_ARCHIVE_CACHE="${DATA_ROOT}/zips/${task}" HF_KEEP_ARCHIVES=0 \
      HF_MAX_WORKERS=1 HF_EXTRACT_WORKERS=1 bash scripts/download_xpolicylab_data.sh "$task")
    # seen 指令池分片：只给构建器看这一个任务
    local view="${DATA_ROOT}/pool_view/${task}"
    rm -rf "$view"; mkdir -p "$view"; ln -s "$raw" "${view}/${task}"
    "$LEROBOT_PY" "${REPO_ROOT}/src/lingbot_patches/seen_instructions.py" build \
      --demo-clean-root "$view" --out "${POOL_DIR}/${task}.json.tmp"
    # LeRobot v3 转换（与 AMD 参考样例 Robotwin-radeon-cloud 的 convert_all_data.sh 同参数）
    rm -rf "${LEROBOT_DIR}/${task}_joint_v30"
    (cd "$ROBOTWIN_ROOT" && HF_LEROBOT_HOME="$LEROBOT_DIR" LEROBOT_VIDEO_BACKEND=pyav LEROBOT_VIDEO_CODEC=h264 \
      "$LEROBOT_PY" XPolicyLab/scripts/transform_lerobot_v30_format.py "demo_clean.${task}.aloha_agilex" \
      --repo_id "${task}_joint_v30" --max_episode 50)
  } > "$log" 2>&1 || { echo "[fail] ${task}; see ${log}" >&2; return 1; }
  if grep -q '^Failed files:' "$log" || [[ ! -f "${LEROBOT_DIR}/${task}_joint_v30/meta/info.json" ]]; then
    echo "[fail] ${task}: conversion incomplete; see ${log}" >&2; return 1
  fi
  mv "${POOL_DIR}/${task}.json.tmp" "${POOL_DIR}/${task}.json"
  rm -rf "$raw" "${DATA_ROOT}/pool_view/${task}" "${DATA_ROOT}/zips/${task}"
  echo "[done] ${task} in $(( $(date +%s) - t0 )) s, lerobot $(du -sh "${LEROBOT_DIR}/${task}_joint_v30" | cut -f1)"
}
export -f prepare_task
export ROBOTWIN_ROOT DATA_ROOT LEROBOT_DIR POOL_DIR LOGS LEROBOT_PY REPO_ROOT DATA_REV

start=$(date +%s)
printf '%s\n' "${tasks[@]}" | xargs -r -P "$JOBS" -I{} bash -c 'prepare_task "$1"' _ {} || true
failed=()
for t in "${tasks[@]}"; do
  [[ -f "${LEROBOT_DIR}/${t}_joint_v30/meta/info.json" && -f "${POOL_DIR}/${t}.json" ]] || failed+=("$t")
done
echo "[prepare] ${#tasks[@]} requested, ${#failed[@]} failed in $(( $(date +%s) - start )) s: ${failed[*]:-none}"
du -sh "$LEROBOT_DIR"; df -h /root/autodl-tmp
[[ ${#failed[@]} -eq 0 ]] || exit 1
$finalize || exit 0

# ---- finalize：全部 50 个任务 ----
raw_manifest="${RUNTIME}/manifest_raw.txt"
: > "$raw_manifest"
while read -r t; do echo "robotwin ${LEROBOT_DIR}/${t}_joint_v30" >> "$raw_manifest"; done \
  < <("$PY" -c "import yaml,sys; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['tasks']))" "$ALL_TASKS_YML")
"$PY" "${REPO_ROOT}/src/data_checks.py" --manifest "$raw_manifest" --all-tasks-yml "$ALL_TASKS_YML" \
  --out "${RUNTIME}/robotwin_demo_clean_joint_v30.txt"
"$LEROBOT_PY" "${REPO_ROOT}/src/lingbot_patches/seen_instructions.py" merge \
  --out "${RUNTIME}/seen_instruction_pool.json" "${POOL_DIR}"/*.json
sha256sum "${RUNTIME}/seen_instruction_pool.json"
"$LEROBOT_PY" "${REPO_ROOT}/src/lingbot_patches/seen_instructions.py" check \
  --pool "${RUNTIME}/seen_instruction_pool.json" --lerobot-root "$LEROBOT_DIR" --min-coverage 0.8 \
  || echo "[prepare] WARN low seen-pool coverage (D-06: if coverage is 0, train with run.seen_instruction_pool=null)" >&2
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  NORM_RATIO="${NORM_RATIO:-0.1}" bash "${REPO_ROOT}/scripts/cloud/norm_stats.sh"   # 全量约 3 h（1 任务 233 s），抽 10%（D-07）
else
  echo "[prepare] no GPU: run 'PLATFORM=autodl bash scripts/cloud/norm_stats.sh' after GPU boot (needs NCCL)"
fi
echo "[prepare] finalize done"

#!/usr/bin/env bash
# 【仅 AMD，初版 AMD 方案，未实测、未用于最终权重】在 Radeon Cloud `robotwin` 镜像实例上执行：校验环境、幂等地打补丁、报告磁盘。
# 用法：bash scripts/cloud/setup.sh            （仓库已同步到实例，见 scripts/local/push_code.sh）
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/RoboTwin}"
SOURCE_DIR="${ROBOTWIN_ROOT}/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2"
LINGBOT_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123
POOL="${POOL:-/workspace/runtime/seen_instruction_pool.json}"
LEROBOT_PY=/opt/lerobot-env/bin/python

fail() { echo "[setup] FAIL: $*" >&2; exit 1; }
ok() { echo "[setup] ok: $*"; }

# 1. 镜像内容（与 AMD external-data entrypoint 的必需路径一致）
for p in \
  /models/robotwin-persistent/assets/objects/objaverse/list.json \
  /models/robotwin-persistent/data/demo_clean \
  /models/robotwin-persistent/data/robotwin_demo_clean_joint_v30.txt \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/model.safetensors.index.json \
  /models/robotwin-persistent/models/moge-2-vitb-normal/model.pt \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/depth/model.pt \
  /models/robotwin-persistent/models/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth \
  /opt/robotwin-env/bin/python; do
  [[ -e "$p" ]] || fail "missing $p"
done
ok "image paths present"

n_tasks=$(find /models/robotwin-persistent/data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)
[[ "$n_tasks" -eq 50 ]] || fail "expected 50 demo_clean tasks, found $n_tasks"
if find /models/robotwin-persistent/data -maxdepth 1 -iname '*random*' | grep -q .; then
  echo "[setup] WARN: randomized data present under /models/robotwin-persistent/data — must never enter training (rules-checklist A)" >&2
fi
ok "50 demo_clean tasks"

# 2. 上游版本
if git -C "${ROBOTWIN_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "${ROBOTWIN_ROOT}" rev-parse HEAD)" == "${ROBOTWIN_COMMIT}" ]] || fail "RoboTwin commit mismatch"
else
  echo "[setup] WARN: ${ROBOTWIN_ROOT} is not a git checkout; cannot verify ${ROBOTWIN_COMMIT}" >&2
fi
[[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" == "${LINGBOT_COMMIT}" ]] || fail "lingbot-vla-v2 commit mismatch"
ok "upstream commits pinned"

# 3. 我们的模块与补丁（scripts/lib/our_patches.sh）
source "${REPO_ROOT}/scripts/lib/our_patches.sh"
apply_our_patches "${SOURCE_DIR}" "${REPO_ROOT}" || fail "patches"
apply_infer_patches "${SOURCE_DIR}" "${REPO_ROOT}" || fail "infer patches"
ok "modules + patches"

# 4. 训练清单：修正旧路径前缀并校验只含 50 个 demo_clean 任务（rules-checklist A）
/opt/robotwin-env/bin/python "${REPO_ROOT}/src/data_checks.py" \
  --manifest /models/robotwin-persistent/data/robotwin_demo_clean_joint_v30.txt \
  --demo-clean-root /models/robotwin-persistent/data/demo_clean \
  --out /workspace/runtime/robotwin_demo_clean_joint_v30.txt || fail "training manifest invalid"

# 5. clean-only 归一化统计量（D-07）
bash "${REPO_ROOT}/scripts/cloud/norm_stats.sh" || fail "norm stats"

# 6. seen 指令池（只读 demo_clean HDF5 的 instruction/instructions，确定性；见 D-06）
if [[ ! -f "$POOL" ]]; then
  "$LEROBOT_PY" "${REPO_ROOT}/src/lingbot_patches/seen_instructions.py" build \
    --demo-clean-root /models/robotwin-persistent/data/demo_clean --out "$POOL"
fi
sha256sum "$POOL"
"$LEROBOT_PY" "${REPO_ROOT}/src/lingbot_patches/seen_instructions.py" check \
  --pool "$POOL" --lerobot-root /models/robotwin-persistent/data/lerobot --min-coverage 0.8 \
  && ok "seen instruction pool" \
  || echo "[setup] WARN: low seen-pool coverage; see D-06 (train with run.seen_instruction_pool=null if 0)" >&2

# 7. GPU 与磁盘
/opt/robotwin-env/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "no GPU visible"
print(f"[setup] ok: torch {torch.__version__} hip={torch.version.hip} gpus={torch.cuda.device_count()}")
PY
df -h / /workspace
mkdir -p /workspace/runtime/outputs/logs
echo "[setup] done"

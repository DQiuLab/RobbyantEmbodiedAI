#!/usr/bin/env bash
# 检查容器内 SAPIEN 的 Vulkan 渲染是否可用（仿真评测的前提）。需 GPU 开机。
# 用法：bash scripts/autodl/test_render.sh
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
ls /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>/dev/null || echo "[render] no Vulkan ICD directory found"
vulkaninfo --summary 2>&1 | head -30 || true
cd "$ROBOTWIN_ROOT"
"$PY" scripts/test_render.py 2>&1 | tail -20

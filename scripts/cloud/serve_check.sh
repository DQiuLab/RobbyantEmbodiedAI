#!/usr/bin/env bash
# 部署加载检查（不需要仿真资产）：起推理服务 → 发随机观测 → 检查动作有限且为 14 维 → 关服务。
# 用于只训练的实例（如 H800）在主训练前确认 convert.sh 产物可部署；真实仿真评测仍在评测实例做。
# 用法：[PLATFORM=autodl] bash scripts/cloud/serve_check.sh <hf_ckpt_dir> [gpu_id] [port]
set -euo pipefail
model="${1:?usage: serve_check.sh hf_ckpt_dir [gpu_id] [port]}"
gpu="${2:-0}"
port="${3:-13391}"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
log="$(mktemp -d)/serve.log"

PYTHONUNBUFFERED=1 bash "${REPO_ROOT}/scripts/cloud/serve.sh" "$model" "$gpu" "$port" > "$log" 2>&1 &
server=$!
trap 'kill ${server} 2>/dev/null || true; pkill -P ${server} 2>/dev/null || true' EXIT

for _ in $(seq 1 90); do   # 加载 + 可能的编译，最多 15 分钟
  kill -0 "$server" 2>/dev/null || { echo "[serve_check] FAIL: server exited"; tail -40 "$log"; exit 1; }
  grep -q "server listening on" "$log" && (echo > "/dev/tcp/127.0.0.1/${port}") 2>/dev/null && break   # logging 走 stderr，不受 stdout 缓冲影响
  sleep 10
done
grep -q "server listening on" "$log" || { echo "[serve_check] FAIL: server not listening"; tail -40 "$log"; exit 1; }

cd "${EXP}/source/lingbot-vla-v2"
"$PY" "${REPO_ROOT}/src/serve_probe.py" "$port"
grep -iE "missing|unexpected" "$log" | head -5 || true
echo "[serve_check] PASS ${model}"

#!/usr/bin/env bash
# 本地 → 实例：同步仓库代码与配置（不含数据、上游源码、生成物）。
# 仅 AMD 路径（/workspace，初版 AMD 方案），AutoDL 未使用。
# 用法：bash scripts/local/push_code.sh <user>@<host> <port>
set -euo pipefail
target="${1:?usage: push_code.sh user@host port}"
port="${2:?usage: push_code.sh user@host port}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

rsync -az --delete -e "ssh -p ${port}" \
  --exclude .git --exclude data --exclude third_party --exclude .deps \
  --exclude submission --exclude __pycache__ --exclude .pytest_cache \
  "${repo}/" "${target}:/workspace/RobbyantEmbodiedAI/"
echo "pushed $(git -C "${repo}" rev-parse --short HEAD) to ${target}:/workspace/RobbyantEmbodiedAI"

# shellcheck shell=bash
# 多个评测客户端按任务动态领取（docs/specs/eval.md E5）。mkdir 原子：同一任务只会被一个客户端领到。
# 用法：clear_claims <dir>；claim_task <dir> <task> && 运行该任务

clear_claims() {
  rm -rf "${1:?}" && mkdir -p "$1"
}

claim_task() {
  mkdir "${1:?}/${2:?}" 2>/dev/null
}

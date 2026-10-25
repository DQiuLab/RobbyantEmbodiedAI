# shellcheck shell=bash
# DCP 完整性守卫（被 train_supervised.sh source）。
# checkpointer 按 model/ → optimizer/ → extra_state/ 顺序写，extra_state/extra_state_rank_0.pt 最后写出，作为完成标志。
# 上游续训直接取编号最大的 global_step_*：保存中途被杀会留下残缺目录，续训必然失败（EXP-20261015-02，10-16 07:10）。

# ckpt_complete <global_step_dir>：完整返回 0
ckpt_complete() { [[ -f "$1/extra_state/extra_state_rank_0.pt" && -f "$1/model/.metadata" ]]; }

# quarantine_incomplete <checkpoints_root>：把残缺的 global_step_* 移到同级 checkpoints_incomplete/（不删除），打印被移走的目录
quarantine_incomplete() {
  local root="$1" d q
  q="$(dirname "$root")/checkpoints_incomplete"
  for d in "$root"/global_step_*; do
    [[ -d "$d" ]] || continue
    ckpt_complete "$d" && continue
    mkdir -p "$q"
    mv "$d" "$q/$(basename "$d")_$(date +%m%d%H%M%S)"
    echo "$d"
  done
}

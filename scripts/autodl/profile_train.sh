#!/usr/bin/env bash
# 训练效率排查：跑一段训练，同时采集 GPU 利用率时间序列、CPU 占用；TRACE=1 时用训练代码内置的
# torch.profiler 记录第 3 个优化步（容器无 ptrace 权限，py-spy 不可用），结束后用 tools/analyze_trace.py 汇总。
# 到 N 步（或 trace 写出）后直接终止（不触发 24 GB 的结束保存）。输出：runtime/profile/<tag>/
# 用法：bash scripts/autodl/profile_train.sh <tag> <steps> [--set k=v ...]
#   例：bash scripts/autodl/profile_train.sh base 12
#       bash scripts/autodl/profile_train.sh compile 12 train.use_compile=true
set -uo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
tag="${1:?usage: profile_train.sh tag steps [k=v ...]}"; steps="${2:?}"; shift 2
P="${RUNTIME}/profile/${tag}"; rm -rf "$P"; mkdir -p "$P"
sets=(); for kv in "$@"; do sets+=(--set "$kv"); done
if [[ "${TRACE:-0}" == "1" ]]; then
  sets+=(--set train.enable_profiling=true --set train.profile_trace_dir="$P/trace")
fi
log() { echo "[profile ${tag} $(date +%H:%M:%S)] $*" | tee -a "$P/summary.txt"; }


log "start: steps=${steps} overrides: $*"
(cd "$REPO_ROOT" && "$PY" -m src.launch_train --config "$TRAIN_CONFIG" "${sets[@]}" \
   --set run.output_dir="$P/out" --set run.max_steps=100000 --set run.save_steps=100000 \
   --set run.require_resume=false --set run.stop_file=null) > "$P/train.log" 2>&1 &
launcher=$!

# 等主训练进程出现并开始出 step
pid=""
for _ in $(seq 1 120); do
  pid=$(ps -eo pid,args | awk '/[t]asks\.vla\.train_lingbotvla/ && !/torch\.distributed\.run/ {print $1; exit}')
  [[ -n "$pid" ]] && grep -aq "Step 2/" "$P/train.log" && break
  kill -0 $launcher 2>/dev/null || { log "FAIL: training exited early"; tail -30 "$P/train.log"; exit 1; }
  sleep 10
done
[[ -n "$pid" ]] || { log "FAIL: no training process"; exit 1; }
log "rank0 pid=${pid}; warmup done, sampling"

nvidia-smi dmon -s um -d 1 > "$P/dmon.txt" 2>&1 &
dmon=$!
( while kill -0 $pid 2>/dev/null; do top -b -n1 -w 200 | sed -n '8,30p' >> "$P/top.txt"; echo "----" >> "$P/top.txt"; sleep 15; done ) &
topper=$!

start_step=$(grep -aoE "Step [0-9]+/" "$P/train.log" | tail -1 | grep -oE "[0-9]+")
while true; do
  cur=$(grep -aoE "Step [0-9]+/" "$P/train.log" | tail -1 | grep -oE "[0-9]+")
  (( cur - start_step >= steps )) && break
  [[ "${TRACE:-0}" == "1" ]] && grep -aq "Profiling result saved" "$P/train.log" && { sleep 5; break; }
  kill -0 $launcher 2>/dev/null || { log "training exited"; break; }
  sleep 10
done
kill $topper $dmon 2>/dev/null
pkill -TERM -P $launcher 2>/dev/null; kill $launcher 2>/dev/null; sleep 5
ps -eo pid,args | awk '/[t]asks\.vla\.train_lingbotvla/ {print $1}' | xargs -r kill -9

# 汇总
grep -aoE "StepTime [0-9.]+s" "$P/train.log" | tail -n "$steps" | grep -oE "[0-9.]+" \
  | awk '{s+=$1; n++; if($1>m)m=$1; if(min==""||$1<min)min=$1} END{printf "StepTime mean=%.1fs min=%.1fs max=%.1fs n=%d\n", s/n, min, m, n}' | tee -a "$P/summary.txt"
awk '$1 ~ /^[0-9]+$/ {u+=$2; n++; if($2<10) idle++} END{printf "GPU util mean=%.0f%%  samples=%d  <10%%: %.0f%%\n", u/n, n, 100*idle/n}' "$P/dmon.txt" | tee -a "$P/summary.txt"
if [[ "${TRACE:-0}" == "1" ]]; then
  t=$(ls -t "$P"/trace/*.json* 2>/dev/null | head -1)
  [[ -n "$t" ]] && "$PY" "$REPO_ROOT/tools/analyze_trace.py" "$t" --top 25 | tee -a "$P/summary.txt" || log "no trace file"
fi
log "done"

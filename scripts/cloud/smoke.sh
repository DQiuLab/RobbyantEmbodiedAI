#!/usr/bin/env bash
# 冒烟测试：全链路打通 + 采集决定主训练参数所需的实测数据。全部通过才开主训练。
#   1 setup → 2 训练 10 步（主配置）→ 3 预置 STOP 文件续训：验证 resume、优雅停止、DCP 轮转
#   → 4 DCP→HF → 5 起服务 + adjust_bottle 1 ep → 6 汇总
# 用法：[PLATFORM=autodl] [SKIP_EVAL=1] bash scripts/cloud/smoke.sh [extra --set overrides for launch_train...]
#   SMOKE_TAG=smoke_MMDD_HHMM：复用已有输出目录（global_step_10 已在则跳过第 2 步）
#   SKIP_EVAL=1：只训练的实例（无仿真资产，如 H800）跳过第 5 步，HF 权重拷到评测实例再测
#   例：bash scripts/cloud/smoke.sh train.optimizer=muon            （D-03 Muon 试验）
#       bash scripts/cloud/smoke.sh train.enable_fp32=false train.use_compile=false   （D-04 回退）
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
tag="${SMOKE_TAG:-smoke_$(date +%m%d_%H%M)}"   # SMOKE_TAG=已有 tag：已有 global_step_10 时从第 3 步续
out="${RUNTIME}/outputs/${tag}"
logs="${RUNTIME}/outputs/logs"
mkdir -p "$logs"
report="${logs}/${tag}.report"
sets=()
for kv in "$@"; do sets+=(--set "$kv"); done

say() { echo "[smoke] $*" | tee -a "$report"; }

# 磁盘峰值采样（每 10 s）
( while true; do echo "$(date +%s) $(df -B1G --output=used "${RUNTIME}" | tail -1) $(df -B1G --output=used / | tail -1)"; sleep 10; done ) > "${logs}/${tag}.disk" &
disk_pid=$!
trap 'kill ${disk_pid} 2>/dev/null || true' EXIT

if [[ "$PLATFORM" == "amd" ]]; then
  say "1/6 setup"
  bash "${REPO_ROOT}/scripts/cloud/setup.sh" | tee -a "$report"
else
  say "1/6 setup skipped on ${PLATFORM} (run scripts/autodl/setup_env.sh / prepare_data.sh first)"
fi

train() {  # $1 = max_steps, $2 = log name, rest = extra --set pairs
  local t0=$(date +%s) max="$1" name="$2"; shift 2
  local extra=(); for kv in "$@"; do extra+=(--set "$kv"); done
  (cd "$REPO_ROOT" && "$PY" -m src.launch_train --config "${TRAIN_CONFIG}" "${sets[@]}" "${extra[@]}" \
      --set run.output_dir="$out" --set run.max_steps="$max" --set run.save_steps=10 --set run.dcp_keep=1) \
      > "${logs}/${tag}_${name}.log" 2>&1 || { say "FAIL: ${name} exited non-zero"; tail -40 "${logs}/${tag}_${name}.log"; exit 1; }
  say "${name}: exit 0 in $(( $(date +%s) - t0 )) s"
}

if [[ -d "${out}/checkpoints/global_step_10" && -f "${logs}/${tag}_train10.log" ]]; then
  say "2/6 skipped: ${out}/checkpoints/global_step_10 exists (SMOKE_TAG=${tag})"
else
  say "2/6 train 10 steps (overrides: $*)"
  train 10 train10
fi
grep -E "global_step|it/s|s/it|step_time|loss" "${logs}/${tag}_train10.log" | tail -5 | tee -a "$report" || true
grep -m 6 "\[seen_instructions\]" "${logs}/${tag}_train10.log" | tee -a "$report" || true
grep "\[seen_instructions\]" "${logs}/${tag}_train10.log" | awk -F"' -> '" '{a=$1; sub(/.*\] '"'"'/,"",a); b=$2; sub(/'"'"'$/,"",b); if (a!=b) n++} END{exit n>0?0:1}' \
  || say "WARN: seen instruction hook never changed a prompt (pool disabled or no coverage) — see D-06; if coverage is 0, train with run.seen_instruction_pool=null"
du -sh "${out}/checkpoints/global_step_10" | tee -a "$report"

say "3/6 resume with a pre-created STOP file: expect load@10 → 1 step → save@11 (rotating 10) → exit"
stop_file="${RUNTIME}/STOP_${tag}"
touch "$stop_file"
train 1000 resume_stop run.stop_file="$stop_file"
rm -f "$stop_file"
log="${logs}/${tag}_resume_stop.log"
grep -q "Load distributed checkpoint .* successfully" "$log" || { say "FAIL: resume did not load checkpoint"; exit 1; }
grep -q "Stop file found at step" "$log" || { say "FAIL: stop file not honoured"; exit 1; }
grep -q "Removed old DCP before save" "$log" || { say "FAIL: DCP rotation did not run"; exit 1; }
[[ ! -d "${out}/checkpoints/global_step_10" && -d "${out}/checkpoints/global_step_11" ]] || { say "FAIL: unexpected checkpoints: $(ls "${out}/checkpoints")"; exit 1; }
grep -o "Distributed checkpoint saved.*" "$log" | tail -1 | tee -a "$report"
say "resume + stop + rotation ok"

if "$PY" -c "import yaml,sys; sys.exit(0 if yaml.safe_load(open(sys.argv[1]))['train']['use_lora'] else 1)" "${out}/lingbotvla_cli.yaml"; then
  say "4/6 merge LoRA → HF"
  bash "${REPO_ROOT}/scripts/cloud/merge.sh" "$out" 11 "${out}/hf/global_step_11" | tee -a "$report"
else
  say "4/6 convert DCP → HF"
  bash "${REPO_ROOT}/scripts/cloud/convert.sh" "$out" 11 | tee -a "$report"
fi

if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  say "5/6 eval skipped (SKIP_EVAL=1): copy ${out}/hf/global_step_11 to an eval instance"
else
  say "5/6 serve + adjust_bottle 1 episode (demo_clean)"
  bash "${REPO_ROOT}/scripts/cloud/eval_subset.sh" "${out}/hf/global_step_11" "${tag}_eval" demo_clean 1 0 adjust_bottle | tee -a "$report"
  grep -q "Final success rate" "${RUNTIME}/outputs/${tag}_eval/logs/eval_demo_clean__adjust_bottle.log" || { say "FAIL: eval produced no result"; exit 1; }
fi

say "6/6 summary"
say "disk peak GiB (${RUNTIME}, /): $(sort -k2 -n "${logs}/${tag}.disk" | tail -1 | awk '{print $2", "$3}')"
df -h / "${RUNTIME}" | tee -a "$report"
say "PASS. 记录运行结果后，删除 ${out}（需人工确认）再开主训练。"

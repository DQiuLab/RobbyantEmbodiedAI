#!/usr/bin/env bash
# 单 GPU 评测：起一个服务，评测给定任务（ALL = env_cfg/eval/all_tasks.yml 的 50 个任务）。可重复执行：已有结果的任务跳过。
# 日志可直接喂给 tools/results.py。
# EVAL_VIDEO=true：录评测视频（RoboTwin 存到 eval_result/ 下的本次结果目录），只用于关键快照（D-12）。
# CLIENTS=N：同一服务挂 N 个客户端进程，按任务动态领取（docs/specs/eval.md E5）；每个任务仍是独立的 --seed 0 调用。
# ORDER_FROM="<glob>"：按历史 .out 中的单 episode 耗时把长任务排前（src/task_order.py），减少多客户端收尾空等。
# HANG_SEC=3600：任务日志这么久没更新就判定卡死，杀掉后整任务重跑，最多 HANG_RETRIES=2 次。
# ATTACH=1：不起服务，复用 PORT 上已在跑的服务；任务倒序领取，跳过任何已存在的日志（给旧版串行循环临时加客户端）。
# 用法：[PLATFORM=autodl] [PORT=13600] [CLIENTS=1] [ORDER_FROM=glob] [ATTACH=0] [EVAL_VIDEO=true] bash scripts/cloud/eval_subset.sh <hf_ckpt_dir> <run_name> <demo_clean|demo_randomized> <episodes> <gpu_id> <task...|ALL>
set -euo pipefail
model="${1:?usage: eval_subset.sh hf_ckpt run_name task_config episodes gpu_id task...|ALL}"
run_name="${2:?}"; task_config="${3:?}"; episodes="${4:?}"; gpu="${5:?}"; shift 5
[[ $# -gt 0 ]] || { echo "no tasks given" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
source "${REPO_ROOT}/scripts/lib/task_claim.sh"
run_dir="${RUNTIME}/outputs/${run_name}"
port="${PORT:-$((13500 + gpu))}"   # 同一张卡上并行多个评测时用 PORT 区分
clients="${CLIENTS:-1}"; attach="${ATTACH:-0}"
source "${REPO_ROOT}/configs/eval/clients.env"   # 显存上限（E5）：超过时截断并提示
cap="$MAX_CLIENTS_CLEAN"; [[ "$task_config" == demo_randomized ]] && cap="$MAX_CLIENTS_RANDOMIZED"
if (( clients > cap )); then echo "[eval] CLIENTS=${clients} > ${cap} for ${task_config}; using ${cap}" >&2; clients=$cap; fi
hang_sec="${HANG_SEC:-3600}"; hang_poll="${HANG_POLL:-1}"; hang_retries="${HANG_RETRIES:-2}"   # 卡死看门狗（日志无更新秒数 / 检查间隔）
mkdir -p "${run_dir}/logs"

tasks=("$@")
if [[ "${tasks[0]}" == "ALL" ]]; then
  mapfile -t tasks < <("$PY" -c "import yaml,sys; print('\n'.join(yaml.safe_load(open(sys.argv[1]))['tasks']))" "${ROBOTWIN_ROOT}/env_cfg/eval/all_tasks.yml")
fi
if [[ -n "${ORDER_FROM:-}" ]]; then
  # shellcheck disable=SC2086
  mapfile -t tasks < <("$PY" "${REPO_ROOT}/src/task_order.py" --history ${ORDER_FROM} --tasks "${tasks[@]}")
fi
if [[ "$attach" == 1 ]]; then
  mapfile -t tasks < <(printf '%s\n' "${tasks[@]}" | tac)
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT
if [[ "$attach" != 1 ]]; then
  bash "${REPO_ROOT}/scripts/cloud/serve.sh" "$model" "$gpu" "$port" >> "${run_dir}/logs/server_gpu${gpu}_${port}.log" 2>&1 &
  server_pid=$!
  pids+=("$server_pid")
  until curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1; do
    kill -0 "$server_pid" 2>/dev/null || { echo "server died; see ${run_dir}/logs/server_gpu${gpu}_${port}.log" >&2; exit 1; }
    sleep 5
  done
else
  curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null || { echo "ATTACH=1 but no server on ${port}" >&2; exit 1; }
fi

claims="${run_dir}/claims_${port}_$$"
clear_claims "$claims"
visible_gpus "$gpu"
export PYTHONFAULTHANDLER=1   # 看门狗发 SIGABRT 时打印调用栈（定位确定性卡死，10-18）
export PYTHONUNBUFFERED=1   # 任务日志实时落盘，否则按 8 KB 块缓冲，监控会把长任务误判为卡死（10-17）
cd "${ROBOTWIN_ROOT}"

# 单个任务：后台运行 eval_policy.sh（独立进程组），日志 HANG_SEC 秒无更新视为卡死（客户端阻塞在 websocket recv，
# 10-18 两个任务各卡 3–6.6 h），杀掉整组并返回 2；正常结束返回 eval_policy.sh 的退出码。
run_task() {
  local task="$1" log="$2" p
  setsid bash scripts/eval_policy.sh \
    --task_name "$task" --task_config "$task_config" \
    --policy_name LingBot-VLA-v2 --protocol lingbot_vla_v2 \
    --host 127.0.0.1 --port "$port" --device_id 0 --seed 0 \
    --test_num "$episodes" --expert_check true --accept_expert_info_on_failure true \
    --eval_batch false --additional_info eval_video_log=${EVAL_VIDEO:-false} \
    > "${log}.part" 2>&1 &
  p=$!
  while kill -0 "$p" 2>/dev/null; do
    sleep "$hang_poll"
    if (( $(date +%s) - $(stat -c %Y "${log}.part") > hang_sec )) && kill -0 "$p" 2>/dev/null; then
      # 先发 SIGABRT：PYTHONFAULTHANDLER=1 时 Python 会把所有线程的调用栈写进日志，便于定位卡在哪里
      kill -ABRT -- "-$p" 2>/dev/null; sleep 5
      kill -- "-$p" 2>/dev/null; sleep 2; kill -9 -- "-$p" 2>/dev/null
      wait "$p" 2>/dev/null
      return 2
    fi
  done
  wait "$p"
}

client_loop() {
  local task log start rc attempt
  for task in "${tasks[@]}"; do
    log="${run_dir}/logs/eval_${task_config}__${task}.log"
    if [[ "$attach" == 1 ]] && [[ -f "$log" || -f "${log}.part" ]]; then
      continue   # 另一个循环已完成或正在跑
    fi
    if [[ -f "$log" ]] && grep -q "Final success rate" "$log"; then
      continue
    fi
    claim_task "$claims" "$task" || continue
    start=$(date +%s)
    # 卡死只按"日志停止更新"判断，与成败无关；从第一个 seed 开始整任务重跑，最多 HANG_RETRIES 次（docs/specs/eval.md E5）
    for attempt in $(seq 1 $((hang_retries + 1))); do
      rc=0; run_task "$task" "$log" || rc=$?
      [[ $rc == 2 ]] || break
      mv -f "${log}.part" "${log}.hang.$(date +%s)"
      echo "[eval] HANG ${task_config}/${task} attempt ${attempt}: no log output for ${hang_sec}s; killed" >&2
    done
    if [[ $rc == 2 ]]; then
      echo "[eval] WARN ${task_config}/${task} hung $((hang_retries + 1)) times; left for a later resume" >&2
      continue
    fi
    [[ $rc == 0 ]] || echo "[eval] WARN ${task_config}/${task} exited non-zero; see ${log}" >&2
    if [[ "$attach" == 1 ]]; then
      mv -n "${log}.part" "$log"   # 旧循环可能已写了同名日志：不覆盖
    else
      mv -f "${log}.part" "$log"
    fi
    echo "[eval] ${task_config}/${task}: $(grep -o 'Final success rate:.*' "$log" | tail -1) ($(( $(date +%s) - start )) s)"
  done
}

client_pids=()
for _ in $(seq "$clients"); do
  client_loop &
  client_pids+=("$!"); pids+=("$!")
done
wait "${client_pids[@]}" || true   # 单个客户端循环被杀（如手动替换）不能让主进程在 set -e 下退出、连带关掉服务（10-19 01:40 事故）
rm -rf "$claims"
if [[ "$attach" != 1 ]]; then
  # 有 ATTACH 进来的客户端还连着本服务时，等它们跑完再退出（退出时 trap 会关服务）
  while pgrep -f "eval_policy_xpolicylab.py .*--port ${port}( |$)" >/dev/null; do sleep "$hang_poll"; done
fi

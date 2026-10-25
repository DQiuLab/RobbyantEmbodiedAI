#!/usr/bin/env bash
# 复现训练（docs/specs/submission.md S3）：50 任务 RoboTwin 2.0 Aloha-AgileX demo_clean → 最终提交权重（全局 20000 步）。
# 参考环境：AutoDL 单卡 H800 80 GB（CUDA），数据盘 /root/autodl-tmp ≥ 200 GB；路径与 configs/train/*.yaml 一致。
# 两阶段（D-13）：stage 1 按 20k 步余弦计划训练、在 7500 步保存退出并合并 LoRA；stage 2 以该合并权重为起点再训 12500 步。
# 每一步幂等，中断后重跑即续上。DRY_RUN=1 只打印命令。
# 用法：bash train.sh 2>&1 | tee -a /root/autodl-tmp/train_sh.log
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PLATFORM=autodl STREAM=1   # STREAM=1：低内存流式合并（src/stream_merge.py）；train_supervised.sh 默认不关机（SHUTDOWN=1 才关）
RUNTIME=/root/autodl-tmp/runtime
S1_HF="${RUNTIME}/outputs/vlmlora_run/hf/global_step_7500"
S2_INIT="${RUNTIME}/recover/global_step_7500"              # configs/train/autodl_vlmlora_s2.yaml 的 model.init_path
FINAL="${RUNTIME}/outputs/vlmlora_run_s2/hf/global_step_12500"

run() { echo "+ $*"; [[ "${DRY_RUN:-0}" == 1 ]] || "$@"; }

run bash scripts/autodl/setup_env.sh                       # conda 环境、固定 commit 源码、AMD 复现补丁、我们的补丁
run bash scripts/autodl/fetch_assets.sh                    # robbyant/lingbot-vla-v2-6b（固定 revision）、Qwen3-VL 配置、仿真资产
run env JOBS="${JOBS:-4}" bash scripts/autodl/prepare_data.sh ALL --finalize   # demo_clean → LeRobot v3、seen 指令池、clean-only 统计量

if [[ "${DRY_RUN:-0}" == 1 || ! -f "${S1_HF}/model.safetensors.index.json" ]]; then
  run env TRAIN_CONFIG=configs/train/autodl_vlmlora.yaml bash scripts/autodl/train_supervised.sh   # stage 1，结束时合并 7500
fi
run mkdir -p "$(dirname "$S2_INIT")"
if [[ "${DRY_RUN:-0}" == 1 ]] || [[ ! -e "$S2_INIT" && ! -L "$S2_INIT" ]]; then
  run ln -s "$S1_HF" "$S2_INIT"
fi

if [[ "${DRY_RUN:-0}" == 1 || ! -f "${FINAL}/model.safetensors.index.json" ]]; then
  run env TRAIN_CONFIG=configs/train/autodl_vlmlora_s2.yaml bash scripts/autodl/train_supervised.sh  # stage 2，结束时合并 12500
fi
echo "final checkpoint: ${FINAL}"

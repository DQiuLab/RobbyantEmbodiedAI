#!/usr/bin/env bash
# 下载固定 revision 的模型与 RoboTwin 仿真资产（默认经 hf-mirror.com；HF_ENDPOINT=https://huggingface.co 时走 AutoDL 学术加速）。可重复执行。
#   - robbyant/lingbot-vla-v2-6b：不下载 depth/、dino_video/ 两个 teacher（D-08 不用 teacher，省 2.7 GB）
#   - Qwen3-VL-4B-Instruct：只要 config/tokenizer
#   - RoboTwin2.0 assets：background_texture/embodiments/objects 三个 zip，解压后删 zip
#   - SKIP_SIM_ASSETS=1：只训练、不评测的实例（如 H800）跳过仿真资产（约 16 GB）
#   - SHARDS_FROM=modelscope：6 个 safetensors 分片改从 ModelScope 并行下载（西北B 实测 hf-mirror 0.4 MB/s、ModelScope 3.6 MB/s），
#     按 HF 固定 revision 的 LFS sha256 逐一校验；其余小文件仍从 HF_ENDPOINT 下载
# 用法：[SKIP_SIM_ASSETS=1] bash scripts/autodl/fetch_assets.sh 2>&1 | tee -a /root/autodl-tmp/fetch_assets.log
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"

MODEL_REV=11c703bf6a5c1f45b3b69168482da11fdbba53d7
QWEN_REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
DATA_REV=a967b852afa21a9cbf19a198f7e653109042e87c
HF_CLI="$(dirname "$PY")/huggingface-cli"
MODELS="${EXP}/models"
export HF_HOME="${DATA_ROOT}/hf_home"            # 缓存放数据盘（AutoDL 文档建议）
export HF_HUB_ENABLE_HF_TRANSFER=0
# 默认走 hf-mirror.com 直连（实测比学术加速访问 huggingface.co 稳定）；HF_ENDPOINT=https://huggingface.co 时改用学术加速
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60
if [[ "$HF_ENDPOINT" == "https://huggingface.co" && -f /etc/network_turbo ]]; then
  source /etc/network_turbo >/dev/null
  export no_proxy="${no_proxy:-},mirrors.aliyun.com,pypi.org,files.pythonhosted.org"; export NO_PROXY="$no_proxy"
fi
mkdir -p "$MODELS" "${DATA_ROOT}/asset_zips"

log() { echo "[fetch_assets $(date +%H:%M:%S)] $*"; }
retry() {  # 断点续传：huggingface-cli 会复用 .incomplete 文件
  local n; for n in 1 2 3 4 5; do "$@" && return 0; log "retry ${n}/5: $1 $2 $3"; sleep $((n * 20)); done; return 1
}
log "HF_ENDPOINT=${HF_ENDPOINT}"

# HF robbyant/lingbot-vla-v2-6b @ MODEL_REV 的 LFS oid（huggingface.co/api/models/.../tree/<rev>，2026-10-14 读取；
# ModelScope Robbyant/lingbot-vla-v2-6b master 的 Sha256 与之逐一相同）
SHARD_SHA256="4afb52b06a13df8b738a156ae5c8196d3bfe6b3ca931cecbd701e44cb9674e45  model-00001-of-00006.safetensors
ec131afa26a340db94c0dba8ec00e990be5f3d842ce6532070f0c8e26a067501  model-00002-of-00006.safetensors
7dccb068ca66c11fa514476d64661eeead56e2e80e1c0572c9ea82aa0d9ecf27  model-00003-of-00006.safetensors
1c2cb78066b69ae11255db851df95071e2cd69a3a4bc5020b3fd3b13b17819fb  model-00004-of-00006.safetensors
8fe36bf1f4f617869954bdfa1ad12e16abd8b0235a5fa99c88d571d4cddf4a17  model-00005-of-00006.safetensors
3cf613d592dad64b1e2a1b1bb34a6f556a7fa05eacdb9972d73e0ea4882555a0  model-00006-of-00006.safetensors"
model_dir="${MODELS}/robbyant_lingbot-vla-v2-6b"
excludes=("depth/*" "dino_video/*")
if [[ "${SHARDS_FROM:-hf}" == "modelscope" ]]; then
  mkdir -p "$model_dir"
  marker="${model_dir}/.shards_verified"
  if [[ ! -f "$marker" ]]; then
    log "safetensors shards from ModelScope (parallel)"
    pids=()
    while read -r sum f; do
      [[ -f "${model_dir}/${f}" ]] && echo "${sum}  ${model_dir}/${f}" | sha256sum -c --status && continue
      # ModelScope 用 master（无与 HF revision 对应的固定版本号），内容由下方 sha256 严格校验兜底
      ( retry curl -sSfL -C - --retry 5 -o "${model_dir}/${f}" \
          "https://www.modelscope.cn/models/Robbyant/lingbot-vla-v2-6b/resolve/master/${f}" ) &
      pids+=($!)
    done <<<"$SHARD_SHA256"
    for p in "${pids[@]}"; do wait "$p" || log "shard download pid ${p} failed"; done
    (cd "$model_dir" && sha256sum -c --strict <<<"$SHARD_SHA256") || { echo "shard sha256 mismatch" >&2; exit 1; }
    touch "$marker"
  fi
  excludes+=("model-*.safetensors")
fi
log "lingbot-vla-v2-6b @ ${MODEL_REV:0:8} (without teachers)"
retry "$HF_CLI" download robbyant/lingbot-vla-v2-6b --revision "$MODEL_REV" \
  --exclude "${excludes[@]}" --local-dir "$model_dir"
[[ "$(find "${MODELS}/robbyant_lingbot-vla-v2-6b" -maxdepth 1 -name 'model-*.safetensors' | wc -l)" -eq 6 ]] \
  || { echo "expected 6 safetensors shards" >&2; exit 1; }

log "Qwen3-VL-4B-Instruct config/tokenizer @ ${QWEN_REV:0:8}"
retry "$HF_CLI" download Qwen/Qwen3-VL-4B-Instruct --revision "$QWEN_REV" \
  --include '*.json' '*.txt' '*.jinja' --local-dir "${MODELS}/Qwen3-VL-4B-Instruct-config-tokenizer"

if [[ "${SKIP_SIM_ASSETS:-0}" == "1" ]]; then
  log "skip RoboTwin sim assets (SKIP_SIM_ASSETS=1)"
else
  for z in embodiments objects background_texture; do
    marker="${ROBOTWIN_ROOT}/assets/.${z}.done"
    [[ -f "$marker" ]] && { log "skip asset ${z}"; continue; }
    log "asset ${z}.zip"
    retry "$HF_CLI" download TianxingChen/RoboTwin2.0 "${z}.zip" --repo-type dataset --revision "$DATA_REV" \
      --local-dir "${DATA_ROOT}/asset_zips"
    "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
      "${DATA_ROOT}/asset_zips/${z}.zip" "${ROBOTWIN_ROOT}/assets"
    rm -f "${DATA_ROOT}/asset_zips/${z}.zip"
    touch "$marker"
  done
  (cd "$ROBOTWIN_ROOT" && "$PY" scripts/update_embodiment_config_path.py)
  test -f "${ROBOTWIN_ROOT}/assets/objects/objaverse/list.json"
fi
rm -rf "${HF_HOME}" "${MODELS}"/*/.cache
du -sh "${MODELS}"/* "${ROBOTWIN_ROOT}/assets" 2>/dev/null || true
df -h /root/autodl-tmp
log "done"

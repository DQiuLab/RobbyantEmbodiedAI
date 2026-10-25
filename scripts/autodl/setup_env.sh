#!/usr/bin/env bash
# AutoDL 容器实例一次性装环境（D-08）。按赛事介绍列出的 AMD 代码参考样例（Robotwin-radeon-cloud）镜像的结构复刻，GPU 栈换成 CUDA：
#   系统依赖 → conda 环境 rt（训练/推理/仿真）与 lerobot（数据转换）→ 固定 commit 的源码 → AMD 补丁 → 我们的补丁 → 校验。
# 每一步完成后写标记，可重复执行（失败后修复再跑即可续上）。无卡模式下可运行，GPU 相关校验会跳过。
# 用法：bash scripts/autodl/setup_env.sh 2>&1 | tee /root/autodl-tmp/setup_env.log
set -euo pipefail
export PLATFORM=autodl
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
source "${REPO_ROOT}/scripts/lib/our_patches.sh"

# 固定版本（与 Robotwin-radeon-cloud Reproduce_Guide 一致；AMD 仓库自身固定到本项目读过的 commit）
ROBOTWIN_COMMIT=266f3aadf505a4f7fe9af0faa41a20f5f47cd123
XPOLICYLAB_COMMIT=c37109c500be67d0dea6b36bf7337bbd26e763cd
LINGBOT_COMMIT=951475ae1b1d87553e7dc47c97b53a3d695c0d13
AMD_REPO_COMMIT=27b2f6fcb6c715e2cbb6db4431381f87c94b86ed
TORCH_SPEC=(torch==2.8.0 torchvision==0.23.0)
FLASH_ATTN_RELEASE=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3

BASE=/root/autodl-tmp
ENVS="${BASE}/envs"
SRC="${BASE}/src"
STATE="${RUNTIME}/.setup"
AMD_REPO="${SRC}/Robotwin-radeon-cloud"
LINGBOT="${EXP}/source/lingbot-vla-v2"
mkdir -p "$ENVS" "$SRC" "$STATE" "$RUNTIME/outputs/logs" "$DATA_ROOT"

log() { echo "[setup_env $(date +%H:%M:%S)] $*"; }
step() {  # step <name> <command...>
  local name="$1"; shift
  if [[ -f "${STATE}/${name}.done" ]]; then log "skip ${name}"; return 0; fi
  log "begin ${name}"; local t0=$(date +%s)
  "$@"
  touch "${STATE}/${name}.done"; log "done ${name} ($(( $(date +%s) - t0 )) s)"
}
retry() {  # retry <cmd...>：GitHub 经学术加速偶发 503 / 直连偶发超时，最多 5 次
  local n
  for n in 1 2 3 4 5; do "$@" && return 0; log "retry ${n}/5 failed: $*"; sleep $((n * 15)); done
  return 1
}
apply_once() {  # apply_once <repo> <patch>：已打过则跳过（这三个 AMD 补丁各自作用于不同仓库，可用 --reverse --check 判断）
  if git -C "$1" apply --reverse --check "$2" 2>/dev/null; then log "already applied $(basename "$2")"; else git -C "$1" apply "$2"; fi
}
turbo_on() {  # AutoDL 学术加速：github / huggingface。pip 源（阿里云）必须绕过代理，否则 "from versions: none"
  [[ -f /etc/network_turbo ]] && source /etc/network_turbo >/dev/null || true
  export no_proxy="${no_proxy:-},mirrors.aliyun.com,pypi.org,files.pythonhosted.org,download.pytorch.org"
  export NO_PROXY="$no_proxy"
}
turbo_off() { unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true; }

CONDA="$(command -v conda || echo /root/miniconda3/bin/conda)"
[[ -x "$CONDA" ]] || { echo "conda not found; choose a Miniconda base image" >&2; exit 1; }
RT_PIP=("${ENVS}/rt/bin/python" -m pip)
LR_PIP=("${ENVS}/lerobot/bin/python" -m pip)

apt_deps() {
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    git git-lfs curl wget ffmpeg libgl1 libglib2.0-0 libegl1 libvulkan1 vulkan-tools mesa-vulkan-drivers \
    build-essential ninja-build cmake pkg-config tmux rsync
}

create_envs() {
  [[ -x "${ENVS}/rt/bin/python" ]] || "$CONDA" create -y -p "${ENVS}/rt" python=3.12 pip
  [[ -x "${ENVS}/lerobot/bin/python" ]] || "$CONDA" create -y -p "${ENVS}/lerobot" python=3.12 pip
}

rt_torch() {
  "${RT_PIP[@]}" install --upgrade 'pip<26' 'setuptools<81' wheel
  # 官方源慢时退到阿里云 pytorch wheel 镜像（第三方镜像，未校验 sha256，只在下一行断言版本号）
  "${RT_PIP[@]}" install "${TORCH_SPEC[@]}" --index-url https://download.pytorch.org/whl/cu128 \
    || "${RT_PIP[@]}" install "${TORCH_SPEC[@]}" -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
  "${ENVS}/rt/bin/python" -c "import torch; assert torch.__version__.startswith('2.8.0') and torch.version.cuda.startswith('12.8'), torch.__version__"
}

amd_repo() {
  turbo_on
  [[ -d "${AMD_REPO}/.git" ]] || retry git clone https://github.com/ZiguanWang/Robotwin-radeon-cloud.git "$AMD_REPO"
  retry git -C "$AMD_REPO" fetch -q origin "$AMD_REPO_COMMIT" || true
  git -C "$AMD_REPO" checkout -q "$AMD_REPO_COMMIT"
  turbo_off
}

rt_requirements() {
  # AMD 镜像同款依赖（numpy 1.26 ABI、sapien 3.0.3、mplib 0.2.1；lerobot 等用 --no-deps 防止覆盖版本）
  local nodeps="${AMD_REPO}/docker/requirements-no-deps.txt" git_req
  "${RT_PIP[@]}" install --prefer-binary -r "${AMD_REPO}/docker/requirements.txt"
  # PyPI 部分必须成功；两个 GitHub 依赖（utils3d、pipeline）只有 MoGe 深度 teacher 用到，D-08 不用 teacher → 失败只告警
  grep -vE '^\s*(#|$)|git\+' "$nodeps" > "${SRC}/requirements-no-deps-pypi.txt"
  "${RT_PIP[@]}" install --no-deps -r "${SRC}/requirements-no-deps-pypi.txt"
  "${RT_PIP[@]}" install --prefer-binary open3d==0.19.0
  if [[ "${INSTALL_TEACHER_DEPS:-0}" != "1" ]]; then
    log "skip GitHub deps utils3d/pipeline (MoGe teacher only; set INSTALL_TEACHER_DEPS=1 for TEACHER_MODE=full)"
    return 0
  fi
  turbo_on
  while read -r git_req; do
    for attempt in 1 2 3; do
      "${RT_PIP[@]}" install --no-deps "$git_req" && break
      log "retry ${attempt}/3 failed: ${git_req}"; sleep 20
      [[ $attempt -eq 3 ]] && log "WARN: skipped ${git_req} (only needed for TEACHER_MODE=full)"
    done
  done < <(grep -E 'git\+' "$nodeps" | sed 's/[[:space:]]*#.*//')
  turbo_off
}

rt_flash_attn() {
  local abi wheel
  abi=$("${ENVS}/rt/bin/python" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  wheel="flash_attn-2.8.3+cu12torch2.8cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"
  turbo_on
  # GitHub Release 预编译 wheel：未校验 sha256，只检查文件非空
  [[ -s "${SRC}/${wheel}" ]] || retry curl -fL --retry 3 -o "${SRC}/${wheel}" "${FLASH_ATTN_RELEASE}/${wheel}"
  turbo_off
  "${RT_PIP[@]}" install --no-deps "${SRC}/${wheel}"
}

lerobot_env() {
  "${LR_PIP[@]}" install --upgrade 'pip<26' 'setuptools<81' wheel
  "${LR_PIP[@]}" install --prefer-binary 'lerobot[dataset]==0.6.0' 'h5py==3.14.0'
}

sources() {
  turbo_on
  if [[ ! -d "${ROBOTWIN_ROOT}/.git" ]]; then
    # 目录可能已被 fetch_assets 先建出 experiments/.../models（两者并行时）：clone 到旁边再把 .git 移进来，
    # 下面的 checkout 填出工作区，已有的未跟踪文件不受影响
    local tmp="${ROBOTWIN_ROOT}.clone"
    rm -rf "$tmp"
    retry git clone --no-checkout https://github.com/RoboTwin-Platform/RoboTwin.git "$tmp"
    mkdir -p "$ROBOTWIN_ROOT"
    mv "${tmp}/.git" "${ROBOTWIN_ROOT}/.git"
    rmdir "$tmp"
  fi
  git -C "$ROBOTWIN_ROOT" checkout -q "$ROBOTWIN_COMMIT"
  retry git -C "$ROBOTWIN_ROOT" submodule update --init --recursive
  [[ "$(git -C "${ROBOTWIN_ROOT}/XPolicyLab" rev-parse HEAD)" == "$XPOLICYLAB_COMMIT" ]] || { echo "XPolicyLab commit mismatch" >&2; return 1; }
  mkdir -p "${EXP}/source"
  [[ -d "${LINGBOT}/.git" ]] || retry git clone https://github.com/robbyant/lingbot-vla-v2.git "$LINGBOT"
  git -C "$LINGBOT" checkout -q "$LINGBOT_COMMIT"
  turbo_off
  # AMD 补丁：RoboTwin（MPLib 回退、expert_check、lingbot_vla_v2 协议）、XPolicyLab（LeRobot 0.6）、lingbot（LeRobot/部署兼容、LoRA）
  apply_once "$ROBOTWIN_ROOT" "${AMD_REPO}/docker/patches/robotwin-rocm-reproduction.patch"
  apply_once "${ROBOTWIN_ROOT}/XPolicyLab" "${AMD_REPO}/docker/patches/xpolicylab-lerobot-v30.patch"
  apply_once "$LINGBOT" "${AMD_REPO}/docker/patches/lingbot-vla-v2-rocm.patch"
  cp -a "${AMD_REPO}/docker/assets/experiments/lingbot_vla_v2_6b_robotwin/." "${EXP}/"
}

our_patches() { apply_our_patches "$LINGBOT" "$REPO_ROOT" && apply_infer_patches "$LINGBOT" "$REPO_ROOT"; }

editable_installs() {
  "${RT_PIP[@]}" install -e "${ROBOTWIN_ROOT}/XPolicyLab"
  "${RT_PIP[@]}" install --no-deps -e "$LINGBOT"
}

lingbot_extras() {
  # lingbot-vla-v2 tools/create_train_env.sh 额外安装、但 AMD 依赖清单没有的包（--no-deps，不动已钉的版本）
  "${RT_PIP[@]}" install --no-deps numpydantic==1.9.0 torchcodec==0.6.0 seaborn==0.13.2 ipdb==0.13.13
  # 同 rt_torch：官方源失败时退到阿里云镜像（未校验 sha256）
  "${RT_PIP[@]}" install --no-deps torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 \
    || "${RT_PIP[@]}" install --no-deps torchaudio==2.8.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
}

verify() {
  "${ENVS}/rt/bin/python" - <<'PY'
import torch, torchvision, numpy, sapien, mplib, lerobot, peft, flash_attn, numpydantic
import lingbotvla.utils.normalize, lingbotvla.data.vla_data.base_dataset  # 曾因缺 numpydantic 失败
assert numpy.__version__ == "1.26.4", numpy.__version__
assert lerobot.__version__ == "0.6.0", lerobot.__version__
print("rt:", torch.__version__, "cuda", torch.version.cuda, "flash_attn", flash_attn.__version__, "sapien", sapien.__version__)
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("GPU:", torch.cuda.get_device_name(), "capability", cap, "mem GiB", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
    from flash_attn import flash_attn_func
    q = torch.randn(1, 16, 4, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, q, q)
    print("flash_attn kernel OK on this GPU")
else:
    print("no GPU visible (无卡模式): GPU checks skipped — rerun verify with a GPU")
PY
  "${ENVS}/lerobot/bin/python" -c "import lerobot, h5py, numpy; print('lerobot env:', lerobot.__version__, 'numpy', numpy.__version__)"
}

cleanup() {
  "${RT_PIP[@]}" cache purge || true
  "${LR_PIP[@]}" cache purge || true
  "$CONDA" clean -a -y || true
}

step apt_deps apt_deps
step create_envs create_envs
step rt_torch rt_torch
step amd_repo amd_repo
step rt_requirements rt_requirements
step rt_flash_attn rt_flash_attn
step lerobot_env lerobot_env
step sources sources
step our_patches our_patches
step editable_installs editable_installs
step lingbot_extras lingbot_extras
cleanup
verify     # 不加标记：换到 GPU 开机后再跑一次本脚本即可完成 GPU 校验
df -h / /root/autodl-tmp
log "all done"

# shellcheck shell=bash
# 把 src/lingbot_patches 的模块复制进上游 lingbot-vla-v2，并按顺序打我们的挂钩补丁（须在 AMD rocm 补丁之后）。
# 幂等：打完后写入标记文件（补丁内容哈希）；哈希一致则跳过。补丁之间有相邻上下文，不能逐个用 --reverse --check 判断。
# 用法：apply_our_patches <lingbot_source_dir> <repo_root>
apply_our_patches() {
  local src="$1" repo="$2" name patch
  local names=(dcp-rotation seen-instructions run-control compile-flex fsdp-single-gpu vlm-lora-expert-full)
  local marker="${src}/.our_patches_applied"
  local digest
  digest=$(cd "${repo}/src/patches" && sha256sum $(for name in "${names[@]}"; do echo "lingbot-vla-v2-${name}.patch"; done) | sha256sum | cut -d' ' -f1)

  cp "${repo}/src/lingbot_patches/dcp_rotation.py" "${src}/lingbotvla/utils/dcp_rotation.py"
  cp "${repo}/src/lingbot_patches/run_control.py" "${src}/lingbotvla/utils/run_control.py"
  cp "${repo}/src/lingbot_patches/seen_instructions.py" "${src}/lingbotvla/data/vla_data/seen_instructions.py"

  if [[ -f "$marker" ]]; then
    [[ "$(cat "$marker")" == "$digest" ]] && { echo "[patch] already applied (${digest:0:12})"; return 0; }
    echo "[patch] marker exists but patches changed; reset the source tree (git checkout -- tasks lingbotvla) and re-apply the AMD patch first" >&2
    return 1
  fi
  for name in "${names[@]}"; do
    patch="${repo}/src/patches/lingbot-vla-v2-${name}.patch"
    git -C "$src" apply --check "$patch" || { echo "[patch] cannot apply $(basename "$patch")" >&2; return 1; }
    git -C "$src" apply "$patch" && echo "[patch] applied $(basename "$patch")"
  done
  echo "$digest" > "$marker"
}

# 推理侧补丁（只动 deploy/，不影响训练代码与上面的标记）。单独标记，可在已打训练补丁的源码树上追加。
# 用法：apply_infer_patches <lingbot_source_dir> <repo_root>
apply_infer_patches() {
  local src="$1" repo="$2" name patch
  local names=(infer-num-steps)
  local marker="${src}/.our_infer_patches_applied"
  local digest
  digest=$(cd "${repo}/src/patches" && sha256sum $(for name in "${names[@]}"; do echo "lingbot-vla-v2-${name}.patch"; done) | sha256sum | cut -d' ' -f1)
  if [[ -f "$marker" ]]; then
    [[ "$(cat "$marker")" == "$digest" ]] && { echo "[patch] infer patches already applied (${digest:0:12})"; return 0; }
    echo "[patch] infer marker exists but patches changed; git checkout -- deploy/lingbot_vla_v2_policy.py and re-apply" >&2
    return 1
  fi
  for name in "${names[@]}"; do
    patch="${repo}/src/patches/lingbot-vla-v2-${name}.patch"
    git -C "$src" apply --check "$patch" || { echo "[patch] cannot apply $(basename "$patch")" >&2; return 1; }
    git -C "$src" apply "$patch" && echo "[patch] applied $(basename "$patch")"
  done
  echo "$digest" > "$marker"
}

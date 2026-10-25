#!/usr/bin/env bash
# 仅用 clean 数据重算归一化统计量（D-07）。官方 assets/norm_stats/robotwin.json 由 clean+randomized 计算，不能使用。
# 用法：[PLATFORM=...] bash scripts/cloud/norm_stats.sh [manifest] [out]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/platform.sh"
manifest="${1:-${RUNTIME}/robotwin_demo_clean_joint_v30.txt}"
out="${2:-${RUNTIME}/norm_stats_clean.json}"
SOURCE_DIR="${EXP}/source/lingbot-vla-v2"
nproc="${NPROC:-$("$PY" -c 'import torch; print(max(1, torch.cuda.device_count()))')}"
ratio="${NORM_RATIO:-1.0}"   # <1：上游按固定种子 42 抽样（compute_norm_stats.compute_norm），结果可复现

[[ -f "$out" ]] && { echo "[norm] exists: $out"; sha256sum "$out"; exit 0; }
grep -qi random "$manifest" && { echo "[norm] randomized entry in $manifest" >&2; exit 1; }

cd "${SOURCE_DIR}"
start=$(date +%s)
"$PY" -m torch.distributed.run --standalone --nproc-per-node="$nproc" scripts/compute_norm_stats.py \
  configs/vla/norm_compute/post_data.yaml \
  --data.robot_name robotwin \
  --data.train_path "$manifest" \
  --data.norm_path "$out" \
  --data.data_ratio_for_norm_compute "$ratio"
echo "[norm] ratio=${ratio} computed in $(( $(date +%s) - start )) s"

"$PY" - "$out" assets/norm_stats/robotwin.json <<'PY'
import json, sys
ours, official = (json.load(open(p)) for p in sys.argv[1:3])
assert set(ours["norm_stats"]) == set(official["norm_stats"]), (set(ours["norm_stats"]), set(official["norm_stats"]))
for key, ref in official["norm_stats"].items():
    stats = ours["norm_stats"][key]
    assert set(ref) <= set(stats), (key, set(ref) - set(stats))   # 我们的版本另含 min/max
    for field, values in ref.items():
        assert len(stats[field]) == len(values), (key, field)
print(f"[norm] keys/shapes match official; frames: ours={ours['count']} official(clean+randomized)={official['count']}")
PY
sha256sum "$out"

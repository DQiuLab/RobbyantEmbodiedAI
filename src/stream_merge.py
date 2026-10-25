"""Low-memory LoRA DCP -> merged HF safetensors (streams tensors group by group).

Same result as ``merge_lora_dcp.py`` in the AMD reference sample Robotwin-radeon-cloud (merge formula ``W = base + alpha/rank * B @ A`` in fp32,
then cast; saved as bf16), but reads the DCP in groups of at most ``--group-gb`` instead of loading the
whole model. Needed because on the training instance the trainer holds ~84 GiB of the 120 GiB container
and the trainer was SIGKILLed during a full in-memory merge (EXP-20261015-02, 10-15 11:18; suspected memory quota).

Usage (CPU only; run with CUDA_VISIBLE_DEVICES="" and a prlimit address-space cap):
  python src/stream_merge.py --checkpoint <run>/checkpoints/global_step_N --base-model <base hf dir> \
      --output <run>/hf/global_step_N --rank 32 --alpha 64 [--group-gb 2] [--shard-gb 5]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PREFIX = "state.model."
LORA_SUFFIXES = (".base_layer.weight", ".base_layer.bias", ".lora_A.default.weight", ".lora_B.default.weight")
WEIGHT_EXTS = (".safetensors", ".bin", ".index.json")


def unit_of(name: str) -> str:
    """Module prefix that must be loaded together (a LoRA layer's base weight/bias and A/B)."""
    for suffix in LORA_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def plan_groups(sizes: dict[str, int], group_bytes: int) -> list[list[str]]:
    """Pack keys (in given order) into groups of <= group_bytes without splitting a LoRA unit."""
    units: dict[str, list[str]] = {}
    for key in sizes:
        units.setdefault(unit_of(key), []).append(key)
    groups, cur, cur_bytes = [], [], 0
    for keys in units.values():
        b = sum(sizes[k] for k in keys)
        if cur and cur_bytes + b > group_bytes:
            groups.append(cur)
            cur, cur_bytes = [], 0
        cur.extend(keys)
        cur_bytes += b
    if cur:
        groups.append(cur)
    return groups


def merge_group(tensors: dict, alpha: float, rank: int) -> dict:
    """Merge LoRA layers inside one group; names are model keys (PREFIX stripped). Returns new dict."""
    out = dict(tensors)
    for a_key in [k for k in tensors if k.endswith(".lora_A.default.weight")]:
        prefix = a_key[: -len(".lora_A.default.weight")]
        base = out.pop(prefix + ".base_layer.weight")
        a = out.pop(a_key)
        b = out.pop(prefix + ".lora_B.default.weight")
        out[prefix + ".weight"] = (base.float() + (alpha / rank) * (b.float() @ a.float())).to(base.dtype)
        if prefix + ".base_layer.bias" in out:
            out[prefix + ".bias"] = out.pop(prefix + ".base_layer.bias")
    leftovers = [k for k in out if ".lora_" in k or ".base_layer." in k]
    if leftovers:
        raise ValueError(f"unmerged LoRA keys: {leftovers[:5]}")
    return out


def main(argv: list[str] | None = None) -> int:
    import torch
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--group-gb", type=float, default=2.0)
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args(argv)

    model_dir = a.checkpoint / "model"
    if a.output.exists():
        raise FileExistsError(a.output)
    tmp = a.output.with_name(a.output.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    md = FileSystemReader(str(model_dir)).read_metadata().state_dict_metadata
    sizes = {}
    for key, m in md.items():
        if key.startswith(PREFIX) and isinstance(m, TensorStorageMetadata):
            numel = 1
            for d in m.size:
                numel *= d
            sizes[key] = numel * torch.tensor([], dtype=m.properties.dtype).element_size()

    shard, shard_bytes, shards, weight_map, total = {}, 0, [], {}, 0

    def flush():
        nonlocal shard, shard_bytes
        if shard:
            name = f"part-{len(shards):05d}.safetensors"
            save_file(shard, str(tmp / name), metadata={"format": "pt"})
            shards.append((name, list(shard)))
            shard, shard_bytes = {}, 0

    n_merged = 0
    for group in plan_groups(sizes, int(a.group_gb * 2**30)):
        sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in group}
        _load_state_dict(sd, storage_reader=FileSystemReader(str(model_dir)), planner=DefaultLoadPlanner(), no_dist=True)
        n_merged += sum(k.endswith(".lora_A.default.weight") for k in sd)
        merged = merge_group({k[len(PREFIX):]: v for k, v in sd.items()}, a.alpha, a.rank)
        del sd
        for name, t in merged.items():
            t = t.to(torch.bfloat16).contiguous()
            shard[name] = t
            shard_bytes += t.numel() * t.element_size()
            total += t.numel() * t.element_size()
            if shard_bytes >= a.shard_gb * 1e9:
                flush()
        del merged
    flush()

    n = len(shards)
    for i, (name, keys) in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{n:05d}.safetensors"
        (tmp / name).rename(tmp / final)
        weight_map.update({k: final for k in keys})
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True) + "\n")
    for asset in a.base_model.iterdir():
        if asset.is_file() and not asset.name.endswith(WEIGHT_EXTS):
            shutil.copy2(asset, tmp / asset.name)
    tmp.rename(a.output)
    print(f"Merged {n_merged} LoRA layers into {a.output} ({n} shards, {total / 1e9:.1f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

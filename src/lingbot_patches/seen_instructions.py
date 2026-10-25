"""Seen-pool instruction replacement (installed as lingbotvla/data/vla_data/seen_instructions.py).

RD 2.1 of the official challenge forum allows replacing a clean trajectory's instruction with another instruction
from the same task's seen pool. The converted LeRobot data stores only the HDF5
`instruction` field (= seen[0]) per episode. This module

1. builds a pool {first instruction -> that episode's HDF5 `instructions` list}
   from the raw demo_clean HDF5 files only (never the instructions/*.json files,
   which also contain unseen instructions), and
2. at training time swaps each sample's task text for a random member of its
   own episode's list.

Keys are the per-episode first instruction; if two episodes share a first
instruction but have different lists, that key is dropped (left vs right arm
variants would otherwise get wrong labels) and reported in `conflicts`.
Enabled by LINGBOT_SEEN_INSTRUCTION_POOL=<pool.json>; unset = upstream behaviour.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ENV_VAR = "LINGBOT_SEEN_INSTRUCTION_POOL"


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def build_pool(demo_clean_root) -> dict:
    import h5py

    root = Path(demo_clean_root)
    files = sorted(root.glob("*/aloha_agilex/data/episode_*.hdf5"))
    lists: dict[str, list[list[str]]] = {}
    for path in files:
        with h5py.File(path, "r") as f:
            first = _decode(f["instruction"][()])
            instructions = [_decode(s) for s in f["instructions"][()]]
        if first not in instructions:
            raise ValueError(f"{path}: instruction {first!r} not in its instructions list")
        lists.setdefault(first, []).append(instructions)

    pool, conflicts = {}, []
    for first in sorted(lists):
        variants = {tuple(v) for v in lists[first]}
        if len(variants) == 1:
            pool[first] = list(lists[first][0])
        else:
            conflicts.append(first)
    return {
        "version": 1,
        "episodes": len(files),
        "sources": [str(p.relative_to(root)) for p in files],
        "pool": pool,
        "conflicts": conflicts,
    }


def merge_pools(pools: list[dict]) -> dict:
    """Combine per-task pools; a key that conflicts anywhere (or has different lists) is dropped."""
    conflicts = set().union(*(p["conflicts"] for p in pools)) if pools else set()
    merged: dict[str, list[str]] = {}
    for p in pools:
        for key, lst in p["pool"].items():
            if key in merged and merged[key] != lst:
                conflicts.add(key)
            merged.setdefault(key, lst)
    return {
        "version": 1,
        "episodes": sum(p["episodes"] for p in pools),
        "sources": [s for p in pools for s in p["sources"]],
        "pool": {k: v for k, v in sorted(merged.items()) if k not in conflicts},
        "conflicts": sorted(conflicts),
    }


class SeenInstructionSampler:
    def __init__(self, pool_path, rng=None):
        data = json.loads(Path(pool_path).read_text(encoding="utf-8"))
        self.pool: dict[str, list[str]] = data["pool"]
        # Default to the module-level RNG: torch DataLoader reseeds `random` per worker,
        # whereas a Random() created here would be forked with identical state into every worker.
        self.rng = rng or random

        self.calls = 0

    def __call__(self, task: str) -> str:
        candidates = self.pool.get(task)
        new = self.rng.choice(candidates) if candidates else task
        self.calls += 1
        if self.calls <= 3:  # evidence for smoke.sh that the hook fires inside dataloader workers
            print(f"[seen_instructions] {task!r} -> {new!r}", file=sys.stderr, flush=True)
        return new


def sampler_from_env() -> SeenInstructionSampler | None:
    path = os.environ.get(ENV_VAR, "")
    return SeenInstructionSampler(path) if path else None


def coverage(pool: dict, task_texts: list[str]) -> dict:
    keys, conflicts = pool["pool"], set(pool["conflicts"])
    return {
        "tasks": len(task_texts),
        "replaceable": sum(t in keys for t in task_texts),
        "conflicts": sum(t in conflicts for t in task_texts),
        "missing": [t for t in task_texts if t not in keys and t not in conflicts],
    }


def lerobot_task_texts(lerobot_root) -> list[str]:
    """Task strings stored in LeRobot v3 meta/tasks.parquet (index = task text)."""
    import pandas as pd

    texts = []
    for path in sorted(Path(lerobot_root).glob("*/meta/tasks.parquet")):
        texts.extend(str(t) for t in pd.read_parquet(path).index)
    return texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the pool from raw demo_clean HDF5 files")
    b.add_argument("--demo-clean-root", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("merge", help="merge per-task pools built before raw data was deleted")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("pools", type=Path, nargs="+")
    c = sub.add_parser("check", help="check that training task texts are keys of the pool")
    c.add_argument("--pool", type=Path, required=True)
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--lerobot-root", type=Path)
    src.add_argument("--task-list", type=Path, help="one task text per line (tests)")
    c.add_argument("--min-coverage", type=float, default=0.8)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        pool = build_pool(args.demo_clean_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        sizes = [len(v) for v in pool["pool"].values()]
        print(f"episodes={pool['episodes']} keys={len(pool['pool'])} conflicts={len(pool['conflicts'])} "
              f"pool_size min/max={min(sizes, default=0)}/{max(sizes, default=0)}")
        return 0

    if args.cmd == "merge":
        pool = merge_pools([json.loads(p.read_text(encoding="utf-8")) for p in args.pools])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"merged {len(args.pools)} pools: episodes={pool['episodes']} keys={len(pool['pool'])} conflicts={len(pool['conflicts'])}")
        return 0

    pool = json.loads(args.pool.read_text(encoding="utf-8"))
    texts = (lerobot_task_texts(args.lerobot_root) if args.lerobot_root
             else args.task_list.read_text(encoding="utf-8").splitlines())
    report = coverage(pool, texts)
    ratio = report["replaceable"] / max(report["tasks"], 1)
    print(f"tasks={report['tasks']} replaceable={report['replaceable']} ({ratio:.1%}) "
          f"conflicts={report['conflicts']} missing={len(report['missing'])} e.g. {report['missing'][:3]}")
    return 0 if ratio >= args.min_coverage else 1


if __name__ == "__main__":
    sys.exit(main())

"""Summarise a torch.profiler chrome trace: GPU busy fraction, top kernels, top CPU ops, launch counts.

Usage: python tools/analyze_trace.py <trace.json[.gz]> [--top 20]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def merged_busy(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def gemm_precision(name: str) -> str | None:
    """Classify a GPU kernel name as a bf16 / tf32-fp32 GEMM (or another GEMM); None for non-GEMM kernels."""
    n = name.lower()
    if "gemm" not in n:
        return None
    if "bf16" in n or "f16" in n.replace("f32", ""):
        return "bf16"
    if "s1688" in n or "tf32" in n or "sgemm" in n:
        return "tf32/fp32"
    return "other-gemm"


def summarise(events: list[dict], top: int = 20) -> dict:
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu_ops = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    runtime = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
    xs = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    wall = (max(e["ts"] + e["dur"] for e in xs) - min(e["ts"] for e in xs)) if xs else 0.0
    gpu_busy = merged_busy([(e["ts"], e["ts"] + e["dur"]) for e in kernels])

    kdur, kcnt = defaultdict(float), Counter()
    for e in kernels:
        kdur[e["name"][:90]] += e["dur"]
        kcnt[e["name"][:90]] += 1
    cdur = defaultdict(float)
    for e in cpu_ops:
        cdur[e["name"][:90]] += e["dur"]
    rdur, rcnt = defaultdict(float), Counter()
    for e in runtime:
        rdur[e["name"]] += e["dur"]
        rcnt[e["name"]] += 1
    durs = sorted(e["dur"] for e in kernels)
    prec: dict[str, float] = defaultdict(float)
    for e in kernels:
        k = gemm_precision(e["name"])
        if k:
            prec[k] += e["dur"] / 1e6
    return {
        "wall_s": wall / 1e6,
        "gpu_busy_s": gpu_busy / 1e6,
        "gpu_busy_frac": gpu_busy / wall if wall else 0.0,
        "n_kernels": len(kernels),
        "median_kernel_us": durs[len(durs) // 2] if durs else 0.0,
        "kernels_under_10us_frac": sum(d < 10 for d in durs) / len(durs) if durs else 0.0,
        "gemm_precision_s": dict(prec),
        "top_kernels": sorted(((n, kdur[n] / 1e6, kcnt[n]) for n in kdur), key=lambda x: -x[1])[:top],
        "top_cpu_ops": sorted(((n, d / 1e6) for n, d in cdur.items()), key=lambda x: -x[1])[:top],
        "top_runtime": sorted(((n, rdur[n] / 1e6, rcnt[n]) for n in rdur), key=lambda x: -x[1])[:8],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(argv)
    s = summarise(load(a.trace), a.top)
    print(f"wall {s['wall_s']:.1f}s | GPU busy {s['gpu_busy_s']:.1f}s ({100*s['gpu_busy_frac']:.0f}%) | "
          f"kernels {s['n_kernels']} | median {s['median_kernel_us']:.0f}us | <10us {100*s['kernels_under_10us_frac']:.0f}%")
    print("GEMM by precision: " + ", ".join(f"{k} {v:.2f}s" for k, v in sorted(s["gemm_precision_s"].items())))
    print("\n== top GPU kernels (s, count)")
    for n, d, c in s["top_kernels"]:
        print(f"{d:8.2f} {c:8d}  {n}")
    print("\n== top CPU ops (inclusive s)")
    for n, d in s["top_cpu_ops"]:
        print(f"{d:8.2f}  {n}")
    print("\n== CUDA runtime calls (s, count)")
    for n, d, c in s["top_runtime"]:
        print(f"{d:8.2f} {c:8d}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

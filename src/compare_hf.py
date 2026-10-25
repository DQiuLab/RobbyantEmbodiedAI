"""Compare two sharded HF safetensors dirs tensor by tensor (lazy, low memory).

Exit 0 iff both have the same key set and every tensor is bit-identical (shape, dtype, values).
Usage: python src/compare_hf.py <dir_a> <dir_b>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def weight_map(d: Path) -> dict[str, str]:
    return json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]


def main(argv: list[str]) -> int:
    import torch
    from safetensors import safe_open

    a, b = Path(argv[0]), Path(argv[1])
    wa, wb = weight_map(a), weight_map(b)
    if set(wa) != set(wb):
        print(f"key sets differ: only_a={sorted(set(wa) - set(wb))[:5]} only_b={sorted(set(wb) - set(wa))[:5]}")
        return 1
    bad = 0
    by_file: dict[str, list[str]] = {}
    for key in wa:
        by_file.setdefault(wa[key], []).append(key)
    for fa, keys in sorted(by_file.items()):   # one mmap of A at a time; B opened per key
        with safe_open(str(a / fa), framework="pt") as ha:
            for key in sorted(keys):
                ta = ha.get_tensor(key)
                with safe_open(str(b / wb[key]), framework="pt") as hb:
                    tb = hb.get_tensor(key)
                if ta.shape != tb.shape or ta.dtype != tb.dtype or not torch.equal(ta, tb):
                    bad += 1
                    if bad <= 5:
                        diff = (ta.float() - tb.float()).abs().max().item() if ta.shape == tb.shape else "shape"
                        print(f"MISMATCH {key}: {tuple(ta.shape)} {ta.dtype} vs {tuple(tb.shape)} {tb.dtype} maxdiff={diff}")
    print(f"compared {len(wa)} tensors, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

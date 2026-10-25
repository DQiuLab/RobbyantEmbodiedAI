"""Order eval tasks longest-first from earlier eval_subset.sh outputs, so multi-client runs don't end on one long task.

Reads `[eval] demo_x/<task>: Final success rate: a/b = ..% (N s)` lines; per-episode time = N / b, averaged over files.
Tasks without history keep their original relative order and go after the timed ones (docs/specs/eval.md E5).

Usage: python src/task_order.py --history <out>... --tasks <task>...
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

LINE = re.compile(r"^\[eval\] demo_\w+/(\S+): Final success rate: (\d+)/(\d+) .*\((\d+) s\)\s*$")


def episode_seconds(paths: list[Path]) -> dict[str, float]:
    samples: dict[str, list[float]] = defaultdict(list)
    for p in paths:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = LINE.match(line)
            if m and int(m.group(3)) > 0:
                samples[m.group(1)].append(int(m.group(4)) / int(m.group(3)))
    return {t: sum(v) / len(v) for t, v in samples.items()}


def order(tasks: list[str], secs: dict[str, float]) -> list[str]:
    timed = sorted((t for t in tasks if t in secs), key=lambda t: -secs[t])
    return timed + [t for t in tasks if t not in secs]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", nargs="*", default=[])
    ap.add_argument("--tasks", nargs="+", required=True)
    a = ap.parse_args(argv)
    print("\n".join(order(a.tasks, episode_seconds([Path(p) for p in a.history if Path(p).is_file()]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

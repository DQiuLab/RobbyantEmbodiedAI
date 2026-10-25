"""Paired per-episode comparison of two eval runs (docs/specs/eval.md E6).

Each RoboTwin task log prints, after every episode, `Success rate: X/Y => P%, current seed: S` (with ANSI colours).
Episode outcome = whether X increased. Episodes are paired by (task, seed): same seed = same scene, so the
comparison removes scene difficulty. Discordant pairs are tested with an exact two-sided sign test.

Corrected metrics (docs/specs/eval.md E6): executed-episode success rate (with the evaluation script used here, unstable scenes that never ran
are counted in the `Final` denominator), per-task "phantom" counts, task-clustered bootstrap 95% CI of the paired
difference, and a leave-one-task-out check.

Usage: python tools/compare_runs.py <logs_dir_a> <logs_dir_b> --config demo_clean [--max-eps 10] [--tasks-file f]
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EPISODE = re.compile(r"Success rate: (\d+)/(\d+) => [0-9.]+%, current seed: (\d+)")
FINAL = re.compile(r"Final success rate: (\d+)/(\d+)")
INFRA = re.compile(r"Policy rollout error|cannot create buffer|out of memory")   # 资源错误：整个任务不参与比较
LOW_PAIRS = 5


def episodes(log: Path, max_eps: int | None = None) -> dict[int, bool]:
    """seed -> success for each finished episode in a task log (in order, truncated to max_eps)."""
    text = ANSI.sub("", log.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n"))
    out: dict[int, bool] = {}
    prev = 0
    for s, n, seed in EPISODE.findall(text):
        s, n = int(s), int(n)
        if n != len(out) + 1:
            raise ValueError(f"{log}: episode counter jumped to {n} after {len(out)} episodes")
        out[int(seed)] = s > prev
        prev = s
        if max_eps and len(out) >= max_eps:
            break
    return out


def reported_attempts(log: Path) -> int | None:
    """Denominator of the last `Final success rate` line (includes episodes that never ran), or None if unfinished."""
    m = FINAL.findall(log.read_text(encoding="utf-8", errors="ignore"))
    return int(m[-1][1]) if m else None


def paired_diff(rows: list[dict]) -> float:
    pairs = sum(r["pairs"] for r in rows)
    return sum(r["b"] - r["a"] for r in rows) / pairs if pairs else 0.0


def bootstrap_ci(rows: list[dict], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Task-clustered bootstrap 95% CI of the paired difference (resample tasks with replacement)."""
    rows = [r for r in rows if r["pairs"]]
    if not rows:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs = sorted(paired_diff([rng.choice(rows) for _ in rows]) for _ in range(n))
    return diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]


def leave_one_out(rows: list[dict]) -> tuple[str, float]:
    """Drop the task contributing most to the sign of the difference; return (task, diff without it)."""
    rows = [r for r in rows if r["pairs"]]
    if not rows:
        return ("", 0.0)
    d = paired_diff(rows)
    key = (lambda r: r["b"] - r["a"]) if d >= 0 else (lambda r: r["a"] - r["b"])
    top = max(rows, key=key)
    return top["task"], paired_diff([r for r in rows if r is not top])


def sign_test_p(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def find_log(logs: Path, config: str, task: str) -> Path | None:
    for name in (f"eval_{config}__{task}.log", f"eval_{config}__{task}.log.part"):
        if (logs / name).is_file():
            return logs / name
    return None


def compare(a_dir: Path, b_dir: Path, config: str, tasks: list[str], max_eps: int | None) -> dict:
    rows = []
    tot = {"pairs": 0, "a": 0, "b": 0, "a_only": 0, "b_only": 0,
           "exec_a": 0, "succ_a": 0, "exec_b": 0, "succ_b": 0, "phantom_a": 0, "phantom_b": 0}
    for t in tasks:
        la, lb = find_log(a_dir, config, t), find_log(b_dir, config, t)
        if not la or not lb:
            rows.append({"task": t, "missing": True})
            continue
        if any(INFRA.search(l.read_text(encoding="utf-8", errors="ignore")) for l in (la, lb)):
            rows.append({"task": t, "missing": True, "infra": True})
            continue
        ea, eb = episodes(la, max_eps), episodes(lb, max_eps)
        common = sorted(set(ea) & set(eb))
        ra, rb = reported_attempts(la), reported_attempts(lb)
        r = {"task": t, "pairs": len(common), "a": sum(ea[s] for s in common), "b": sum(eb[s] for s in common),
             "a_only": sum(ea[s] and not eb[s] for s in common), "b_only": sum(eb[s] and not ea[s] for s in common),
             "exec_a": len(ea), "succ_a": sum(ea.values()), "exec_b": len(eb), "succ_b": sum(eb.values()),
             # 记录分母 − 实跑数（只在未截断时有意义）
             "phantom_a": max(0, (ra or 0) - len(ea)) if not max_eps or (ra or 0) <= max_eps else 0,
             "phantom_b": max(0, (rb or 0) - len(eb)) if not max_eps or (rb or 0) <= max_eps else 0}
        r["low"] = r["pairs"] < LOW_PAIRS
        rows.append(r)
        for k in tot:
            tot[k] += r[k]
    good = [r for r in rows if not r.get("missing")]
    tot["p"] = sign_test_p(tot["a_only"], tot["b_only"])
    tot["diff"] = paired_diff(good)
    tot["ci"] = bootstrap_ci(good)
    tot["loo"] = leave_one_out(good)
    return {"rows": rows, "total": tot}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-eps", type=int, default=None)
    ap.add_argument("--tasks-file", default="configs/eval/sweep_tasks.txt")
    x = ap.parse_args(argv)
    tasks = [l.strip() for l in Path(x.tasks_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    res = compare(Path(x.a), Path(x.b), x.config, tasks, x.max_eps)
    print(f"{'task':28s} pairs   A   B  A-only B-only  execA execB phantomA phantomB")
    for r in res["rows"]:
        if r.get("missing"):
            print(f"{r['task']:28s} ({'excluded: infra error' if r.get('infra') else 'missing'})")
            continue
        flag = "  LOW-N" if r["low"] else ""
        print(f"{r['task']:28s} {r['pairs']:5d} {r['a']:3d} {r['b']:3d} {r['a_only']:7d} {r['b_only']:6d}"
              f"  {r['exec_a']:5d} {r['exec_b']:5d} {r['phantom_a']:8d} {r['phantom_b']:8d}{flag}")
    t = res["total"]
    if t["pairs"]:
        lo, hi = t["ci"]
        print(f"EXECUTED  A={t['succ_a']}/{t['exec_a']} ({t['succ_a']/max(1,t['exec_a']):.1%})  "
              f"B={t['succ_b']}/{t['exec_b']} ({t['succ_b']/max(1,t['exec_b']):.1%})  "
              f"phantom A={t['phantom_a']} B={t['phantom_b']}")
        print(f"PAIRED    n={t['pairs']}  A={t['a']} ({t['a']/t['pairs']:.1%})  B={t['b']} ({t['b']/t['pairs']:.1%})  "
              f"diff={t['diff']:+.1%}  95%CI(task bootstrap)=[{lo:+.1%}, {hi:+.1%}]  "
              f"leave-out {t['loo'][0]}: {t['loo'][1]:+.1%}  sign-test p={t['p']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

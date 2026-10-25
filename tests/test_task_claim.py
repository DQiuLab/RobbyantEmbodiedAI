import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "scripts/lib/task_claim.sh"


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {LIB}; {cmd}"], capture_output=True, text=True)


def test_each_task_claimed_exactly_once_under_concurrency(tmp_path):
    d = tmp_path / "claims"
    tasks = " ".join(f"t{i}" for i in range(40))
    # 8 个并发客户端争抢 40 个任务
    worker = f'for t in {tasks}; do if claim_task {d} $t; then echo $t; fi; done'
    r = run(f"clear_claims {d}; for k in $(seq 8); do ( {worker} ) & done; wait")
    assert r.returncode == 0, r.stderr
    got = r.stdout.split()
    assert sorted(got) == sorted(tasks.split())   # 无遗漏、无重复


def test_clear_claims_resets_stale_locks(tmp_path):
    d = tmp_path / "claims"
    assert run(f"clear_claims {d}; claim_task {d} a").returncode == 0
    assert run(f"claim_task {d} a").returncode != 0
    assert run(f"clear_claims {d}; claim_task {d} a").returncode == 0

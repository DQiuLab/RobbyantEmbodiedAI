import subprocess
import sys
from pathlib import Path

from src.task_order import episode_seconds, order

SCRIPT = Path(__file__).resolve().parents[1] / "src/task_order.py"


def write(p: Path, lines: list[str]) -> Path:
    p.write_text("\n".join(lines) + "\n")
    return p


def test_episode_seconds_averages_per_episode_across_files(tmp_path):
    a = write(tmp_path / "a.out", [
        "[eval] demo_clean/put_bottles_dustbin: Final success rate: 0/1 = 0.0% (1200 s)",
        "[eval] demo_clean/click_bell: Final success rate: 2/2 = 100.0% (100 s)",
        "[eval] WARN demo_clean/x exited non-zero; see log",
        "[eval] skip demo_clean/y (done)",
    ])
    b = write(tmp_path / "b.out", ["[eval] demo_randomized/put_bottles_dustbin: Final success rate: 1/1 = 100.0% (800 s)"])
    s = episode_seconds([a, b])
    assert s == {"put_bottles_dustbin": 1000.0, "click_bell": 50.0}


def test_order_longest_first_unknown_last_in_original_order():
    secs = {"b": 10.0, "d": 300.0, "a": 50.0}
    assert order(["a", "b", "c", "d", "e"], secs) == ["d", "a", "b", "c", "e"]


def test_cli_without_history_keeps_order(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--history", str(tmp_path / "missing.out"), "--tasks", "x", "y"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split() == ["x", "y"]

import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "scripts/lib/ckpt_guard.sh"


def make_ckpt(root: Path, step: int, complete: bool) -> Path:
    d = root / f"global_step_{step}"
    (d / "model").mkdir(parents=True)
    (d / "model" / "__0_0.distcp").write_bytes(b"x")
    if complete:
        (d / "model" / ".metadata").write_bytes(b"m")
        (d / "optimizer").mkdir()
        (d / "extra_state").mkdir()
        (d / "extra_state" / "extra_state_rank_0.pt").write_bytes(b"e")
    return d


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"set -euo pipefail; source {LIB}; {cmd}"], capture_output=True, text=True)


def test_quarantine_moves_only_incomplete_and_keeps_data(tmp_path):
    root = tmp_path / "run" / "checkpoints"
    good = make_ckpt(root, 7500, complete=True)
    bad = make_ckpt(root, 8000, complete=False)
    r = run(f"quarantine_incomplete {root}")
    assert r.returncode == 0, r.stderr
    assert good.is_dir() and not bad.exists()
    moved = list((tmp_path / "run" / "checkpoints_incomplete").iterdir())
    assert len(moved) == 1 and moved[0].name.startswith("global_step_8000_")
    assert (moved[0] / "model" / "__0_0.distcp").read_bytes() == b"x"   # moved, not deleted
    assert str(bad) in r.stdout


def test_quarantine_noop_when_all_complete_or_empty(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    assert run(f"quarantine_incomplete {root}").stdout == ""
    make_ckpt(root, 500, complete=True)
    r = run(f"quarantine_incomplete {root}")
    assert r.returncode == 0 and r.stdout == "" and (root / "global_step_500").is_dir()


def test_missing_model_metadata_counts_as_incomplete(tmp_path):
    d = make_ckpt(tmp_path, 1, complete=True)
    (d / "model" / ".metadata").unlink()
    assert run(f"ckpt_complete {d}").returncode != 0

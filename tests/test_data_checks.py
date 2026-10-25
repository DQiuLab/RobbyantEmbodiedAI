from pathlib import Path

import pytest

from src import data_checks

TASKS = ["adjust_bottle", "lift_pot"]


def manifest(*paths):
    return "".join(f"robotwin {p}\n" for p in paths)


def make_lerobot(root: Path, *tasks):
    for t in tasks:
        (root / f"{t}_joint_v30" / "meta").mkdir(parents=True)


def test_normalize_rewrites_stale_prefix():
    src = manifest("/workspace/RoboTwin/data/lerobot/adjust_bottle_joint_v30")
    assert data_checks.normalize_manifest(src) == manifest("/RoboTwin/data/lerobot/adjust_bottle_joint_v30")


def test_validate_accepts_clean_manifest(tmp_path):
    make_lerobot(tmp_path, *TASKS)
    text = manifest(*(tmp_path / f"{t}_joint_v30" for t in TASKS))
    assert data_checks.validate_manifest(text, TASKS) == sorted(TASKS)


@pytest.mark.parametrize("bad,match", [
    ("robotwin {root}/adjust_bottle_randomized_joint_v30", "random"),
    ("robotwin {root}/adjust_bottle-aloha-agilex_randomized_500-1000", "random"),
    ("robotwin {root}/not_a_task_joint_v30", "not_a_task"),
    ("aloha {root}/lift_pot_joint_v30", "robot"),
])
def test_validate_rejects_bad_entries(tmp_path, bad, match):
    make_lerobot(tmp_path, *TASKS, "not_a_task", "adjust_bottle_randomized")
    (tmp_path / "adjust_bottle-aloha-agilex_randomized_500-1000" / "meta").mkdir(parents=True)
    text = manifest(tmp_path / "adjust_bottle_joint_v30") + bad.format(root=tmp_path) + "\n"
    with pytest.raises(ValueError, match=match):
        data_checks.validate_manifest(text, TASKS)


def test_validate_rejects_missing_path(tmp_path):
    make_lerobot(tmp_path, "adjust_bottle")
    text = manifest(tmp_path / "adjust_bottle_joint_v30", tmp_path / "lift_pot_joint_v30")
    with pytest.raises(ValueError, match="does not exist"):
        data_checks.validate_manifest(text, TASKS)


def test_validate_requires_every_task_exactly_once(tmp_path):
    make_lerobot(tmp_path, *TASKS)
    with pytest.raises(ValueError, match="lift_pot"):
        data_checks.validate_manifest(manifest(tmp_path / "adjust_bottle_joint_v30"), TASKS)
    dup = manifest(*(tmp_path / f"{t}_joint_v30" for t in TASKS + ["lift_pot"]))
    with pytest.raises(ValueError, match="duplicate"):
        data_checks.validate_manifest(dup, TASKS)


def test_cli_writes_normalized_manifest(tmp_path):
    lerobot = tmp_path / "lerobot"
    make_lerobot(lerobot, *TASKS)
    demo = tmp_path / "demo_clean"
    for t in TASKS:
        (demo / t).mkdir(parents=True)
    src = tmp_path / "src.txt"
    src.write_text(manifest(*(lerobot / f"{t}_joint_v30" for t in TASKS)))
    out = tmp_path / "out.txt"
    assert data_checks.main(["--manifest", str(src), "--demo-clean-root", str(demo), "--out", str(out)]) == 0
    assert out.read_text() == src.read_text()


def test_cli_accepts_all_tasks_yml(tmp_path):
    lerobot = tmp_path / "lerobot"
    make_lerobot(lerobot, *TASKS)
    tasks_yml = tmp_path / "all_tasks.yml"
    tasks_yml.write_text("tasks:\n  - adjust_bottle\n  - lift_pot\n")
    src = tmp_path / "src.txt"
    src.write_text(manifest(*(lerobot / f"{t}_joint_v30" for t in TASKS)))
    out = tmp_path / "out.txt"
    assert data_checks.main(["--manifest", str(src), "--all-tasks-yml", str(tasks_yml), "--out", str(out)]) == 0

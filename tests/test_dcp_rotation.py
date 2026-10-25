import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "src/lingbot_patches/dcp_rotation.py"
spec = importlib.util.spec_from_file_location("dcp_rotation", MODULE)
dcp_rotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dcp_rotation)


def make(root: Path, *steps: int, extra=()):
    for s in steps:
        d = root / f"global_step_{s}"
        d.mkdir(parents=True)
        (d / "__0_0.distcp").write_bytes(b"x")
    for name in extra:
        (root / name).mkdir(parents=True)
    return root


def remaining(root: Path):
    return sorted(p.name for p in root.iterdir())


def test_keep_all_by_default_is_noop(tmp_path):
    make(tmp_path, 100, 200)
    assert dcp_rotation.rotate_before_save(tmp_path, keep=-1) == []
    assert remaining(tmp_path) == ["global_step_100", "global_step_200"]


def test_keep_one_deletes_all_existing_before_save(tmp_path):
    make(tmp_path, 100, 200)
    removed = dcp_rotation.rotate_before_save(tmp_path, keep=1)
    assert sorted(p.name for p in removed) == ["global_step_100", "global_step_200"]
    assert remaining(tmp_path) == []


def test_keep_two_leaves_newest_one_before_save(tmp_path):
    make(tmp_path, 100, 1000, 200)
    dcp_rotation.rotate_before_save(tmp_path, keep=2)
    assert remaining(tmp_path) == ["global_step_1000"]


def test_ignores_non_checkpoint_entries(tmp_path):
    make(tmp_path, 100, extra=("hf_ckpt", "global_step_x", "async_hf_failures"))
    (tmp_path / "loss.jsonl").write_text("{}")
    dcp_rotation.rotate_before_save(tmp_path, keep=1)
    assert remaining(tmp_path) == ["async_hf_failures", "global_step_x", "hf_ckpt", "loss.jsonl"]


def test_never_deletes_the_step_about_to_be_saved(tmp_path):
    make(tmp_path, 100, 200)
    dcp_rotation.rotate_before_save(tmp_path, keep=1, saving_step=200)
    assert remaining(tmp_path) == ["global_step_200"]


def test_missing_dir_is_noop(tmp_path):
    assert dcp_rotation.rotate_before_save(tmp_path / "nope", keep=1) == []


@pytest.mark.parametrize("value,expected", [(None, -1), ("", -1), ("1", 1), ("3", 3)])
def test_keep_from_env(value, expected, monkeypatch):
    if value is None:
        monkeypatch.delenv("LINGBOT_DCP_KEEP", raising=False)
    else:
        monkeypatch.setenv("LINGBOT_DCP_KEEP", value)
    assert dcp_rotation.keep_from_env() == expected


@pytest.mark.parametrize("value", ["0", "-2", "abc"])
def test_keep_from_env_rejects_invalid(value, monkeypatch):
    monkeypatch.setenv("LINGBOT_DCP_KEEP", value)
    with pytest.raises(ValueError):
        dcp_rotation.keep_from_env()

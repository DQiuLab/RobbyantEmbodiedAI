import yaml

from src import relocate_snapshot as rs


def make_snapshot(d, old="/old/place/hf/global_step_1"):
    (d / "robot_configs").mkdir(parents=True)
    (d / "norm_stats.json").write_text("{}")
    (d / "robot_configs" / "robotwin.yaml").write_text(yaml.safe_dump({"arm": 1, "norm_stats": f"{old}/norm_stats.json"}))
    (d / "lingbotvla_cli.yaml").write_text(yaml.safe_dump({
        "data": {"norm_stats_file": f"{old}/norm_stats.json", "robot_config_root": f"{old}/robot_configs", "image_augment": True},
        "model": {"tokenizer_path": "/keep/me"}}))


def test_relocate_rewrites_only_norm_and_robot_paths_and_is_idempotent(tmp_path):
    d = tmp_path / "snap"
    make_snapshot(d)
    changed = rs.relocate(d)
    assert set(changed) == {"robot_configs/robotwin.yaml", "lingbotvla_cli.yaml"}
    robot = yaml.safe_load((d / "robot_configs" / "robotwin.yaml").read_text())
    cli = yaml.safe_load((d / "lingbotvla_cli.yaml").read_text())
    assert robot == {"arm": 1, "norm_stats": str(d.resolve() / "norm_stats.json")}
    assert cli["data"]["norm_stats_file"] == str(d.resolve() / "norm_stats.json")
    assert cli["data"]["robot_config_root"] == str(d.resolve() / "robot_configs")
    assert cli["data"]["image_augment"] is True and cli["model"]["tokenizer_path"] == "/keep/me"
    assert rs.relocate(d) == []


def test_relocate_requires_bundled_norm_stats(tmp_path):
    d = tmp_path / "snap"
    make_snapshot(d)
    (d / "norm_stats.json").unlink()
    import pytest
    with pytest.raises(FileNotFoundError):
        rs.relocate(d)

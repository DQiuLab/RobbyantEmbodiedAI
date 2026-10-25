import json
from pathlib import Path

import pytest
import yaml

from src import bundle_ckpt


def setup(tmp_path, mismatch=False):
    norm = tmp_path / "norm.json"
    norm.write_text(json.dumps({"norm_stats": {}, "count": 1}))
    robot_dir = tmp_path / "robot"
    robot_dir.mkdir()
    robot = {"states": [], "actions": [], "images": [], "norm_stats": str(tmp_path / "other.json" if mismatch else norm)}
    (robot_dir / "robotwin.yaml").write_text(yaml.safe_dump(robot))
    cli = tmp_path / "train" / "lingbotvla_cli.yaml"
    cli.parent.mkdir()
    cli.write_text(yaml.safe_dump({
        "data": {"norm_stats_file": str(norm), "robot_config_root": str(robot_dir), "joints": ["{'arm.position': 14}"]},
        "train": {"use_lora": True, "lora_rank": 32},
    }))
    hf = tmp_path / "hf"
    hf.mkdir()
    return cli, hf


def test_bundle_copies_and_rewrites_paths(tmp_path):
    cli, hf = setup(tmp_path)
    bundle_ckpt.bundle(cli, hf)
    out = yaml.safe_load((hf / "lingbotvla_cli.yaml").read_text())
    assert out["data"]["norm_stats_file"] == str(hf / "norm_stats.json")
    assert out["data"]["robot_config_root"] == str(hf / "robot_configs")
    assert out["data"]["joints"] == ["{'arm.position': 14}"]
    assert json.loads((hf / "norm_stats.json").read_text())["count"] == 1
    robot = yaml.safe_load((hf / "robot_configs" / "robotwin.yaml").read_text())
    assert robot["norm_stats"] == str(hf / "norm_stats.json")


def test_bundle_rejects_train_deploy_norm_mismatch(tmp_path):
    cli, hf = setup(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="norm"):
        bundle_ckpt.bundle(cli, hf)


def test_bundle_marks_lora_merge(tmp_path):
    cli, hf = setup(tmp_path)
    bundle_ckpt.bundle(cli, hf, lora_merged=True)
    out = yaml.safe_load((hf / "lingbotvla_cli.yaml").read_text())
    assert out["train"]["use_lora"] is False          # weights are already merged; deploy must not re-inject LoRA
    note = json.loads((hf / "SUBMISSION_NOTE.json").read_text())
    assert note["lora_merged"] is True and note["lora_rank"] == 32


def test_cli(tmp_path):
    cli, hf = setup(tmp_path)
    assert bundle_ckpt.main(["--train-cli", str(cli), "--hf-dir", str(hf)]) == 0
    assert (hf / "lingbotvla_cli.yaml").exists()

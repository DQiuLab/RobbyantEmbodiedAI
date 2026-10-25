"""Make a deployable HF checkpoint self-contained (D-07, D-08).

Deployment reads `data.norm_stats_file` from lingbotvla_cli.yaml, training read the
robot config's `norm_stats`; both must be the clean-only file. This copies the
training config, robot config and norm stats into the checkpoint directory and
points the paths there, so judges can run the checkpoint without our runtime dirs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml


def bundle(train_cli: Path, hf_dir: Path, lora_merged: bool = False) -> None:
    train_cli, hf_dir = Path(train_cli), Path(hf_dir)
    cfg = yaml.safe_load(train_cli.read_text(encoding="utf-8"))
    norm_src = Path(cfg["data"]["norm_stats_file"])
    robot_src = Path(cfg["data"]["robot_config_root"]) / "robotwin.yaml"
    robot = yaml.safe_load(robot_src.read_text(encoding="utf-8"))
    if Path(robot["norm_stats"]) != norm_src:
        raise ValueError(f"training norm stats {robot['norm_stats']} != deploy norm stats {norm_src}")

    shutil.copy2(norm_src, hf_dir / "norm_stats.json")
    (hf_dir / "robot_configs").mkdir(exist_ok=True)
    robot["norm_stats"] = str(hf_dir / "norm_stats.json")
    (hf_dir / "robot_configs" / "robotwin.yaml").write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")

    cfg["data"]["norm_stats_file"] = str(hf_dir / "norm_stats.json")
    cfg["data"]["robot_config_root"] = str(hf_dir / "robot_configs")
    if lora_merged:
        rank = cfg["train"].get("lora_rank")
        cfg["train"]["use_lora"] = False  # adapters are merged into the weights
        note = {"lora_merged": True, "lora_rank": rank, "lora_alpha": cfg["train"].get("lora_alpha"),
                "lora_scope": cfg["train"].get("lora_scope"), "source_config": str(train_cli)}
        (hf_dir / "SUBMISSION_NOTE.json").write_text(json.dumps(note, indent=1) + "\n", encoding="utf-8")
    (hf_dir / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cli", type=Path, required=True, help="<train output_dir>/lingbotvla_cli.yaml")
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--lora-merged", action="store_true")
    args = parser.parse_args(argv)
    bundle(args.train_cli, args.hf_dir, args.lora_merged)
    print(f"[bundle] config, robot config and norm stats bundled in {args.hf_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

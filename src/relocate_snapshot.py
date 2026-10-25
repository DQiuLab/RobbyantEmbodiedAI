"""Point a bundled HF snapshot's norm-stats / robot-config paths at the snapshot's current directory.

`bundle_ckpt.py` writes absolute paths (robot_configs/*.yaml `norm_stats`, lingbotvla_cli.yaml
`data.norm_stats_file` and `data.robot_config_root`) for the directory the snapshot was created in.
After copying a snapshot to another instance/directory those paths dangle and the policy server fails
in reset() with FileNotFoundError (EXP-20261015-02, 10-17 07:31). Idempotent; touches only those keys.

Usage: python src/relocate_snapshot.py <hf_dir> [<hf_dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def relocate(hf_dir: Path) -> list[str]:
    hf_dir = hf_dir.resolve()
    norm = hf_dir / "norm_stats.json"
    if not norm.is_file():
        raise FileNotFoundError(norm)
    changed = []
    for robot_yaml in sorted((hf_dir / "robot_configs").glob("*.yaml")):
        robot = yaml.safe_load(robot_yaml.read_text(encoding="utf-8"))
        if robot.get("norm_stats") != str(norm):
            robot["norm_stats"] = str(norm)
            robot_yaml.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
            changed.append(str(robot_yaml.relative_to(hf_dir)))
    cli_path = hf_dir / "lingbotvla_cli.yaml"
    cli = yaml.safe_load(cli_path.read_text(encoding="utf-8"))
    want = {"norm_stats_file": str(norm), "robot_config_root": str(hf_dir / "robot_configs")}
    if any(cli["data"].get(k) != v for k, v in want.items()):
        cli["data"].update(want)
        cli_path.write_text(yaml.safe_dump(cli, sort_keys=False), encoding="utf-8")
        changed.append(cli_path.name)
    return changed


def main(argv: list[str]) -> int:
    for d in argv:
        print(f"{d}: {relocate(Path(d)) or 'already relocated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

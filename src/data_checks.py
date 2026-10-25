"""Normalise and validate the training manifest (rules-checklist A: clean data only).

A manifest written on the persistent volume of the initial AMD Radeon Cloud plan may carry stale
/workspace/RoboTwin/ prefixes (train_full_sft.sh in the AMD reference sample rewrites them the same way). Validation guarantees the
file that actually feeds training lists exactly the 50 demo_clean tasks, each
once, as existing `<task>_joint_v30` LeRobot directories, and nothing randomized.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

STALE_PREFIX = "/workspace/RoboTwin/"
ROOT_PREFIX = "/RoboTwin/"
SUFFIX = "_joint_v30"


def normalize_manifest(text: str) -> str:
    return text.replace(STALE_PREFIX, ROOT_PREFIX)


def validate_manifest(text: str, clean_tasks: list[str]) -> list[str]:
    seen: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        robot, _, path = line.strip().partition(" ")
        if robot != "robotwin":
            raise ValueError(f"line {lineno}: robot config {robot!r} != 'robotwin'")
        name = Path(path).name
        if "random" in path.lower():
            raise ValueError(f"line {lineno}: randomized data in manifest: {path}")
        if not name.endswith(SUFFIX) or name[: -len(SUFFIX)] not in clean_tasks:
            raise ValueError(f"line {lineno}: {name} is not <demo_clean task>{SUFFIX}")
        if not Path(path).is_dir():
            raise ValueError(f"line {lineno}: {path} does not exist")
        task = name[: -len(SUFFIX)]
        if task in seen:
            raise ValueError(f"line {lineno}: duplicate task {task}")
        seen.append(task)
    missing = sorted(set(clean_tasks) - set(seen))
    if missing:
        raise ValueError(f"manifest misses tasks: {missing}")
    return sorted(seen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo-clean-root", type=Path, help="task list = sub-directories")
    source.add_argument("--all-tasks-yml", type=Path, help="RoboTwin env_cfg/eval/all_tasks.yml (raw data may be deleted)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.demo_clean_root:
        tasks = sorted(p.name for p in args.demo_clean_root.iterdir() if p.is_dir())
    else:
        import yaml
        tasks = sorted(yaml.safe_load(args.all_tasks_yml.read_text(encoding="utf-8"))["tasks"])
    text = normalize_manifest(args.manifest.read_text(encoding="utf-8"))
    validated = validate_manifest(text, tasks)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"manifest ok: {len(validated)} clean tasks → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

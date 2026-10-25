"""Bound the number of DCP checkpoints on disk (installed as lingbotvla/utils/dcp_rotation.py).

Upstream train_lingbotvla.py writes a new full DCP (~70 GB with AdamW) at every
save and never deletes old ones. Written for the 100 GB /workspace of Radeon Cloud (initial AMD plan,
not used for the final weights); the AutoDL configs use it too (dcp_keep in configs/train/). With
LINGBOT_DCP_KEEP=N, older checkpoints are deleted *before* each save so that at
most N exist afterwards. Unset or empty keeps upstream behaviour (keep all).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

STEP_DIR = re.compile(r"global_step_(\d+)")


def keep_from_env() -> int:
    raw = os.environ.get("LINGBOT_DCP_KEEP", "")
    if raw == "":
        return -1
    keep = int(raw)
    if keep < 1:
        raise ValueError(f"LINGBOT_DCP_KEEP must be >= 1, got {raw!r}")
    return keep


def rotate_before_save(checkpoint_root, keep: int, saving_step: int | None = None) -> list[Path]:
    root = Path(checkpoint_root)
    if keep < 0 or not root.is_dir():
        return []
    steps = []
    for entry in root.iterdir():
        match = STEP_DIR.fullmatch(entry.name)
        if entry.is_dir() and match and int(match.group(1)) != saving_step:
            steps.append((int(match.group(1)), entry))
    steps.sort(key=lambda item: item[0], reverse=True)
    doomed = [path for _, path in steps[keep - 1:]]
    for path in doomed:
        shutil.rmtree(path)
    return doomed

"""Run-control guards for long single-run training (installed as lingbotvla/utils/run_control.py).

- LINGBOT_REQUIRE_RESUME=1: fail fast when checkpoints exist but none loaded.
  Upstream silently trains from scratch in that case, and with LINGBOT_DCP_KEEP=1
  the next save would then delete the only good checkpoint.
- LINGBOT_STOP_FILE=<path>: when the file exists, the training loop saves and exits
  at the end of the current step (graceful stop for stage ends and the train cutoff).
- LINGBOT_STOP_AT_STEP=<n>: same graceful stop once global_step >= n. Reproduces stage 1, which followed a
  20k-step cosine schedule but ended at step 7500 (D-13, docs/specs/submission.md S3).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

STEP_DIR = re.compile(r"global_step_(\d+)")  # same pattern as dcp_rotation / upstream resume


def check_resumed(checkpoint_root, loaded: bool) -> None:
    if os.environ.get("LINGBOT_REQUIRE_RESUME", "") != "1" or loaded:
        return
    root = Path(checkpoint_root)
    found = sorted(p.name for p in root.iterdir() if p.is_dir() and STEP_DIR.fullmatch(p.name)) if root.is_dir() else []
    if found:
        raise RuntimeError(f"LINGBOT_REQUIRE_RESUME=1 but no checkpoint loaded from {root}: {found}")


def stop_requested(step: int | None = None) -> bool:
    path = os.environ.get("LINGBOT_STOP_FILE", "")
    if path and Path(path).exists():
        return True
    stop_at = os.environ.get("LINGBOT_STOP_AT_STEP", "")
    return bool(stop_at) and step is not None and step >= int(stop_at)

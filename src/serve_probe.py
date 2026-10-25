"""Probe a running lingbot_vla_v2 policy server with random observations (no simulator needed).

Run from the lingbot-vla-v2 source dir (imports ``deploy``). Exits non-zero unless every returned action is
finite and 14-dimensional (Aloha-AgileX). Used by scripts/cloud/serve_check.sh.

Usage: python src/serve_probe.py <port> [n_requests]
"""

from __future__ import annotations

import sys

import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_DIM = 14


def random_observation(rng: np.random.Generator) -> dict:
    obs = {f"observation.images.{c}": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for c in CAMERAS}
    obs["observation.state"] = np.zeros(ACTION_DIM, dtype=np.float32)
    obs["task"] = "adjust the bottle upright"
    return obs


def check_action(action) -> str | None:
    """Return an error message for a bad action chunk, None if it is usable."""
    a = np.asarray(action, dtype=np.float64)
    if a.size == 0:
        return "empty action"
    if a.shape[-1] != ACTION_DIM:
        return f"action dim {a.shape[-1]} != {ACTION_DIM}"
    if not np.isfinite(a).all():
        return "non-finite action values"
    return None


def main(argv: list[str]) -> int:
    from deploy.websocket_client_policy import WebsocketClientPolicy

    port = int(argv[0])
    n = int(argv[1]) if len(argv) > 1 else 3
    rng = np.random.default_rng(0)
    client = WebsocketClientPolicy("127.0.0.1", port)
    client.infer({"reset": True, "robo_name": "robotwin"})
    for i in range(n):
        action = client.infer(random_observation(rng))["action"]
        a = np.asarray(action, dtype=np.float64)
        print(f"[serve_probe] infer {i}: shape {a.shape} min {a.min():.4f} max {a.max():.4f}", flush=True)
        err = check_action(action)
        if err:
            print(f"[serve_probe] FAIL: {err}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

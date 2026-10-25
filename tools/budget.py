"""Training step budget under AMD credits and wall-clock limits.

Initial AMD Radeon Cloud plan, not used for the final weights (the AutoDL runs set max_steps directly).

Spec: docs/specs/training.md §T3. Defaults come from configs/budget.yaml.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


class BudgetError(ValueError):
    pass


@dataclass(frozen=True)
class BudgetInput:
    start: datetime
    train_end: datetime
    step_seconds: float
    gpus: int
    stage1_credits: float
    stage2_credits: float
    stage1_overhead_h: float      # smoke test, conversion, sanity eval inside stage 1
    stage2_overhead_h: float      # evals, conversions inside stage 2 (credits only)
    stage_gap_h: float            # queue wait between stage 1 and stage 2
    cap_steps: int
    round_to: int
    pre_train_end_overhead_h: float = 0.0  # stage-2 work that pauses training before train_end


@dataclass(frozen=True)
class BudgetResult:
    stage1_stop_step: int
    wall_clock_steps: int
    credit_steps: int
    max_steps: int
    train_hours: float


def compute(p: BudgetInput) -> BudgetResult:
    steps_per_hour = 3600.0 / p.step_seconds
    window_h = (p.train_end - p.start).total_seconds() / 3600.0
    wall_train_h = window_h - p.stage_gap_h - p.stage1_overhead_h - p.pre_train_end_overhead_h
    credit_train_h = (p.stage1_credits + p.stage2_credits) / p.gpus - p.stage1_overhead_h - p.stage2_overhead_h
    if wall_train_h <= 0 or credit_train_h <= 0:
        raise BudgetError(f"no training time left: wall={wall_train_h:.1f} h, credits={credit_train_h:.1f} h")

    wall_clock_steps = int(wall_train_h * steps_per_hour)
    credit_steps = int(credit_train_h * steps_per_hour)
    max_steps = min(wall_clock_steps, credit_steps, p.cap_steps) // p.round_to * p.round_to
    if max_steps <= 0:
        raise BudgetError("max_steps rounds down to zero")

    stage1_train_h = p.stage1_credits / p.gpus - p.stage1_overhead_h
    stage1_stop_step = min(int(stage1_train_h * steps_per_hour) // p.round_to * p.round_to, max_steps)
    return BudgetResult(
        stage1_stop_step=stage1_stop_step,
        wall_clock_steps=wall_clock_steps,
        credit_steps=credit_steps,
        max_steps=max_steps,
        train_hours=max_steps / steps_per_hour,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/budget.yaml")
    parser.add_argument("--start", required=True, help="training start, ISO local time")
    parser.add_argument("--train-end", help="override train_end from config, ISO local time")
    parser.add_argument("--step-seconds", type=float, help="override measured step time")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.train_end:
        cfg["train_end"] = args.train_end
    if args.step_seconds:
        cfg["step_seconds"] = args.step_seconds
    cfg["start"] = datetime.fromisoformat(args.start)
    cfg["train_end"] = datetime.fromisoformat(str(cfg["train_end"]))
    result = compute(BudgetInput(**cfg))
    for name, value in vars(result).items():
        print(f"{name}: {value:.1f}" if isinstance(value, float) else f"{name}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

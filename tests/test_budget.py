from datetime import datetime

import pytest

from tools import budget


def plan(**overrides):
    params = dict(
        start=datetime(2026, 10, 13, 12, 0),
        train_end=datetime(2026, 10, 22, 12, 0),
        step_seconds=24.68,
        gpus=8,
        stage1_credits=600,
        stage2_credits=1600,
        stage1_overhead_h=6.0,
        stage2_overhead_h=50.0,
        stage_gap_h=12.0,
        cap_steps=30000,
        round_to=500,
    )
    params.update(overrides)
    return budget.compute(budget.BudgetInput(**params))


def test_stage1_stop_step_from_credits():
    # 600 credits / 8 GPUs = 75 h; minus 6 h overhead = 69 h of training.
    result = plan()
    assert result.stage1_stop_step == int(69 * 3600 / 24.68) // 500 * 500


def test_wall_clock_bound():
    # 216 h window - 12 h gap - 6 h stage-1 overhead = 198 h training (stage-2 eval happens after train_end).
    result = plan()
    assert result.wall_clock_steps == int(198 * 3600 / 24.68)


def test_credit_bound():
    # 2200 / 8 = 275 h - 6 - 50 = 219 h.
    result = plan()
    assert result.credit_steps == int(219 * 3600 / 24.68)


def test_max_steps_is_min_of_bounds_rounded_down():
    result = plan()
    expected = min(result.wall_clock_steps, result.credit_steps, 30000) // 500 * 500
    assert result.max_steps == expected
    assert result.max_steps % 500 == 0


def test_cap_applies_when_time_is_plentiful():
    result = plan(start=datetime(2026, 10, 1), stage1_credits=10000, stage2_credits=10000)
    assert result.max_steps == 30000


def test_stage1_never_exceeds_max_steps():
    result = plan(start=datetime(2026, 10, 20, 0, 0))
    assert result.stage1_stop_step <= result.max_steps


def test_late_start_with_no_window_raises():
    with pytest.raises(budget.BudgetError):
        plan(start=datetime(2026, 10, 22, 0, 0))


def test_four_gpus_double_step_time_halves_credit_hours():
    eight = plan()
    four = plan(gpus=4, step_seconds=41.39)
    assert four.max_steps < eight.max_steps


def test_cli_prints_summary(capsys):
    code = budget.main([
        "--start", "2026-10-13T12:00", "--train-end", "2026-10-22T12:00", "--step-seconds", "24.68",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "max_steps" in out and "stage1_stop_step" in out

from pathlib import Path

import pytest

from tools import compare_runs as cr


def log_text(outcomes: list[tuple[int, bool]]) -> str:
    """Mimic RoboTwin output (ANSI colours, \\r progress) for a list of (seed, success)."""
    lines, s = [], 0
    for i, (seed, ok) in enumerate(outcomes, 1):
        s += ok
        lines.append(f"step: \x1b[92m1 / 400\x1b[0m\rstep: 2 / 400\r")
        lines.append(f"Success rate: \x1b[96m{s}/{i}\x1b[0m => \x1b[95m{100*s/i:.1f}%\x1b[0m, current seed: \x1b[90m{seed}\x1b[0m\n")
    lines.append(f"Final success rate: {s}/{len(outcomes)} = {100*s/max(1,len(outcomes)):.1f}%\n")
    return "".join(lines)


def write(d: Path, task: str, outcomes, config="demo_clean", suffix=".log") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"eval_{config}__{task}{suffix}").write_text(log_text(outcomes))


def test_episodes_parses_outcomes_and_truncates(tmp_path):
    write(tmp_path, "t", [(100000, True), (100001, False), (100003, True)])
    assert cr.episodes(tmp_path / "eval_demo_clean__t.log") == {100000: True, 100001: False, 100003: True}
    assert cr.episodes(tmp_path / "eval_demo_clean__t.log", max_eps=2) == {100000: True, 100001: False}


def test_episodes_rejects_counter_jump(tmp_path):
    (tmp_path / "x.log").write_text("Success rate: 1/1 => 100.0%, current seed: 5\nSuccess rate: 1/3 => 33.3%, current seed: 7\n")
    with pytest.raises(ValueError):
        cr.episodes(tmp_path / "x.log")


def test_compare_pairs_by_seed_and_counts_discordant(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "t1", [(1, True), (2, False), (3, False), (4, True)])
    write(b, "t1", [(1, True), (2, True), (3, True), (5, True)])          # seed 4/5 unpaired
    write(a, "t2", [(1, False)]); write(b, "t2", [(1, False)], suffix=".log.part")
    res = cr.compare(a, b, "demo_clean", ["t1", "t2", "t3"], None)
    t1 = res["rows"][0]
    assert (t1["pairs"], t1["a"], t1["b"], t1["a_only"], t1["b_only"]) == (3, 1, 3, 0, 2)
    assert res["rows"][2] == {"task": "t3", "missing": True}
    assert res["total"]["pairs"] == 4 and res["total"]["b_only"] == 2


def test_sign_test_p():
    assert cr.sign_test_p(0, 0) == 1.0
    assert cr.sign_test_p(0, 5) == pytest.approx(2 / 32)
    assert cr.sign_test_p(3, 3) == 1.0


def test_phantom_counts_unstable_episodes_in_final_denominator(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "t", [(1, True), (2, True)])
    text = log_text([(1, True), (3, False)]).replace("Final success rate: 1/2", "Final success rate: 1/5")
    b.mkdir(); (b / "eval_demo_clean__t.log").write_text(text)          # 3 次未执行却计入分母
    r = cr.compare(a, b, "demo_clean", ["t"], None)["rows"][0]
    assert (r["exec_b"], r["succ_b"], r["phantom_b"], r["phantom_a"]) == (2, 1, 3, 0)


def test_bootstrap_ci_and_leave_one_out():
    rows = [{"task": f"t{i}", "pairs": 10, "a": 5, "b": 6} for i in range(8)] + [{"task": "big", "pairs": 10, "a": 0, "b": 10}]
    d = cr.paired_diff(rows)
    lo, hi = cr.bootstrap_ci(rows)
    assert lo <= d <= hi and lo > 0
    task, d_wo = cr.leave_one_out(rows)
    assert task == "big" and d_wo == pytest.approx(0.1)


def test_low_sample_flag(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "t", [(1, True), (2, False)]); write(b, "t", [(1, False), (2, False)])
    r = cr.compare(a, b, "demo_clean", ["t"], None)["rows"][0]
    assert r["low"] is True


def test_task_with_infra_error_is_excluded(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write(a, "t", [(1, True)]); write(b, "t", [(1, False)])
    p = b / "eval_demo_clean__t.log"; p.write_text("Policy rollout error:\n" + p.read_text())
    r = cr.compare(a, b, "demo_clean", ["t"], None)
    assert r["rows"][0] == {"task": "t", "missing": True, "infra": True} and r["total"]["pairs"] == 0

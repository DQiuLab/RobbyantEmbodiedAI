import json
from pathlib import Path

import pytest

from tests.conftest import load_tasks
from tools import results


def template_tasks():
    return load_tasks()


def write_log(run_dir: Path, task_config: str, task: str, body: str) -> None:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"eval_{task_config}__{task}.log").write_text(body, encoding="utf-8")


def fill_run(run_dir: Path, task_configs=("demo_clean", "demo_randomized"), succ=7, n=10):
    for cfg in task_configs:
        for task in template_tasks():
            write_log(run_dir, cfg, task, f"noise\nFinal success rate: {succ}/{n} = {100*succ/n:.1f}%\n")


def test_template_has_50_unique_tasks_matching_both_settings(results_template):
    data = json.loads(results_template.read_text(encoding="utf-8"))
    clean = list(data["results"]["clean"])
    assert len(clean) == 50 and len(set(clean)) == 50
    assert clean == list(data["results"]["randomized"])


def test_parse_log_takes_last_match_and_handles_carriage_returns(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("Final success rate: 1/2 = 50.0%\rprogress\rFinal success rate: 3/10 = 30.0%\n", encoding="utf-8")
    assert results.parse_log(log) == (3, 10)


def test_parse_log_without_result_raises(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("crashed\n", encoding="utf-8")
    with pytest.raises(results.ResultsError):
        results.parse_log(log)


@pytest.mark.parametrize("marker", ["Policy rollout error:", "RuntimeError: cannot create buffer", "CUDA out of memory"])
def test_parse_log_rejects_infra_errors(tmp_path, marker):
    log = tmp_path / "eval_demo_clean__a.log"
    log.write_text(f"step 1\n{marker}\nFinal success rate: 3/10 = 30.0%\n")
    with pytest.raises(results.ResultsError, match="infrastructure error"):
        results.parse_log(log)


def test_collect_maps_task_configs_to_settings(tmp_path):
    write_log(tmp_path, "demo_clean", "lift_pot", "Final success rate: 4/5 = 80%\n")
    write_log(tmp_path, "demo_randomized", "lift_pot", "Final success rate: 2/5 = 40%\n")
    got = results.collect([tmp_path])
    assert got["clean"]["lift_pot"] == {"attempts": 5, "successes": 4}
    assert got["randomized"]["lift_pot"] == {"attempts": 5, "successes": 2}


def test_collect_rejects_unknown_task_config(tmp_path):
    write_log(tmp_path, "demo_weird", "lift_pot", "Final success rate: 4/5 = 80%\n")
    with pytest.raises(results.ResultsError):
        results.collect([tmp_path])


def test_collect_rejects_duplicate_across_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_log(a, "demo_clean", "lift_pot", "Final success rate: 4/5 = 80%\n")
    write_log(b, "demo_clean", "lift_pot", "Final success rate: 5/5 = 100%\n")
    with pytest.raises(results.ResultsError, match="lift_pot"):
        results.collect([a, b])


def test_collect_merges_disjoint_runs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    fill_run(a, task_configs=("demo_clean",))
    fill_run(b, task_configs=("demo_randomized",), succ=3)
    got = results.collect([a, b])
    assert len(got["clean"]) == 50 and len(got["randomized"]) == 50
    assert got["randomized"]["lift_pot"]["successes"] == 3


def test_build_preserves_template_order_and_team_id(tmp_path, results_template):
    fill_run(tmp_path)
    out = results.build(results_template, "team_123", results.collect([tmp_path]))
    assert out["schema_version"] == 1
    assert out["team_id"] == "team_123"
    assert list(out["results"]["clean"]) == template_tasks()
    assert out["results"]["clean"]["hanging_mug"] == {"attempts": 10, "successes": 7}


def test_build_requires_complete_results_by_default(tmp_path, results_template):
    write_log(tmp_path, "demo_clean", "lift_pot", "Final success rate: 4/5 = 80%\n")
    with pytest.raises(results.ResultsError, match="missing"):
        results.build(results_template, "team_123", results.collect([tmp_path]))


def test_build_allow_partial_fills_zero(tmp_path, results_template):
    write_log(tmp_path, "demo_clean", "lift_pot", "Final success rate: 4/5 = 80%\n")
    out = results.build(results_template, "team_123", results.collect([tmp_path]), allow_partial=True)
    assert out["results"]["clean"]["lift_pot"] == {"attempts": 5, "successes": 4}
    assert out["results"]["randomized"]["lift_pot"] == {"attempts": 0, "successes": 0}


def test_build_rejects_task_not_in_template(tmp_path, results_template):
    fill_run(tmp_path)
    write_log(tmp_path, "demo_clean", "not_a_task", "Final success rate: 1/1 = 100%\n")
    with pytest.raises(results.ResultsError, match="not_a_task"):
        results.build(results_template, "team_123", results.collect([tmp_path]))


@pytest.mark.parametrize("team_id", ["<team_id>", "replace_with_your_team_id", ""])
def test_validate_rejects_placeholder_team_id(tmp_path, results_template, team_id):
    fill_run(tmp_path)
    with pytest.raises(results.ResultsError, match="team_id"):
        results.build(results_template, team_id, results.collect([tmp_path]))


def test_placeholder_set_covers_template_and_official_placeholders():
    assert {"<team_id>", "replace_with_your_team_id"} <= results.PLACEHOLDER_TEAM_IDS


@pytest.mark.parametrize("entry", [
    {"attempts": 5, "successes": 6},
    {"attempts": -1, "successes": 0},
    {"attempts": 5.0, "successes": 1},
    {"attempts": 5},
])
def test_validate_rejects_bad_entries(entry, results_template):
    data = json.loads(results_template.read_text(encoding="utf-8"))
    data["team_id"] = "team_123"
    data["results"]["clean"]["lift_pot"] = entry
    with pytest.raises(results.ResultsError):
        results.validate(data, results_template)


def test_cli_writes_valid_json(tmp_path, results_template):
    run = tmp_path / "run"
    fill_run(run)
    cfg = tmp_path / "submission.yaml"
    cfg.write_text(f"team_name: t\nteam_id: team_123\ntemplate_json: {results_template}\n", encoding="utf-8")
    out = tmp_path / "results.json"
    assert results.main(["--config", str(cfg), "--run-dir", str(run), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    results.validate(written, results_template)

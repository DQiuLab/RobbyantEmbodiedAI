import importlib.util
import json
import random
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py")

MODULE = Path(__file__).resolve().parents[1] / "src/lingbot_patches/seen_instructions.py"
spec = importlib.util.spec_from_file_location("seen_instructions", MODULE)
seen_instructions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seen_instructions)


def make_episode(root: Path, task: str, ep: int, instructions: list[str], first: str | None = None):
    d = root / task / "aloha_agilex" / "data"
    d.mkdir(parents=True, exist_ok=True)
    with h5py.File(d / f"episode_{ep:07d}.hdf5", "w") as f:
        f["instruction"] = (first if first is not None else instructions[0]).encode()
        f["instructions"] = [s.encode() for s in instructions]


def test_build_pool_keys_by_first_instruction(tmp_path):
    make_episode(tmp_path, "lift_pot", 0, ["a0", "a1", "a2"])
    make_episode(tmp_path, "lift_pot", 1, ["b0", "b1"])
    pool = seen_instructions.build_pool(tmp_path)
    assert pool["pool"] == {"a0": ["a0", "a1", "a2"], "b0": ["b0", "b1"]}
    assert pool["conflicts"] == []
    assert pool["episodes"] == 2


def test_build_pool_is_deterministic_and_sorted(tmp_path):
    make_episode(tmp_path, "b_task", 1, ["y", "z"])
    make_episode(tmp_path, "a_task", 0, ["x", "w"])
    first = json.dumps(seen_instructions.build_pool(tmp_path), sort_keys=True)
    second = json.dumps(seen_instructions.build_pool(tmp_path), sort_keys=True)
    assert first == second
    assert seen_instructions.build_pool(tmp_path)["sources"] == [
        "a_task/aloha_agilex/data/episode_0000000.hdf5", "b_task/aloha_agilex/data/episode_0000001.hdf5"]


def test_collision_with_identical_lists_is_kept(tmp_path):
    make_episode(tmp_path, "t", 0, ["same", "other"])
    make_episode(tmp_path, "t", 1, ["same", "other"])
    pool = seen_instructions.build_pool(tmp_path)
    assert pool["pool"] == {"same": ["same", "other"]}
    assert pool["conflicts"] == []


def test_collision_with_different_lists_is_dropped_and_reported(tmp_path):
    make_episode(tmp_path, "t", 0, ["same", "left arm"])
    make_episode(tmp_path, "t", 1, ["same", "right arm"])
    pool = seen_instructions.build_pool(tmp_path)
    assert "same" not in pool["pool"]
    assert pool["conflicts"] == ["same"]


def test_first_instruction_must_be_in_list(tmp_path):
    make_episode(tmp_path, "t", 0, ["a", "b"], first="zzz")
    with pytest.raises(ValueError, match="zzz"):
        seen_instructions.build_pool(tmp_path)


def test_never_reads_instruction_json_files(tmp_path):
    make_episode(tmp_path, "t", 0, ["a", "b"])
    unseen_dir = tmp_path / "t" / "aloha_agilex" / "instructions"
    unseen_dir.mkdir()
    (unseen_dir / "episode0.json").write_text(json.dumps({"seen": ["a"], "unseen": ["UNSEEN"]}))
    pool = seen_instructions.build_pool(tmp_path)
    assert "UNSEEN" not in json.dumps(pool)


def test_sampler_replaces_within_pool_only(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"pool": {"a0": ["a0", "a1", "a2"]}, "conflicts": []}))
    sampler = seen_instructions.SeenInstructionSampler(path, rng=random.Random(0))
    draws = {sampler("a0") for _ in range(200)}
    assert draws == {"a0", "a1", "a2"}


def test_sampler_keeps_unknown_task_text(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"pool": {"a0": ["a0", "a1"]}, "conflicts": []}))
    sampler = seen_instructions.SeenInstructionSampler(path, rng=random.Random(0))
    assert sampler("not in pool") == "not in pool"


def test_from_env_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("LINGBOT_SEEN_INSTRUCTION_POOL", raising=False)
    assert seen_instructions.sampler_from_env() is None


def test_from_env_loads_pool(tmp_path, monkeypatch):
    path = tmp_path / "pool.json"
    path.write_text(json.dumps({"pool": {"a0": ["a0", "a1"]}, "conflicts": []}))
    monkeypatch.setenv("LINGBOT_SEEN_INSTRUCTION_POOL", str(path))
    assert seen_instructions.sampler_from_env()("a0") in {"a0", "a1"}


def test_cli_writes_pool(tmp_path):
    make_episode(tmp_path / "demo", "t", 0, ["a", "b"])
    out = tmp_path / "pool.json"
    assert seen_instructions.main(["build", "--demo-clean-root", str(tmp_path / "demo"), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["pool"] == {"a": ["a", "b"]}


def test_coverage_counts_task_texts_found_in_pool():
    pool = {"pool": {"a": ["a", "b"], "c": ["c"]}, "conflicts": ["x"]}
    report = seen_instructions.coverage(pool, ["a", "c", "x", "zzz"])
    assert report == {"tasks": 4, "replaceable": 2, "conflicts": 1, "missing": ["zzz"]}


def test_check_fails_when_coverage_below_threshold(tmp_path):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps({"pool": {"a": ["a", "b"]}, "conflicts": []}))
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("a\nnope\n")
    assert seen_instructions.main(["check", "--pool", str(pool), "--task-list", str(tasks), "--min-coverage", "0.9"]) == 1
    assert seen_instructions.main(["check", "--pool", str(pool), "--task-list", str(tasks), "--min-coverage", "0.5"]) == 0


def test_merge_pools_unions_and_detects_cross_fragment_conflicts():
    a = {"version": 1, "episodes": 2, "sources": ["t1/e0"], "pool": {"x": ["x", "y"], "same": ["same", "l"]}, "conflicts": ["c1"]}
    b = {"version": 1, "episodes": 3, "sources": ["t2/e0"], "pool": {"z": ["z"], "same": ["same", "r"], "x": ["x", "y"]}, "conflicts": []}
    merged = seen_instructions.merge_pools([a, b])
    assert merged["pool"] == {"x": ["x", "y"], "z": ["z"]}
    assert merged["conflicts"] == ["c1", "same"]
    assert merged["episodes"] == 5 and merged["sources"] == ["t1/e0", "t2/e0"]


def test_merge_conflict_in_one_fragment_drops_key_from_others():
    a = {"version": 1, "episodes": 1, "sources": [], "pool": {}, "conflicts": ["k"]}
    b = {"version": 1, "episodes": 1, "sources": [], "pool": {"k": ["k", "v"]}, "conflicts": []}
    assert "k" not in seen_instructions.merge_pools([a, b])["pool"]


def test_cli_merge(tmp_path):
    for i, pool in enumerate([{"x": ["x"]}, {"y": ["y"]}]):
        (tmp_path / f"p{i}.json").write_text(json.dumps({"version": 1, "episodes": 1, "sources": [], "pool": pool, "conflicts": []}))
    out = tmp_path / "pool.json"
    assert seen_instructions.main(["merge", "--out", str(out), str(tmp_path / "p0.json"), str(tmp_path / "p1.json")]) == 0
    assert json.loads(out.read_text())["pool"] == {"x": ["x"], "y": ["y"]}

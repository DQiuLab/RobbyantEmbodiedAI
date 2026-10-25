import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "src/lingbot_patches/run_control.py"
spec = importlib.util.spec_from_file_location("run_control", MODULE)
run_control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_control)


def test_require_resume_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGBOT_REQUIRE_RESUME", raising=False)
    (tmp_path / "global_step_100").mkdir()
    run_control.check_resumed(tmp_path, loaded=False)  # no raise


def test_require_resume_raises_when_checkpoint_exists_but_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGBOT_REQUIRE_RESUME", "1")
    (tmp_path / "global_step_100").mkdir()
    with pytest.raises(RuntimeError, match="global_step_100"):
        run_control.check_resumed(tmp_path, loaded=False)


def test_require_resume_ok_when_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGBOT_REQUIRE_RESUME", "1")
    (tmp_path / "global_step_100").mkdir()
    run_control.check_resumed(tmp_path, loaded=True)


def test_require_resume_ok_on_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setenv("LINGBOT_REQUIRE_RESUME", "1")
    run_control.check_resumed(tmp_path / "missing", loaded=False)
    (tmp_path / "hf").mkdir()
    run_control.check_resumed(tmp_path, loaded=False)


def test_stop_file_path_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LINGBOT_STOP_FILE", raising=False)
    assert run_control.stop_requested() is False
    stop = tmp_path / "STOP"
    monkeypatch.setenv("LINGBOT_STOP_FILE", str(stop))
    assert run_control.stop_requested() is False
    stop.touch()
    assert run_control.stop_requested() is True


def test_stop_at_step(monkeypatch, tmp_path):
    monkeypatch.delenv("LINGBOT_STOP_FILE", raising=False)
    monkeypatch.setenv("LINGBOT_STOP_AT_STEP", "7500")
    assert not run_control.stop_requested(7499)
    assert run_control.stop_requested(7500)
    assert run_control.stop_requested(7501)


def test_stop_at_step_unset_and_file_still_works(monkeypatch, tmp_path):
    monkeypatch.delenv("LINGBOT_STOP_AT_STEP", raising=False)
    stop = tmp_path / "STOP"
    monkeypatch.setenv("LINGBOT_STOP_FILE", str(stop))
    assert not run_control.stop_requested(10**6)
    stop.touch()
    assert run_control.stop_requested(1)
    assert run_control.stop_requested()          # 旧调用方式（不传 step）仍可用

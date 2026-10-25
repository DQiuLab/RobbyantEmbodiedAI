"""eval.sh 端到端（桩服务 + 桩 eval_policy.sh），docs/specs/submission.md S4。"""
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from tests.conftest import load_tasks, write_template
from tests.test_eval_subset import free_port, make_sandbox

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_REL = "Robbyant-Embodied-AI-Challenge/必读_模板和目录要求/灵波2026/初赛评测结果模板_JSON版.json"   # configs/submission.yaml 中的路径；主办方模板不随仓库分发，测试里写入生成的同结构模板


def setup(tmp: Path) -> tuple[Path, Path]:
    repo = make_sandbox(tmp)
    shutil.copytree(REPO / "tools", repo / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "eval.sh", repo / "eval.sh")
    write_template(repo / TEMPLATE_REL)
    cfg = yaml.safe_load((repo / "configs/submission.yaml").read_text())
    cfg["team_id"] = "team_test"
    (repo / "configs/submission.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True))
    tasks = load_tasks()
    (tmp / "RoboTwin/env_cfg/eval").mkdir(parents=True)
    (tmp / "RoboTwin/env_cfg/eval/all_tasks.yml").write_text(yaml.safe_dump({"tasks": tasks}))
    ckpt = tmp / "ckpt" / "global_step_12500"
    (ckpt / "robot_configs").mkdir(parents=True)
    (ckpt / "norm_stats.json").write_text("{}")
    (ckpt / "robot_configs/robotwin.yaml").write_text(yaml.safe_dump({"norm_stats": "/old/place/norm_stats.json"}))
    (ckpt / "lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data": {"norm_stats_file": "/old", "robot_config_root": "/old"}, "model": {}}))
    return repo, ckpt


def test_eval_sh_both_settings_writes_results(tmp_path):
    repo, ckpt = setup(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "PLATFORM": "fake", "CALLS": str(tmp_path / "calls.txt"),
           "SERVE_DIR": str(tmp_path / "srv"), "PORT": str(free_port()), "CLIENTS": "4"}
    r = subprocess.run(["bash", str(repo / "eval.sh"), str(ckpt), "both", "1", "0"], env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = json.loads((tmp_path / "runtime/outputs/eval_global_step_12500_results.json").read_text())
    assert out["team_id"] == "team_test"
    assert all(v == {"attempts": 1, "successes": 1} for s in ("clean", "randomized") for v in out["results"][s].values())
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert len(calls) == 100 and all("seed=0" in c for c in calls)
    # 权重内路径已改写为 checkpoint 自身目录
    assert yaml.safe_load((ckpt / "robot_configs/robotwin.yaml").read_text())["norm_stats"] == str(ckpt / "norm_stats.json")


def test_eval_sh_rejects_bad_setting(tmp_path):
    repo, ckpt = setup(tmp_path)
    r = subprocess.run(["bash", str(repo / "eval.sh"), str(ckpt), "dirty"], env={"PATH": "/usr/bin:/bin", "PLATFORM": "fake"},
                       capture_output=True, text=True)
    assert r.returncode == 2


def test_eval_sh_tasks_subset(tmp_path):
    repo, ckpt = setup(tmp_path)
    env = {"PATH": "/usr/bin:/bin", "PLATFORM": "fake", "CALLS": str(tmp_path / "calls.txt"),
           "SERVE_DIR": str(tmp_path / "srv"), "PORT": str(free_port()), "TASKS": "click_bell beat_block_hammer"}
    r = subprocess.run(["bash", str(repo / "eval.sh"), str(ckpt), "clean", "2", "0"], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert sorted(c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines()) == ["beat_block_hammer", "click_bell"]

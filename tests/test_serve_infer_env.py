"""serve.sh 按 INFER_ENV 选择推理参数文件（docs/specs/eval.md E6）。"""
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run_serve(tmp: Path, **env) -> str:
    repo = tmp / "repo"
    shutil.copytree(REPO / "scripts", repo / "scripts")
    shutil.copytree(REPO / "configs", repo / "configs")
    rt = tmp / "RoboTwin"
    (rt / "experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2").mkdir(parents=True)
    stub = tmp / "py"
    stub.write_text("#!/usr/bin/env bash\necho \"ARGS $* NS=${LINGBOT_NUM_STEPS:-unset}\"\n")
    stub.chmod(0o755)
    (repo / "configs/platform/fake.env").write_text(
        f"ROBOTWIN_ROOT={rt}\nPY={stub}\nLEROBOT_PY={stub}\nRUNTIME={tmp}/rt\nDATA_ROOT={tmp}/d\n"
        "ENV_PROFILE=configs/env/cuda.env\nTRAIN_CONFIG=configs/train/autodl_lora.yaml\n")
    model = tmp / "model"
    model.mkdir()
    (model / "lingbotvla_cli.yaml").write_text("model: {}\n")
    r = subprocess.run(["bash", str(repo / "scripts/cloud/serve.sh"), str(model), "0", "1234"],
                       env={"PATH": "/usr/bin:/bin", "PLATFORM": "fake", **env}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_default_infer_env(tmp_path):
    out = run_serve(tmp_path)
    assert "--use_length 25" in out and "--chunk_ret True" in out


def test_sweep_infer_env_overrides_use_length(tmp_path):
    out = run_serve(tmp_path, INFER_ENV="configs/eval/sweep/ul10.env")
    assert "--use_length 10" in out and "--chunk_ret True" in out


def test_num_steps_exported_only_when_set(tmp_path):
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    assert "NS=5" in run_serve(tmp_path / "a", INFER_ENV="configs/eval/sweep/ns5.env")
    assert "NS=unset" in run_serve(tmp_path / "b", LINGBOT_NUM_STEPS="7")   # 默认参数文件不设，且清掉外部残留

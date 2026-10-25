"""train.sh 的调用顺序（DRY_RUN，docs/specs/submission.md S3）。"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_train_sh_dry_run_order():
    r = subprocess.run(["bash", str(REPO / "train.sh")], env={"PATH": "/usr/bin:/bin", "DRY_RUN": "1"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    cmds = [l[2:] for l in r.stdout.splitlines() if l.startswith("+ ")]
    joined = "\n".join(cmds)
    order = ["setup_env.sh", "fetch_assets.sh", "prepare_data.sh ALL --finalize",
             "TRAIN_CONFIG=configs/train/autodl_vlmlora.yaml",
             "ln -s /root/autodl-tmp/runtime/outputs/vlmlora_run/hf/global_step_7500 /root/autodl-tmp/runtime/recover/global_step_7500",
             "TRAIN_CONFIG=configs/train/autodl_vlmlora_s2.yaml"]
    pos = [joined.index(o) for o in order]
    assert pos == sorted(pos)
    assert "final checkpoint: /root/autodl-tmp/runtime/outputs/vlmlora_run_s2/hf/global_step_12500" in r.stdout


def test_stage2_init_path_matches_train_sh():
    import yaml
    s2 = yaml.safe_load((REPO / "configs/train/autodl_vlmlora_s2.yaml").read_text())
    assert s2["model"]["init_path"] == "/root/autodl-tmp/runtime/recover/global_step_7500"
    s1 = yaml.safe_load((REPO / "configs/train/autodl_vlmlora.yaml").read_text())
    assert s1["run"]["output_dir"] == "/root/autodl-tmp/runtime/outputs/vlmlora_run"
    assert s2["run"]["output_dir"] == "/root/autodl-tmp/runtime/outputs/vlmlora_run_s2"


def test_shutdown_is_opt_in():
    """关机默认关闭：只有 SHUTDOWN=1 时 supervisor / 流水线才执行 /usr/bin/shutdown；train.sh 不开启。"""
    import re
    for rel in ("scripts/autodl/train_supervised.sh", "scripts/autodl/train_after_prep.sh"):
        text = (REPO / rel).read_text()
        assert "NO_SHUTDOWN" not in text, rel
        body = text.split("finish()", 1)[1].split("\n}", 1)[0]
        guard = body.index('"${SHUTDOWN:-0}" != "1"')
        assert guard < body.index("/usr/bin/shutdown"), rel
        code = [l for l in text.splitlines() if not l.lstrip().startswith("#")]
        assert sum("/usr/bin/shutdown" in l for l in code) == 1, rel   # 只在 finish() 内
    train_sh = (REPO / "train.sh").read_text()
    assert "NO_SHUTDOWN" not in train_sh
    train_code = "\n".join(l.split("#", 1)[0] for l in train_sh.splitlines())   # 注释里可以提到开关
    assert not re.search(r"\bSHUTDOWN=1", train_code)

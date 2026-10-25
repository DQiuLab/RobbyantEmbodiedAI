import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tests.conftest import load_tasks, write_template
from tools import package as pk


@pytest.fixture(autouse=True)
def _allow_dirty_tree(monkeypatch):
    # 测试在开发中的工作区运行，允许未提交改动；真实打包时默认拒绝
    monkeypatch.setenv("PACKAGE_ALLOW_DIRTY", "1")

REPO = Path(__file__).resolve().parents[1]
TASKS = load_tasks()


def make_logs(tmp: Path, config: str, n_tasks: int = len(TASKS)) -> Path:
    d = tmp / f"run_{config}"
    (d / "logs").mkdir(parents=True)
    for t in TASKS[:n_tasks]:
        (d / "logs" / f"eval_{config}__{t}.log").write_text("Final success rate: 3/10 = 30.0%\n")
    return d


def make_weights(tmp: Path) -> tuple[Path, Path]:
    w = tmp / "w" / "global_step_12500"
    (w / "robot_configs").mkdir(parents=True)
    (w / "model-00001-of-00001.safetensors").write_bytes(b"x" * 100)
    (w / "norm_stats.json").write_text("{}")
    (w / "robot_configs/robotwin.yaml").write_text(yaml.safe_dump({"norm_stats": "/old/norm_stats.json"}))
    (w / "lingbotvla_cli.yaml").write_text(yaml.safe_dump({"data": {"norm_stats_file": "/old", "robot_config_root": "/old"}}))
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(w)}" for p in sorted(w.rglob("*")) if p.is_file()]
    sha = tmp / "w" / "global_step_12500.sha256"
    sha.write_text("\n".join(lines) + "\n")
    return w, sha


def make_cfg(tmp: Path) -> Path:
    c = yaml.safe_load((REPO / "configs/submission.yaml").read_text())
    template = write_template(tmp / "template" / "results_template.json")
    c.update(team_name="BotLab", team_id="team_test", template_json=str(template))
    p = tmp / "submission.yaml"
    p.write_text(yaml.safe_dump(c, allow_unicode=True))
    return p


def docs(tmp: Path) -> tuple[Path, Path]:
    (tmp / "README.md").write_text("# readme\n"); (tmp / "report.md").write_text("# report\n")
    return tmp / "README.md", tmp / "report.md"


def test_package_with_weights(tmp_path):
    w, sha = make_weights(tmp_path)
    readme, report = docs(tmp_path)
    root = pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                      readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha)
    assert root.name == "BotLab_初赛提交材料"
    res = json.loads((root / "01_评测结果/results.json").read_text())
    assert res["team_id"] == "team_test" and res["results"]["randomized"][TASKS[0]] == {"attempts": 10, "successes": 3}
    code = root / "02_代码材料"
    for f in ("README.md", "train.sh", "eval.sh", "src/stream_merge.py", "tools/results.py", "configs/eval/infer.env"):
        assert (code / f).exists(), f
    assert not (code / "scripts/autodl/bench").exists() and not list(code.rglob("__pycache__"))
    assert len((code / "COMMIT").read_text().strip().split("-")[0]) == 40
    pkg_cfg = yaml.safe_load((code / "configs/submission.yaml").read_text())
    assert (code / pkg_cfg["template_json"]).is_file()
    ck = root / "03_模型权重/checkpoint/global_step_12500"
    assert (ck / "model-00001-of-00001.safetensors").read_bytes() == b"x" * 100
    assert yaml.safe_load((ck / "robot_configs/robotwin.yaml").read_text())["norm_stats"] == str(ck.resolve() / "norm_stats.json")
    assert (root / "04_复现与调优说明/reproduction_report.md").exists()


def test_package_rejects_corrupted_weights(tmp_path):
    w, sha = make_weights(tmp_path)
    (w / "model-00001-of-00001.safetensors").write_bytes(b"y" * 100)
    readme, report = docs(tmp_path)
    with pytest.raises(pk.PackageError, match="sha256 mismatch"):
        pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                   readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha)


def test_package_link_mode_and_partial(tmp_path):
    readme, report = docs(tmp_path)
    link = tmp_path / "LINK.md"; link.write_text("weights at ...\n")
    run = make_logs(tmp_path, "demo_clean", 10)
    with pytest.raises(pk.results.ResultsError):
        pk.package(tmp_path / "out1", [run], readme, report, make_cfg(tmp_path), weights_link=link)
    root = pk.package(tmp_path / "out2", [run], readme, report, make_cfg(tmp_path),
                      weights_link=link, allow_partial=True)
    assert (root / "03_模型权重/checkpoint/LINK.md").read_text() == "weights at ...\n"


def test_package_refuses_existing_output_and_bad_args(tmp_path):
    readme, report = docs(tmp_path)
    link = tmp_path / "LINK.md"; link.write_text("x")
    runs = [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")]
    pk.package(tmp_path / "out", runs, readme, report, make_cfg(tmp_path), weights_link=link)
    with pytest.raises(pk.PackageError, match="already exists"):
        pk.package(tmp_path / "out", runs, readme, report, make_cfg(tmp_path), weights_link=link)
    with pytest.raises(pk.PackageError, match="exactly one"):
        pk.package(tmp_path / "o3", runs, readme, report, make_cfg(tmp_path))


def test_package_refuses_unfilled_placeholders(tmp_path):
    readme, report = docs(tmp_path)
    report.write_text("USE_LENGTH = 【待填】\n")
    link = tmp_path / "LINK.md"; link.write_text("x")
    with pytest.raises(pk.PackageError, match="placeholders"):
        pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean")], readme, report, make_cfg(tmp_path),
                   weights_link=link, allow_partial=True)


def test_make_zip_roundtrip(tmp_path):
    import zipfile
    w, sha = make_weights(tmp_path)
    readme, report = docs(tmp_path)
    root = pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                      readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha)
    z = pk.make_zip(root)
    assert z.name == "BotLab_初赛提交材料.zip"
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        assert all(n.startswith("BotLab_初赛提交材料/") for n in names)
        assert "BotLab_初赛提交材料/01_评测结果/results.json" in names
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
        assert zf.read("BotLab_初赛提交材料/03_模型权重/checkpoint/global_step_12500/model-00001-of-00001.safetensors") == b"x" * 100
    with pytest.raises(pk.PackageError, match="already exists"):
        pk.make_zip(root)


def test_make_zip_small_files_before_big(tmp_path, monkeypatch):
    """小文件排在大分片之前（spec：避免 README 等落到 2 GiB 之后被 unzip 解成 #U 乱码目录）。"""
    import zipfile
    monkeypatch.setattr(pk, "BIG_FILE", 50)
    root = tmp_path / "BotLab_初赛提交材料"
    for rel, n in [("01_a/r.json", 10), ("03_w/a.safetensors", 100), ("03_w/b.yaml", 10), ("04_r/r.md", 10)]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True); (root / rel).write_bytes(b"x" * n)
    with zipfile.ZipFile(pk.make_zip(root)) as zf:
        names = [n.split("/", 1)[1] for n in zf.namelist()]
    assert names == ["01_a/r.json", "03_w/b.yaml", "04_r/r.md", "03_w/a.safetensors"]


def test_package_link_weights_hardlinks_shards_only(tmp_path):
    w, sha = make_weights(tmp_path)
    readme, report = docs(tmp_path)
    root = pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                      readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha, link_weights=True)
    ck = root / "03_模型权重/checkpoint/global_step_12500"
    shard = "model-00001-of-00001.safetensors"
    assert (ck / shard).stat().st_ino == (w / shard).stat().st_ino
    assert (ck / "lingbotvla_cli.yaml").stat().st_ino != (w / "lingbotvla_cli.yaml").stat().st_ino
    # relocate 只改包内副本，源目录仍与清单一致
    for rel, digest in pk.read_manifest(sha).items():
        assert hashlib.sha256((w / rel).read_bytes()).hexdigest() == digest, rel
    assert yaml.safe_load((ck / "robot_configs/robotwin.yaml").read_text())["norm_stats"] == str(ck.resolve() / "norm_stats.json")


def test_package_link_weights_refuses_copy_fallback(tmp_path, monkeypatch):
    import errno
    w, sha = make_weights(tmp_path)
    readme, report = docs(tmp_path)

    def no_link(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")
    monkeypatch.setattr(pk.os, "link", no_link)
    with pytest.raises(pk.PackageError, match="hard link"):
        pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                   readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha, link_weights=True)


def test_make_zip_dest_dir_and_verify_weights(tmp_path):
    w, sha = make_weights(tmp_path)
    readme, report = docs(tmp_path)
    root = pk.package(tmp_path / "out", [make_logs(tmp_path, "demo_clean"), make_logs(tmp_path, "demo_randomized")],
                      readme, report, make_cfg(tmp_path), weights=w, weights_sha=sha, link_weights=True)
    dest = tmp_path / "other_disk"
    dest.mkdir()
    z = pk.make_zip(root, dest)
    assert z == dest / "BotLab_初赛提交材料.zip" and not (root.parent / z.name).exists()
    assert pk.verify_zip_weights(z, "global_step_12500", sha) == 4
    bad = tmp_path / "bad.sha256"
    bad.write_text(sha.read_text().replace(hashlib.sha256(b"x" * 100).hexdigest(), "0" * 64))
    with pytest.raises(pk.PackageError, match="sha256 mismatch in zip"):
        pk.verify_zip_weights(z, "global_step_12500", bad)
    extra = tmp_path / "extra.sha256"
    extra.write_text(sha.read_text() + f"{'1' * 64}  ./missing.bin\n")
    with pytest.raises(pk.PackageError, match="missing"):
        pk.verify_zip_weights(z, "global_step_12500", extra)

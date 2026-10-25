"""Assemble the preliminary-round submission package (docs/specs/submission.md S2).

Requires the organizers' results template (configs/submission.yaml `template_json`; get it from the
challenge page, it is not distributed with this repository).

<out>/<team_name>_初赛提交材料/
  01_评测结果/results.json            built from eval logs with tools/results.py (S1 checks, incl. infra-error rejection)
  02_代码材料/                        code whitelist + README + train.sh + eval.sh (results template copied in)
  03_模型权重/checkpoint/             weights dir (sha256-verified copy, paths relocated) or LINK.md
  04_复现与调优说明/reproduction_report.md

Usage:
  python tools/package.py --out submission --run-dir <clean_logs_run> --run-dir <rand_logs_run> \
      --readme docs/submission/README.md --report docs/submission/reproduction_report.md \
      (--weights <hf_dir> --weights-sha <sha256_manifest> | --weights-link <LINK.md>) [--allow-partial]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import results  # noqa: E402
from src.relocate_snapshot import relocate  # noqa: E402

CODE_DIRS = ["scripts", "src", "configs", "tools", "tests"]
CODE_FILES = ["pytest.ini", "train.sh", "eval.sh"]
EXCLUDE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "bench")
TEMPLATE_IN_PACKAGE = "configs/reference/results_template.json"
BIG_FILE = 64 << 20   # zip 中排在最后的大文件（权重分片）阈值，见 make_zip
PLACEHOLDER = "【待填"   # README / 报告里未填的占位符；打包时必须清零
REQUIRED = [
    "01_评测结果/results.json",
    "02_代码材料/README.md", "02_代码材料/train.sh", "02_代码材料/eval.sh", "02_代码材料/configs", "02_代码材料/src",
    "03_模型权重/checkpoint",
    "04_复现与调优说明/reproduction_report.md",
]


class PackageError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    # 1 MiB 块并对 ENOMEM 重试：某些挂载文件系统大块读取偶发 Errno 12（ENOMEM，10-20）
    import errno
    import time
    for attempt in range(5):
        h = hashlib.sha256()
        try:
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError as e:
            if e.errno != errno.ENOMEM or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def read_manifest(path: Path) -> dict[str, str]:
    """sha256sum output ("<hash>  ./rel/path") -> {rel/path: hash}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.strip().removeprefix("./")] = digest
    return out


def _link_shard(src: str, dst: str) -> None:
    """copytree copy_function for --link-weights: hard-link *.safetensors, copy the rest (relocate rewrites yaml in place)."""
    if not src.endswith(".safetensors"):
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError as e:   # 不退回复制：那正是要避免的 12 GB 写入
        raise PackageError(f"cannot hard link {src} -> {dst} ({e}); package dir must be on the weights' filesystem") from e


def copy_weights(src: Path, manifest: Path, dest: Path, link: bool = False) -> int:
    expected = read_manifest(manifest)
    files = {str(p.relative_to(src)): p for p in src.rglob("*") if p.is_file()}
    if set(files) != set(expected):
        raise PackageError(f"weights file set differs from manifest: extra={sorted(set(files) - set(expected))} "
                           f"missing={sorted(set(expected) - set(files))}")
    for rel, p in files.items():
        if sha256(p) != expected[rel]:
            raise PackageError(f"sha256 mismatch in source weights: {rel}")
    shutil.copytree(src, dest, copy_function=_link_shard if link else shutil.copy2)
    for rel in files:
        if sha256(dest / rel) != expected[rel]:
            raise PackageError(f"sha256 mismatch after copy: {rel}")
    relocate(dest)   # 复制后改写权重内配置路径（只改 yaml，模型分片不变）
    return len(files)


def git_commit() -> str:
    """HEAD commit of this repo; refuses a dirty tree so the packaged code matches the recorded commit."""
    import subprocess
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty and not os.environ.get("PACKAGE_ALLOW_DIRTY"):
        raise PackageError(f"uncommitted changes; commit first so COMMIT matches the packaged code:\n{dirty}")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


def make_zip(root: Path, dest_dir: Path | None = None) -> Path:
    """<root>.zip (in dest_dir, default next to root) with root as the top-level folder; STORED (weights are
    incompressible), ZIP64; verified by re-reading."""
    zpath = (dest_dir or root.parent) / f"{root.name}.zip"
    if zpath.exists():
        raise PackageError(f"{zpath} already exists (not overwritten)")
    # 小文件在前、大分片在后：偏移超过 2 GiB 的条目带 ZIP64 字段，locale 缺失时 Debian unzip 会把这类条目的中文路径
    # 解成 #U 乱码目录（EXP-20261024-01）；这样 README、报告和权重配置都在 2 GiB 以内
    files = sorted(sorted(p for p in root.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size >= BIG_FILE)
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in files:
            z.write(p, str(Path(root.name) / p.relative_to(root)))
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        if bad is not None:
            raise PackageError(f"zip CRC check failed: {bad}")
        sizes = {i.filename: i.file_size for i in z.infolist()}
    for p in files:
        name = str(Path(root.name) / p.relative_to(root))
        if sizes.get(name) != p.stat().st_size:
            raise PackageError(f"zip size mismatch: {name}")
    return zpath


def verify_zip_weights(zpath: Path, weights_name: str, manifest: Path) -> int:
    """Stream-hash every weights file inside the zip against the manifest (instead of unzipping another 12 GB).
    Relocated *.yaml differ from the manifest by design: check their path fields point into the packaged dir."""
    expected = read_manifest(manifest)
    with zipfile.ZipFile(zpath) as z:
        top = z.namelist()[0].split("/", 1)[0]
        prefix = f"{top}/03_模型权重/checkpoint/{weights_name}/"
        names = {n.removeprefix(prefix) for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")}
        if names != set(expected):
            raise PackageError(f"zip weights file set differs from manifest: extra={sorted(names - set(expected))} "
                               f"missing={sorted(set(expected) - names)}")
        for rel, digest in expected.items():
            with z.open(prefix + rel) as f:
                if rel.endswith(".yaml"):
                    doc = yaml.safe_load(f.read())
                    paths = [doc.get("norm_stats")] if "norm_stats" in doc else \
                        [doc["data"]["norm_stats_file"], doc["data"]["robot_config_root"]]
                    if not all(f"/03_模型权重/checkpoint/{weights_name}/" in str(p) for p in paths):
                        raise PackageError(f"relocated paths in {rel} do not point into {weights_name}: {paths}")
                    continue
                h = hashlib.sha256()
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise PackageError(f"sha256 mismatch in zip: {rel}")
    return len(expected)


def package(out: Path, run_dirs: list[Path], readme: Path, report: Path, config: Path = REPO / "configs/submission.yaml",
            weights: Path | None = None, weights_sha: Path | None = None, weights_link: Path | None = None,
            allow_partial: bool = False, link_weights: bool = False) -> Path:
    if (weights is None) == (weights_link is None):
        raise PackageError("give exactly one of --weights (with --weights-sha) or --weights-link")
    if weights is not None and weights_sha is None:
        raise PackageError("--weights requires --weights-sha")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    team_name = cfg["team_name"]
    if not team_name or team_name.startswith("REPLACE"):
        raise PackageError("team_name not set in configs/submission.yaml")
    template = REPO / cfg["template_json"]
    for doc in (readme, report):
        if PLACEHOLDER in doc.read_text(encoding="utf-8"):
            raise PackageError(f"{doc} still contains placeholders ({PLACEHOLDER}...】)")
    root = out / f"{team_name}_初赛提交材料"
    if root.exists():
        raise PackageError(f"{root} already exists; choose another --out (not overwritten)")

    data = results.build(template, str(cfg["team_id"]), results.collect(run_dirs), allow_partial)
    (root / "01_评测结果").mkdir(parents=True)
    (root / "01_评测结果/results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code = root / "02_代码材料"
    for d in CODE_DIRS:
        shutil.copytree(REPO / d, code / d, ignore=EXCLUDE)
    for f in CODE_FILES:
        shutil.copy2(REPO / f, code / f)
    shutil.copy2(readme, code / "README.md")
    (code / "COMMIT").write_text(git_commit() + "\n", encoding="utf-8")   # 报告"代码版本"引用此文件
    shutil.copy2(template, code / TEMPLATE_IN_PACKAGE)
    pkg_cfg = dict(cfg, template_json=TEMPLATE_IN_PACKAGE)   # 提交包内不含赛事原始材料目录
    (code / "configs/submission.yaml").write_text(yaml.safe_dump(pkg_cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    ckdir = root / "03_模型权重/checkpoint"
    if weights is not None:
        ckdir.parent.mkdir(parents=True)
        copy_weights(weights, weights_sha, ckdir / weights.name, link=link_weights)
        shutil.copy2(weights_sha, ckdir / f"{weights.name}.sha256")
    else:
        ckdir.mkdir(parents=True)
        shutil.copy2(weights_link, ckdir / "LINK.md")

    (root / "04_复现与调优说明").mkdir()
    shutil.copy2(report, root / "04_复现与调优说明/reproduction_report.md")

    missing = [r for r in REQUIRED if not (root / r).exists()]
    if missing:
        raise PackageError(f"missing required items: {missing}")
    return root


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "submission")
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--readme", type=Path, default=REPO / "docs/submission/README.md")
    ap.add_argument("--report", type=Path, default=REPO / "docs/submission/reproduction_report.md")
    ap.add_argument("--config", type=Path, default=REPO / "configs/submission.yaml")
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--weights-sha", type=Path)
    ap.add_argument("--weights-link", type=Path)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--link-weights", action="store_true", help="hard-link *.safetensors instead of copying (same filesystem)")
    ap.add_argument("--zip", action="store_true", help="also write <root>.zip (STORED, ZIP64) and verify it")
    ap.add_argument("--zip-dir", type=Path, help="write the zip here instead of next to the package (e.g. another disk)")
    a = ap.parse_args(argv)
    try:
        root = package(a.out, a.run_dir, a.readme, a.report, a.config, a.weights, a.weights_sha, a.weights_link, a.allow_partial,
                       a.link_weights)
    except (PackageError, results.ResultsError) as e:
        print(f"package: {e}", file=sys.stderr)
        return 1
    print(f"package written: {root}")
    if a.zip:
        try:
            zpath = make_zip(root, a.zip_dir)
            print(f"zip written: {zpath} ({zpath.stat().st_size / 2**30:.2f} GiB), CRC re-read OK", flush=True)
            if a.weights is not None:
                n = verify_zip_weights(zpath, a.weights.name, a.weights_sha)
                print(f"zip weights sha256 OK ({n} files vs {a.weights_sha.name})")
        except PackageError as e:
            print(f"package: {e}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

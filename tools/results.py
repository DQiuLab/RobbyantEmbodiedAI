"""Build and validate the official results.json from run_clean_benchmark.py outputs.

Spec: docs/specs/submission.md §S1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

# Same pattern as run_clean_benchmark.read_result (AMD reference sample Robotwin-radeon-cloud listed in the challenge intro).
FINAL_RATE = re.compile(r"Final success rate:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%")
SETTING_OF_CONFIG = {"demo_clean": "clean", "demo_randomized": "randomized"}
SETTINGS = ("clean", "randomized")
# 未填写的 team_id：本仓库配置的占位符与主办方模板里的占位符
PLACEHOLDER_TEAM_IDS = frozenset({"<team_id>", "replace_with_your_team_id"})
# 评测机资源错误：RoboTwin 把这类 episode 记为失败，但不是策略表现（docs/specs/submission.md S1 第 8 条）
INFRA_ERROR = re.compile(r"Policy rollout error|cannot create buffer|out of memory")


class ResultsError(ValueError):
    pass


def parse_log(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    matches = FINAL_RATE.findall(text)
    if not matches:
        raise ResultsError(f"no 'Final success rate' in {path}")
    infra = INFRA_ERROR.search(text)
    if infra:
        raise ResultsError(f"infrastructure error {infra.group(0)!r} in {path}; rerun this task with fewer clients")
    successes, attempts, _ = matches[-1]
    return int(successes), int(attempts)


def collect(run_dirs: list[Path]) -> dict[str, dict[str, dict[str, int]]]:
    collected: dict[str, dict[str, dict[str, int]]] = {s: {} for s in SETTINGS}
    sources: dict[tuple[str, str], Path] = {}
    for run_dir in run_dirs:
        for log in sorted((Path(run_dir) / "logs").glob("eval_*__*.log")):
            config, _, task = log.name[len("eval_"):-len(".log")].partition("__")
            if config not in SETTING_OF_CONFIG:
                raise ResultsError(f"unknown task_config {config!r} in {log}")
            setting = SETTING_OF_CONFIG[config]
            key = (setting, task)
            if key in sources:
                raise ResultsError(f"duplicate result for {setting}/{task}: {sources[key]} and {log}")
            successes, attempts = parse_log(log)
            sources[key] = log
            collected[setting][task] = {"attempts": attempts, "successes": successes}
    return collected


def _template(template_path: Path) -> dict:
    return json.loads(Path(template_path).read_text(encoding="utf-8"))


def validate(data: dict, template_path: Path) -> None:
    template = _template(template_path)
    if data.get("schema_version") != template["schema_version"]:
        raise ResultsError("schema_version does not match template")
    team_id = data.get("team_id")
    if not isinstance(team_id, str) or not team_id or team_id in PLACEHOLDER_TEAM_IDS:
        raise ResultsError("team_id must be set to the real Tianchi team id")
    for setting in SETTINGS:
        expected = list(template["results"][setting])
        got = data.get("results", {}).get(setting, {})
        if set(got) != set(expected):
            extra = sorted(set(got) - set(expected))
            missing = sorted(set(expected) - set(got))
            raise ResultsError(f"{setting}: task set differs from template; extra={extra} missing={missing}")
        for task, entry in got.items():
            if set(entry) != {"attempts", "successes"}:
                raise ResultsError(f"{setting}/{task}: entry must have exactly attempts and successes")
            attempts, successes = entry["attempts"], entry["successes"]
            if type(attempts) is not int or type(successes) is not int:
                raise ResultsError(f"{setting}/{task}: attempts/successes must be integers")
            if attempts < 0 or successes < 0 or successes > attempts:
                raise ResultsError(f"{setting}/{task}: invalid counts {successes}/{attempts}")


def build(template_path: Path, team_id: str, collected: dict, allow_partial: bool = False) -> dict:
    template = _template(template_path)
    out = {"schema_version": template["schema_version"], "team_id": team_id, "results": {}}
    missing = []
    for setting in SETTINGS:
        expected = list(template["results"][setting])
        unknown = sorted(set(collected.get(setting, {})) - set(expected))
        if unknown:
            raise ResultsError(f"{setting}: tasks not in template: {unknown}")
        out["results"][setting] = {}
        for task in expected:
            entry = collected.get(setting, {}).get(task)
            if entry is None:
                missing.append(f"{setting}/{task}")
                entry = {"attempts": 0, "successes": 0}
            out["results"][setting][task] = dict(entry)
    if missing and not allow_partial:
        raise ResultsError(f"missing {len(missing)} results: {missing[:5]}...")
    if missing:
        print(f"warning: {len(missing)} results missing, filled with 0/0: {missing}", file=sys.stderr)
    validate(out, template_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/submission.yaml"))
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = build(Path(config["template_json"]), str(config["team_id"]), collect(args.run_dir), args.allow_partial)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for setting in SETTINGS:
        rows = data["results"][setting].values()
        s, n = sum(r["successes"] for r in rows), sum(r["attempts"] for r in rows)
        print(f"{setting}: {s}/{n}" + (f" = {100 * s / n:.2f}%" if n else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

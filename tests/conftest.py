"""共享 fixture：在 tmp 里生成与提交结果 JSON 同结构的模板（主办方模板不随仓库分发）。"""
import json
from pathlib import Path

import pytest

TASKS_FILE = Path(__file__).resolve().parent / "fixtures/robotwin_tasks.txt"   # RoboTwin 2.0 的 50 个公开任务名，按字母序
SCHEMA_VERSION = 1
TEMPLATE_TEAM_ID = "<team_id>"


def load_tasks() -> list[str]:
    return [line.strip() for line in TASKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_template(path: Path) -> Path:
    tasks = load_tasks()
    data = {
        "schema_version": SCHEMA_VERSION,
        "team_id": TEMPLATE_TEAM_ID,
        "results": {s: {t: {"attempts": 0, "successes": 0} for t in tasks} for s in ("clean", "randomized")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def robotwin_tasks() -> list[str]:
    return load_tasks()


@pytest.fixture(scope="session")
def results_template(tmp_path_factory) -> Path:
    return write_template(tmp_path_factory.mktemp("template") / "results_template.json")

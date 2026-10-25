"""eval_subset.sh 多客户端：用桩服务与桩 eval_policy.sh 在临时目录里跑（docs/specs/eval.md E5）。"""
import shutil
import socket
import time
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STUB_EVAL = """#!/usr/bin/env bash
# 与真实 eval_policy.sh 一样以 eval_policy_xpolicylab.py 命名的进程运行（供服务方等待已挂接客户端）
if [[ -z "${STUB_INNER:-}" ]]; then STUB_INNER=1 exec -a "python eval_policy_xpolicylab.py $*" bash "$0" "$@"; fi
# 桩：记录调用参数，随机耗时，输出与 RoboTwin 相同格式的结果行
task=""; seed=""; port=""
while [[ $# -gt 0 ]]; do case "$1" in --task_name) task=$2;; --seed) seed=$2;; --port) port=$2;; esac; shift; done
echo "$task seed=$seed port=$port unbuf=${PYTHONUNBUFFERED:-}" >> "$CALLS"
# HANG_TASKS 中的任务第一次调用时不输出并挂住（模拟 websocket 卡死）
if [[ " ${HANG_TASKS:-} " == *" $task "* && ! -f "$CALLS.hung.$task" ]]; then
  touch "$CALLS.hung.$task"; trap 'echo "GOT_ABRT fh=${PYTHONFAULTHANDLER:-}"; exit 134' ABRT
  sleep 600 & echo $! > "$CALLS.sleep_pid.$task"; wait
fi
sleep 0.$((RANDOM % 3))
echo "Final success rate: 1/1 = 100.0%"
"""

STUB_SERVE = """#!/usr/bin/env bash
mkdir -p "$SERVE_DIR" && touch "$SERVE_DIR/healthz"
exec python3 -m http.server "$3" --bind 127.0.0.1 --directory "$SERVE_DIR"
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_sandbox(tmp: Path) -> Path:
    repo = tmp / "repo"
    shutil.copytree(REPO / "scripts", repo / "scripts")
    shutil.copytree(REPO / "configs", repo / "configs")
    shutil.copytree(REPO / "src", repo / "src", ignore=shutil.ignore_patterns("__pycache__"))
    rt = tmp / "RoboTwin"
    (rt / "scripts").mkdir(parents=True)
    (rt / "scripts" / "eval_policy.sh").write_text(STUB_EVAL)
    (repo / "scripts" / "cloud" / "serve.sh").write_text(STUB_SERVE)
    (repo / "configs" / "platform" / "fake.env").write_text(
        f"ROBOTWIN_ROOT={rt}\nPY=python3\nLEROBOT_PY=python3\nRUNTIME={tmp}/runtime\n"
        f"DATA_ROOT={tmp}/data\nENV_PROFILE=configs/env/cuda.env\nTRAIN_CONFIG=configs/train/autodl_lora.yaml\n")
    return repo


def run_eval(repo: Path, tmp: Path, tasks: list[str], **env) -> subprocess.CompletedProcess:
    e = {"PATH": "/usr/bin:/bin", "PLATFORM": "fake", "CALLS": str(tmp / "calls.txt"),
         "SERVE_DIR": str(tmp / "srv"), **{k: str(v) for k, v in env.items()}}
    return subprocess.run(["bash", str(repo / "scripts/cloud/eval_subset.sh"), "/m", "run", "demo_clean", "1", "0", *tasks],
                          env=e, capture_output=True, text=True, timeout=120)


def test_clients_run_every_task_once_with_seed_zero(tmp_path):
    repo = make_sandbox(tmp_path)
    tasks = [f"task{i}" for i in range(12)]
    r = run_eval(repo, tmp_path, tasks, CLIENTS=4, PORT=free_port())
    assert r.returncode == 0, r.stderr
    calls = (tmp_path / "calls.txt").read_text().split("\n")
    calls = [c for c in calls if c]
    assert sorted(c.split()[0] for c in calls) == sorted(tasks)          # 每个任务恰好一次
    assert all("seed=0" in c for c in calls)                              # seed 与串行一致
    assert all("unbuf=1" in c for c in calls)                             # 日志不缓冲
    logs = tmp_path / "runtime/outputs/run/logs"
    assert sorted(p.name for p in logs.glob("eval_*.log")) == sorted(f"eval_demo_clean__{t}.log" for t in tasks)
    assert not list(logs.glob("*.part"))
    assert r.stdout.count("Final success rate") == len(tasks)


def test_resume_skips_finished_and_reruns_partial(tmp_path):
    repo = make_sandbox(tmp_path)
    logs = tmp_path / "runtime/outputs/run/logs"
    logs.mkdir(parents=True)
    (logs / "eval_demo_clean__a.log").write_text("Final success rate: 0/1 = 0.0%\n")
    (logs / "eval_demo_clean__b.log").write_text("crashed midway\n")
    r = run_eval(repo, tmp_path, ["a", "b", "c"], CLIENTS=2, PORT=free_port())
    assert r.returncode == 0, r.stderr
    ran = sorted(c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines())
    assert ran == ["b", "c"]


def test_attach_reuses_server_and_skips_any_existing_log(tmp_path):
    repo = make_sandbox(tmp_path)
    port = free_port()
    srv = subprocess.Popen(["bash", str(repo / "scripts/cloud/serve.sh"), "/m", "0", str(port)],
                           env={"PATH": "/usr/bin:/bin", "SERVE_DIR": str(tmp_path / "srv")})
    try:
        subprocess.run(["bash", "-c", f"until curl -fsS http://127.0.0.1:{port}/healthz >/dev/null 2>&1; do sleep 0.2; done"], timeout=30)
        logs = tmp_path / "runtime/outputs/run/logs"
        logs.mkdir(parents=True)
        (logs / "eval_demo_clean__a.log").write_text("in progress by the old serial loop\n")
        r = run_eval(repo, tmp_path, ["a", "b", "c"], CLIENTS=2, ATTACH=1, PORT=port)
        assert r.returncode == 0, r.stderr
        ran = sorted(c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines())
        assert ran == ["b", "c"]
        assert srv.poll() is None   # ATTACH 模式退出时不杀已有服务
    finally:
        srv.kill()


def test_attach_fails_without_server(tmp_path):
    repo = make_sandbox(tmp_path)
    r = run_eval(repo, tmp_path, ["a"], ATTACH=1, PORT=free_port())
    assert r.returncode != 0 and "no server" in r.stderr


def test_order_from_runs_long_tasks_first(tmp_path):
    repo = make_sandbox(tmp_path)
    hist = tmp_path / "old.out"
    hist.write_text("[eval] demo_clean/b: Final success rate: 0/1 = 0.0% (900 s)\n"
                    "[eval] demo_clean/c: Final success rate: 1/1 = 100.0% (60 s)\n")
    r = run_eval(repo, tmp_path, ["a", "b", "c"], CLIENTS=1, PORT=free_port(), ORDER_FROM=str(tmp_path / "*.out"))
    assert r.returncode == 0, r.stderr
    ran = [c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines()]
    assert ran == ["b", "c", "a"]


def test_hang_watchdog_kills_and_reruns_task_once(tmp_path):
    repo = make_sandbox(tmp_path)
    r = run_eval(repo, tmp_path, ["a", "b"], CLIENTS=2, PORT=free_port(), HANG_TASKS="b", HANG_SEC=2, HANG_POLL=0.5)
    assert r.returncode == 0, r.stderr
    ran = sorted(c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines())
    assert ran == ["a", "b", "b"]                       # b 卡死一次后整任务重跑
    assert "HANG demo_clean/b attempt 1" in r.stderr
    logs = tmp_path / "runtime/outputs/run/logs"
    assert "Final success rate" in (logs / "eval_demo_clean__b.log").read_text()
    hang = list(logs.glob("eval_demo_clean__b.log.hang.*"))
    assert len(hang) == 1 and "GOT_ABRT fh=1" in hang[0].read_text()   # 先 SIGABRT（faulthandler 打栈），再杀
    pid = int((tmp_path / "calls.txt.sleep_pid.b").read_text())
    assert subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode != 0   # 卡死进程（整个进程组）已被杀


def test_attach_skips_task_in_progress_part_log(tmp_path):
    repo = make_sandbox(tmp_path)
    port = free_port()
    srv = subprocess.Popen(["bash", str(repo / "scripts/cloud/serve.sh"), "/m", "0", str(port)],
                           env={"PATH": "/usr/bin:/bin", "SERVE_DIR": str(tmp_path / "srv")})
    try:
        subprocess.run(["bash", "-c", f"until curl -fsS http://127.0.0.1:{port}/healthz >/dev/null 2>&1; do sleep 0.2; done"], timeout=30)
        logs = tmp_path / "runtime/outputs/run/logs"
        logs.mkdir(parents=True)
        (logs / "eval_demo_clean__a.log.part").write_text("running in the main loop\n")
        r = run_eval(repo, tmp_path, ["a", "b"], ATTACH=1, PORT=port)
        assert r.returncode == 0, r.stderr
        ran = [c.split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines()]
        assert ran == ["b"]
    finally:
        srv.kill()


def test_main_loop_keeps_server_until_attached_clients_finish(tmp_path):
    repo = make_sandbox(tmp_path)
    port = free_port()
    env = {"PATH": "/usr/bin:/bin", "PLATFORM": "fake", "CALLS": str(tmp_path / "calls.txt"),
           "SERVE_DIR": str(tmp_path / "srv"), "PORT": str(port)}
    # 一个"外部挂接"的慢客户端：用同样的命令行特征模拟
    slow = subprocess.Popen(["bash", "-c", f"exec -a 'python eval_policy_xpolicylab.py --port {port} ' sleep 4"])
    t0 = time.time()
    r = subprocess.run(["bash", str(repo / "scripts/cloud/eval_subset.sh"), "/m", "run", "demo_clean", "1", "0", "a"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 >= 3.5          # 主循环等挂接客户端结束后才退出
    slow.wait()


def test_clients_capped_for_randomized(tmp_path):
    repo = make_sandbox(tmp_path)
    cap = int([l.split("=")[1] for l in (repo / "configs/eval/clients.env").read_text().splitlines()
               if l.startswith("MAX_CLIENTS_RANDOMIZED=")][0])
    e = {"PATH": "/usr/bin:/bin", "PLATFORM": "fake", "CALLS": str(tmp_path / "calls.txt"),
         "SERVE_DIR": str(tmp_path / "srv"), "CLIENTS": "8", "PORT": str(free_port())}
    r = subprocess.run(["bash", str(repo / "scripts/cloud/eval_subset.sh"), "/m", "run", "demo_randomized", "1", "0", "a", "b"],
                       env=e, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert f"CLIENTS=8 > {cap} for demo_randomized; using {cap}" in r.stderr

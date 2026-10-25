# Spec: 评测

## E1. 协议（与官方一致，不修改）

- 设定：`demo_clean`（`eval_instruction: seen`）与 `demo_randomized`（`eval_instruction: unseen`），来源 RoboTwin `env_cfg/task_config/*.yml`@266f3aa。
- 判定：`check_success()` 返回值（赛事官方论坛 RD 4.2）。
- 任务集合：RoboTwin `env_cfg/eval/all_tasks.yml` 50 任务，与官方 JSON 模板任务集合一致（2026-10-11 已核对）。
- 自测环境：CUDA（AutoDL RTX 4080 SUPER）+ MPLib + `expert_check=true` + `accept_expert_info_on_failure=true`（专家失败时仍保留该场景做策略评估，与赛事官方 Q&A 13 一致），`--seed 0`，结果写入 README（初版 AMD 方案为 ROCm + MPLib）。

## E2. 工具

- 全量（初版 AMD 方案，未用于最终自测）：AMD 参考样例的 `run_clean_benchmark.py`（`--task-config demo_clean|demo_randomized|both`，`--resume`，按任务动态分配 GPU）。只支持全部 50 任务。
- 子集：逐任务调用 RoboTwin `scripts/eval_policy.sh`（`scripts/cloud/eval_subset.sh` 封装），用于 sanity 与推理参数对比。
- 汇总：`tools/results.py`。

## E3. 评测预算（初版 AMD 方案，未用于最终权重；估时依据 AMD 参考样例 Reproduce_Guide 记录的 8× W7900 实测 15 h 12 min / 1000 ep，我们未在 AMD 上运行）

| 用途 | 规模 | 估时 |
|---|---|---:|
| 阶段一 sanity | 10 任务 × 10 ep，clean | ~1.5 h |
| 中途方向性 | 50 任务 × 10 ep × 2 设定 | ~15 h |
| 推理参数对比 | 10 任务 × 10 ep × 每组参数 | ~1.5 h/组 |
| 最终自测 | 50 任务 × 20 ep × 2 设定 | ~30 h |

## E4. 推理可调参数（单一权重，均需在 README 披露，RD 2.4）

| 参数 | 位置 | AMD 默认 |
|---|---|---|
| `use_length`（每个 50 步 chunk 实际执行步数） | `launch_official_server.sh` → `LingbotVLAV2Policy` | 25 |
| 精度 bf16 / fp32 | `LingbotVLAV2Policy(use_bf16, use_fp32)` | bf16 |
| 去噪步数 `num_steps` | 模型 config | 10 |

禁止：多 seed 投票、按任务路由、失败重置（`docs/rules-checklist.md` C）。

## E5. 单卡多客户端（2026-10-17，D-14）

事实（代码 + 实测，10-17）：
- 服务端已是 chunk 模式：`deploy.lingbot_vla_v2_policy` 的 argparse 默认 `--chunk_ret True`，每次调用都前向一次并返回前 `USE_LENGTH`=25 步动作；官方客户端（`eval_policy_xpolicylab.py` 的 `direct_lingbot` 分支）整段执行。日志 `calls=16`/400 步印证。调用之间不保留状态（`global_step`/`last_action_chunk` 在 chunk 模式下每次被覆盖），`reset` 只按 `robo_name` 重建同一份 FeatureTransform。
- `WebsocketPolicyServer._handler` 在事件循环里同步调用 `infer`，请求天然串行，多个客户端连同一服务不会交错。
- 单个 episode 推理耗时约 19 s / 墙钟约 3 min，GPU 大部分时间空闲；仿真客户端每个占 ~2.2–2.7 GB 显存（渲染）+ 1 核 CPU。容器配额 16 核、62 GB 内存（当时的评测实例；复现演练的克隆实例为 12 核）。

规格：
- `configs/eval/infer.env` 显式写 `CHUNK_RET=True`（与原默认相同，不改变行为，便于 README 披露），`serve.sh` 透传。
- `eval_subset.sh` 支持 `CLIENTS=N`：一个服务 + N 个客户端进程，按任务动态领取（`mkdir` 原子锁，`scripts/lib/task_claim.sh`），每个任务仍是独立的 `eval_policy.sh --seed 0` 调用，seed→episode 映射与串行完全一致。不使用 `--eval_batch true`（其 worker 走 XPolicyLab 路径且改变 seed 分配）。
- 日志先写 `<log>.part`，完成后改名为 `<log>`；已有含 `Final success rate` 的日志则跳过（可续跑）。
- `ORDER_FROM="<glob>"`：用历史 `.out` 的单 episode 耗时把长任务排在前面先跑（`src/task_order.py`），减少收尾时只剩一个客户端在跑的空等（10-17 全局 19000 randomized：put_bottles_dustbin 单独拖了约 20 min）。
- **卡死看门狗（10-18）**：评测偶发某个任务长时间无输出（10-18 clean 30 ep 中 blocks_ranking_size 卡了 6.6 h、move_playingcard_away 卡了 3 h，当时只从进程状态看到主线程停在 `wait_woken`，像是阻塞在 websocket recv 等服务回复，没有调用栈，原因未确认）。10-18 晚用 faulthandler 抓到的调用栈显示卡在专家检查阶段的 MPLib `IK` / `plan_qpos`（策略执行之前），IK 带随机初值，非确定性，整任务重跑有效。`HANG_SEC`（默认 3600；10-19 前为 1200，会误杀专家规划超过 20 分钟的慢任务，如 randomized move_can_pot 单 episode 约 22 分钟）秒内任务日志没有更新（`PYTHONUNBUFFERED=1` 后每步都会写日志），就先对进程组发 SIGABRT（`PYTHONFAULTHANDLER=1`，Python 把各线程调用栈写进日志，用于定位卡在哪里），再杀掉整个进程组，从第一个 seed 起整任务重跑，最多 `HANG_RETRIES`（默认 2）次（即一次运行内最多中止 3 次），仍卡就留给续跑（10-18：pick_diverse_bottles randomized 在同一位置 3 次中卡 2 次，非确定性，重跑有效）。判据只看日志是否停止更新，与 episode 成败无关；卡死那次的部分日志保留为 `.hang.<ts>`，不计入结果。
- `ATTACH=1`：不起服务，复用 `PORT` 上已在跑的服务，任务倒序领取，且跳过任何已存在的日志（含正在写的 `.part`）（供旧版串行循环运行中临时加客户端）。
- 容量：32 GB 卡上 1 服务（12.5 GB）+ N 客户端（每个按峰值 ~2.7 GB 计）→ N ≤ 6–7。10-17 实测：5 个客户端时推理延迟 p95 ≤ 2.1 s、episode 墙钟不变；7 个客户端时吞吐约为原"2 服务 × 1 客户端"的 3.5 倍。
- **容量修正（10-17 21:00 实测）**：每个客户端连续跑很多 episode 时渲染显存会涨到 2.1–2.7 GB（之后的 clean 扫描中最高约 3.1 GB，见 `configs/eval/clients.env`）。8 个客户端时总显存打到 32.2 GB，出现 `RuntimeError: cannot create buffer`（RoboTwin 记为 episode 失败）和客户端卡死。**上限定为 N=6**（12.5 + 6×2.7 ≈ 28.7 GB）。**randomized 场景单客户端最高 3.8 GB**（随机背景与杂物，10-18 16:45 实测 6 客户端到 30.9 GB），randomized 上限先定为 N=4，10-18 20:10 又测到单客户端最高 5.5 GB（place_a2b_*），改为 **N=3**（≈ 29 GB）。上限写在 `configs/eval/clients.env`，`eval_subset.sh` 超限时自动截断。开发期间人工监控 rollout/渲染错误、显存 > 30.5 GB、任务日志 15 min 未更新三项指标；`tools/results.py` 拒收含这类错误的日志（submission S1 第 8 条）。
- 合规：仍是单一权重、无路由、seed 不变、不挑结果（`docs/rules-checklist.md` C）；不修改 RoboTwin 评测代码。

## E6. 推理参数扫描（2026-10-18）

- 目的：在单一权重（全局 20000）上找更好的推理参数（赛事官方论坛 RD 2.4 允许，须在 README 披露）。选定后，最终自测与 `eval.sh` 都用同一组参数（RD 4.2：官方复测用选手 eval.sh 的参数）。
- 子集：`configs/eval/sweep_tasks.txt`，16 个任务（按 clean 30 ep 成功率分四档，每档用固定种子抽 4 个；子集的 30 ep 基线为 262/480 = 54.6%，接近当时已完成的 48 个任务的 50.2%；最终 50 个任务为 50.6%）。
- 每个变体：16 任务 × 10 ep，`--seed 0`，clean 与 randomized 都跑。同一 seed 对应同一场景，按 episode 配对比较（`tools/compare_runs.py`）。clean 基线 `USE_LENGTH=25` 直接取 clean 30 ep 日志的前 10 个 episode（同一批 seed），不重跑。
- 变体（一次只改一个因素）：
  1. `USE_LENGTH` ∈ {10, 15, 35, 50}（`configs/eval/sweep/ul*.env`，由 `INFER_ENV` 选择）；实际执行：10 明显更差、提前停止，15 随之放弃；35、50 跑了 clean，randomized 只跑了 35；
  2. 去噪步数 `num_steps` ∈ {5, 20}（默认 10）：补丁 `src/patches/lingbot-vla-v2-infer-num-steps.patch` 读取环境变量 `LINGBOT_NUM_STEPS`，参数文件里写 `NUM_STEPS`，由 `serve.sh` 导出；补丁由 `apply_infer_patches` 单独打，用单独的标记文件，不影响训练补丁的标记；
  3. fp32 推理（训练为 fp32；显存允许时）：未执行（`fp32.env` 保留）；
  4. 视结果再考虑 temporal ensemble 等：未执行。
- **修正指标（10-18，因 AMD 补丁把不稳定场景计入分母，见 EXP-20261015-02）**：选参数时不用 `Final success rate`，而用 `tools/compare_runs.py` 输出的：
  1. 实跑成功率 = 成功数 / 实际执行的 episode 数（逐 episode 的 `Success rate x/y` 行）；同时列出每个任务"记录分母 − 实跑数"（未执行却计失败的数目）。
  2. 配对差值：只在两边都实跑的 (task, seed) 上比较；列出每个任务的配对数，配对数 < 5 的任务标为"样本不足"。
  3. 按任务聚类的 bootstrap 95% 置信区间（以任务为单位重采样，固定随机种子，2000 次），代替把 episode 当独立样本的符号检验（符号检验仍列出供参考）。
  4. 留一任务稳健性：去掉对差值贡献最大的任务后重新计算配对差值。
- 判定：配对差值的 95% 置信区间下限 > 0、clean 与 randomized 方向一致、留一任务后仍为正，三条同时满足才采用；否则保持默认值。

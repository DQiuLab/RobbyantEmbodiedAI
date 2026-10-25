# Spec: 提交材料生成

依据：主办方《模板和目录要求》（比赛页面，不随仓库分发）；赛题页 §2.3–2.4。

## S1. results.json（`tools/results.py`）

输入：
- 一个或多个评测运行目录（`scripts/cloud/eval_subset.sh` 或初版 AMD 方案 `run_clean_benchmark.py` 的输出目录），任务日志位于 `logs/eval_<task_config>__<task>.log`。
- 主办方《初赛评测结果模板_JSON版》（比赛页面，不随仓库分发；需自行获取并放到 `configs/submission.yaml` 的 `template_json` 路径），任务名与顺序的唯一来源。
- `configs/submission.yaml` 中的 `team_id`。

规则：
1. 每个日志取**最后一行** `Final success rate: S/N = X%` 作为该任务 `successes=S`、`attempts=N`（与 `run_clean_benchmark.read_result` 相同口径）。
2. `task_config` 映射：`demo_clean → clean`，`demo_randomized → randomized`；其他值报错。
3. 多个运行目录可合并（例如 clean 与 randomized 分开跑），但同一 (setting, task) 出现两次即报错，不静默覆盖。
4. 输出与模板结构完全一致：`schema_version`、`team_id`、`results.clean`、`results.randomized`，任务键顺序与模板一致。
5. 默认要求 100 个 (setting, task) 全部有结果；`--allow-partial` 时缺失项保持 `0/0` 并在 stderr 列出。
6. 校验：任务集合 == 模板任务集合；`attempts`、`successes` 为非负整数且 `successes ≤ attempts`；`team_id` 不能是模板占位符。
7. 不对日志里的数字做任何修改（赛规 §10.1：禁止伪造评测结果）。
8. 日志中出现评测基础设施错误（`Policy rollout error`、`cannot create buffer`、`out of memory`）即报错，不生成结果：RoboTwin 会把这类 episode 记为失败，但它们反映的是评测机资源问题，不是策略表现（10-17 8 客户端显存打满事故，EXP-20261015-02）。处理方法是降低并发后重跑该任务，不得手工剔除个别 episode。

## S2. 提交包（`tools/package.py`）

目录结构按赛题页 §2.3（不参与真机任务，故无 `05_`）：

```
<team_name>_初赛提交材料/
├── 01_评测结果/results.json
├── 02_代码材料/{README.md, train.sh, eval.sh, configs/, src/}
├── 03_模型权重/checkpoint/        # 权重目录（复制并校验 sha256）；LINK.md 仅作备用（10-19：提交上限 100000M，包约 12 GB，直接放权重）
├── 04_复现与调优说明/reproduction_report.md
└── 06_…/                            # 主办方的可选创作活动目录
```

- 打包前用 S1 校验 results.json。
- 缺少任何必需文件即失败，并列出缺失项。
- 产物写入 `--out` 指定目录（默认 `submission/`，gitignore）。
- 代码白名单：`scripts/ src/ configs/ tools/ tests/ pytest.ini` + 提交版 `README.md`（来自 `docs/submission/README.md`）+ 根目录 `train.sh`、`eval.sh`；排除 `scripts/autodl/bench/`、`__pycache__`、`outputs/`、`third_party/`、`data/`、`submission/`。
- 权重目录模式：逐文件 sha256 与给定清单（`global_step_20000.sha256`）一致才算成功；复制后用 `src/relocate_snapshot.py` 改写权重内路径。
- `--link-weights`：`*.safetensors` 用硬链接放进包（不占额外空间；10-22 打包时把 12 GB 权重再复制一份导致磁盘空间不足），其余小文件照常复制（`relocate` 原地改写 yaml，不能与源文件共用 inode）。跨文件系统无法硬链接时直接报错，不退回复制。
- 06_ 为可选的创作活动目录，不参加，不生成。
- `--zip`：生成 `<team_name>_初赛提交材料.zip`（ZIP64、STORED 不压缩，safetensors 几乎不可压缩），默认写在目录旁，`--zip-dir` 可指定其他磁盘；zip 内顶层为该目录；条目顺序为小文件（< 64 MiB）在前、大文件（权重分片）在后，各自按路径排序。原因是 Debian unzip 在 locale 缺失时，会把同时带 UTF-8 标志和 ZIP64 字段的条目（偏移超过 2 GiB）的中文路径解成 `#U` 乱码目录（EXP-20261024-01 问题 2）。这个问题在打包层面无法根治：Info-ZIP `zip -fz` 也一样，去掉 UTF-8 标志又会让中文 Windows 解压乱码（10-25 在中文 Windows 上实测：带标志时资源管理器和 `Expand-Archive` 都正确，不带标志时资源管理器解出乱码）。小文件放在前面，可以保证 README、报告、results.json 和权重配置都落在 2 GiB 以内、正常解出；README 写明解压方式；生成后重新读取 zip，逐文件比对 CRC 与大小，失败即报错。权重目录模式下再从 zip 内流式计算每个权重文件的 sha256 与清单比对（`relocate` 改写过的 `*.yaml` 除外，只核对其路径字段指向包内目录），代替在打包盘上解压复核（解压会再占 12 GB）。天池提交上限 100000M。上传前另在空间充足的机器上按 README 写明的解压方式解压一次并 `sha256sum -c` 复核（10-25 补交时在评测机上执行）。

## S3. train.sh（复现训练）

- 参考环境：AutoDL 单卡 H800 80 GB、CUDA，路径 `/root/autodl-tmp`（与 `configs/train/*.yaml` 一致）。
- 依次调用（每步幂等，可重复执行续跑）：`scripts/autodl/setup_env.sh` → `scripts/autodl/fetch_assets.sh` → `scripts/autodl/prepare_data.sh ALL --finalize` → stage 1（`configs/train/autodl_vlmlora.yaml`，20k 余弦计划，`run.stop_at_step: 7500` 优雅停止）→ `STREAM=1 scripts/cloud/merge.sh` 得到 stage 1 合并权重 → stage 2（`configs/train/autodl_vlmlora_s2.yaml`，`init_path` 指向该权重）→ merge step 12500 → 最终权重目录。
- `stop_at_step`：`launch_train.py` 导出 `LINGBOT_STOP_AT_STEP`，run-control 补丁在 `global_step >= stop_at` 时保存并退出（与 stop 文件同一路径）。
- `DRY_RUN=1` 只打印将执行的命令（测试用）。
- 非逐位复现：原始 stage 1 经过多次中断续训（D-13），数据顺序不保证一致，README 写明。

## S4. eval.sh（复现评测）

- 用法：`bash eval.sh <checkpoint_dir> [clean|randomized|both] [episodes=100] [gpu=0]`。
- 先 `src/relocate_snapshot.py <checkpoint_dir>`，再按设定调用 `scripts/cloud/eval_subset.sh`（推理参数取 `configs/eval/infer.env`；`--seed 0 --expert_check true --accept_expert_info_on_failure true`；默认 `CLIENTS=1`，可通过环境变量设 `CLIENTS`），最后 `tools/results.py` 生成 `results.json`。
- 平台路径来自 `PLATFORM`（默认 `autodl`，即 setup_env.sh 搭出的布局）。
- `TASKS="a b"`：只评测指定任务（默认 ALL）；非全量时 results.json 以 `--allow-partial` 生成。

## S5. README / 复现报告必含披露项

对照 `docs/rules-checklist.md` E 节与赛规：
1. LoRA（VLM 注意力，r32/α64）+ 非 VLM 参数全量训练，提交为合并回基座的全量权重（赛事官方 Q&A 4）。
2. 两阶段训练（D-13）与两条命令；起点 `robbyant/lingbot-vla-v2-6b`；只用 50 任务 demo_clean。
3. clean-only 归一化统计量（D-07）、seen 指令池内替换（D-06）、图像增广（D-05）。
4. 推理参数：`USE_LENGTH`、`CHUNK_RET=True`（上游默认）、bf16、去噪步数；参数扫描方法与结论（E6），说明扫描使用评测集、存在选择偏差。
5. 评测环境：AutoDL RTX 4080 SUPER + CUDA + MPLib + `expert_check=true` + `accept_expert_info_on_failure=true`，RoboTwin@266f3aa + 赛事介绍列出的 AMD 代码参考样例（`Robotwin-radeon-cloud@27b2f6f`）的复现补丁；单卡多客户端仅为加速（E5），seed 与 episode 对应不变。
6. 对 LingBot-VLA-v2 的补丁清单与用途（`src/patches/`）；RoboTwin 评测代码：我们未修改。
7. 统计口径（中性、简短）：results.json 取评测脚本原始输出；附按实际执行 episode 计算的两个总数供参考。


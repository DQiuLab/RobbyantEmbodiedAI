# Spec: 训练

## T1. 基线（初版方案，已被 D-08 取代；复用 AMD 参考样例）

- 入口：赛事介绍列出的 AMD 代码参考样例（`Robotwin-radeon-cloud`）的 `train_full_sft.sh` + `lingbotvla_cli.yaml`，上游 commit 固定见 `scripts/cloud/setup.sh`。
- 起点：`robbyant/lingbot-vla-v2-6b` @ `11c703bf6a5c1f45b3b69168482da11fdbba53d7`。
- `TEACHER_MODE=full`，`GPU_COUNT=8`，`MICRO_BATCH_SIZE=16`，`GLOBAL_BATCH_SIZE=256`，lr 1e-4 → 5e-5 cosine。

## T2. 已核实的代码事实（2026-10-11 本地阅读）

| 事实 | 位置 | 影响 |
|---|---|---|
| `train_full_sft.sh` 写死 `--train.enable_resume false` | AMD `train_full_sft.sh` | 续训必须改为可配置 |
| 每次保存写新的 `checkpoints/global_step_N/`，**不删除旧 DCP** | `tasks/vla/train_lingbotvla.py` 保存分支；`checkpointer.py` 无清理 | 第二次保存即超出 100 GB `/workspace` → 必须自己管理 |
| resume 自动选 `output_dir/checkpoints` 下最大 step，失败则逐个回退 | `train_lingbotvla.py` L631–675 | 续训只需同一 `output_dir` + `enable_resume=true` |
| `data.image_augment` 已实现（亮度 0.7–1.3、对比度 0.6–1.4、饱和度 0.5–1.5，同一样本各视角共享参数），默认关闭 | `lingbotvla/data/vla_data/transform.py` | 打开即可，零代码改动；合规（赛事官方 Q&A 18-1） |
| 数据转换先查 HDF5 的 `instruction`（单字符串 = 该轨迹 seen 池第 1 条）再查 `instructions`，因此每条轨迹固定只用 `seen[0]`，结果确定 | XPolicyLab `transform_lerobot_v30_format.py::_find_instructions`；本地核对 adjust_bottle ep0 | 训练每条轨迹只见 1 条指令，而评测每个 episode 随机抽 seen/unseen（`build_instruction` 的 `np.random.choice`）→ 语言分布不匹配 |
| HDF5 `instructions` = 该轨迹 seen 池去重（ep0：97 条，全部 ∈ JSON `seen`，0 条 ∈ `unseen`）；unseen 只在 `instructions/episode*.json` | 本地核对 adjust_bottle ep0 | seen 池内替换（赛事官方论坛 RD 2.1 允许）可直接以 HDF5 `instructions` 为唯一来源 |
| 官方 `assets/norm_stats/robotwin.json` 的 count=6,062,592；`assets/training_data/robotwin.txt` 列有 50 randomized + 50 clean，上游 README 用它生成该文件（由此推断其含 randomized 数据，见 D-07） | 上游 `assets/`、`lingbotvla/data/vla_data/README.md` | 不能使用 → D-07 重算 |
| 持久化清单可能含旧前缀 `/workspace/RoboTwin/` | AMD `train_full_sft.sh` 中的 sed | `src/data_checks.py` 修正并校验 |
| LR 调度总步数 = min(train_steps×epochs, max_steps)，cosine lambda 在每次启动时按当前 `max_steps` 重建；resume 只恢复步数 | `train_lingbotvla.py` L588–593；HF `get_cosine_schedule_with_warmup` | 阶段间改 `max_steps` 安全，LR 曲线在续训点切换到新曲线 |
| 评测 instruction 类型由 task config 决定：`demo_clean → seen`，`demo_randomized → unseen` | RoboTwin `env_cfg/task_config/*.yml` `eval_instruction` | 与赛规 A.5 一致，无需修改 |
| AMD 配置与官方 `robotwin.yaml` 的 align/训练参数有差异（如 `use_cosine_loss`、`block_future_depth_to_action`、`enable_fp32`、`use_compile`） | 两份 YAML 对比 | 见 decisions D-04 |

## T3. 步数预算（`tools/budget.py`；为初版 AMD 方案设计，credits 指 AMD 赛事算力）

输入：开训时刻、实测 step 时长、训练截止时刻、GPU 数、两阶段 credits、各阶段非训练开销（冒烟、评测、转换）、阶段间隙预留、保存开销。
输出：
- `stage1_stop_step`：阶段一 credits 扣除开销后能跑到的步数；
- `max_steps`：min(上限, 墙钟可负担步数, credits 可负担步数)，向下取整到 `round_to`；
- 若 `max_steps < stage1_stop_step` 则阶段一直接以 `max_steps` 结束。

## T4. 待冒烟验证（初版 AMD 方案；随 D-08 放弃，未执行）

1. DCP 实际大小、保存耗时、保存时磁盘峰值；`/` 根盘可用空间。
2. resume：10 步 → 保存 → 以 `enable_resume=true` 续 1 步，日志出现 `Load distributed checkpoint ... successfully`。
3. Muon 在 ROCm 上可运行性、显存、step 时长（≤30 分钟）。
4. `image_augment=true` 下 step 时长无明显上升（CPU 侧增广）。

## T5. AutoDL 单卡 LoRA（D-08；已被 D-10、D-11 取代，最终方案为 `autodl_vlmlora*.yaml`：`lora_scope=vlm_expert_full`、global batch 128、两阶段训练，见 D-11/D-13 与 submission spec S3）

- 配置：`configs/train/autodl_lora.yaml`；平台路径：`configs/platform/autodl.env`；两者一致性有测试（`tests/test_launch_train.py`）。
- 与 AMD LoRA 示例的差异：rank 32 / alpha 64（示例 8/16）、单卡、global batch 32、`enable_mixed_precision=false`、颜色增广、seen 指令替换、clean-only 统计量。
- 数据：`scripts/autodl/prepare_data.sh` 逐任务"下载→指令池分片→LeRobot v3→删原始数据"；finalize 生成并校验清单、合并指令池（`seen_instructions.py merge`，跨任务冲突同样剔除）、计算统计量。
- 显存/吞吐：`scripts/autodl/smoke_mem.sh` 用 1 个任务测出；`max_steps` 由 `tools/budget.py` 按实测重算后写回配置。
- 产物：`scripts/cloud/merge.sh` → 合并权重 + `SUBMISSION_NOTE.json`（rank/alpha/scope）。

## T6. 训练效率开关（D-09）

| 开关 | 作用 | 默认 |
|---|---|---|
| `train.use_compile` | 整模型 torch.compile（官方取值 true），消除 flex attention 的 eager 回退 | true（实测后改） |
| `run.compile_flex` | 只编译 flex_attention（`LINGBOT_COMPILE_FLEX`），比整模型编译慢，作为备选 | false |
| `run.fsdp_single_gpu` | 单卡也套 FSDP2，使 bf16 `MixedPrecisionPolicy` 生效（`LINGBOT_FSDP_SINGLE_GPU`）；多卡时不起作用 | false（最终保持官方 fp32） |

- 衡量方法：`scripts/autodl/profile_train.sh` 输出 s/step 与 dmon，`TRACE=1` 时用 `tools/analyze_trace.py` 汇总 kernel 和 GEMM 精度。
- 换卡或改训练范围（例如全量训练 action expert）后，micro batch 要在目标卡上重新测；编译和精度开关与卡型无关。

## T7. AutoDL 单卡全量训练 action expert（D-10；训练范围已被 D-11 取代，以下实例准备、检查与无人值守流程沿用到最终方案）

- 配置：`configs/train/autodl_expert.yaml`，与 `autodl_lora.yaml` 共用平台、数据、统计量、指令池（有测试）。
- 新实例准备：
  1. 无卡模式下运行 `setup_env.sh`；
  2. 运行 `SKIP_SIM_ASSETS=1 fetch_assets.sh`（只训练，不需要仿真资产）；
  3. 运行 `pull_runtime.sh <旧实例>`（校验和一致才算通过）。
- 测速：`profile_train.sh` 加 `TRAIN_CONFIG=configs/train/autodl_expert.yaml`，组合 micro batch 8/16/32 与 `run.fsdp_single_gpu` 开/关。开 bf16 时同时设置 `train.enable_fp32=false`。每组 10 步左右，比较每元样本数。
- 产物：DCP → `scripts/cloud/convert.sh` → HF 权重，下载回本地后拷到北京B 的 4080 实例评测。
- 主训练前的检查（H800 上无法做仿真评测）：`scripts/cloud/serve_check.sh <hf_dir>` 起推理服务，用 `src/serve_probe.py` 发随机观测，要求返回的动作有限且为 14 维。
- 无人值守主训练：`scripts/autodl/train_supervised.sh`。
  - 失败时自动续训，最多 `MAX_ATTEMPTS` 次；遇到 OOM 后把 micro batch 降到 8（gbs 不变）。
  - 按 `SNAPSHOT_STEPS`，在对应 DCP 被轮转删除之前，用 CPU 转出 HF 快照，供事后挑选 checkpoint。
  - 设置 `SHUTDOWN=1` 时，训练成功、遇到 STOP、或重试耗尽，三种情况结束后都会关机；默认不关机。

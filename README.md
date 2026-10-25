# BotLab · 蚂蚁灵波具身大模型挑战赛初赛总结

> **English summary.** This is BotLab's entry for the preliminary round of the Ant Lingbo Embodied AI Challenge (2026). Starting from `robbyant/lingbot-vla-v2-6b`, we trained a single policy on a single H800 GPU, using only the RoboTwin 2.0 Aloha-AgileX `demo_clean` data (50 tasks × 50 demos). We trained LoRA on the VLM attention layers, fully fine-tuned all non-VLM parameters, and ran 20k steps in two stages, for about 58 GPU-hours and about ¥800 in total.
>
> Self-evaluation (30 episodes per task): **clean 50.6%, randomized 25.5%**.
>
> The repository contains the code, patches, configs, tests, specs, decision records and a technical report (Chinese) needed to reproduce the entry.

本仓库是 BotLab 参加[蚂蚁灵波具身大模型挑战赛](https://tianchi.aliyun.com/competition/entrance/532514)初赛的方案与复现材料，作为参赛总结公开：

- 训练、评测、打包的全部代码和对上游的补丁；
- 配置与测试；
- 规格文档、决策日志；
- 一份技术报告。

## 结果

最终权重：全局 20000 步，默认推理参数，每任务每个设定自测 30 次，RTX 4080 SUPER 上运行：

| 设定 | 成功 / 测试次数 | 成功率 | 只计实际执行的 episode |
|---|---|---|---|
| clean（seen 指令） | 759 / 1500 | 50.6% | 759 / 1433 = 53.0% |
| randomized（unseen 指令） | 383 / 1500 | 25.5% | 383 / 1268 = 30.2% |

"成功 / 测试次数"一栏取评测脚本原始输出，与提交的 results.json 一致；该脚本把初始状态不稳定、未执行策略的场景也计入测试次数，"只计实际执行的 episode"一栏去掉了这部分。逐任务结果见[技术报告](docs/report/report.pdf) §5.2。

## 方法要点

- **起点**：`robbyant/lingbot-vla-v2-6b`。不使用 `lingbot-vla-v2-6b-robotwin`，不使用 teacher 或蒸馏。
- **数据**：只用 50 任务 × 50 条 clean 示范。
  - 归一化统计量只用 clean 数据重新计算；
  - 训练时，任务文本在同一条轨迹自带的 seen 指令之间随机替换（seen 池内替换）。
- **参数化**：VLM 注意力层 LoRA（r32/α64），非 VLM 参数（动作专家等）全量训练，可训练参数占 29%。为此新增了 `lora_scope=vlm_expert_full`（`src/patches/`）。
- **训练**：单卡 H800，fp32/TF32，开启 `torch.compile`，global batch 128。
  - stage 1 在保存第 8000 步时被平台中断，之后没有可续训的 checkpoint；
  - stage 2 以 stage 1 第 7500 步的合并权重为起点，再训 12500 步。
- **推理**：单一权重，上游默认参数（执行长度 25、去噪 10 步、bf16）。我们扫描过推理参数，没有发现可靠的改进。

## 主要工程经验（详见技术报告）

- 长时间单卡训练的稳定性：
  - 容器内存看门狗与页缓存的关系；
  - checkpoint 轮转"先删后写"的坑；
  - 两阶段恢复。
- **单卡多客户端评测**：一个推理服务同时服务多个仿真客户端，吞吐约 3.5 倍；显存容量上限的估计。
- 评测卡死的定位：MPLib 规划卡在 IK 阶段；用 faulthandler 抓调用栈，配合看门狗自动重跑。
- 推理参数扫描：按 (任务, seed) 配对比较，按任务做 bootstrap 置信区间，结论是"参数基本不敏感"（负结果）。
- 提交前的**评委视角复现演练**：
  - 只凭提交包在干净机器上从零复现，逐 seed 一致率 77.3%；
  - 从零准备的训练数据与原始版本逐字节一致；
  - 发现路径含中文、体积超过 2 GiB 的 zip 在 locale 缺失的 Linux 上会解压乱码，并给出缓解办法。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `train.sh` / `eval.sh` | 一键训练 / 评测（提交包中的入口） |
| `configs/` | 训练配置、推理参数、robot config、平台路径、提交配置；`reference/` 为上游配置副本，官方 HF 配置因来源未声明许可证不随仓库分发，只给链接与 sha256（见其中 README） |
| `src/` | 训练启动器、LoRA 流式合并、权重打包与路径重定位；`patches/` 为对 lingbot-vla-v2 的 7 个补丁 |
| `scripts/autodl/` | AutoDL 上的环境搭建、下载、数据准备、无人值守训练 |
| `scripts/cloud/` | 合并、推理服务、评测（单卡多客户端、卡死看门狗） |
| `tools/` | results.json 生成与校验、配对比较、提交打包 |
| `tests/` | pytest（192 项，不需要 GPU；其中 `test_seen_instructions.py` 的 16 项需要 h5py） |
| `docs/report/` | 技术报告（[PDF](docs/report/report.pdf)） |
| `docs/submission/` | 提交包内的 README 与复现报告 |
| `docs/specs/`、`docs/decisions.md` | 规格文档、决策日志 |
| `docs/provenance.md` | 发布范围与版本说明 |

## 复现

完整步骤见 [`docs/submission/README.md`](docs/submission/README.md)，即提交包里的复现说明，包括基础镜像、资源需求、解压、权重校验、评测耗时等。最简流程：

```bash
bash scripts/autodl/setup_env.sh      # conda 环境、固定 commit 的上游源码、补丁
bash scripts/autodl/fetch_assets.sh   # 基座权重、tokenizer、仿真资产
bash train.sh                         # 复现训练（单卡 H800 80 GB，约 60 h）
CLIENTS=6 bash eval.sh <checkpoint_dir> both 100 0   # 评测
python3 -m pytest tests/ -q           # 本地测试
```

说明：
- 脚本默认使用 AutoDL 的路径约定（数据盘 `/root/autodl-tmp`），只在 NVIDIA CUDA 上实测过。
- 生成提交包（`tools/package.py`）需要主办方的结果模板。赛题、规则与模板见[比赛页面](https://tianchi.aliyun.com/competition/entrance/532514)；这些材料版权属于主办方，不随本仓库分发，打包时请放到 `configs/submission.yaml` 里 `template_json` 指向的路径。
- 提交的权重没有放在本仓库中。
- 发布范围、文中实验编号与提交哈希的含义、两次提交的 zip sha256，见 [`docs/provenance.md`](docs/provenance.md)。

## 局限

- 没有做全量微调：受预算限制。我们推测这是成绩的主要瓶颈，未做验证。
- 自测每任务 30 次，单任务的统计误差较大。
- 训练经历过中断续训，不能逐位复现。

## 关于协作方式

本项目由 BotLab 与 Claude（Anthropic）协作完成。项目规范见 [`CLAUDE.md`](CLAUDE.md)：先写规格再实现，每次云端运行都记账、写记录，改动后必须跑测试。

## 致谢与许可

感谢蚂蚁灵波与赛事主办方提供基座模型、数据与评测框架，感谢赛事介绍列出的 AMD 代码参考样例（[Robotwin-radeon-cloud](https://github.com/ZiguanWang/Robotwin-radeon-cloud)），感谢 [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) 团队开源仿真平台。

本仓库以 [Apache-2.0](LICENSE) 许可发布；第三方组件说明见 [NOTICE](NOTICE)。

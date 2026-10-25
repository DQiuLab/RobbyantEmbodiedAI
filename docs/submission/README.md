# BotLab · 蚂蚁灵波具身大模型挑战赛初赛 · 代码材料

> 本文件为提交包内说明；包内路径相对于提交包，权重未随本仓库发布。

基于 `robbyant/lingbot-vla-v2-6b`（HF revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`），只用 RoboTwin 2.0 Aloha-AgileX 50 任务 `demo_clean` 数据（每任务 50 条）训练一份 cotrain 权重，在 clean / randomized 两种设定下评测。

## 1. 快速开始

| 目的 | 命令 | 说明 |
|---|---|---|
| 复现训练 | `bash train.sh` | 参考环境：AutoDL 单卡 H800 80 GB（CUDA），数据盘 `/root/autodl-tmp` ≥ 200 GB；训练约 58 h（20000 步），另需环境与数据准备约 5 h |
| 复现评测 | `CLIENTS=6 bash eval.sh <checkpoint_dir> both 100 0` | 输出 `/root/autodl-tmp/runtime/outputs/eval_<ckpt>_results.json`；耗时与 `CLIENTS` 见下文 |
| 本地测试 | `python3 -m pytest tests/ -q` | 不需要 GPU |

**评测耗时与并行**：
- `CLIENTS=N` 让同一个推理服务同时服务 N 个评测进程。每个任务仍是独立的 `--seed 0` 运行，结果与串行一致（§7）。默认 `CLIENTS=1`，即串行。
- 单卡 32 GB 的上限在 `configs/eval/clients.env` 中设定：clean 6 个，randomized 3 个，超过上限会自动降到上限。依据是推理服务约占 12.5 GB，每个评测进程的渲染：clean 2–3 GB，randomized 最高 5.5 GB。我们试过 clean 8 个，显存会打满（32.2 GB）并出现渲染错误。
- 以下耗时是按自测折算的估计（每 episode 约 4–4.5 客户端·分钟），并非实际跑过 100 ep：
  - RTX 4080 SUPER 上 100 ep 时，clean（6 客户端）约 2–2.5 天，randomized（3 客户端）约 4.5–5 天；
  - 实测参照：自测每设定 30 ep；评委视角演练每任务 2 ep，`CLIENTS=4`（randomized 自动降为 3）下 clean、randomized 各约 2 h。
- 用 `TASKS="task_a task_b ..."` 可以只评测部分任务，用来分到多台机器上跑（多卡和多机并行我们没有实测）。

`train.sh` 与 `eval.sh` 都可重复执行，中断后重跑即续上。`DRY_RUN=1 bash train.sh` 只打印将执行的命令。

**路径约定**：所有脚本与 `configs/train/*.yaml`、`configs/platform/autodl.env` 都以 `/root/autodl-tmp` 为数据盘根目录（AutoDL 的默认数据盘）。在其他机器上，把一块 ≥ 200 GB 的盘挂到这个路径即可，例如 `mkdir -p /data/robbyant && ln -s /data/robbyant /root/autodl-tmp`，然后把本目录放到 `/root/autodl-tmp/RobbyantEmbodiedAI`。

**最小评测复现**（已有提交的权重，不重新训练）：

```bash
bash scripts/autodl/setup_env.sh          # 环境、RoboTwin@266f3aa、AMD 参考样例补丁、我们的补丁
bash scripts/autodl/fetch_assets.sh       # Qwen3-VL 配置与 tokenizer、RoboTwin 仿真资产（基座权重评测时不需要，也会一并下载）
bash eval.sh <本提交包>/03_模型权重/checkpoint/global_step_20000 both 100 0
```

**运行环境与兼容性**：

| 项 | 要求 / 实测情况 |
|---|---|
| 操作系统 | Linux x86_64，以 root 运行（`setup_env.sh` 用 apt 安装系统库，路径固定为 `/root/autodl-tmp`）。实测为 Ubuntu 22.04。Windows 和 macOS 只能用来解压、阅读，不能运行（仿真依赖 SAPIEN/Vulkan，WSL2 未测） |
| GPU | **只在 NVIDIA CUDA 上实测过**：训练用 H800 80 GB，评测用 RTX 4080 SUPER 32 GB |
| AMD GPU（ROCm） | 未实测。代码保留了 `PLATFORM=amd`（`configs/platform/amd.env`、`configs/env/rocm.env`）。这条路径面向 AMD Radeon Cloud 的 `robotwin` 镜像，沿用赛事介绍列出的 AMD 代码参考样例（Robotwin-radeon-cloud@27b2f6f）的设置，我们没有在 AMD 上运行过。我们**预计（未验证）**也可以按 AMD 参考样例的原生流程加载本权重，只需把 norm stats 和 robot config 指向权重目录内的 `norm_stats.json`、`robot_configs/`。理由：权重是与硬件无关的 bf16 safetensors；推理参数全部为上游默认值；唯一影响推理的补丁 `infer-num-steps` 在默认配置下不生效（§5、§6） |
| 磁盘 | 只做评测约 105 GB：两个 conda 环境约 40 GB，基座模型 24 GB（评测不需要，但会一并下载），仿真资产 16 GB，再加 zip 和解压后的提交包各约 12.5 GB。复现训练 ≥ 200 GB |
| 网络 | 默认用国内镜像：HF 走 `hf-mirror.com`，torch 下载失败时回退到阿里云镜像，pip 源沿用机器自身的配置。在海外机器上，设 `HF_ENDPOINT=https://huggingface.co` 即可直连 |

**基础镜像与资源**：
- 镜像：AutoDL 基础镜像 Miniconda / Ubuntu 22.04 / CUDA 11.8。镜像自带的 CUDA 和 PyTorch 都不使用：`setup_env.sh` 安装的 torch 2.8.0+cu128 自带 CUDA 运行时，只要求 NVIDIA 驱动支持 CUDA 12.8（≥ 570）。评测机的驱动版本是 580.105.08。
- `setup_env.sh` 和 `fetch_assets.sh` 不需要 GPU，可以在 AutoDL 无卡模式下运行，2026-10-24 在干净镜像上实测共约 2.5 h。GPU 相关的自检会跳过，换到 GPU 模式后再运行一次 `setup_env.sh` 即可补上。
- `prepare_data.sh`（只有训练需要）要在 GPU 模式下运行：
  - 统计量计算要用 GPU；
  - 默认 `JOBS=4` 时，四个 LeRobot 转换进程并发。在无卡模式 2 GB 内存的上限下，这些进程会被系统杀掉；
  - 在 12 核、62 GB 内存的 4080 实例上实测约 2.2 h；
  - 内存较小时，请调低 `JOBS`。

**解压**：zip 内的路径含中文，偏移超过 2 GiB 的条目带 ZIP64 扩展字段。请用以下任一方式解压：

```bash
LANG=C.UTF-8 unzip BotLab_初赛提交材料.zip
python3 -m zipfile -e BotLab_初赛提交材料.zip .
```

AutoDL 镜像设置了 `LANG=en_US.UTF-8`，但没有安装这个 locale。在这样的环境里直接 `unzip`：
- 偏移超过 2 GiB 的第 3–7 个权重分片会被解到乱码目录 `BotLab_#U521d#U8d5b…/`；
- unzip 报 `mismatching "local" filename`，退出码为 1；
- 文件内容本身完好，其余文件（本 README、报告、results.json、权重配置和清单）都排在 zip 前部，能正常解出。

遇到这种情况，删掉解出的两个目录，用上面任一命令重新解压即可。这是 Debian unzip 对"UTF-8 文件名 + ZIP64 扩展字段"条目的处理方式，与打包工具无关：路径含中文、体积超过 2 GiB（Python zipfile）或 4 GiB（Info-ZIP zip）的 zip 都会遇到。

以下方式不受影响：`python3 -m zipfile` 和 locale 正常的 Linux `unzip`，已用本包实测；Windows 10 中文版的资源管理器和 PowerShell `Expand-Archive`，用强制 ZIP64 的小样本实测。macOS 和 7-Zip 按 zip 规范支持 UTF-8 标志和 ZIP64，未实测。

**校验权重**：

```bash
cd 03_模型权重/checkpoint/global_step_20000 && sha256sum -c ../global_step_20000.sha256
```

预期结果是 23 项中 21 项 OK，权重分片全部 OK。`lingbotvla_cli.yaml` 和 `robot_configs/robotwin.yaml` 两项 FAILED 属于正常：这两个文件里有绝对路径，打包时已改写为打包目录，`eval.sh` 加载前还会再次改写为实际目录（`src/relocate_snapshot.py`）。

## 2. 依赖与环境

- `scripts/autodl/setup_env.sh` 一次性搭建全部环境（conda 环境 `rt` 用于训练/推理/仿真，`lerobot` 用于数据转换）：
  - RoboTwin `266f3aa`（含 XPolicyLab 子模块）、lingbot-vla-v2 `951475a`；
  - 赛事介绍列出的 AMD 代码参考样例（`ZiguanWang/Robotwin-radeon-cloud@27b2f6f`）的三个补丁（`robotwin-rocm-reproduction.patch`、`xpolicylab-lerobot-v30.patch`、`lingbot-vla-v2-rocm.patch`），把 ROCm 方案移植到 CUDA；
  - 我们对 lingbot-vla-v2 的补丁（见 §5）。
- 版本：torch 2.8.0 / torchvision 0.23.0，flash-attn 2.8.3，CUDA 环境；Python 依赖固定在 `setup_env.sh` 中。
- `scripts/autodl/fetch_assets.sh` 下载固定 revision 的基座权重、Qwen3-VL-4B 配置与 tokenizer、RoboTwin 仿真资产。
- `scripts/autodl/prepare_data.sh` 下载 `demo_clean`（数据集 revision `a967b852`）并转换为 LeRobot v3，构建 seen 指令池，用 clean 数据计算归一化统计量。

## 3. 目录

```
train.sh / eval.sh          一键训练 / 评测
configs/train/              autodl_vlmlora.yaml（stage 1）、autodl_vlmlora_s2.yaml（stage 2）
configs/eval/infer.env      推理参数（单一权重）
configs/robot_autodl/       robot config；configs/env/cuda.env 环境变量
configs/reference/          上游 / 官方配置原样副本与 results.json 模板（只读）
scripts/autodl/             环境、下载、数据准备、无人值守训练（train_supervised.sh）
scripts/cloud/              merge.sh（LoRA 合并）、serve.sh（推理服务）、eval_subset.sh（评测）
scripts/lib/                平台路径、补丁应用、checkpoint 完整性检查
src/                        launch_train.py、stream_merge.py、bundle_ckpt.py、relocate_snapshot.py 等；patches/ 为对上游的补丁
tools/                      results.py（生成 results.json）、compare_runs.py（配对比较）、package.py（打包）
tests/                      pytest
```

## 4. 训练方法

- **起点**：`robbyant/lingbot-vla-v2-6b`（不使用 `lingbot-vla-v2-6b-robotwin`，不使用 teacher / 蒸馏）。
- **数据**：50 任务 × 50 条 `demo_clean`；不使用 randomized 数据、unseen 指令、外部数据或任何仿真 rollout。训练清单由 `src/data_checks.py` 校验（恰好 50 个 clean 任务、无 randomized 条目）。
- **参数化**：VLM 注意力层 LoRA（`q/k/v/o_proj`、视觉 `attn.qkv/attn.proj`，rank 32 / alpha 64）+ 非 VLM 参数（action expert 等）全量训练（我们新增的 `lora_scope=vlm_expert_full`）。训练精度 fp32/TF32（`enable_fp32: true`，与官方配方一致），`use_compile: true`。
- **超参**：global batch 128（micro batch 16），AdamW，lr 1e-4 余弦衰减到 5e-5，seed 42，`image_augment: true`。
- **两阶段**（实际训练中 stage 1 在保存 step 8000 时被平台终止，见复现报告）：
  - stage 1：按 20000 步余弦计划训练，在 step 7500 保存并退出（`run.stop_at_step: 7500`），LoRA 合并回全量权重；
  - stage 2：以 stage 1 合并权重为起点，lr 8.46e-5（stage 1 在 7500 步时的值），2% 预热后余弦降到 5e-5，训练 12500 步，Adam 状态重置。
  - 合计 20000 步（全局），最终提交权重为 stage 2 step 12500 的合并权重。
- **提交权重形态**：LoRA 合并回基座的全量权重（bf16 safetensors，约 12 GB；赛事官方《常见 Q&A》赛题相关第 4 条允许），合并工具 `src/stream_merge.py`（与 AMD 参考样例的 `merge_lora_dcp.py` 逐张量一致，1624 个张量已核对）。
- **合规的数据增强**：
  - 归一化统计量只用 50 个 clean 数据集重算（官方 `assets/norm_stats/robotwin.json` 的帧数远多于 clean 数据，上游生成它的训练清单含 randomized 数据集，我们据此推断它含 randomized 数据，未使用），随权重提交为 `norm_stats.json`；
  - 训练时把任务文本替换为同一条轨迹 HDF5 `instructions` 中的另一条 seen 指令（赛事官方论坛 RD 2.1 允许的"seen 池内替换"，不新增配对，从不读取含 unseen 的指令 JSON）；
  - 框架自带颜色增广（亮度/对比度/饱和度）。
- **复现说明**：原始 stage 1 经历过多次中断续训，数据顺序不保证与 `train.sh` 逐位一致；`train.sh` 复现的是同一配置与流程。

## 5. 对 LingBot-VLA-v2 官方代码的修改

全部为 `src/patches/` 下的 git 补丁，由 `scripts/lib/our_patches.sh` 在 AMD 补丁之后按顺序应用：

| 补丁 | 作用 | 是否影响推理 |
|---|---|---|
| `dcp-rotation` | 只保留最近 N 个训练 checkpoint（磁盘限制） | 否 |
| `seen-instructions` | 训练期 seen 指令池内替换（`src/lingbot_patches/seen_instructions.py`） | 否 |
| `run-control` | 续训时找不到 checkpoint 即报错；stop 文件 / `stop_at_step` 优雅停止 | 否 |
| `compile-flex` | flex attention 单独编译开关（最终配置未启用） | 否 |
| `fsdp-single-gpu` | 单卡套用 FSDP2 的 bf16 开关（最终配置未启用） | 否 |
| `vlm-lora-expert-full` | 新增 `lora_scope=vlm_expert_full` | 否 |
| `infer-num-steps` | 推理服务读取 `LINGBOT_NUM_STEPS` 覆盖去噪步数（未设置时为默认 10） | 仅在设置时 |

## 6. 推理参数（单一权重，无按任务路由）

`configs/eval/infer.env`：

| 参数 | 值 | 说明 |
|---|---|---|
| `USE_LENGTH` | 25 | 每次推理生成 50 步动作块，执行前 25 步后重新推理（AMD 参考样例默认值） |
| `CHUNK_RET` | True | 每次调用返回整段动作（上游 argparse 默认值） |
| 精度 | bf16 | `USE_BF16=True` |
| 去噪步数 | 10 | 上游默认值（未设置 `NUM_STEPS`） |
| `torch.compile` | 关 | |

推理参数选择方法：在 16 个任务的固定子集上按 (任务, seed) 配对比较若干组参数，按任务聚类的 bootstrap 95% 置信区间下限 > 0、clean 与 randomized 方向一致、去掉贡献最大的任务后仍为正，三条同时满足才采用。该扫描使用了评测场景，存在一定选择偏差。结论：全部保持默认值。

| 变体（相对默认） | clean 配对差值（95% CI） | randomized 配对差值（95% CI） |
|---|---|---|
| `USE_LENGTH` 10 | 前 3 个任务 2/30 vs 12/30，明显更差，提前停止 | 未测 |
| `USE_LENGTH` 35 | +5.4%（−5.8%, +16.2%） | −4.5%（−13.5%, +4.8%） |
| `USE_LENGTH` 50 | +4.1%（−6.1%, +15.4%） | 未测（clean 与 35 相当） |
| 去噪 5 步 | −0.7%（−6.5%, +4.1%） | 未测（clean 无提升，不可能被采用） |
| 去噪 20 步 | +2.9%（−5.5%, +10.7%） | 未测（clean CI 下限 ≤ 0） |

配对差值只统计两边都实际执行的 episode（见 §7 统计口径），置信区间以任务为单位 bootstrap。

## 7. 评测

- 环境：AutoDL RTX 4080 SUPER 32 GB，CUDA，MPLib 规划器；RoboTwin `266f3aa` + AMD 参考样例补丁；`--seed 0 --expert_check true --accept_expert_info_on_failure true`（与 AMD 参考样例的 Reproduce_Guide 一致）。
- 自测规模：clean / randomized 各 50 任务 × 30 episode（《常见 Q&A》赛题相关第 10 条：未测满 100 次可以提交，以官方复测为准）。
- 并行：一个推理服务同时服务多个评测进程（`CLIENTS`），每个任务仍是独立的 `--seed 0` 运行，seed 与 episode 对应关系与串行完全一致；推理服务在 `CHUNK_RET=True` 下对调用之间无状态，请求串行处理。
- **RoboTwin 官方评测代码：我们没有修改。** 使用的 `scripts/eval_policy_xpolicylab.py` 是 AMD 参考样例补丁版。
- **统计口径**：results.json 直接取评测脚本输出的 `Final success rate`。该版本脚本中，初始状态不稳定（`UnStableError`）、未执行策略的场景也计入测试次数；如只统计实际执行的 episode，clean 为 759/1433（52.97%），randomized 为 383/1268（30.21%），供参考。
- **专家规划卡死**：评测中偶发某个任务长时间无输出。用 Python faulthandler 抓到的调用栈显示卡在专家检查阶段（`eval_policy_xpolicylab.py:826` → 任务 `play_once` → MPLib `planner.py` 的 `IK` / `plan_qpos`），即策略执行之前，与模型和推理服务无关；MPLib 的 IK 带随机初始值，因此是非确定性的。我们的评测脚本（`scripts/cloud/eval_subset.sh`）在任务日志 60 分钟无更新时整任务重跑（同样的 `--seed 0`，seed 序列不变），一次运行内最多重跑 2 次（即最多中止 3 次），仍卡则留给续跑；只看日志是否停止更新，与 episode 成败无关。最终自测中被看门狗中止过的有：randomized `move_can_pot`（中止 3 次，即首次运行加 2 次重跑；当时阈值为 20 分钟，误判了专家规划较慢的正常运行，之后阈值改为 60 分钟，续跑时完成）、randomized `put_bottles_dustbin`（中止 1 次）；clean `blocks_ranking_size`、`move_playingcard_away`（评测客户端阻塞，人工终止后重跑）。每次重跑都从第一个 seed 开始，被中止那次的部分结果不计入。
- **重跑的随机性**：同一权重、同样 `--seed 0` 重跑时，逐 episode 的成败并不完全一致。原因有两个：flow matching 采样带随机噪声；MPLib IK 的初值是随机的。
  - 同一台机器上，同一权重几次运行之间的逐 seed 一致率为 70–77%。
  - 2026-10-24，我们在另一台新装环境的 4080 上，只凭提交包和本 README 从零复现（每任务 2 次）。与自测的逐 seed 一致率 clean、randomized 均为 77.3%。
  - 因此，复测和自测的逐任务成功率会有波动，汇总成功率更有参考意义。

## 8. 自测结果

最终权重 `03_模型权重/checkpoint/global_step_20000`（全局 20000 步，即 stage 2 的 step 12500），默认推理参数，每任务每设定 30 次：

| 设定 | 成功 / 测试次数（results.json） | 成功率 |
|---|---|---|
| clean | 759 / 1500 | 50.60% |
| randomized | 383 / 1500 | 25.53% |

逐任务结果见 `01_评测结果/results.json`。

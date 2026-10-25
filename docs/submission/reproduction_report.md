# 复现与调优说明

> 本文件为提交包内说明；包内路径相对于提交包，权重未随本仓库发布。

## 1. 基础信息

- 团队名称：BotLab
- 使用模型版本：`robbyant/lingbot-vla-v2-6b`（HF revision `11c703bf6a5c1f45b3b69168482da11fdbba53d7`）
- 代码版本 / Commit：见 `02_代码材料/COMMIT`（打包时由 `tools/package.py` 写入的 git commit；上游依赖的固定版本见 README §2）
- 提交 checkpoint 名称：`global_step_20000`（目录名按全局步数；即 stage 2 的 step 12500，两阶段训练合计 20000 步；LoRA 合并回基座的全量权重，bf16 safetensors）

## 2. 训练复现

- 使用数据集：RoboTwin 2.0 Aloha-AgileX 50 任务 `demo_clean`，每任务 50 条（HF 数据集 revision `a967b852afa21a9cbf19a198f7e653109042e87c`），转换为 LeRobot v3（`scripts/autodl/prepare_data.sh`）。
- 是否使用 randomized 数据训练：否。归一化统计量也只用 clean 数据计算（官方 `robotwin.json` 由含 randomized 数据集的上游清单生成（推断，依据其帧数与上游说明），未使用）。
- 一键训练脚本：`train.sh`（环境 → 下载 → 数据 → stage 1 → 合并 → stage 2 → 合并）。
- 训练配置文件：`configs/train/autodl_vlmlora.yaml`（stage 1）、`configs/train/autodl_vlmlora_s2.yaml`（stage 2）；启动器 `src/launch_train.py`，无人值守包装 `scripts/autodl/train_supervised.sh`。
- robot config：`configs/robot_autodl/robotwin.yaml`；提交权重内的 `robot_configs/robotwin.yaml` 与 `norm_stats.json` 为训练时实际使用的副本。
- 硬件与耗时：AutoDL 单卡 H800 PCIe 80 GB，约 9.4–11 s/step，20000 步约 58 h。基础镜像为 AutoDL Miniconda / Ubuntu 22.04 / CUDA 11.8；CUDA 运行时随 torch 2.8.0+cu128 安装，要求驱动 ≥ 570。数据准备需在 GPU 模式下运行（README §1）。
- 训练输入的可复现性（2026-10-24 实测）：在干净实例上按 `prepare_data.sh` 从零下载和转换，得到的训练清单、seen 指令池与原始训练所用的文件逐字节一致；clean 统计量 count 同为 54,978，数值最大相对差 2.6e-6（浮点累加顺序）。
- 两阶段的由来（如实说明）：原计划单阶段 20000 步。stage 1 在保存 step 8000 的 checkpoint 时训练进程被 SIGKILL（推测为保存期间容器内存超过 120 GiB 上限，未证实），当时的 checkpoint 轮转策略先删旧再写新，导致没有可续训的 checkpoint。我们把 step 7500 的 LoRA 合并权重（已推送到评测机并通过 sha256 校验）作为 stage 2 的起点，lr 取 stage 1 在 7500 步时的值 8.46e-5，2% 预热后余弦降到 5e-5，训练剩余 12500 步，Adam 状态重置。`train.sh` 用 `stop_at_step: 7500` 复现这一过程。
- 非逐位复现：stage 1 中途有多次中断续训，数据加载顺序不保证与一次跑完一致。

## 3. 评测复现

- 一键评测脚本：`eval.sh`
- checkpoint 加载方式：`eval.sh <checkpoint_dir>` 先用 `src/relocate_snapshot.py` 把权重内配置里的绝对路径改为该目录，再由 `scripts/cloud/serve.sh` 启动 lingbot-vla-v2 推理服务（`deploy.lingbot_vla_v2_policy`），推理参数来自 `configs/eval/infer.env`。
- clean setting 评测命令：`bash eval.sh <checkpoint_dir> clean 100 0`
- randomized setting 评测命令：`bash eval.sh <checkpoint_dir> randomized 100 0`
- 评测环境：AutoDL RTX 4080 SUPER 32 GB，CUDA，MPLib；RoboTwin `266f3aa` + AMD 参考样例（Robotwin-radeon-cloud@27b2f6f）补丁；`--seed 0 --expert_check true --accept_expert_info_on_failure true`。自测为每任务每设定 30 次（赛事官方《常见 Q&A》赛题相关第 10 条：未测满 100 次可以提交，以官方复测为准）；单卡上用一个推理服务同时服务多个评测进程加速（README §7）。
- 复现演练（2026-10-24）：在另一台干净的 4080 实例上，只凭提交包和 README 从零复现。解压 → `setup_env.sh` → `fetch_assets.sh` → `eval.sh`（每任务 2 次）全程无需人工干预。与自测相比，逐 seed 一致率 clean、randomized 均为 77.3%，处在同权重重跑的正常波动范围内（70–77%，README §7）。解压方式与权重校验方法见 README §1。

## 4. 代码改动说明

- 是否修改 LingBot-VLA 2.0 官方代码：是，7 个补丁（`src/patches/`），清单与作用见 README §5。其中影响训练的有：seen 指令池内替换、`lora_scope=vlm_expert_full`；影响推理的只有 `infer-num-steps`（未设置时不生效）。
- 是否修改 RoboTwin 官方评测代码：否。使用 AMD 参考样例补丁版 `scripts/eval_policy_xpolicylab.py`，未做任何改动。
- 如有修改，请说明修改内容和对应文件位置：见 README §5；统计口径见 README §7。

## 5. 调优说明

- 训练参数调整：
  - 单卡算力下，采用 VLM 注意力 LoRA（r32/α64）+ 非 VLM 参数全量训练，替代官方全量训练（官方 global batch 1024 × 50k 步，超出预算）；global batch 128，20000 步。
  - 保持官方 fp32/TF32 训练精度与 `use_compile`（编译带来约 3–5 倍提速）。
  - 不使用 teacher（官方配方中的深度与视频对齐 teacher）。
- 数据处理方式：clean-only 归一化统计量；训练期 seen 指令池内替换（同一条轨迹的 seen 指令之间随机替换）；颜色增广。
- 模型结构改动：无。只新增 LoRA 作用范围选项。
- 推理策略：单一权重；chunk 执行长度 `USE_LENGTH` = 25，bf16，去噪步数 10（均为上游默认，扫描后未发现可靠的改进）；无按任务路由、无 ensemble、无多 seed 投票、无失败重置。参数扫描方法见 README §6。
- 主要提升点：
  - 训练过程中 clean 成功率随步数上升：全局 7500 步约 32%，15000 步约 43%，19000/20000 步约 48–52%（每任务 1–2 次，粗测）；randomized 在 30–34% 之间基本持平。
  - 失败集中在多步任务（如 stack_blocks_three、handover_block）与精细定位任务（如 stamp_seal、move_stapler_pad、place_mouse_pad）。
  - 最终自测（每任务 30 次）：clean 759/1500 = 50.60%，randomized 383/1500 = 25.53%。
  - 推理参数扫描（`USE_LENGTH` 10/35/50、去噪步数 5/20，16 任务子集按 seed 配对比较）：除执行长度过短明显变差外均在噪声范围内，保持默认值（README §6）。

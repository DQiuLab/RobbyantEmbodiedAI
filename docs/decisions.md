# 决策日志

格式：`## D-NN 标题` → 日期 / 决策 / 依据 / 来源 / 状态。

## D-01 单主线全参数 SFT，复用 AMD 参考样例

- 日期：2026-10-11
- 决策：从 `lingbot-vla-v2-6b` 全参数 SFT，`TEACHER_MODE=full`，8 卡，global batch 256；阶段一、阶段二同一条 run 续训；训练/转换/评测脚本直接复用赛事介绍列出的 AMD 代码参考样例（Robotwin-radeon-cloud，下称 AMD 参考样例）的固定版本。
- 依据：总算力 2200 credits，30k 步需 ~1474–1650 credits，不允许多条探索 run；AMD 参考样例已在 ROCm 上打通全链路。
- 来源：AMD 算力规则文档（参赛队材料，未公开）；AMD 参考样例 `Reproduce_Guide.md` §8.1。
- 状态：已被 D-08 取代（初版 AMD 方案，未用于最终权重）

## D-02 按实测速度规划，目标 ~20k 步

- 日期：2026-10-11
- 决策：规划用实测 24.68 s/step（8×W7900，mb16×ga2）与评测 15h12m / 1000 ep；`max_steps` 由 `tools/budget.py` 在开训时按"训练 ≤ 约 10-22 结束"倒推。
- 依据：AMD 算力规则文档（参赛队材料，未公开）的估算（61 h/10k、5 h/1000 ep）与实测记录不一致，取保守值。
- 来源：`Reproduce_Guide.md` §6 墙钟表、§8.1 容量表；AMD 算力规则文档 §2。
- 状态：已被 D-08 取代（初版 AMD 方案，未用于最终权重）

## D-03 优化器默认 AdamW，冒烟时验证 Muon

- 日期：2026-10-11
- 决策：默认 AdamW；冒烟阶段 ≤30 分钟试 Muon，可跑且显存/速度不差则切换。
- 依据：AdamW 是 AMD 在 ROCm 上验证的路径；Muon 是官方配方且 optimizer state 更小（缓解 100 GB 磁盘）。
- 来源：`configs/vla/robotwin/robotwin.yaml`（`optimizer: muon`）；`Reproduce_Guide.md` §8.1（`OPTIMIZER=adamw`）。
- 状态：已定，最终训练使用 AdamW（`configs/train/autodl_vlmlora*.yaml`）；Muon 未采用

## D-04 align 参数对齐官方 robotwin 训练配置，而非 AMD 脚本

- 日期：2026-10-11
- 决策：`block_future_depth_to_action=false`、`block_suffix_to_future_video=false`、`use_cosine_loss=true`（与官方一致）；AMD 脚本额外的 `head_type` / `shared_query_head_type` / `target_type` / `block_warmup_*` 在 commit 951475a 的代码中无引用，删除。`enable_fp32=true`、`use_compile=true` 两项涉及显存/速度，冒烟时测试，能跑则与官方一致。
- 依据：官方 93.52%/92.80% 的结果出自该配置；AMD 参考样例的脚本用于容量/兼容性测试，没有给出这几项取值对效果的影响。
- 来源：HF `robbyant/lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml`（仅读取超参，不使用其权重，合规；不随仓库分发，见 configs/reference/README.md）；`configs/vla/robotwin/robotwin.yaml`@951475a；AMD `train_full_sft.sh` `full` 分支；`grep` 结果见 `docs/specs/training.md` T2。
- 状态：已定；`enable_fp32`、`use_compile` 最终均开启（D-09、D-11）

## D-05 打开 `data.image_augment`

- 日期：2026-10-11
- 决策：主训练开启框架自带的颜色增广（亮度/对比度/饱和度，各视角共享参数）。
- 依据：clean2random 测试含随机背景、杂物桌面、随机光照（`demo_randomized.yml`），训练只有 clean 数据，颜色增广是零代码、合规（赛事官方 Q&A 第 18-1 条，下称 QA）的泛化手段。官方配置未开启（其训练混用 randomized 数据，无此需求）。
- 风险：无本赛题上的消融证据；冒烟确认 step 时长无明显上升。
- 来源：`lingbotvla/data/vla_data/transform.py::sample_visual_augmentation_params`；RoboTwin `env_cfg/task_config/demo_randomized.yml`。
- 状态：已定

## D-06 训练期 seen 指令池内替换

- 日期：2026-10-11
- 决策：主训练开启（`run.seen_instruction_pool`）。每个样本的任务文本以均匀概率替换为**同一条轨迹**的 HDF5 `instructions` 列表中的一条；指令池由 `setup.sh` 从 `demo_clean` HDF5 确定性构建，键为该轨迹的 `instruction`（= seen[0]）。键冲突且列表不同的轨迹不替换。
- 依据：
  - 训练数据每条轨迹只用 `seen[0]`，而评测每个 episode 从 seen（clean）/ unseen（randomized）池随机抽一条，训练与评测的语言分布不一致（`docs/specs/training.md` T2）。
  - 赛事官方论坛的规则解读（下称 RD）2.1 明确允许"seen 池内指令替换"（不新增配对、零环境调用、仍是 BC）。
  - 本地核对 adjust_bottle：50 条轨迹各自 91–99 条 seen 指令，全部 ∈ JSON `seen`、0 条 ∈ `unseen`；首条指令 47 个唯一值，3 组冲突且列表不同（多为左右臂差异）→ 冲突键不替换，避免错标。
  - 2026-10-11 追加核对 click_bell / press_stapler / open_laptop：逐轨迹 HDF5 `instructions` ⊆ JSON `seen`，与 `unseen` 交集为 0；可替换轨迹比例 92% / 100% / 100%（click_bell 2 组冲突），每条轨迹指令池 82–99 条。4 个任务合计可替换 ≈ 95%。
- 合规边界：只读 HDF5 的 `instruction`/`instructions` 两个字段，从不读取含 unseen 的 `instructions/*.json`（有单测）。
- 风险：无本赛题消融；`setup.sh` 覆盖率检查（LeRobot 任务文本 ∈ 指令池键 ≥ 80%）与 `smoke.sh` 的 `[seen_instructions]` 日志确认挂钩生效。
- 来源：`src/lingbot_patches/seen_instructions.py`、`src/patches/lingbot-vla-v2-seen-instructions.patch`、`tests/test_seen_instructions.py`。
- 状态：已定

## D-07 归一化统计量只用 clean 数据重算

- 日期：2026-10-11
- 决策：不用官方 `assets/norm_stats/robotwin.json`；`setup.sh` 用上游 `scripts/compute_norm_stats.py` 在 50 个 clean LeRobot 数据集上重算，写入运行目录下的 `norm_stats_clean.json`（AMD 方案为 `/workspace/runtime/`，AutoDL 为 `configs/platform/autodl.env` 的 `RUNTIME`），训练与部署均使用该文件（随 HF 权重一起提交）。
- 依据（已知事实）：
  - 官方 `robotwin.json` 的 `count` = 6,062,592 帧；
  - 上游 `assets/training_data/robotwin.txt` 共 99 行，其中 50 行 `*-aloha-agilex_randomized_500-1000`、50 行 clean；
  - 上游 `lingbotvla/data/vla_data/README.md` 的重算示例正是用 `robotwin.txt` 生成 `robotwin.json`。
- 推断：`robotwin.json` 由 clean + randomized 数据计算（clean 50×50 条轨迹的帧数远小于 600 万）。使用它构成对 randomized 数据的间接使用，违反 RD 3.1"任何比例、任何形式"。AMD 参考样例使用该文件；本方案不使用。
- 影响：`bounds_99_woclip` 归一化对分布变化不敏感，预计性能影响小；`norm_stats.sh` 校验键与形状与官方一致，并打印帧数作为合规证据。
- 来源：上游 `lingbot-vla-v2@951475a` 上述文件；`scripts/cloud/norm_stats.sh`。
- 状态：已定

## D-08 改用 AutoDL 单卡 LoRA（初赛算力方案 v2）

- 日期：2026-10-12
- 背景：AMD 算力发放时间晚于 10-26 初赛截止；没有其他可用 GPU，预算有限。
- 决策：
  - 平台：AutoDL 北京B区 vGPU-32GB（¥1.58/时，16 核 / 62 GB，数据盘可扩容）。容器内不能用 Docker，按 AMD 镜像的结构手动装环境（`scripts/autodl/setup_env.sh`）。
  - 训练：LoRA，`lora_scope=action_expert`，rank 32 / alpha 64，不用 teacher（`align_params` 为空），`post_training=false`，与 AMD 参考样例 Reproduce_Guide §7.1 跑通的 LoRA 路径一致；`enable_mixed_precision=false`，基座保持 bf16（约 12 GB，否则 fp32 约 24 GB）。D-05/06/07 保留。
  - 显存不够时的降级顺序：micro batch 4→2→1 → fsdp1 + CPU offload → 迁移到同区 vGPU-48GB（`scripts/autodl/smoke_mem.sh`）。
  - 规划器：CUDA 上同样用 MPLib（`configs/env/cuda.env`），不编译 CuRobo。自测环境注明 `CUDA + MPLib + expert_check=true`。
  - 提交：LoRA 合并回基座的全量权重（`scripts/cloud/merge.sh`，Q&A 4 允许），配置、robot config、统计量一起打包（`src/bundle_ckpt.py`）。
  - 开机方式：装环境、下载用无卡模式（0.5 核 / 2 GB / ¥0.1/时，autodl.com/docs/save_money）；转换数据、冒烟、训练、评测用 GPU 开机。
- 代价（已知）：LoRA 只调 action expert 注意力层、无 teacher、训练步数少，预期成绩明显低于官方配方。目标是交出合规、可复现的作品。
- 来源：AutoDL 控制台；autodl.com/docs/{env,price,fs,network_turbo,save_money}；`train_lingbotvla.py` L398–423；`torch_parallelize.py` L108、L439；AMD `merge_lora_dcp.py`。
- 状态：平台与提交方式已定；显存、吞吐、渲染已于 10-13 实测（见下）；训练范围已被 D-10、D-11 取代
- 2026-10-13 补充：AutoDL 实测 1 个任务（adjust_bottle，7,188 帧）计算 233 s，全部 50 个任务约 36 万帧——官方文件 606 万帧，约为 clean 数据的 17 倍，进一步印证其包含 randomized 数据。全量计算单卡约 3 h，因此 AutoDL 上用上游 `data_ratio_for_norm_compute=0.1`（上游固定种子 42 抽样，可复现）。我们的文件比官方多 `min`/`max` 字段，其余字段与维度一致。
- 2026-10-13 补充（GPU 首次开机实测，RTX 4080 SUPER 32 GB，sm_89）：
  - flash-attn 2.8.3 内核可用；SAPIEN 渲染 "Render Well"。
  - `enable_mixed_precision=false`（bf16 基座）在 `embed_suffix` 的 `state_proj` 处报 `mat1 and mat2 must have the same dtype (Float vs BFloat16)` → 改回 AMD 验证的 fp32 权重 + bf16 计算。
  - fp32 布局 micro batch 2、gbs 32、梯度检查点：峰值 27.2 GB；约 38–48 s/优化步（同时有数据转换占 CPU）。LoRA 可训练参数 15.3 M / 6.24 B（0.25%）。
  - 容器 `ulimit -n` soft=1024 导致 dataloader `Too many open files` → `launch_train.py` 启动前提到 hard（65535）。
  - 预期：6 h 训练约 540 步（约 1.7 万样本，约 clean 帧数的 5%），成绩会很低；与 D-08 预期一致。
- 2026-10-13 主训练启动（10:52 CST）：全量 50 任务；指令池 2,500 轨迹 / 2,340 键 / 69 冲突，任务文本可替换率 97.1%；统计量抽样 10% 共 54,978 帧（clean 全量约 55 万帧，官方 606 万帧）。稳态 35 s/step；GPU 利用率采样仅约 17%、CPU 也空闲 → 下一轮排查数据加载/同步瓶颈。

## D-09 训练效率：整模型编译 + 单卡 bf16 混合精度

- 日期：2026-10-13
- 背景：主训练 35.8 s/step、GPU 利用率约 17%（EXP-20261013-01）。排查记录见 EXP-20261013-02。
- 已知事实：
  - 不开编译时，flex attention 走逐元素索引的回退实现，约占每步 40%。开 `train.use_compile=true`（官方 robotwin.yaml 的取值）并把 micro batch 调到 4 后：6.9 s/step，dmon 利用率 88%。
  - 主要 GEMM 是 `cutlass_80_tensorop_s1688gemm`（TF32/fp32），不是 bf16。原因：上游 `ParallelState.fsdp_enabled` 是 `fsdp_size > 1`，单卡时不走 FSDP。`build_parallelize_model` 先 `model.float()`，又没有 `MixedPrecisionPolicy`，结果整个模型都按 fp32 计算。更正（同日核对官方配置）：官方 HF `lingbotvla_cli.yaml`（D-04）是 `enable_fp32: true`，FSDP2 的 `param_dtype=float32`，即官方训练本身就是 fp32/TF32 计算；单卡 fp32 与官方数值一致，不是缺陷。AMD 参考样例 `Robotwin-radeon-cloud@27b2f6f` 的 `docker/assets/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml` 是 `enable_fp32: false`（bf16 计算）。因此 bf16 是**相对官方的偏离、相对 AMD 的对齐**，纯属提速选项。
- 决策：新增补丁 `lingbot-vla-v2-fsdp-single-gpu.patch`，开关为 `run.fsdp_single_gpu`（环境变量 `LINGBOT_FSDP_SINGLE_GPU=1`），作用是单卡时也套用 FSDP2（mesh 大小为 1），让单卡也能用 AMD 验证过的 bf16 混合精度（提速选项；官方配方为 fp32）。
- 指标：用 s/step 和每元样本数衡量，不用 dmon 利用率。dmon 只表示"有 kernel 在跑的时间占比"，GEMM 变快后这个数可能反而下降。
- 验证门槛：同 seed、同数据跑 20 步，loss 曲线与 fp32 基线处在同一量级、趋势一致，且没有 NaN。不满足就保持 false。
- 状态：parity 通过（EXP-20261013-02）。在 4080（32 GB）上更慢（13.3 s vs 6.9 s，原因是单卡 FSDP 的逐 micro batch 参数拷贝），所以 4080 上保持 false；大显存卡上待实测后决定
- 来源：上游 `lingbot-vla-v2@951475a` 中的 `lingbotvla/distributed/parallel_state.py`（`fsdp_enabled`）、`torch_parallelize.py`（L102–L108、L149、L208–L215）；训练日志中只有 "Enable gradient checkpointing"，没有 "Apply data parallel"

## D-10 主线改为单卡 H800 全量训练 action expert（不带 teacher）

- 日期：2026-10-13
- 背景：判断只对 action expert 做 LoRA 收益有限（EXP-20261013-01）。目标是进决赛，预算上限约 ¥1000。
- 决策：
  - 卡：AutoDL 西北B H800 80 GB，¥9.98/时，20 核，数据盘可扩到 334 GB。
  - 训练：`configs/train/autodl_expert.yaml`。`train_expert_only=true`：VLM（4.44B）冻结，action expert（1.80B）全参训练。第一轮不带 depth/video teacher。其余尽量取官方值：`use_compile`、`enable_fp32`、`post_training`、lr 1e-4 余弦。
  - 数据：用 `scripts/autodl/pull_runtime.sh` 从北京B的旧实例原样拉取 LeRobot 数据、统计量、指令池，并比对校验和。不重新转换。
  - 评测：仍在北京B的 4080 上做，H800 只用来训练。
- 依据：
  - 已知事实：A800（北京B）数据盘最大只到 70 GB；本项目训练实例实测占用为环境 16 GB、模型 24 GB、数据 5.1 GB。
  - 估算：一个全量 DCP 约 40 GB（fp32 全量权重 25 GB + expert 的 Adam 状态 14 GB）；单卡显存需求约 47 GB 加激活。
  - 推测：H800 的 bf16/TF32 算力约为 A800 的 3 倍，单价只贵 25%。是否成立取决于数据加载能否跟上，需要测速确认。
- 风险：
  - 西北B 那台机 8 卡只空 1 卡，无卡模式装环境期间 GPU 可能被别人租走；
  - VLM 冻结可能低于官方的全模型训练（官方 `train_expert_only=false`）；
  - 对全模型训练做全量微调，单卡 80 GB 放不下（fp32 + Adam 约 100 GB）。
- `post_training=false`（与官方的 true 不同）：
  - 已知事实：基座 `model.safetensors.index.json`（rev 11c703bf，经 hf-mirror 读取）共 1708 个键，其中 95 个属于对齐头和对齐 token（`model.*_align_head`、`*_align_embs`、`*_shared_task_proj`），没有顶层的 `expert_visual.` 键。
  - 已知事实：`modeling_lingbot_vla_v2.py` 在 `align_params` 为空时不构建这些模块。`module_utils.load_model_weights` 在 post_training 模式下遇到多余键会抛 `KeyError`；为 false 时只记日志并跳过。
  - 结论：不带 teacher 时必须用 false，这与 AMD LoRA 路径的做法一致。
- 状态：已被 D-11 取代（H800、单卡、`post_training=false` 等保留；训练范围改为 C）。

## D-11 训练范围：只训 expert（A）/ 全模型 2 卡（B）/ VLM LoRA + expert 全参（C）

- 日期：2026-10-14
- 背景：D-10 采用 A（`train_expert_only=true`），需要评估 A 与全模型训练的差距。
- 已知事实：
  - 官方 HF `lingbotvla_cli.yaml` 为 `train_expert_only: false`（全模型）。
  - VLM4VLA（Zhang et al., "VLM4VLA: Revisiting Vision-Language-Models in Vision-Language-Action Models", arXiv 2601.03309，表 3；2026-10-25 核对 v2）：只冻结视觉编码器，SimplerBridge 成功率下降 21–42 个百分点，Calvin 平均完成子任务数下降 1.0–3.0。该实验的 VLM 是普通 VLM，未做过 VLA 预训练。
  - 全模型 fp32 + Adam 约 100 GB，单卡 80 GB 放不下。
- 推测：lingbot 的 VLM 已做过 VLA 预训练，冻结造成的损失应小于 VLM4VLA 的结果，但 randomized 场景（背景、光照、杂物变化）仍可能明显受损。
- 候选：
  - A：`autodl_expert.yaml`，已实测 7.3 s/step。
  - B：`autodl_full.yaml`，2×H800 FSDP2，官方配方。
  - C：`autodl_vlmlora.yaml`，新补丁 `lingbot-vla-v2-vlm-lora-expert-full.patch`：VLM 的视觉和语言注意力层加 LoRA，其余非 VLM 参数全参训练。合并时 `merge_lora_dcp.py` 从 DCP 取全部训练后权重，基座只提供非权重文件。
- 决策规则（开训前）：B、C 各测速约 15 分钟，按预算内能训的步数选。效果无法在开训前对比，原因是跨区传权重加评测要 6 h 以上。
- 测速（EXP-20261015-01）：A 7.3 s/step（1 卡）；C 9.6（1 卡）；C 12.4（2 卡）；B 28.8（2 卡）。按每千步成本计：A ¥20、C ¥27、B ¥160。本机两卡之间无 P2P，多卡没有收益。
- 决策（2026-10-15）：C，20k 步，1 卡 H800，save_steps 500，HF 快照每 2500 步一个。
- 状态：已定，主训练 EXP-20261015-02

## D-12 评测数据留存与评测视频

- 日期：2026-10-15
- 已知事实：
  - 初赛必交材料为 results.json（clean / randomized 下 50 个任务的测试次数与成功次数）、代码、权重、复现与调优说明（《赛题与数据》§2.3–2.4，比赛页面，不随仓库分发）。
  - 视频只出现在真机 Demo（05 目录，可选，本队不做），仿真评测不要求视频。
  - 正式提交的成绩要求每个任务测 100 次。
- 决策（2026-10-15）：
  - 阶段性评测不录视频，只保存日志和成绩。
  - 关键快照（首个评测的快照、最终选定的快照，另加至多 1 个）录视频：`EVAL_VIDEO=true`，每个任务、每个设置录 1 次。用途：失败分析、复现报告配图。
  - 留存方式：
    - 原始日志与视频拉回本地 `outputs/eval/<run>/<step>/`（不进 git）；
    - 每个快照的汇总（总成功率与各任务成功次数）写入实验记录（`EXP-*`，未公开）。
- 待实测：首次评测时用 1 个任务测出录视频的单段大小，以及录视频会让评测慢多少。
- 状态：已定；录视频的单段大小与耗时未单独记录。

## D-13 两阶段训练恢复与 DCP 保护

- 日期：2026-10-16
- 事故：10-16 07:10:48，stage 1 在保存 step 8000 的 DCP 过程中被平台 SIGKILL（exitcode -9，cgroup `oom_kill` 为 0，与 10-15 11:18 相同）。`dcp_rotation` 补丁在 keep=1 时**先删旧 DCP 再写新 DCP**，于是 step 7500 已删、step 8000 只写出 model 的 22 GB（无 `.metadata`、optimizer、extra_state），没有可续训的 DCP。supervisor 的 attempt 3 报 `global_step_8000/model/.metadata` 不存在，已手动停止。
- 决策：
  - **第二阶段**：`configs/train/autodl_vlmlora_s2.yaml`，以 stage 1 step 7500 的合并 HF 快照（4080 上已用 sha256 校验）作为 `init_path`。
    - 其余配置与 stage 1 相同（VLM LoRA + 非 VLM 全参，mb16 / gbs128）。
    - max_steps 12500，即全局步 = 7500 + 本阶段步。
    - lr 8.46e-5 接 stage 1 的余弦，2% 预热后余弦降到 5e-5。
    - Adam 状态重置。
  - **DCP 保护**：
    - `dcp_keep=2`，保存新 DCP 时上一份仍在；
    - `scripts/lib/ckpt_guard.sh`：supervisor 每次尝试前把残缺的 `global_step_*`（缺 `model/.metadata` 或 `extra_state`）移到 `checkpoints_incomplete/`，不删除，有测试覆盖。
  - **内存**：supervisor 每 30 s 记录容器内存（cur/anon/shmem/file），超过 105 GiB 时在 supervisor 日志中报 MEM ALERT。根因（保存时的 CPU 暂存 + 长时间运行后的基线增长，属推测）未定位，有保护后，一次被杀最多损失 500 步。
- 状态：已执行（stage 2 完成，EXP-20261015-02）。
- 代价：丢失 step 7500→8000（约 1.5 h，约 ¥15），以及停机约 ¥5；优化器状态重置。
- 复现影响：最终权重来自两个阶段，复现报告须写明两条命令。stage 1 按 20k 余弦跑到 7500 步后合并；stage 2 从该合并权重再训 12500 步。
- 2026-10-16 补充（内存）：
  - cgroup v2 的 `memory.current` 把页缓存也算进去，`file` 一项中包含 shmem。
  - stage 2 刚启动时 cur 为 110 GiB（anon 21 + shmem 63 + 其他页缓存约 24），当时页缓存主要来自刚拉回和校验的快照。
  - 推测（未证实）：平台看护按 `memory.current` 判断超限；保存 DCP 时写出的 37 GB 页缓存会把总量推过 120 GiB，与两次 SIGKILL 都出现在大量文件 I/O 期间相符。
  - 对策：`src/evict_cache.py` 先 fdatasync 再 posix_fadvise DONTNEED，无需特权，单次清理实测从 110 降到 87 GiB；配合 DataLoader 预取深度 4→2，保存窗口峰值从 118 降到 90 GiB（EXP-20261015-02）；`scripts/autodl/evict_loop.sh` 每 15 s 对输出目录执行一次，训练数据目录除外。

## D-14 评测改为单卡 1 服务 + 多客户端

- 日期：2026-10-17
- 依据：
  - 服务端本来就是 chunk 模式（上游 argparse 默认 `--chunk_ret True`），调用之间无状态；请求在事件循环中串行处理（代码核对见 `docs/specs/eval.md` E5）。
  - 实测（EXP-20261015-02，10-17 16:14）：1 个服务挂 5 个客户端，无报错，推理延迟基本不变（p95 ≤ 2.1 s），episode 墙钟不变；每个客户端占约 2.2–2.7 GB 显存。
- 决策：后续评测（含最终快照）一律用 `CLIENTS=N bash scripts/cloud/eval_subset.sh`，32 GB 卡取 N=6–7，不再在一张卡上起 2 个服务（后经 D-15 修订，最终上限 clean 6、randomized 3，见 `configs/eval/clients.env`）。
- 合规：单一权重、无路由；每个任务仍是独立的 `--seed 0` 运行，seed 与 episode 的对应关系不变，所有结果都计入；不修改 RoboTwin 评测代码（`docs/rules-checklist.md` C）。README 中披露 `USE_LENGTH=25`、`CHUNK_RET=True`，两者都是上游默认行为。
- 风险：若显存峰值超过估计，客户端会报 OOM。首次使用 N=7 时用显存采样确认，失败则降到 N=6。
- 状态：已定。

## D-15 提交全局 20000 权重；正式自测每任务 30 ep（原定 40 ep）

- 日期：2026-10-17
- 决策：
  - 提交权重为 stage 2 step 12500（全局 20000）。依据是 1 ep 对比中 19000 与 20000 的差异在误差范围内，而 20000 是计划终点，不靠噪声挑选（EXP-20261015-02）。
  - 正式自测：clean 与 randomized 各 50 任务 × 40 ep，`--seed 0`，单次运行。种子序列与官方 seed 0 复测前 40 个 episode 相同。1 服务 + 8 客户端，长任务优先（D-14）。
- 依据（费用）：
  - 实测每 episode 约 3.9 客户端·分钟，8 客户端下 40 ep × 50 任务 × 2 设定 ≈ 32 h ≈ ¥51。
  - 两台数据盘到 10-26 约 ¥27。
  - 在上述费用之外留出应对意外的余量。
  - 100 ep（官方建议）约 ¥128，预算不足以测满 100 ep；按 QA 10，未测满可以提交，以官方复测为准。
- 风险：
  - 8 客户端的显存峰值估计约 29.5 GB / 32 GB，评测监控有 OOM 告警，出现 OOM 就降到 7 个客户端续跑（已完成的任务会跳过）。
  - 收尾时间由 put_bottles_dustbin 决定（约 20 min/ep × 40 ≈ 13 h），短于单个设定的总量（约 16 h），长任务优先时影响不大。
- 2026-10-17 21:10 修订：8 客户端时显存打满（32.2 GB），出现渲染错误（共 15 个 episode 被 RoboTwin 记为失败）和客户端卡死，该次 40 ep 运行作废（移到 4080 实例的 `runtime/invalid/`，未删除）。改为 **6 客户端 × 30 ep**：6 客户端下 30 ep 约 32.5 h、¥51，与原预算相当；6 客户端下 40 ep 约 43 h、¥68，应对意外的余量不够，所以没选。此前 1 ep 的各轮评测都检查过，没有这类错误。
- 状态：已执行（clean / randomized 各 50 任务 × 30 ep 完成；randomized 客户端上限后降为 3，见 D-14）。

## D-16 补交一次：权重、结果不变，只更新复现文档（2026-10-25）

- 决策：用剩余 2 次中的 1 次补交。权重 23 个文件和 results.json 与 10-23 的包逐字节相同；README、复现报告、`train.sh`、`tools/package.py` 使用新 commit；zip 条目改为小文件在前。
- 依据：
  - 《赛题与数据》§2.4、QA 赛制 4 规定：复现细节缺失、描述不清，导致评委无法复现核验的，后果由参赛队承担。
  - EXP-20261024-01 的评委视角演练发现，旧 README 缺少基础镜像、解压方式、数据准备内存要求、权重校验说明、重跑随机性说明，runbook 的 CUDA 版本也写错了。
  - 官方不提供中间反馈，保留提交次数没有信息上的收益。新旧包的权重和结果相同，无论官方按哪次提交计，分数都不受影响；唯一的风险是新包打错，用 10-23 的整套校验来控制。
- 兼容性（覆盖 Windows / Linux / AMD）：
  - 乱码问题在打包层面无法根治，只能缓解（条目顺序 + README 说明），见 `docs/specs/submission.md`。
  - Windows 解压已实测。
  - AMD 路径未实测，已在 README 如实写明。
- 来源：队内讨论（10-25）；《赛题与数据》（比赛页面，不随仓库分发）。
- 状态：已执行（10-25 上传）。

## D-17 开源发布范围（2026-10-25）

- 决策：赛后公开发布复现所需的代码、配置、测试、规格、决策记录与技术报告；实验过程记录、排期与费用账本、提交前复核清单、一次性评测编排脚本不公开。发布版以一个初始提交开始，之后的更新以普通提交追加。
- 依据：过程记录面向队内，包含实例、费用流水等与复现无关的信息；复现所需的结论已写入技术报告、`docs/submission/README.md` 与本决策日志。文中出现的实验编号（EXP-*）和提交哈希见 `docs/provenance.md`。
- 状态：已定。
- 来源：队内讨论（10-25）。

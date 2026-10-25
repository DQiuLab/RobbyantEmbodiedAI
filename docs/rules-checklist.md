# 合规清单

来源：赛事官方《常见 Q&A》（下称 QA）、《论坛》（赛题页链接的官方规则解读，下称 RD），均见比赛页面，不随仓库分发。
本文件是核对模板：任何训练 / 推理 / 数据处理改动前逐条核对，结果写进对应的实验记录（`EXP-*`，未公开，见 `docs/provenance.md`），因此这里的复选框保持未勾选。

## A. 数据来源轴

- [ ] 训练数据仅为 RoboTwin 2.0 Aloha-AgileX 50 任务 `demo_clean`，每任务 50 条（QA 赛题1；RD 1.2）
- [ ] 不含任何 `demo_randomized` 数据，任何比例、任何形式（RD 3.1）
- [ ] 不含外部数据（真机、第三方数据集）（RD 1.2）
- [ ] 不经仿真渲染 / 交互生成新数据：无自建 randomization、无额外 rollout、无 `check_success()` 打分筛选（QA 18；RD 3.1）
- [ ] 指令只来自 seen_instructions 池；unseen 指令不以任何形式进训练；不用 LLM 生成新指令（RD 3.1）
- [ ] 无 HER 伪目标重标注（RD 3.1）
- [ ] 归一化统计量只由 clean 数据计算（运行目录下的 `norm_stats_clean.json`，D-07）；不使用官方 `assets/norm_stats/robotwin.json`（含 randomized）
- [ ] 训练清单经 `src/data_checks.py` 校验：恰好 50 个 demo_clean 任务、无 randomized 条目

允许：训练期视觉增广（亮度/颜色噪声）、机器人状态扰动、训练机制内部噪声、jerk/Z-score 质量过滤、seen 池内指令替换、按数据自身统计量重加权（QA 18；RD 2.1）。

## B. 模型起点轴

- [ ] 主干从 `robbyant/lingbot-vla-v2-6b` 初始化（QA 6、14）
- [ ] 未以 `lingbot-vla-v2-6b-robotwin` 作起点、教师、reward / 判据模型（RD 3.2）
- [ ] 引入的公开预训练模块已在 README 注明来源与用法（QA 7）
- 官方配方的 LingBot-Depth / DINO-Video 双教师（`robotwin.yaml` 的 `align_params`）允许（RD 3.4）

## C. 推理形态轴

- [ ] 推理只加载一份权重（QA 9；RD 3.3）
- [ ] 无按任务 ID / instruction 的确定性路由（QA 8、9）
- [ ] 无多 checkpoint ensemble、无多 seed 投票 / TTA（QA 17；RD 3.3）
- [ ] 无"失败检测后重置 episode"（RD 2.4、3.3）

允许（需在 README 披露）：chunk 执行长度、replanning、action smoothing、temporal ensemble、基于自身输出的 confidence gating、精度与编译开关（RD 2.4）。

## D. 学习方法轴

- [ ] 训练全程零环境调用（RD 3.4 三问①）
- [ ] reward / 标签 / 权重全部可由 release 数据内部推导（三问②）
- [ ] 教师或起点不涉及官方 robotwin 权重（三问③）

## E. 文档披露义务（RD 4.3）

- [ ] LoRA 合并等权重形态说明
- [ ] 公开预训练模块来源
- [ ] 评测环境细节（自测为 `CUDA + MPLib + expert_check=true`，RTX 4080 SUPER）
- [ ] 对 RoboTwin 官方评测代码的修改内容与文件位置（模板《代码材料目录要求》第 5 条）

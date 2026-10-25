# RobbyantEmbodiedAI · 项目规范

蚂蚁灵波具身大模型挑战赛初赛：基于 `lingbot-vla-v2-6b`，在 RoboTwin 2.0 Aloha-AgileX 50 任务 clean 数据上训练一份 cotrain 权重并提交。
本文件是开发期间与 AI 助手（Claude）协作时使用的项目规范，随发布版保留供参考。操作手册：`docs/runbook.md`；决策日志：`docs/decisions.md`。

## 目录约定

| 路径 | 放什么 | 规则 |
|---|---|---|
| `Robbyant-Embodied-AI-Challenge/` | 官方赛事材料与模板 | 版权属主办方，不随仓库分发（`.gitignore`）；需要时从比赛页面获取并放到此路径；测试在运行时生成同结构的模板 |
| `docs/rules-checklist.md` | 合规清单 | 任何训练/推理改动前逐条对照 |
| `docs/specs/` | SpecDD 规格（training / eval / submission） | 先改 spec，再改实现 |
| `docs/decisions.md` | 决策日志 | 每条含日期、决策、依据、来源 |
| `configs/` | `train/`、`eval/`、`env/*.env`、`platform/{amd,autodl}.env`、`robot*/`、`submission.yaml`、`budget.yaml` | 参数只在这里定义，脚本不硬编码 |
| `scripts/cloud/` | 平台通用的云端脚本（训练冒烟、转换/合并、服务、评测）；`PLATFORM=amd` / `PLATFORM=autodl` 选路径 | 必须幂等、`set -euo pipefail` |
| `scripts/autodl/` | AutoDL 专用：装环境、下载、数据准备、显存冒烟、渲染测试 | 幂等，可重复执行 |
| `scripts/lib/` | 公共 shell 库：`platform.sh`（按 `PLATFORM` 加载路径）、`our_patches.sh` | 被 cloud/autodl 脚本 source |
| `scripts/local/` | 本地 ↔ 实例同步（rsync over SSH） | 不传 DCP、数据、上游源码 |
| `src/` | 我们对上游的改动：`launch_train.py`、`lingbot_patches/`（可单测的模块）、`patches/`（对上游的 git patch） | 补丁须能在 AMD 补丁之上 `git apply`；改动写进 `docs/submission/README.md` §5 |
| `configs/reference/` | 上游/官方配置的原样副本（首行注明来源） | 只读，用于测试比对 |
| `tools/` | 本地 Python 工具 | 必须有 `tests/` 对应测试 |
| `tests/` | pytest | — |
| `submission/` | `tools/package.py` 生成的提交包 | gitignore，不手工编辑 |

- 数据、checkpoint、原始日志、上游源码副本**不进 git**。上游代码由 `scripts/autodl/setup_env.sh` 按固定 commit 拉取并打我们的补丁；本地阅读用的上游 clone 放 `third_party/`（gitignore）。
- 文件命名：Python `snake_case.py`；shell 统一用 `snake_case.sh`。

## 合规红线（详见 `docs/rules-checklist.md`）

- 训练只用 clean 数据（50 任务 × 50 条）与 seen 指令池；randomized 数据、unseen 指令、外部数据、仿真额外 rollout 一律禁止。
- 起点只能是 `robbyant/lingbot-vla-v2-6b`；禁止用 `lingbot-vla-v2-6b-robotwin` 作起点或教师。
- 推理只能一份权重；禁止按任务 ID / instruction 路由、多 seed 投票、失败重置 episode。

## 云端纪律

- 每次云端运行都记录 commit、配置、step、结果与费用；用完立即关机。
- 数据盘只保留一个 DCP；HF 快照转换后立即下载回本地。
- ROCm / CUDA 差异只通过 `configs/env/{rocm,cuda}.env` 切换。

## 验证命令

```bash
python3 -m pytest tests/ -q          # 本地工具测试，提交前必须全绿
```

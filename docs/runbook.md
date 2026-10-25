# 云端操作手册（AutoDL 版，D-08）

AMD 路径未实测：脚本保留了 `PLATFORM=amd`（初版 AMD 方案，未用于最终权重）。

**正式流程**：训练用 `bash train.sh`，评测用 `bash eval.sh`（见 `docs/submission/README.md`）。下文是分步手册，§6–§7 的命令属于早期 LoRA 方案（D-08）。

**约定：**
- 实例上的仓库路径：`/root/autodl-tmp/RobbyantEmbodiedAI`（数据盘）。
- 每一步都在 `tmux` 里跑，防止 SSH 断开。
- 开机、关机、换开机模式都要记入运行记录（时间与费用）。
- **不用 GPU 时立刻关机，或改用无卡模式开机。**

## 0. 创建实例（控制台）

1. 选北京B区的 **vGPU-32GB（RTX 4080 SUPER）**，按量计费，1 卡。
2. 镜像选基础的 **Miniconda**（Ubuntu 22.04 + CUDA 11.8，10-24 在实例上核实）。我们自己建 conda 环境，不依赖镜像自带的 CUDA 和 PyTorch（torch cu128 自带运行时，驱动需 ≥ 570）。镜像的 `LANG=en_US.UTF-8` 没有安装对应 locale：解压含中文路径的 zip 时要用 `LANG=C.UTF-8 unzip`（EXP-20261024-01 问题 2）。无卡模式只有 2 GB 内存，不能运行 `prepare_data.sh`。
3. 数据盘扩容到 **≥ 200 GB**，先看清每天的单价。用量：两个环境约 40 GB、模型约 26 GB、仿真资产约 16 GB、LeRobot 数据，再加合并权重约 12–24 GB。
4. 创建后实例会以 GPU 模式开机，**马上关机**，再选"无卡模式开机"（¥0.1/时，0.5 核、2 GB 内存）。
   - 风险：无卡模式会释放 GPU，之后正常开机时 GPU 可能被别人占用。遇到这种情况就等一会儿，或者"克隆实例"到同区的其他主机。

## 1. 同步代码（本地）

```bash
# SSH 信息在控制台复制，形如 ssh -p 12345 root@region-xx.autodl.pro
rsync -az -e "ssh -p <port>" --exclude .git --exclude data --exclude third_party --exclude .deps --exclude submission \
  ./ root@<host>:/root/autodl-tmp/RobbyantEmbodiedAI/
```

## 2. 装环境与下载（无卡模式，约 1–3 小时 × ¥0.1）

```bash
cd /root/autodl-tmp/RobbyantEmbodiedAI
tmux new -s setup
bash scripts/autodl/setup_env.sh 2>&1 | tee -a /root/autodl-tmp/setup_env.log
bash scripts/autodl/fetch_assets.sh 2>&1 | tee -a /root/autodl-tmp/fetch_assets.log
```

两个脚本都能重复执行，出错后修复再跑就能接着上次的进度。无卡模式下 `setup_env.sh` 会跳过 GPU 相关的校验。

## 3. GPU 开机：环境校验、渲染、显存决策（约 1–2 小时 × ¥1.58）

关机后正常开机：

```bash
cd /root/autodl-tmp/RobbyantEmbodiedAI && tmux new -s gpu
bash scripts/autodl/setup_env.sh            # 再跑一次，完成 GPU 校验：显卡型号、capability、flash-attn 内核
bash scripts/autodl/test_render.sh          # 输出 "Render Well" 才说明能做仿真评测
bash scripts/autodl/prepare_data.sh adjust_bottle
bash scripts/autodl/smoke_mem.sh adjust_bottle 20
```

| 结果 | 处理 |
|---|---|
| `setup_env` 报 flash-attn 内核失败（例如 sm_120） | 停下来，把日志发回本地处理（改注意力实现），不要在 GPU 计费时现场调试 |
| `test_render` 失败 | 训练照常进行；评测另想办法（换一张能渲染的卡只做评测），并记入实验记录 |
| `smoke_mem` 某一档 OK | 记下这一档的 micro batch、峰值显存、s/it，进入 §4 |
| 全部 OOM | 迁移到北京B区的 vGPU-48GB（¥2.78/时），从 §3 重新开始 |

## 4. 准备全部数据（GPU 开机，16 核）

```bash
JOBS=4 bash scripts/autodl/prepare_data.sh ALL --finalize 2>&1 | tee -a /root/autodl-tmp/prepare_data.log
```

- 产物：
  - `runtime/robotwin_demo_clean_joint_v30.txt`（已校验，恰好 50 个 clean 任务）；
  - `runtime/seen_instruction_pool.json`（打印出覆盖率）；
  - `runtime/norm_stats_clean.json`（打印出帧数）。
- 某个任务失败：看 `runtime/outputs/logs/prepare_data/<task>.log`，修复后重跑同一命令，已完成的任务会跳过。

## 5. 全链路冒烟（GPU）

```bash
PLATFORM=autodl bash scripts/cloud/smoke.sh run.micro_batch_size=<§3 的结果>
```

检查项：训练 10 步 → 预置 STOP 文件续训（验证 resume、优雅停止、DCP 轮转）→ 合并 LoRA → 起服务 → `adjust_bottle` 跑 1 个 episode。

## 6. 定步数并训练（GPU；早期 LoRA 方案，D-08）

> 本节是早期 LoRA 方案（D-08）的流程。最终权重的两阶段训练用 `bash train.sh`（`configs/train/autodl_vlmlora.yaml` → `autodl_vlmlora_s2.yaml`，无人值守包装 `scripts/autodl/train_supervised.sh`）。

**本地：** 按 §3 测到的 s/it 和剩余预算算出 `run.max_steps`，写进 `configs/train/autodl_lora.yaml`，提交 commit，再同步到实例。

**实例：**

```bash
tmux new -s train
PY=/root/autodl-tmp/envs/rt/bin/python
$PY -m src.launch_train --config configs/train/autodl_lora.yaml --dry-run
$PY -m src.launch_train --config configs/train/autodl_lora.yaml \
  2>&1 | tee -a /root/autodl-tmp/runtime/outputs/logs/lora_run.log; /usr/bin/shutdown
```

- 命令末尾的 `; /usr/bin/shutdown`：训练结束（或出错）后自动关机，不再计费（autodl.com/docs/save_money）。
- **中途停止：** `touch /root/autodl-tmp/runtime/STOP`，当前 step 结束后会保存再退出；之后自动关机。**不要 Ctrl-C。**
- **续训：** 先 `rm /root/autodl-tmp/runtime/STOP`，再跑同样的命令。几分钟内应该看到 `Load distributed checkpoint ... successfully`。

## 7. 合并、评测、拉回（GPU；早期 LoRA 方案，D-08）

> 正式评测用 `bash eval.sh <checkpoint_dir> both <n_ep> 0`（多客户端并行见 `docs/submission/README.md` §1）。

```bash
PLATFORM=autodl bash scripts/cloud/merge.sh /root/autodl-tmp/runtime/outputs/lora_run <step>
M=/root/autodl-tmp/runtime/outputs/lora_run/merged/global_step_<step>
PLATFORM=autodl bash scripts/cloud/eval_subset.sh $M final_clean demo_clean 1 0 ALL
PLATFORM=autodl bash scripts/cloud/eval_subset.sh $M final_rand demo_randomized 1 0 ALL
```

评测可以重复执行，已有结果的任务会跳过，所以可以分几次开机跑完。

**本地：**
- `rsync` 把 `merged/` 和 `outputs/final_*` 拉回本地，`tools/results.py` 生成 `results.json`。
- 确认本地文件完整后关机；不再需要的话释放实例，这样数据盘扩容也不再计费。

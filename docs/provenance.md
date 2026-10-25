# 版本说明

## 发布范围

本仓库公开复现所需的内容：代码、补丁、配置、测试、规格文档、决策日志（`docs/decisions.md`）和技术报告（D-17）。首次发布是一个初始提交，之后的更新以普通提交追加。

开发过程中的实验记录（`EXP-*`）、排期与费用账本、提交前复核清单没有公开。文档中出现的实验编号（如 `EXP-20261015-02`）只作标识，相关结论已写入技术报告、`docs/submission/README.md` 和决策日志。

## 文中的提交哈希

文档和提交包 `02_代码材料/COMMIT` 中出现的提交哈希（如 `6bef0fc`、`b9f5901`）指开发仓库，在本仓库中无法解析。两次提交以 zip 的 sha256 为准：

| 提交 | 开发仓库提交 | zip sha256 |
|---|---|---|
| 第 1 次（2026-10-23） | `6bef0fc` | `9ef481184d6230b5dee38bb40ef593b45719e4fcc1f79787bacbc8dfebba76c2` |
| 第 2 次（2026-10-25，D-16） | `b9f5901` | `2efd95c831f9bf6801dcabba2b6d4699c6f60ea1eaa4f32019f7795c17bf92a3` |

## 与提交包的差异

- 主办方赛事材料（赛题、规则、模板）版权属于主办方，不随仓库分发，见[比赛页面](https://tianchi.aliyun.com/competition/entrance/532514)。打包需要的结果模板请自行获取，放到 `configs/submission.yaml` 中 `template_json` 指向的路径；测试在运行时生成同结构的模板，不依赖主办方文件。
- 来源未声明许可证的配置副本不随仓库分发：`configs/reference/official_robotwin_cli.yaml`（来自 HF `robbyant/lingbot-vla-v2-6b-robotwin`）只在 `configs/reference/README.md` 中给出链接和 sha256，缺失时对应测试自动跳过；AMD 参考样例的配置副本 `configs/reference/amd_base_cli.yaml` 同样不分发。
- 实例地址、端口、本机路径等环境信息改为占位符。
- 与第 2 次提交相比，训练、推理、评测与打包的逻辑没有改动。代码上的调整有：
  - `scripts/autodl/train_supervised.sh`、`train_after_prep.sh` 改为默认不关机（设 `SHUTDOWN=1` 才关机），`train.sh` 相应去掉 `NO_SHUTDOWN=1`；`train_after_prep.sh` 的输出目录改为从训练配置读取；
  - `scripts/autodl/push_loop.sh` 的 ssh 参数与路径改为与 `push_snapshot.sh` 一致；
  - `tools/results.py` 把 `configs/submission.yaml` 的占位符 `<team_id>` 也视为未填写；
  - `src/patches/lingbot-vla-v2-infer-num-steps.patch` 改为 git diff 格式，只改了注释；
  - 测试改为运行时生成结果模板；其余为注释与文档措辞。

# configs/reference

上游与官方配置的原样副本（首行注明来源），只读，用于测试比对。

- `upstream_robot_robotwin.yaml`：`Robbyant/lingbot-vla-v2@951475a` 的 `configs/robot_configs/robotwin.yaml`（Apache-2.0）。
- `official_robotwin_cli.yaml`：官方 RoboTwin 配置，来源 <https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin/resolve/main/lingbotvla_cli.yaml>（2026-10-11 下载，sha256 `e120a869a438b8a180492b4e6a822ced01c455e06b25bc13cedec768275cea34`）。该 HF 仓库未声明许可证，本仓库不分发此文件；需要时下载到本目录，`tests/test_launch_train.py` 中对应的对齐测试会自动启用（缺失时跳过）。只读取其中的超参数，不使用该仓库的权重。

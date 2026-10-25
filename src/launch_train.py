"""Launch LingBot-VLA-v2 training (full SFT or LoRA) from configs/train/*.yaml.

Replaces train_full_sft.sh of the AMD reference sample (Robotwin-radeon-cloud, listed in the
challenge introduction): same launcher shape, but resume,
align params, augmentation and checkpoint rotation come from config files.
Spec: docs/specs/training.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _bool(value) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def parse_env_file(path: Path) -> dict[str, str]:
    env = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def _fill(obj, **paths):
    if isinstance(obj, dict):
        return {k: _fill(v, **paths) for k, v in obj.items()}
    if isinstance(obj, str):
        return obj.format(**paths)
    return obj


def load_config(path: Path, repo: Path = REPO) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if cfg.get("align_params_file"):
        align = yaml.safe_load((repo / cfg["align_params_file"]).read_text(encoding="utf-8"))
        cfg["align_params"] = _fill(align, model_root=cfg["platform"]["model_root"])
    else:
        cfg["align_params"] = None  # no depth / video teacher
    return cfg


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    """Apply `section.key=value` overrides; values are parsed as YAML scalars. Keys must already exist."""
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must be section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        *parents, leaf = dotted.split(".")
        node = cfg
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key {dotted!r}")
        node[leaf] = yaml.safe_load(raw)
    return cfg


def build_command(cfg: dict, repo: Path = REPO) -> tuple[list[str], dict[str, str]]:
    platform, run = cfg["platform"], cfg["run"]
    gpu_count, micro, global_bs = int(run["gpu_count"]), int(run["micro_batch_size"]), int(run["global_batch_size"])
    if global_bs % (gpu_count * micro):
        raise ValueError(f"global_batch_size {global_bs} not divisible by gpu_count*micro_batch_size {gpu_count * micro}")
    gpu_ids = str(run["gpu_ids"])
    if len(gpu_ids.split(",")) != gpu_count:
        raise ValueError(f"gpu_ids {gpu_ids!r} must list exactly {gpu_count} devices")

    model_root = platform["model_root"]
    overrides: dict[str, object] = {
        "--data.train_path": platform["data_list"],
        "--data.num_workers": run["num_workers"],
        "--data.prefetch_factor": run["prefetch_factor"],
        "--train.output_dir": run["output_dir"],
        "--train.data_parallel_shard_size": gpu_count,
        "--train.micro_batch_size": micro,
        "--train.gradient_accumulation_steps": global_bs // (gpu_count * micro),
        "--train.global_batch_size": global_bs,
        "--train.max_steps": run["max_steps"],
        "--train.save_steps": run["save_steps"],
        "--train.enable_resume": run["enable_resume"],
        "--model.model_path": cfg["model"].get("init_path") or f"{model_root}/{cfg['model']['model_dir']}",
        "--model.config_path": f"{model_root}/{cfg['model']['model_dir']}",
        "--model.tokenizer_path": f"{model_root}/{cfg['model']['tokenizer_dir']}",
        "--model.post_training": cfg["model"]["post_training"],
    }
    if cfg["align_params"] is not None:
        overrides["--train.align_params"] = json.dumps(cfg["align_params"], separators=(",", ":"))
    for key, value in cfg["model"].items():  # remaining model keys (e.g. attention implementation) pass through
        if key not in ("model_dir", "tokenizer_dir", "init_path", "post_training"):
            overrides[f"--model.{key}"] = value
    for key, value in cfg["train"].items():
        overrides[f"--train.{key}"] = value
    for key, value in cfg["data"].items():
        if key == "robot_config_root":
            value = str((repo / value).resolve())
        overrides[f"--data.{key}"] = value

    argv = [platform["python"], "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpu_count}",
            "-m", "tasks.vla.train_lingbotvla", platform["base_config"]]
    for flag, value in overrides.items():
        argv += [flag, _bool(value)]

    env = parse_env_file(repo / platform["env_profile"])
    visible_var = env.pop("GPU_VISIBLE_VAR")
    env[visible_var] = gpu_ids
    if run.get("dcp_keep") is not None:
        env["LINGBOT_DCP_KEEP"] = str(run["dcp_keep"])
    if run.get("require_resume"):
        env["LINGBOT_REQUIRE_RESUME"] = "1"
    if run.get("stop_file"):
        env["LINGBOT_STOP_FILE"] = str(run["stop_file"])
    if run.get("stop_at_step"):
        env["LINGBOT_STOP_AT_STEP"] = str(run["stop_at_step"])
    if run.get("compile_flex"):
        env["LINGBOT_COMPILE_FLEX"] = "1"
    if run.get("fsdp_single_gpu"):
        env["LINGBOT_FSDP_SINGLE_GPU"] = "1"
    if run.get("seen_instruction_pool"):
        env["LINGBOT_SEEN_INSTRUCTION_POOL"] = str(run["seen_instruction_pool"])
    return argv, env


def raise_nofile_limit() -> int:
    """Raise the open-file soft limit to the hard limit.

    AutoDL containers default to soft=1024; dataloader workers (video decoding + shared-memory
    tensors) exceed it and training dies with 'Too many open files'. Returns the new soft limit.
    """
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    return hard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/train/main.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = apply_overrides(load_config(args.config), args.set)
    command, env = build_command(cfg)
    print(" ".join(f"{k}={v}" for k, v in sorted(env.items())))
    print(shlex.join(command))
    stop_file = cfg["run"].get("stop_file")
    if stop_file and Path(stop_file).exists():
        print(f"WARNING: stop file {stop_file} exists; training will save and exit after one step", file=sys.stderr)
    if args.dry_run:
        return 0

    Path(cfg["run"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **env}
    for var in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        if var not in env:
            full_env.pop(var, None)
    print(f"open-file limit raised to {raise_nofile_limit()}")
    os.chdir(cfg["platform"]["source_dir"])
    os.execvpe(command[0], command, full_env)


if __name__ == "__main__":
    sys.exit(main())

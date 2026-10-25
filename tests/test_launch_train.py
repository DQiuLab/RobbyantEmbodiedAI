import json
from pathlib import Path

import pytest
import yaml

from src import launch_train

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "configs/train/main.yaml"
OFFICIAL = REPO / "configs/reference/official_robotwin_cli.yaml"


def build(**run_overrides):
    cfg = launch_train.load_config(MAIN, REPO)
    cfg["run"].update(run_overrides)
    return launch_train.build_command(cfg, REPO)


def argv_value(argv, flag):
    idx = argv.index(flag)
    return argv[idx + 1]


def strip_paths(d):
    if isinstance(d, dict):
        return {k: strip_paths(v) for k, v in d.items() if not k.endswith("_path")}
    return d


def test_align_params_match_official_config_except_paths():
    if not OFFICIAL.exists():
        pytest.skip("官方 lingbotvla_cli.yaml 未随仓库分发，见 configs/reference/README.md")
    official = yaml.safe_load(OFFICIAL.read_text(encoding="utf-8"))["train"]["align_params"]
    ours = yaml.safe_load((REPO / "configs/train/align_official.yaml").read_text(encoding="utf-8"))
    assert strip_paths(ours) == strip_paths(official)


def test_align_params_paths_are_filled_from_model_root():
    argv, _ = build()
    align = json.loads(argv_value(argv, "--train.align_params"))
    model_root = "/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models"
    assert align["depth"]["moge_path"] == f"{model_root}/moge-2-vitb-normal/model.pt"
    assert align["video"]["ckpt_path"].startswith(model_root)
    assert "{model_root}" not in json.dumps(align)


def test_uses_python_module_launcher_not_torchrun():
    argv, _ = build()
    assert argv[:3] == ["/opt/robotwin-env/bin/python", "-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in argv
    assert argv[argv.index("-m", 3) + 1] == "tasks.vla.train_lingbotvla"


@pytest.mark.parametrize("flag,expected", [
    ("--train.enable_resume", "true"),
    ("--train.optimizer", "adamw"),
    ("--train.use_lora", "false"),
    ("--train.max_steps", "30000"),
    ("--train.save_steps", "1000"),
    ("--train.micro_batch_size", "16"),
    ("--train.global_batch_size", "256"),
    ("--train.gradient_accumulation_steps", "2"),
    ("--train.data_parallel_shard_size", "8"),
    ("--train.enable_fp32", "true"),
    ("--train.use_compile", "true"),
    ("--train.output_dir", "/workspace/runtime/outputs/main_run"),
    ("--data.image_augment", "true"),
    ("--data.use_future_image", "true"),
    ("--data.train_path", "/workspace/runtime/robotwin_demo_clean_joint_v30.txt"),
    ("--data.norm_stats_file", "/workspace/runtime/norm_stats_clean.json"),
    ("--model.post_training", "true"),
    ("--model.model_path", "/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b"),
])
def test_cli_overrides(flag, expected):
    argv, _ = build()
    assert argv_value(argv, flag) == expected


def test_every_flag_appears_once():
    argv, _ = build()
    flags = [a for a in argv if a.startswith("--train.") or a.startswith("--data.") or a.startswith("--model.")]
    assert len(flags) == len(set(flags))


def test_env_contains_rocm_profile_dcp_keep_and_devices():
    _, env = build()
    assert env["LINGBOT_DCP_KEEP"] == "1"
    assert env["HIP_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert env["ROBOTWIN_EE_PLANNER"] == "mplib"
    assert env["AITER_TRITON_ONLY"] == "1"
    assert "GPU_VISIBLE_VAR" not in env


def test_cuda_profile_sets_cuda_visible_devices(tmp_path):
    cfg = launch_train.load_config(MAIN, REPO)
    cfg["platform"]["env_profile"] = "configs/env/cuda.env"
    _, env = launch_train.build_command(cfg, REPO)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "HIP_VISIBLE_DEVICES" not in env


def test_rejects_indivisible_batch():
    with pytest.raises(ValueError, match="global_batch_size"):
        build(global_batch_size=250)


def test_rejects_gpu_ids_count_mismatch():
    with pytest.raises(ValueError, match="gpu_ids"):
        build(gpu_ids="0,1,2,3")


def test_dcp_keep_none_leaves_env_unset():
    _, env = build(dcp_keep=None)
    assert "LINGBOT_DCP_KEEP" not in env


def test_parse_env_file_ignores_comments(tmp_path):
    f = tmp_path / "x.env"
    f.write_text("# c\nA=1\n\nB = two words\n", encoding="utf-8")
    assert launch_train.parse_env_file(f) == {"A": "1", "B": "two words"}


def test_dry_run_prints_command(capsys):
    assert launch_train.main(["--config", str(MAIN), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "tasks.vla.train_lingbotvla" in out and "LINGBOT_DCP_KEEP=1" in out


def test_apply_overrides_parses_yaml_values():
    cfg = launch_train.load_config(MAIN, REPO)
    launch_train.apply_overrides(cfg, ["run.max_steps=10", "train.optimizer=muon", "run.dcp_keep=null",
                                       "train.use_compile=false", "run.output_dir=/workspace/x"])
    assert cfg["run"]["max_steps"] == 10
    assert cfg["train"]["optimizer"] == "muon"
    assert cfg["run"]["dcp_keep"] is None
    assert cfg["train"]["use_compile"] is False
    assert cfg["run"]["output_dir"] == "/workspace/x"


def test_apply_overrides_rejects_unknown_key():
    cfg = launch_train.load_config(MAIN, REPO)
    with pytest.raises(KeyError):
        launch_train.apply_overrides(cfg, ["run.max_stepz=10"])


def test_apply_overrides_rejects_missing_equals():
    cfg = launch_train.load_config(MAIN, REPO)
    with pytest.raises(ValueError):
        launch_train.apply_overrides(cfg, ["run.max_steps"])


def test_cli_set_flows_into_command(capsys):
    assert launch_train.main(["--dry-run", "--set", "run.max_steps=10", "--set", "train.optimizer=muon"]) == 0
    out = capsys.readouterr().out
    assert "--train.max_steps 10" in out and "--train.optimizer muon" in out


def test_run_control_env():
    _, env = build()
    assert env["LINGBOT_REQUIRE_RESUME"] == "1"
    assert env["LINGBOT_STOP_FILE"] == "/workspace/runtime/STOP"
    _, env = build(require_resume=False, stop_file=None)
    assert "LINGBOT_REQUIRE_RESUME" not in env and "LINGBOT_STOP_FILE" not in env


def test_seen_instruction_pool_env():
    _, env = build()
    assert env["LINGBOT_SEEN_INSTRUCTION_POOL"] == "/workspace/runtime/seen_instruction_pool.json"
    _, env = build(seen_instruction_pool=None)
    assert "LINGBOT_SEEN_INSTRUCTION_POOL" not in env


def test_init_path_overrides_model_path_only():
    cfg = launch_train.load_config(MAIN, REPO)
    launch_train.apply_overrides(cfg, ["model.init_path=/workspace/runtime/outputs/main_run/hf/global_step_9000"])
    argv, _ = launch_train.build_command(cfg, REPO)
    assert argv_value(argv, "--model.model_path") == "/workspace/runtime/outputs/main_run/hf/global_step_9000"
    assert argv_value(argv, "--model.config_path").endswith("/models/robbyant_lingbot-vla-v2-6b")
    align = json.loads(argv_value(argv, "--train.align_params"))
    assert align["depth"]["morgbd_path"].endswith("/models/robbyant_lingbot-vla-v2-6b/depth/model.pt")


def test_robot_config_matches_upstream_except_norm_stats():
    ours = yaml.safe_load((REPO / "configs/robot/robotwin.yaml").read_text(encoding="utf-8"))
    upstream = yaml.safe_load((REPO / "configs/reference/upstream_robot_robotwin.yaml").read_text(encoding="utf-8"))
    assert upstream.pop("norm_stats") == "assets/norm_stats/robotwin.json"
    norm = ours.pop("norm_stats")
    assert ours == upstream
    cfg = launch_train.load_config(MAIN, REPO)
    assert norm == cfg["data"]["norm_stats_file"]  # training (robot config) and deploy (norm_stats_file) must agree


def test_robot_config_root_is_absolute_repo_path():
    argv, _ = build()
    assert argv_value(argv, "--data.robot_config_root") == str(REPO / "configs/robot")


LORA = REPO / "configs/train/autodl_lora.yaml"


def build_lora():
    return launch_train.build_command(launch_train.load_config(LORA, REPO), REPO)


def test_lora_config_has_no_align_params_and_lora_flags():
    argv, env = build_lora()
    assert "--train.align_params" not in argv
    assert argv_value(argv, "--train.use_lora") == "true"
    assert argv_value(argv, "--train.lora_scope") == "action_expert"
    assert argv_value(argv, "--train.enable_mixed_precision") == "true"   # D-08 补充：bf16 基座 dtype 不匹配
    assert argv_value(argv, "--data.use_future_image") == "false"
    assert argv_value(argv, "--model.post_training") == "false"
    assert "--nproc-per-node=1" in argv
    assert argv_value(argv, "--train.gradient_accumulation_steps") == "16"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["ROBOTWIN_EE_PLANNER"] == "mplib"


def test_lora_robot_config_matches_upstream_and_norm_file():
    ours = yaml.safe_load((REPO / "configs/robot_autodl/robotwin.yaml").read_text(encoding="utf-8"))
    upstream = yaml.safe_load((REPO / "configs/reference/upstream_robot_robotwin.yaml").read_text(encoding="utf-8"))
    upstream.pop("norm_stats")
    norm = ours.pop("norm_stats")
    assert ours == upstream
    cfg = launch_train.load_config(LORA, REPO)
    assert norm == cfg["data"]["norm_stats_file"]


def test_autodl_config_agrees_with_platform_env():
    env = launch_train.parse_env_file(REPO / "configs/platform/autodl.env")
    cfg = launch_train.load_config(LORA, REPO)
    assert cfg["platform"]["python"] == env["PY"]
    assert cfg["platform"]["robotwin_root"] == env["ROBOTWIN_ROOT"]
    assert cfg["platform"]["data_list"].startswith(env["RUNTIME"])
    assert cfg["run"]["output_dir"].startswith(env["RUNTIME"])
    assert cfg["platform"]["env_profile"] == env["ENV_PROFILE"]


def test_model_section_passthrough_for_attention_fallback():
    argv, _ = build_lora()
    assert argv_value(argv, "--model.attn_implementation") == "flash_attention_2"
    assert argv_value(argv, "--model.vit_attn_implementation") == "flash_attention_2"
    cfg = launch_train.apply_overrides(launch_train.load_config(LORA, REPO),
                                       ["model.attn_implementation=sdpa", "model.vit_attn_implementation=sdpa"])
    argv, _ = launch_train.build_command(cfg, REPO)
    assert argv_value(argv, "--model.attn_implementation") == "sdpa"
    assert "--model.model_dir" not in argv and "--model.init_path" not in argv


def test_raise_nofile_limit_to_hard_limit():
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(1024, hard), hard))
        new_soft = launch_train.raise_nofile_limit()
        assert new_soft == hard == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_compile_flex_env():
    cfg = launch_train.load_config(LORA, REPO)
    _, env = launch_train.build_command(cfg, REPO)
    assert "LINGBOT_COMPILE_FLEX" not in env
    launch_train.apply_overrides(cfg, ["run.compile_flex=true"])
    _, env = launch_train.build_command(cfg, REPO)
    assert env["LINGBOT_COMPILE_FLEX"] == "1"


def test_fsdp_single_gpu_env():
    cfg = launch_train.load_config(LORA, REPO)
    _, env = launch_train.build_command(cfg, REPO)
    assert "LINGBOT_FSDP_SINGLE_GPU" not in env
    launch_train.apply_overrides(cfg, ["run.fsdp_single_gpu=true"])
    _, env = launch_train.build_command(cfg, REPO)
    assert env["LINGBOT_FSDP_SINGLE_GPU"] == "1"


EXPERT = REPO / "configs/train/autodl_expert.yaml"


def test_expert_config_trains_action_expert_fully_without_teacher():
    cfg = launch_train.load_config(EXPERT, REPO)
    argv, env = launch_train.build_command(cfg, REPO)
    assert "--train.align_params" not in argv                              # 第一轮不带 teacher
    assert argv_value(argv, "--train.use_lora") == "false"
    assert argv_value(argv, "--train.train_expert_only") == "true"         # VLM 冻结，action expert 全参
    assert argv_value(argv, "--train.use_compile") == "true"               # D-09
    assert argv_value(argv, "--train.enable_fp32") == "true"               # 官方配方的计算精度
    assert argv_value(argv, "--model.post_training") == "false"            # 无 teacher：true 会因对齐头多余键报错（D-10）
    assert argv_value(argv, "--data.use_future_image") == "false"
    assert argv_value(argv, "--data.image_augment") == "true"              # D-05
    assert "--nproc-per-node=1" in argv
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["LINGBOT_SEEN_INSTRUCTION_POOL"]                            # D-06


def test_expert_config_shares_platform_and_data_with_lora():
    lora, expert = (launch_train.load_config(p, REPO) for p in (LORA, EXPERT))
    assert expert["platform"] == lora["platform"]
    assert expert["data"]["norm_stats_file"] == lora["data"]["norm_stats_file"]          # D-07
    assert expert["data"]["robot_config_root"] == lora["data"]["robot_config_root"]
    assert expert["run"]["seen_instruction_pool"] == lora["run"]["seen_instruction_pool"]
    assert expert["run"]["output_dir"] != lora["run"]["output_dir"]


FULL = REPO / "configs/train/autodl_full.yaml"
VLMLORA = REPO / "configs/train/autodl_vlmlora.yaml"


def test_full_config_is_official_full_model_on_two_gpus():
    argv, env = launch_train.build_command(launch_train.load_config(FULL, REPO), REPO)
    assert argv_value(argv, "--train.train_expert_only") == "false"
    assert argv_value(argv, "--train.use_lora") == "false"
    assert "--nproc-per-node=2" in argv
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_vlmlora_config_uses_vlm_expert_full_scope():
    argv, _ = launch_train.build_command(launch_train.load_config(VLMLORA, REPO), REPO)
    assert argv_value(argv, "--train.use_lora") == "true"
    assert argv_value(argv, "--train.lora_scope") == "vlm_expert_full"
    assert argv_value(argv, "--train.train_expert_only") == "false"   # 上游：use_lora 要求 false
    assert "--nproc-per-node=1" in argv


def test_variant_configs_share_everything_but_training_scope():
    expert = launch_train.load_config(EXPERT, REPO)
    for path in (FULL, VLMLORA):
        cfg = launch_train.load_config(path, REPO)
        assert cfg["platform"] == expert["platform"]
        assert cfg["data"] == expert["data"]
        assert cfg["model"] == expert["model"]
        assert cfg["align_params_file"] == expert["align_params_file"]
        assert cfg["run"]["global_batch_size"] == expert["run"]["global_batch_size"]
        assert cfg["run"]["output_dir"] != expert["run"]["output_dir"]


S2 = REPO / "configs/train/autodl_vlmlora_s2.yaml"


def test_stage2_continues_from_7500_snapshot_with_safe_rotation():
    s1, s2 = (launch_train.load_config(p, REPO) for p in (VLMLORA, S2))
    argv, env = launch_train.build_command(s2, REPO)
    assert s2["model"]["init_path"].endswith("recover/global_step_7500")
    assert argv_value(argv, "--model.model_path") == s2["model"]["init_path"]
    assert s2["run"]["max_steps"] == 20000 - 7500
    assert s2["run"]["dcp_keep"] == 2 and env["LINGBOT_DCP_KEEP"] == "2"   # 保存中途被杀时上一份仍在（D-13）
    assert s2["train"]["lr"] == 8.46e-5 and s2["train"]["lr_min"] == s1["train"]["lr_min"]
    assert 0 < s2["train"]["lr_warmup_ratio"] <= 0.05
    assert s2["run"]["output_dir"] != s1["run"]["output_dir"]
    for section in ("platform", "data", "align_params_file"):
        assert s2[section] == s1[section]
    for k in ("use_lora", "lora_scope", "lora_rank", "lora_alpha", "train_expert_only", "enable_fp32", "use_compile", "seed"):
        assert s2["train"][k] == s1["train"][k], k
    assert s2["run"]["micro_batch_size"] == s1["run"]["micro_batch_size"]
    assert s2["run"]["global_batch_size"] == s1["run"]["global_batch_size"]


def test_stop_at_step_env():
    _, env = build(stop_at_step=7500)
    assert env["LINGBOT_STOP_AT_STEP"] == "7500"
    _, env = build()
    assert "LINGBOT_STOP_AT_STEP" not in env


def test_stage1_config_stops_at_7500():
    cfg = launch_train.load_config(REPO / "configs/train/autodl_vlmlora.yaml", REPO)
    assert cfg["run"]["stop_at_step"] == 7500 and cfg["run"]["max_steps"] == 20000

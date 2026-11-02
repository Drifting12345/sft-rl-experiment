#!/usr/bin/env python3
"""Launch single-node verl GRPO/DAPO from an original Qwen base model."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(ROOT / "verl"))
from eval_qwen3_igsm import DATA_NAME, atomic_json, digest, run_lock
from prepare_data import prepare


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["grpo", "dapo"], required=True)
    parser.add_argument("--hardware", choices=["4090-48g", "3090-24g"], default="4090-48g")
    parser.add_argument("--train-data", type=Path, required=True, help="New raw iGSM JSONL pool; split 90/10")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-0.6B", help="Original model under zfy/models")
    parser.add_argument("--heldout-data", type=Path, nargs="+", default=[ROOT / "data" / DATA_NAME])
    parser.add_argument("--gpus", type=int, default=8, help="Single-node GPU count; use 1 only for a separate debug run")
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32, help="Prompt groups per update")
    parser.add_argument("--mini-batch-size", type=int, default=8, help="Prompt groups; verl multiplies this by rollout n")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, help="Optional explicit optimization-step limit")
    parser.add_argument("--validate-every", type=int, help="Defaults to one nominal epoch")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--max-generation-batches", type=int, default=16, help="Real per-step DAPO prompt-generation budget")
    parser.add_argument("--generation-timeout", type=float, default=1800, help="DAPO sampling-poll deadline in seconds")
    parser.add_argument("--resume", type=Path, help="Complete global_step_N checkpoint directly under this run/checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="CPU data/config checks only; do not initialize Ray or load weights")
    args = parser.parse_args()
    positive = [args.gpus, args.rollout_tp, args.train_batch_size, args.mini_batch_size,
        args.samples_per_prompt, args.epochs, args.max_prompt_length, args.max_response_length,
        args.learning_rate, args.early_stopping_patience, args.max_generation_batches, args.generation_timeout]
    if min(positive) <= 0 or args.samples_per_prompt < 2:
        parser.error("Positive limits and at least two samples per prompt are required")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.validate_every is not None and args.validate_every < 1:
        parser.error("validate-every must be positive")
    if args.gpus % args.rollout_tp or args.train_batch_size % args.mini_batch_size:
        parser.error("gpus must divide into rollout TP groups, and train batch must divide into mini-batches")
    if args.mini_batch_size * args.samples_per_prompt % args.gpus:
        parser.error("mini-batch prompt count * samples-per-prompt must be divisible by actor GPU count")
    return args


def original_model(path: Path) -> Path:
    model = path.resolve()
    if not model.is_relative_to((ROOT / "models").resolve()) or (model / "adapter_config.json").exists():
        raise ValueError("RL must start from an original model under zfy/models, not an SFT output or adapter")
    if not (model / "config.json").is_file() or not any(model.glob("*.safetensors")):
        raise ValueError(f"Original model weights/config are missing: {model}")
    return model


def build_config(args, data: dict):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base=None, config_dir=str(HERE / "config")):
        base = compose(config_name="base")
    cfg = OmegaConf.merge(base, OmegaConf.load(HERE / "config" / f"{args.algorithm}.yaml"),
                         OmegaConf.load(HERE / "config/hardware" / f"{args.hardware}.yaml"))
    if data["train_count"] < args.train_batch_size:
        raise ValueError("The training partition is smaller than train-batch-size; provide more data or lower the batch")
    steps_per_epoch = data["train_count"] // args.train_batch_size
    frequency = args.validate_every or steps_per_epoch
    model = original_model(args.model)
    config = json.loads((model / "config.json").read_text())
    if config["num_attention_heads"] % args.rollout_tp:
        raise ValueError("Model attention heads must divide evenly over rollout tensor parallelism")
    output = args.output_dir.resolve()
    updates = {
        "actor_rollout_ref.model.path": str(model), "critic.model.path": str(model),
        "actor_rollout_ref.actor.optim.lr": args.learning_rate,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.mini_batch_size,
        "actor_rollout_ref.actor.data_loader_seed": args.seed,
        "actor_rollout_ref.rollout.n": args.samples_per_prompt,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.rollout_tp,
        "actor_rollout_ref.rollout.prompt_length": args.max_prompt_length,
        "actor_rollout_ref.rollout.response_length": args.max_response_length,
        "actor_rollout_ref.rollout.max_model_len": args.max_prompt_length + args.max_response_length,
        "actor_rollout_ref.rollout.seed": args.seed,
        "data.train_files": data["train_file"], "data.val_files": data["val_file"],
        "data.train_batch_size": args.train_batch_size, "data.max_prompt_length": args.max_prompt_length,
        "data.max_response_length": args.max_response_length, "data.seed": args.seed,
        "reward.reward_manager.module.path": str(HERE / "reward.py"),
        "reward.custom_reward_function.path": str(HERE / "reward.py"),
        "trainer.nnodes": 1, "trainer.n_gpus_per_node": args.gpus,
        "trainer.total_epochs": args.epochs, "trainer.total_training_steps": args.max_steps,
        "trainer.test_freq": frequency, "trainer.save_freq": frequency,
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.experiment_name": f"igsm-{args.algorithm}-{args.hardware}",
        "trainer.resume_mode": "resume_path" if args.resume else "disable",
        "trainer.resume_from_path": str(args.resume.resolve()) if args.resume else None,
        "igsm_rl.run_dir": str(output), "igsm_rl.patience": args.early_stopping_patience,
        "igsm_rl.max_generation_batches": args.max_generation_batches,
        "igsm_rl.generation_timeout_s": args.generation_timeout,
        "transfer_queue.enable": True,
    }
    for name, value in updates.items():
        OmegaConf.update(cfg, name, value, merge=False)
    cfg.actor_rollout_ref.rollout.max_num_batched_tokens = max(
        cfg.actor_rollout_ref.rollout.max_num_batched_tokens, args.max_prompt_length + args.max_response_length)
    cfg.ray_kwargs.ray_init.runtime_env = OmegaConf.create({"py_executable": sys.executable,
        "env_vars": {"PYTHONPATH": os.pathsep.join([str(HERE), str(ROOT / "verl")]),
                     "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                     "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}})
    OmegaConf.resolve(cfg)
    return cfg


def gpu_preflight(args):
    import torch
    if torch.cuda.device_count() != args.gpus:
        raise RuntimeError(f"Requested {args.gpus} GPUs; CUDA_VISIBLE_DEVICES exposes {torch.cuda.device_count()}")
    minimum_gib = 44 if args.hardware == "4090-48g" else 22
    expected = "4090" if args.hardware == "4090-48g" else "3090"
    for index in range(args.gpus):
        device = torch.cuda.get_device_properties(index)
        if expected not in device.name or device.total_memory < minimum_gib * 1024**3:
            raise RuntimeError(f"GPU {index} ({device.name}) does not match hardware profile {args.hardware}")


def main():
    args = arguments()
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", RAY_USAGE_STATS_ENABLED="0",
        VLLM_NO_USAGE_STATS="1", WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1", VERL_DISABLE_FLASH_ATTN_CE="1", RAY_ADDRESS="local")
    model, output = original_model(args.model), args.output_dir.resolve()
    data = prepare(args.train_data, output / "data", model, heldout_data=args.heldout_data,
        seed=args.seed, dry_run=True, max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_model_len=args.max_prompt_length + args.max_response_length)
    from omegaconf import OmegaConf
    cfg = build_config(args, data)
    # Framework validation is static: no Ray cluster or model workers are started.
    from verl.utils.config import validate_config
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    validate_config(cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))
    source_files = sorted([*HERE.glob("*.py"), *HERE.glob("*.sh"), *HERE.glob("config/**/*.yaml")])
    sources = {str(path.relative_to(HERE)): digest(path.read_bytes()) for path in source_files}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    resume_independent_cfg = json.loads(json.dumps(cfg_dict))
    resume_independent_cfg["trainer"]["resume_mode"] = "disable"
    resume_independent_cfg["trainer"]["resume_from_path"] = None
    manifest = {"schema_version": 1, "algorithm": args.algorithm, "hardware": args.hardware,
        "original_model": str(model), "dataset": data, "sources_sha256": sources,
        "configuration": resume_independent_cfg,
        "versions": {name: version(name) for name in ["torch", "transformers", "vllm", "ray", "hydra-core"]}}
    print(json.dumps({"algorithm": args.algorithm, "hardware": args.hardware,
        "gpus": args.gpus, "rollout_replicas": args.gpus // args.rollout_tp,
        "train_count": data["train_count"], "validation_count": data["validation_count"],
        "reward": "strict binary final-answer accuracy", "thinking": False,
        "validation_interval": cfg.trainer.test_freq, "config": cfg_dict}, indent=2), flush=True)
    if args.dry_run:
        print("RL_DRY_RUN_PASSED: no output directory, Ray cluster, model weights or training started", flush=True)
        return
    gpu_preflight(args)
    if args.resume:
        checkpoint = args.resume.resolve()
        if checkpoint.parent != output / "checkpoints" or not (checkpoint / "igsm_rl_state.json").is_file():
            raise ValueError("Resume needs a complete checkpoint and early-stop state inside this run/checkpoints")
        if json.loads((output / "run_config.json").read_text()) != manifest:
            raise ValueError("Resume data, source, environment or configuration differs from the original run")
        integrity = json.loads((output / "data_integrity.json").read_text())
        for filename, sha in integrity.items():
            if digest((output / "data" / filename).read_bytes()) != sha:
                raise ValueError(f"Prepared RL data changed: {filename}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        if not args.resume:
            prepared = prepare(args.train_data, output / "data", model, heldout_data=args.heldout_data,
                seed=args.seed, max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_response_length,
                max_model_len=args.max_prompt_length + args.max_response_length)
            atomic_json(output / "data_integrity.json", prepared["parquet_sha256"])
            atomic_json(output / "run_config.json", manifest)
            OmegaConf.save(cfg, output / "resolved_config.yaml")
            for path in source_files:
                target = output / "source" / path.relative_to(HERE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        from verl.trainer.main_ppo import run_ppo
        from task_runner import RLTaskRunner
        import ray
        try:
            run_ppo(cfg, task_runner_class=RLTaskRunner)
        finally:
            # This entry point starts its own local Ray cluster, never attaches to another job.
            if ray.is_initialized():
                ray.shutdown()
        print(f"RL_TRAINING_FINISHED: {output}", flush=True)


if __name__ == "__main__":
    main()

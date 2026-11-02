#!/usr/bin/env python3
"""Single-GPU Qwen3 LoRA SFT with a held-out split and validation-loss early stopping."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from eval_qwen3_igsm import (  # noqa: E402
    DATA_NAME, atomic_json, digest, read_examples, render_messages, run_lock,
)


def read_records(path: Path, *, solutions: bool) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    records, seen_ids, seen_problems = [], set(), set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        location = f"{path}:{line_number}"
        if not isinstance(row, dict) or not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError(f"{location}: missing problem text")
        row = dict(row)
        row["id"] = str(row.get("id", f"{path.name}:{line_number}"))
        key = digest(" ".join(row["problem"].split()).encode())
        if row["id"] in seen_ids or key in seen_problems:
            raise ValueError(f"{location}: duplicate id or question")
        seen_ids.add(row["id"]); seen_problems.add(key)
        row["_problem_key"] = key
        if "seed" in row and (type(row["seed"]) is not int or row["seed"] < 0):
            raise ValueError(f"{location}: seed must be a nonnegative integer")
        if solutions:
            answer = str(row.get("answer", "")).strip()
            if not re.fullmatch(r"[0-9]+", answer) or len(answer) > 2 or not 0 <= int(answer) < 23:
                raise ValueError(f"{location}: answer must be an integer in 0..22")
            row["answer"] = int(answer)
            if not isinstance(row.get("solution"), str) or not row["solution"].strip():
                raise ValueError(f"{location}: a worked solution is required")
            if "solution_validated" in row and row["solution_validated"] is not True:
                raise ValueError(f"{location}: solution is marked invalid")
        records.append(row)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records, digest(raw)


def reject_overlap(left: list[dict], right: list[dict], description: str) -> None:
    problems = {row["_problem_key"] for row in right}
    seeds = {row["seed"] for row in right if "seed" in row}
    collisions = [row["id"] for row in left if row["_problem_key"] in problems
                  or ("seed" in row and row["seed"] in seeds)]
    if collisions:
        raise ValueError(f"{description}: overlapping questions or generator seeds: {collisions[:5]}")


def split_records(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    """Keep the same generator seed in one partition, targeting 90/10 by row count."""
    groups = {}
    for index, row in enumerate(rows):
        key = f"seed:{row['seed']}" if "seed" in row else f"problem:{row['_problem_key']}"
        groups.setdefault(key, []).append(index)
    if len(groups) < 10:
        raise ValueError("Automatic 90/10 splitting requires at least 10 independent question/seed groups")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    cumulative = [0]
    for key in keys:
        cumulative.append(cumulative[-1] + len(groups[key]))
    boundary = min(range(1, len(keys)), key=lambda n: (abs(cumulative[n] - len(rows) * 0.1), cumulative[n]))
    validation_indices = {index for key in keys[:boundary] for index in groups[key]}
    train = [row for index, row in enumerate(rows) if index not in validation_indices]
    validation = [row for index, row in enumerate(rows) if index in validation_indices]
    reject_overlap(train, validation, "Automatic train/validation split")
    manifest = {"seed": seed, "requested_validation_fraction": 0.1,
        "actual_validation_fraction": len(validation) / len(rows), "independent_groups": len(groups),
        "train": [], "validation": []}
    for index, row in enumerate(rows):
        manifest["validation" if index in validation_indices else "train"].append({
            "source_index": index, "id": row["id"], "problem_sha256": row["_problem_key"],
            "generator_seed": row.get("seed")})
    return train, validation, manifest


def target_text(row: dict) -> str:
    solution = row["solution"].strip()
    lines = solution.splitlines()
    if lines[-1].strip().startswith("answer:"):
        match = re.fullmatch(r"answer:\s*([0-9]+)", lines[-1].strip())
        if match is None or int(match[1]) != row["answer"]:
            raise ValueError(f"{row['id']}: final answer conflicts with the label")
        solution = "\n".join(lines[:-1]).strip()
    if not solution or any(line.strip().startswith("answer:") for line in solution.splitlines()):
        raise ValueError(f"{row['id']}: expected a worked solution and one final answer")
    return f"{solution}\nanswer:{row['answer']}"


def encode_record(row: dict, tokenizer, template: str, examples: list[dict],
                  max_length: int, max_response_tokens: int) -> tuple[dict, dict]:
    target = target_text(row)
    for text in [row["problem"], target]:
        if any(token in text for token in [*tokenizer.all_special_tokens, "<think>", "</think>"]):
            raise ValueError(f"{row['id']}: special control tokens inside data")
    messages = render_messages(template, row, examples)
    prefix = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    complete = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": target}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prompt_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    full_ids = tokenizer(complete, add_special_tokens=False, truncation=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{row['id']}: training template does not match the inference prefix")
    # Keep the assistant end token; discard only the template separator after it.
    try:
        end = full_ids.index(tokenizer.eos_token_id, len(prompt_ids))
    except ValueError as exc:
        raise ValueError(f"{row['id']}: no assistant EOS token") from exc
    full_ids = full_ids[:end + 1]
    response_tokens = len(full_ids) - len(prompt_ids)
    if len(full_ids) > max_length or response_tokens > max_response_tokens:
        raise ValueError(f"{row['id']}: {len(full_ids)} total / {response_tokens} response tokens "
                         f"exceed {max_length} / {max_response_tokens}; no truncation is allowed")
    if response_tokens <= 1:
        raise ValueError(f"{row['id']}: empty target")
    feature = {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
               "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}
    return feature, {"id": row["id"], "prompt_tokens": len(prompt_ids),
                     "response_tokens": response_tokens, "total_tokens": len(full_ids)}


@dataclass
class CompletionCollator:
    pad_token_id: int

    def __call__(self, features):
        import torch
        longest = max(len(row["input_ids"]) for row in features)
        length = (longest + 7) // 8 * 8
        batch = {key: [] for key in ["input_ids", "attention_mask", "labels"]}
        for row in features:
            padding = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="New iGSM JSONL pool, automatically split 90/10")
    parser.add_argument("--heldout-data", type=Path, nargs="+", default=[ROOT / "data" / DATA_NAME],
                        help="Protected test sets checked for train/validation question and seed overlap")
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-0.6B")
    parser.add_argument("--prompt", type=Path, default=EXP / "prompt.txt")
    parser.add_argument("--examples", type=Path, default=EXP / "prompt_examples.json")
    parser.add_argument("--no-examples", action="store_true", help="Train with a single user message instead")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--early-stopping-patience", type=int, default=2, help="Consecutive validation-loss increases; ties reset the counter")
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Validate/tokenize on CPU; do not load model weights or train")
    args = parser.parse_args()
    if args.max_length < 8 or args.max_length % 8 or args.max_response_tokens < 2:
        parser.error("max-length must be a positive multiple of 8; max-response-tokens >= 2")
    if min(args.epochs, args.learning_rate, args.micro_batch_size, args.gradient_accumulation_steps) <= 0:
        parser.error("epochs, learning-rate and batch sizes must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("max-steps must be -1 or positive")
    if min(args.lora_r, args.lora_alpha, args.early_stopping_patience) < 1:
        parser.error("LoRA rank/alpha and early-stopping patience must be positive")
    if not 0 <= args.lora_dropout < 1:
        parser.error("lora-dropout must be in [0, 1)")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This entry point supports one GPU; do not launch it with torchrun")
    return args


def main():
    args = arguments()
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    pool, train_sha = read_records(args.train_data, solutions=True)
    heldout_info = []
    for path in args.heldout_data:
        protected, sha = read_records(path, solutions=False)
        reject_overlap(pool, protected, f"SFT data pool vs held-out {path}")
        heldout_info.append({"path": str(path.resolve()), "sha256": sha, "count": len(protected)})
    train_rows, validation, split_manifest = split_records(pool, args.seed)
    split_sha = digest(json.dumps(split_manifest, sort_keys=True, separators=(",", ":")).encode())
    template = args.prompt.read_text(encoding="utf-8")
    example_path = None if args.no_examples else args.examples
    examples, examples_sha, examples_raw = read_examples(example_path)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.resolve(), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_config = json.loads((args.model / "config.json").read_text())
    if args.max_length > model_config["max_position_embeddings"]:
        raise ValueError("max-length exceeds the model context window")
    datasets, lengths = {}, {}
    for name, rows in [("train", train_rows), ("validation", validation)]:
        pairs = [encode_record(row, tokenizer, template, examples, args.max_length, args.max_response_tokens) for row in rows]
        datasets[name] = [pair[0] for pair in pairs]
        lengths[name] = [pair[1] for pair in pairs]
    config = {
        "schema_version": 2, "method": "lora_sft", "thinking": False,
        "loss": "target assistant solution + final numeric answer + EOS; prefix/examples/padding masked",
        "train_data": str(args.train_data.resolve()), "train_sha256": train_sha,
        "split_manifest_sha256": split_sha, "split_seed": args.seed,
        "validation_fraction": 0.1, "actual_validation_fraction": len(validation) / len(pool),
        "train_count": len(train_rows), "validation_count": len(validation), "protected_heldout": heldout_info,
        "model": str(args.model.resolve()), "model_config_sha256": digest((args.model / "config.json").read_bytes()),
        "tokenizer_config_sha256": digest((args.model / "tokenizer_config.json").read_bytes()),
        "tokenizer_files_sha256": {name: digest((args.model / name).read_bytes())
            for name in ["tokenizer.json", "vocab.json", "merges.txt", "generation_config.json", "chat_template.jinja"]
            if (args.model / name).is_file()},
        "prompt": str(args.prompt.resolve()), "prompt_sha256": digest(template.encode()),
        "examples": str(example_path.resolve()) if example_path else None, "examples_sha256": examples_sha,
        "script_sha256": digest(Path(__file__).read_bytes()),
        "evaluation_helpers_sha256": digest((EXP / "eval_qwen3_igsm.py").read_bytes()),
        "early_stopping_sha256": digest(Path(__file__).with_name("early_stopping.py").read_bytes()),
        "versions": {name: version(name) for name in ["transformers", "torch", "accelerate", "peft", "tokenizers"]},
        "hyperparameters": {name: getattr(args, name) for name in ["max_length", "max_response_tokens", "epochs",
            "learning_rate", "micro_batch_size", "gradient_accumulation_steps", "max_steps", "seed"]},
        "optimizer": "adamw_torch", "weight_decay": 0.01, "warmup_ratio": 0.03,
        "lr_scheduler": "cosine", "dtype": "bfloat16", "attention": "sdpa", "gradient_checkpointing": True,
        "lora": {"rank": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": args.lora_target_modules, "bias": "none"},
        "early_stopping": {"metric": "eval_loss", "greater_is_better": False,
                           "patience": args.early_stopping_patience, "comparison": "previous_epoch",
                           "rule": "strict loss increase increments; tie or decrease resets", "frequency": "epoch"},
    }
    print(json.dumps({"config": config, "lengths": {name: {
        "count": len(items), "max_total_tokens": max((x["total_tokens"] for x in items), default=0),
        "max_response_tokens": max((x["response_tokens"] for x in items), default=0),
    } for name, items in lengths.items()}}, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASSED: no model weights loaded, GPU training or output directory created", flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from early_stopping import ConsecutiveWorseningCallback
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This BF16 recipe requires a CUDA GPU with BF16 support")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES for this training entry point")
    output = args.output_dir.resolve()
    checkpoint = args.resume_from_checkpoint.resolve() if args.resume_from_checkpoint else None
    if checkpoint:
        if checkpoint.parent != output or not re.fullmatch(r"checkpoint-[0-9]+", checkpoint.name):
            raise ValueError("Resume checkpoint must be a checkpoint-N directly inside output-dir")
        if json.loads((output / "sft_config.json").read_text()) != config:
            raise ValueError("Resume inputs/settings/source differ from the original run")
        if json.loads((output / "split_manifest.json").read_text()) != split_manifest:
            raise ValueError("Resume split differs from the saved partition")
        for name in ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]:
            if not (checkpoint / name).is_file():
                raise ValueError(f"Resume checkpoint is missing {name}; model-only export cannot resume training")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; use a fresh directory or explicit resume")
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        if not checkpoint:
            atomic_json(output / "sft_config.json", config)
            atomic_json(output / "token_lengths.json", lengths)
            atomic_json(output / "split_manifest.json", split_manifest)
            (output / "prompt.txt").write_text(template, encoding="utf-8")
            if examples_raw is not None:
                (output / "examples.json").write_bytes(examples_raw)
            shutil.copyfile(__file__, output / "train_sft.py")
            shutil.copyfile(EXP / "eval_qwen3_igsm.py", output / "evaluation_helpers.py")
            shutil.copyfile(Path(__file__).with_name("early_stopping.py"), output / "early_stopping.py")
        set_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(args.model.resolve(), local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, attn_implementation="sdpa")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules, bias="none"))
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        training_args = TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs,
            max_steps=args.max_steps, learning_rate=args.learning_rate, weight_decay=0.01,
            warmup_steps=0.03, lr_scheduler_type="cosine", optim="adamw_torch",
            per_device_train_batch_size=args.micro_batch_size, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            bf16=True, tf32=True, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            restore_callback_states_from_checkpoint=True,
            save_total_limit=2, save_only_model=False, logging_steps=10,
            report_to="none", prediction_loss_only=True, remove_unused_columns=False,
            dataloader_num_workers=0, seed=args.seed, data_seed=args.seed)
        trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
            eval_dataset=datasets["validation"] if validation else None,
            processing_class=tokenizer, data_collator=CompletionCollator(tokenizer.pad_token_id),
            callbacks=[ConsecutiveWorseningCallback(patience=args.early_stopping_patience)])
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        trained = trainer.accelerator.unwrap_model(trainer.model)
        adapter = output / "best_adapter"
        trained.save_pretrained(adapter, safe_serialization=True)
        tokenizer.save_pretrained(adapter)
        merged = trained.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True
        final = output / "final"
        if (final / "adapter_config.json").exists():
            raise ValueError("final/ must contain only a merged model, not a PEFT adapter")
        merged.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        stopping = next(callback for callback in trainer.callback_handler.callbacks if isinstance(callback, ConsecutiveWorseningCallback))
        atomic_json(output / "result.json", {"complete": True, "global_step": trainer.state.global_step,
                    "best_checkpoint": trainer.state.best_model_checkpoint,
                    "best_validation_loss": trainer.state.best_metric, "epoch": trainer.state.epoch,
                    "early_stopping_counter": stopping.consecutive_increases,
                    "early_stopped": stopping.consecutive_increases >= args.early_stopping_patience,
                    "adapter": str(adapter), "final_model": str(final), "metrics": result.metrics})
        print(f"SFT_COMPLETE: {final}", flush=True)


if __name__ == "__main__":
    main()

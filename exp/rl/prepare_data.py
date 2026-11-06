#!/usr/bin/env python3
"""Prepare a seed-disjoint iGSM RL split without truncating inference prefixes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP))
from eval_qwen3_igsm import DATA_NAME, atomic_json, digest, render_messages
from sft.train_sft import read_records, reject_overlap, split_records


def validate_answers(rows: list[dict], description: str) -> None:
    """RL only needs a canonical label, never a worked reference solution."""
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        if not re.fullmatch(r"[0-9]{1,2}", answer) or not 0 <= int(answer) <= 22:
            raise ValueError(f"{description}, {row['id']}: answer must be an integer in 0..22")
        row["answer"] = int(answer)
        if row.get("seed", 0) > 2**63 - 1:
            raise ValueError(f"{description}, {row['id']}: generator seed exceeds int64")


def encode_prompt(row: dict, split: str, tokenizer, template: str,
                  max_prompt_length: int, max_response_length: int, max_model_len: int) -> tuple[dict, dict]:
    controls = [*tokenizer.all_special_tokens, "<think>", "</think>"]
    if any(token and token in row["problem"] for token in controls):
        raise ValueError(f"{row['id']}: special control token inside the question")
    # Only the question is exposed to the renderer: no label or solution field.
    messages = render_messages(template, {"problem": row["problem"]})
    prefix = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    token_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]
    length = len(token_ids)
    if not length or length > max_prompt_length or length + max_response_length > max_model_len:
        raise ValueError(f"{row['id']}: {length} prompt tokens + {max_response_length} response budget "
                         f"exceed {max_prompt_length} prompt / {max_model_len} context limits; "
                         "no truncation is allowed")
    record = {"data_source": "igsm_rl", "prompt": messages,
        "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
        "extra_info": {"id": row["id"], "split": split, "seed": row.get("seed"),
                       "problem_sha256": row["_problem_key"]}}
    stats = {"id": row["id"], "split": split, "prompt_tokens": length,
             "budgeted_total_tokens": length + max_response_length,
             "prefix_sha256": digest(prefix.encode("utf-8"))}
    return record, stats


def prepare(input_path: Path, output_dir: Path, model: Path = ROOT / "models/Qwen3-0.6B", *,
            heldout_data: list[Path] | Path | None = None, seed: int = 42, dry_run: bool = False,
            prompt: Path = EXP / "prompt.txt",
            max_prompt_length: int = 4096, max_response_length: int = 2048,
            max_model_len: int = 6144) -> dict:
    """Return reproducibility metadata; dry-run writes nothing and loads no weights.

    The first heldout path identifies the protected 100-question benchmark;
    additional paths, if provided, are protected in the same way. Existing output
    directories are never reused or overwritten.
    """
    input_path, output_dir, model, prompt = map(Path, (input_path, output_dir, model, prompt))
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lengths = (max_prompt_length, max_response_length, max_model_len)
    if any(type(value) is not int or value < 1 for value in lengths):
        raise ValueError("prompt, response and context limits must be positive integers")
    if max_prompt_length + max_response_length > max_model_len:
        raise ValueError("max_prompt_length + max_response_length exceeds max_model_len")
    if output_dir.exists() and not dry_run:
        raise FileExistsError(f"Output directory already exists; preserve it: {output_dir}")

    pool, source_sha = read_records(input_path, solutions=False)
    validate_answers(pool, str(input_path))
    if heldout_data is None:
        protected_paths = [ROOT / "data" / DATA_NAME]
    elif isinstance(heldout_data, (str, Path)):
        protected_paths = [Path(heldout_data)]
    else:
        protected_paths = list(map(Path, heldout_data))
    if not protected_paths:
        raise ValueError("The protected 100-question heldout dataset is required")
    heldout = []
    for index, path in enumerate(protected_paths):
        protected, sha = read_records(path, solutions=False)
        if index == 0 and len(protected) != 100:
            raise ValueError(f"{path}: primary heldout must contain exactly 100 questions")
        reject_overlap(pool, protected, f"RL data pool vs protected heldout {path}")
        heldout.append({"path": str(path.resolve()), "sha256": sha, "count": len(protected)})
    train, validation, split_manifest = split_records(pool, seed)
    split_sha = digest(json.dumps(split_manifest, sort_keys=True, separators=(",", ":")).encode())

    template = prompt.read_text(encoding="utf-8")
    config_path = model / "config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    context = model_config.get("max_position_embeddings")
    if type(context) is not int or max_model_len > context:
        raise ValueError("max_model_len exceeds the model's declared context window")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.resolve(), local_files_only=True, trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets, token_stats = {}, {}
    for name, rows in (("train", train), ("validation", validation)):
        pairs = [encode_prompt(row, name, tokenizer, template,
                    max_prompt_length, max_response_length, max_model_len) for row in rows]
        datasets[name] = [pair[0] for pair in pairs]
        per_record = [pair[1] for pair in pairs]
        sizes = [item["prompt_tokens"] for item in per_record]
        token_stats[name] = {"count": len(sizes), "min_prompt_tokens": min(sizes),
            "max_prompt_tokens": max(sizes), "mean_prompt_tokens": statistics.mean(sizes),
            "max_budgeted_total_tokens": max(sizes) + max_response_length, "records": per_record}

    manifest = {"schema_version": 1, "method": "igsm_text_rl", "thinking": False,
        "input": str(input_path.resolve()), "source_sha256": source_sha,
        "train_count": len(train), "validation_count": len(validation),
        "split_seed": seed, "split_manifest": split_manifest, "split_manifest_sha256": split_sha,
        "protected_heldout": heldout, "model": str(model.resolve()),
        "model_config_sha256": digest(config_path.read_bytes()),
        "tokenizer_files_sha256": {name: digest((model / name).read_bytes()) for name in
            ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json",
             "generation_config.json", "chat_template.jinja"] if (model / name).is_file()},
        "prompt": str(prompt.resolve()), "prompt_sha256": digest(template.encode("utf-8")),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "evaluation_helpers_sha256": digest((EXP / "eval_qwen3_igsm.py").read_bytes()),
        "split_helpers_sha256": digest((EXP / "sft/train_sft.py").read_bytes()),
        "max_prompt_length": max_prompt_length, "max_response_length": max_response_length,
        "max_model_len": max_model_len, "token_stats": token_stats, "dry_run": bool(dry_run),
        "output_dir": str(output_dir.resolve()), "train_file": str((output_dir / "train.parquet").resolve()),
        "val_file": str((output_dir / "val.parquet").resolve())}
    if dry_run:
        return manifest

    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
        ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
        ("extra_info", pa.struct([("id", pa.string()), ("split", pa.string()),
                                  ("seed", pa.int64()), ("problem_sha256", pa.string())])),
    ])
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, filename in (("train", "train.parquet"), ("validation", "val.parquet")):
        pq.write_table(pa.Table.from_pylist(datasets[name], schema=schema), output_dir / filename)
    manifest["parquet_sha256"] = {name: digest((output_dir / name).read_bytes())
                                  for name in ("train.parquet", "val.parquet")}
    atomic_json(output_dir / "data_manifest.json", manifest)
    return manifest


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-0.6B")
    parser.add_argument("--heldout-data", type=Path, nargs="+", default=[ROOT / "data" / DATA_NAME])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=Path, default=EXP / "prompt.txt")
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--dry-run", action="store_true", help="CPU token checks only; write no output files")
    return parser.parse_args()


if __name__ == "__main__":
    args = arguments()
    result = prepare(args.input, args.output_dir, args.model, heldout_data=args.heldout_data,
        seed=args.seed, dry_run=args.dry_run, prompt=args.prompt,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length, max_model_len=args.max_model_len)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))

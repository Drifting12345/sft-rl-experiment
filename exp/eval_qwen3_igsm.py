#!/usr/bin/env python3
"""Evaluate independent Qwen3 samples, stopping each problem at its first success."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import socket
import time

ROOT = Path(__file__).resolve().parent.parent
DATA_NAME = "igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl"
ANSWER = re.compile(r"answer:[ \t]*([+-]?[0-9]+)[ \t]*")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for retry in range(6):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows indexers can briefly hold a freshly replaced file open.
            if os.name != "nt" or retry == 5:
                raise
            time.sleep(0.02 * (retry + 1))


def read_dataset(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows, ids, problems = [], set(), set()
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Line {number}: expected a JSON object")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Line {number}: missing string id")
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError(f"Line {number}: missing problem")
        answer = str(row.get("answer", "")).strip()
        if not re.fullmatch(r"[0-9]+", answer) or not 0 <= int(answer) < 23:
            raise ValueError(f"Line {number}: answer must be an integer in 0..22")
        if row["id"] in ids or row["problem"] in problems:
            raise ValueError(f"Line {number}: duplicate id or problem")
        ids.add(row["id"])
        problems.add(row["problem"])
        # Keep the reference solution out of the inference pipeline.
        rows.append({"id": row["id"], "problem": row["problem"], "answer": int(answer)})
    if not rows:
        raise ValueError("Dataset is empty")
    return rows, digest(raw)


def render_prompt(template: str, row: dict) -> str:
    if template.count("{problem}") != 1:
        raise ValueError("Prompt must contain exactly one {problem} placeholder")
    return template.replace("{problem}", row["problem"])


def render_messages(template: str, row: dict) -> list[dict]:
    return [{"role": "user", "content": render_prompt(template, row)}]


def parse_answer(text: str, finish_reason: str) -> tuple[int | None, str]:
    # A length cutoff may cut off a digit, e.g. answer:2 from answer:20.
    if finish_reason != "stop":
        return None, "truncated" if finish_reason == "length" else "incomplete_generation"
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None, "unfinished_thinking"
    lines = text.rstrip().splitlines()
    if not lines:
        return None, "empty_answer"
    markers = [line for line in lines if line.strip().startswith("answer:")]
    match = ANSWER.fullmatch(lines[-1].strip())
    if len(markers) != 1 or match is None:
        return None, "invalid_final_answer"
    number = match[1]
    digits = number.lstrip("+-").lstrip("0") or "0"
    if len(digits) > 2:
        return None, "answer_out_of_range"
    value = int(("-" if number.startswith("-") else "") + digits)
    if not 0 <= value < 23:
        return None, "answer_out_of_range"
    return value, "parsed"


def sample_seed(base: int, question_id: str, attempt: int) -> int:
    raw = f"{base}:{question_id}:{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0x7FFFFFFF


def restore_history(path: Path, rows: list[dict], config: dict) -> dict[str, list]:
    history = {row["id"]: [] for row in rows}
    answers = {row["id"]: row["answer"] for row in rows}
    if not path.exists():
        return history
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                question_id = record["id"]
                previous = history[question_id]
                attempt = record["attempt"]
                predicted, parse_status = parse_answer(record["text"], record["finish_reason"])
                valid = (
                    attempt == len(previous) + 1
                    and attempt <= config["max_attempts"]
                    and not any(x["correct"] for x in previous)
                    and record["seed"] == sample_seed(config["seed"], question_id, attempt)
                    and record["expected_answer"] == answers[question_id]
                    and record["predicted_answer"] == predicted
                    and record["parse_status"] == parse_status
                    and record["correct"] == (predicted == answers[question_id])
                )
                if not valid:
                    raise ValueError("inconsistent attempt record")
                previous.append(record)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Invalid attempt log at line {number}; preserve and inspect it") from exc
    return history


def summarize(rows: list[dict], history: dict, config: dict) -> dict:
    questions = []
    for row in rows:
        attempts = history[row["id"]]
        first = next((x["attempt"] for x in attempts if x["correct"]), None)
        questions.append({
            "id": row["id"], "attempts": len(attempts), "first_correct_attempt": first,
            "status": "solved" if first is not None else (
                "exhausted" if len(attempts) == config["max_attempts"] else "pending"
            ),
            "pass_at_k": {str(k): first is not None and first <= k for k in config["pass_k"]},
        })
    complete = all(x["status"] != "pending" for x in questions)
    counts = {str(k): sum(x["pass_at_k"][str(k)] for x in questions) for k in config["pass_k"]}
    lower_bound = {k: count / len(rows) for k, count in counts.items()}
    records = [record for records in history.values() for record in records]
    return {
        "updated_utc": utc_now(), "complete": complete, "num_questions": len(rows),
        "solved": sum(x["status"] == "solved" for x in questions),
        "pending": sum(x["status"] == "pending" for x in questions),
        "exhausted": sum(x["status"] == "exhausted" for x in questions),
        "total_attempts": len(records),
        "attempts_avoided_by_early_stop": sum(
            config["max_attempts"] - x["first_correct_attempt"]
            for x in questions if x["first_correct_attempt"] is not None
        ),
        "truncated_attempts": sum(x["finish_reason"] == "length" for x in records),
        "invalid_answer_attempts": sum(x["parse_status"] != "parsed" for x in records),
        "pass_at_k_counts": counts,
        "pass_at_k": lower_bound if complete else None,
        "current_pass_at_k_lower_bound": lower_bound,
        "metric_definition": "mean(first_correct_attempt <= k); no success within budget is failure",
        "questions": questions,
    }


def evaluate(rows: list[dict], config: dict, output: Path, generate_batch) -> dict:
    """The callback receives (row, attempt, seed) tasks and returns completion dicts."""
    log_path = output / "attempts.jsonl"
    history = restore_history(log_path, rows, config)
    summary = summarize(rows, history, config)
    atomic_json(output / "summary.json", summary)
    with log_path.open("a", encoding="utf-8", newline="\n") as stream:
        while not summary["complete"]:
            pending = [row for row in rows if len(history[row["id"]]) < config["max_attempts"]
                       and not any(x["correct"] for x in history[row["id"]])]
            for start in range(0, len(pending), config["batch_size"]):
                tasks = []
                for row in pending[start:start + config["batch_size"]]:
                    attempt = len(history[row["id"]]) + 1
                    tasks.append((row, attempt, sample_seed(config["seed"], row["id"], attempt)))
                started = time.monotonic()
                completions = generate_batch(tasks)
                elapsed = time.monotonic() - started
                if len(completions) != len(tasks):
                    raise RuntimeError("Backend returned an unexpected number of completions")
                for (row, attempt, seed), completion in zip(tasks, completions, strict=True):
                    predicted, parse_status = parse_answer(completion["text"], completion["finish_reason"])
                    record = {
                        "id": row["id"], "attempt": attempt, "seed": seed,
                        "expected_answer": row["answer"], "predicted_answer": predicted,
                        "correct": predicted == row["answer"], "parse_status": parse_status,
                        "recorded_utc": utc_now(), "batch_elapsed_seconds": elapsed,
                        **completion,
                    }
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    history[row["id"]].append(record)
                stream.flush()
                os.fsync(stream.fileno())
                summary = summarize(rows, history, config)
                atomic_json(output / "summary.json", summary)
                print(json.dumps({k: summary[k] for k in [
                    "solved", "pending", "exhausted", "total_attempts", "current_pass_at_k_lower_bound"
                ]}), flush=True)
    return summary


@contextmanager
def run_lock(output: Path):
    # Linux flock is released by the OS on exit, including an interrupted run.
    import fcntl
    with (output / ".run.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is using this output directory") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"{socket.gethostname()} pid={os.getpid()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / DATA_NAME)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-0.6B")
    parser.add_argument("--prompt", type=Path, default=Path(__file__).with_name("prompt.txt"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and tokenized prompts without a GPU engine")
    parser.add_argument("--limit", type=int, help="Select only the first N questions for a small check")
    parser.add_argument("--pass-k", type=int, nargs="+", default=[1, 3, 8, 50])
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execution-mode", choices=["eager", "cudagraph"], default="eager",
                        help="cudagraph captures decode execution without torch.compile")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.pass_k = sorted(set(args.pass_k))
    if not args.pass_k or min(args.pass_k) < 1 or max(args.pass_k) > args.max_attempts:
        parser.error("pass-k values must be positive and no greater than max-attempts")
    if min(args.batch_size, args.max_tokens, args.max_model_len) < 1:
        parser.error("batch-size and token limits must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 1:
        parser.error("Use positive temperature/top-k and 0 < top-p <= 1")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("gpu-memory-utilization must be between 0 and 1")
    if args.resume and args.output_dir is None:
        parser.error("resume requires output-dir")
    return args


def main() -> None:
    args = arguments()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    rows, data_sha = read_dataset(args.data)
    dataset_count = len(rows)
    if args.limit is not None:
        if args.limit > dataset_count:
            raise ValueError("limit exceeds the dataset size")
        rows = rows[:args.limit]
    template = args.prompt.read_text(encoding="utf-8")
    messages = {row["id"]: render_messages(template, row) for row in rows}
    user_prompts = {question_id: items[-1]["content"] for question_id, items in messages.items()}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model.resolve()), local_files_only=True,
                                             trust_remote_code=False)
    prompts = {}
    for row in rows:
        text = tokenizer.apply_chat_template(
            messages[row["id"]], tokenize=False,
            add_generation_prompt=True, enable_thinking=args.thinking,
        )
        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(token_ids) + args.max_tokens > args.max_model_len:
            raise ValueError(f"{row['id']}: prompt {len(token_ids)} + max-tokens exceeds max-model-len")
        prompts[row["id"]] = {"prompt_token_ids": token_ids}

    model = args.model.resolve()
    config = {
        "schema_version": 1, "data": str(args.data.resolve()), "data_sha256": data_sha,
        "dataset_count": dataset_count, "selected_ids": [row["id"] for row in rows],
        "model": str(model), "model_config_sha256": digest((model / "config.json").read_bytes()),
        "tokenizer_config_sha256": digest((model / "tokenizer_config.json").read_bytes()),
        "generation_config_sha256": digest((model / "generation_config.json").read_bytes()),
        "prompt_sha256": digest(template.encode("utf-8")),
        "script_sha256": digest(Path(__file__).read_bytes()),
        **{name: getattr(args, name) for name in [
            "pass_k", "max_attempts", "max_tokens", "max_model_len", "batch_size",
            "gpu_memory_utilization", "temperature", "top_p", "top_k", "seed", "thinking", "execution_mode",
        ]},
        "versions": {name: version(name) for name in ["transformers", "torch", "vllm"]},
        "answer_format": "lowercase answer: followed by the actual integer in 0..22; unfinished/truncated output is incorrect",
    }
    token_lengths = [len(x["prompt_token_ids"]) for x in prompts.values()]
    print(json.dumps({"dataset_count": dataset_count, "selected_count": len(rows),
                      "prompt_tokens_min": min(token_lengths), "prompt_tokens_max": max(token_lengths),
                      "config": config}, indent=2), flush=True)
    if args.dry_run:
        print("DRY_RUN_PASSED: no GPU engine or inference started", flush=True)
        return

    output = args.output_dir or Path(__file__).parent / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = output.resolve()
    if args.resume:
        if not output.is_dir():
            raise ValueError("Resume output directory does not exist")
    else:
        output.mkdir(parents=True, exist_ok=False)
    with run_lock(output):
        config_path = output / "run_config.json"
        if args.resume:
            if json.loads(config_path.read_text(encoding="utf-8")) != config:
                raise ValueError("Resume configuration differs; use the original inputs/settings/script")
        else:
            atomic_json(config_path, config)
            (output / "prompt.txt").write_text(template, encoding="utf-8")
            with (output / "prompts.jsonl").open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps({"id": row["id"], "user_prompt": user_prompts[row["id"]],
                                             "messages": messages[row["id"]],
                                             **prompts[row["id"]]}, ensure_ascii=False) + "\n")
        print(f"OUTPUT_DIR={output}", flush=True)
        history = restore_history(output / "attempts.jsonl", rows, config)
        summary = summarize(rows, history, config)
        atomic_json(output / "summary.json", summary)
        if not summary["complete"]:
            from vllm import LLM, SamplingParams
            engine = LLM(model=str(model), tokenizer=str(model), dtype="bfloat16",
                         tensor_parallel_size=1, seed=args.seed, enforce_eager=args.execution_mode == "eager",
                         compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}
                         if args.execution_mode == "cudagraph" else {"mode": 0, "cudagraph_mode": "NONE"},
                         gpu_memory_utilization=args.gpu_memory_utilization,
                         max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                         enable_prefix_caching=True, trust_remote_code=False, disable_log_stats=True)
            eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
            eos = [eos] if isinstance(eos, int) else eos

            def generate_batch(tasks):
                params = [SamplingParams(
                    n=1, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    seed=seed, max_tokens=args.max_tokens, stop_token_ids=eos,
                    ignore_eos=False, skip_special_tokens=True,
                ) for _, _, seed in tasks]
                results = engine.generate([prompts[row["id"]] for row, _, _ in tasks], params, use_tqdm=False)
                completions = []
                for result in results:
                    if not result.finished or len(result.outputs) != 1:
                        raise RuntimeError("Backend did not finish exactly one response per question")
                    candidate = result.outputs[0]
                    completions.append({"text": candidate.text, "finish_reason": candidate.finish_reason,
                                        "stop_reason": candidate.stop_reason,
                                        "generated_tokens": len(candidate.token_ids),
                                        "prompt_tokens": len(result.prompt_token_ids)})
                return completions

            summary = evaluate(rows, config, output, generate_batch)
        print("EVALUATION_COMPLETE " + json.dumps({"output_dir": str(output), "pass_at_k": summary["pass_at_k"],
                                                  "total_attempts": summary["total_attempts"]}), flush=True)


if __name__ == "__main__":
    main()

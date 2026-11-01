"""Generate configurable iGSM hard examples with a solution-only GPT-2 length filter."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

PARAMS = dict(max_op=21, max_edge=28, perm_level=5, detail_level=0,
              op=None, be_shortest=True)
IdGen = tokenizer = fix_seed = true_correct = None
MIN_TOKENS = SAMPLE_TIMEOUT = 0


def initialize(repo, minimum, timeout, params):
    global IdGen, tokenizer, fix_seed, true_correct, MIN_TOKENS, SAMPLE_TIMEOUT, PARAMS
    PARAMS = params
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    torch.set_num_threads(1)
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import tokenizer, fix_seed
    from tools.tools_test import true_correct
    # This tokenizer is used without a GPT-2 model; never truncate long solutions.
    tokenizer.model_max_length = 10**9
    MIN_TOKENS, SAMPLE_TIMEOUT = minimum, timeout
    signal.signal(signal.SIGALRM, on_timeout)


def on_timeout(signum, frame):
    raise TimeoutError("candidate generation exceeded its time limit")


def candidate(seed):
    signal.alarm(SAMPLE_TIMEOUT)
    try:
        fix_seed(seed)
        item = IdGen(**PARAMS)
        item.gen_prob(list(range(23)), p_format="pq")
        length = len(item.sol_token)
        if length < MIN_TOKENS:
            return dict(reason="too_short", seed=seed, solution_tokens=length)
        encoded = tokenizer.encode(item.sol, add_special_tokens=False, truncation=False)
        assert encoded == item.sol_token
        correct, diagnostic, _ = true_correct(item.sol, problem=item.problem)
        if not correct:
            return dict(reason="invalid_solution", seed=seed)
        answer = str(item.problem.ans)
        assert tokenizer.decode(item.ans_token).strip() == answer
        return dict(
            reason="accepted", seed=seed, difficulty="hard", problem=item.prob,
            solution=item.sol, answer=answer, solution_token_count=length,
            problem_token_count=len(item.prob_token), answer_token_count=len(item.ans_token),
            n_op=int(item.problem.n_op), generator_params=PARAMS,
            problem_sha256=hashlib.sha256(item.prob.strip().encode()).hexdigest(),
            solution_validated=True,
        )
    except TimeoutError:
        return dict(reason="timeout", seed=seed)
    except Exception as exc:
        return dict(reason="exception", seed=seed, error=type(exc).__name__ + ": " + str(exc)[:200])
    finally:
        signal.alarm(0)


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audit(path, count, minimum):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == count
    assert len({row["problem_sha256"] for row in rows}) == count
    lengths = []
    for row in rows:
        actual = len(tokenizer.encode(row["solution"], add_special_tokens=False, truncation=False))
        assert actual == row["solution_token_count"] and actual >= minimum
        assert row["solution_validated"] is True and row["difficulty"] == "hard"
        assert row["generator_params"] == PARAMS
        assert 1 <= row["n_op"] <= PARAMS["max_op"]
        assert PARAMS["op"] is None or row["n_op"] == PARAMS["op"]
        assert row["problem_sha256"] == hashlib.sha256(row["problem"].strip().encode()).hexdigest()
        lengths.append(actual)
    return dict(count=count, unique_problems=count, all_length_checks_passed=True,
                all_solutions_validated=True, min_solution_tokens=min(lengths),
                max_solution_tokens=max(lengths), mean_solution_tokens=statistics.mean(lengths),
                median_solution_tokens=statistics.median(lengths),
                operation_histogram=dict(Counter(row["n_op"] for row in rows)))


def main():
    global PARAMS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--min-solution-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=100000)
    parser.add_argument("--sample-timeout", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--max-op", type=int, default=21)
    parser.add_argument("--max-edge", type=int, default=28)
    parser.add_argument("--op", type=int, default=None)
    args = parser.parse_args()
    assert args.count > 0 and args.min_solution_tokens > 0 and args.workers > 0
    PARAMS = dict(PARAMS, max_op=args.max_op, max_edge=args.max_edge, op=args.op)
    assert args.max_op > 0 and args.max_edge > 1
    assert args.op is None or 1 <= args.op <= args.max_op
    output = args.output.resolve()
    partial = output.with_suffix(".partial.jsonl")
    audit_path = output.with_suffix(".audit.json")
    progress_path = output.with_suffix(".progress.json")
    for path in (output, partial, audit_path, progress_path):
        if path.exists():
            raise FileExistsError(f"Will not overwrite an existing run: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    initialize(args.repo, args.min_solution_tokens, args.sample_timeout, PARAMS)
    started = time.monotonic()
    reasons, seen, examples = Counter(), set(), []
    processed, last_seed = 0, None
    source_commit = subprocess.check_output(["git", "-C", args.repo, "rev-parse", "HEAD"], text=True).strip()

    def progress():
        return dict(accepted=len(examples), target=args.count, processed=processed,
                    last_seed=last_seed, elapsed_seconds=round(time.monotonic()-started, 2),
                    rejection_counts=dict(reasons), min_solution_tokens=args.min_solution_tokens)

    with partial.open("x", encoding="utf-8") as stream:
        with mp.get_context("spawn").Pool(args.workers, initializer=initialize,
                initargs=(args.repo, args.min_solution_tokens, args.sample_timeout, PARAMS)) as pool:
            # Ordered results make selection reproducible despite parallel execution.
            seeds = range(args.seed, args.seed + args.max_attempts)
            for row in pool.imap(candidate, seeds, chunksize=1):
                processed += 1
                last_seed = row["seed"]
                reason = row.pop("reason")
                if reason == "accepted" and row["problem_sha256"] in seen:
                    reason = "duplicate"
                reasons[reason] += 1
                if reason == "accepted":
                    seen.add(row["problem_sha256"])
                    row["id"] = f"igsm-hard-long-{len(examples)+1:04d}"
                    examples.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                if processed % 100 == 0 or reason == "accepted":
                    write_json(progress_path, progress())
                    print(json.dumps(progress()), flush=True)
                if len(examples) == args.count:
                    break
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("run time limit reached; accepted examples kept in partial file")
    if len(examples) != args.count:
        raise RuntimeError(f"Only accepted {len(examples)} of {args.count}; see partial file")
    checked = audit(partial, args.count, args.min_solution_tokens)
    # Publish without overwriting, and only after re-reading and re-tokenizing every row.
    os.link(partial, output)
    partial.unlink()
    checked.update(progress())
    checked.update(
        completed_at=datetime.now(timezone.utc).isoformat(), generator_params=PARAMS,
        op_style="light" if args.op is None else "fixed", baseline_hard_params=dict(max_op=21,max_edge=28,perm_level=5,detail_level=0),
        customized_difficulty=PARAMS["max_op"]!=21 or PARAMS["max_edge"]!=28 or args.op is not None,
        hash_bins=list(range(23)), p_format="pq", tokenizer="gpt2",
        token_count_scope="solution only; excludes question, final answer field and special markers",
        truncation=False, padding=False, source_commit=source_commit,
        seed_start=args.seed, workers=args.workers,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        package_versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "networkx")},
    )
    write_json(audit_path, checked)
    write_json(progress_path, dict(progress(), complete=True))
    print("DATASET_COMPLETE", json.dumps(checked), flush=True)


if __name__ == "__main__":
    main()

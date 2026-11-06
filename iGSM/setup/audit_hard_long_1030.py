"""Independently re-tokenize and regenerate every delivered iGSM example."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("dataset", type=Path)
cli.add_argument("--perm-level", type=int, choices=range(7), default=5)
args = cli.parse_args()
dataset = args.dataset.resolve()
repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
os.chdir(repo)
import torch
torch.set_num_threads(1)
from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from tools.tools_test import true_correct

tokenizer.model_max_length = 10**9
rows = [json.loads(line) for line in dataset.read_text().splitlines()]
if len(rows) != 100 or len({r["problem_sha256"] for r in rows}) != 100:
    raise ValueError("Expected exactly 100 unique questions")
params = dict(max_op=52, max_edge=80, op=52, perm_level=args.perm_level, detail_level=0, be_shortest=True)
lengths = []
for index, row in enumerate(rows, 1):
    if row["generator_params"] != params or row["n_op"] != 52:
        raise ValueError(f"Parameter mismatch at {index}")
    count = len(tokenizer.encode(row["solution"], add_special_tokens=False, truncation=False))
    if count < 1024 or count != row["solution_token_count"] or "..." in row["solution"]:
        raise ValueError(f"Length or placeholder check failed at {index}")
    fix_seed(row["seed"])
    sample = IdGen(**params)
    sample.gen_prob(list(range(23)), p_format="pq")
    if sample.prob != row["problem"] or sample.sol != row["solution"]:
        raise ValueError(f"Regenerated text differs at {index}")
    if str(sample.problem.ans) != row["answer"]:
        raise ValueError(f"Answer differs at {index}")
    correct, diagnostic, _ = true_correct(row["solution"], problem=sample.problem)
    if not correct:
        diagnostic.display()
        raise ValueError(f"Solution verification failed at {index}")
    lengths.append(count)
    if index % 20 == 0:
        print(f"INDEPENDENT_CHECK {index}/100", flush=True)

report = dict(
    verified_count=len(rows), unique_questions=100, all_solution_lengths_ge1024=True,
    all_regenerated_text_matches=True, all_solutions_reverified=True,
    no_placeholder_variables=True, tokenizer="gpt2", truncation=False,
    min_solution_tokens=min(lengths), max_solution_tokens=max(lengths),
    mean_solution_tokens=statistics.mean(lengths), median_solution_tokens=statistics.median(lengths),
    generator_params=params, dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
    verified_at=datetime.now(timezone.utc).isoformat(),
)
report_path = dataset.with_suffix(".independent-audit.json")
with report_path.open("x") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print("INDEPENDENT_AUDIT_PASSED", json.dumps(report), flush=True)

"""Convert iGSM JSONL to verl Parquet, without leaking solutions into prompts."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

INSTRUCTION = "Solve the problem step by step. End with a single line: #### <integer answer>."


def make_record(row, index, split):
    problem = row.get("problem", row.get("question"))
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"Row {index}: missing problem/question")
    if row.get("verified") is False:
        raise ValueError(f"Row {index}: explicitly unverified sample")
    answer = str(row.get("answer", "")).strip()
    if not re.fullmatch(r"[+-]?\d+", answer):
        raise ValueError(f"Row {index}: answer must be an integer")
    problem = problem.strip()
    return {
        "data_source": "igsm",
        "prompt": [{"role": "user", "content": problem + "\n\n" + INSTRUCTION}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(int(answer))},
        "extra_info": {"split": split, "index": index, "problem_sha256": hashlib.sha256(problem.encode()).hexdigest()},
    }


def split_records(rows, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    unique = {}
    for index, row in enumerate(rows):
        record = make_record(row, index, "pending")
        key = record["extra_info"]["problem_sha256"]
        if key in unique and unique[key]["reward_model"] != record["reward_model"]:
            raise ValueError("The same problem has conflicting answers")
        unique.setdefault(key, record)
    records = list(unique.values())
    if len(records) < 2:
        raise ValueError("Need at least two distinct problems for a disjoint train/validation split")
    random.Random(seed).shuffle(records)
    n_val = min(len(records) - 1, max(1, round(len(records) * val_fraction)))
    val, train = records[:n_val], records[n_val:]
    for split, items in (("train", train), ("val", val)):
        for i, item in enumerate(items):
            item["extra_info"].update(split=split, index=i)
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # A new directory avoids accidentally replacing an existing dataset.
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory")
    with args.input.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    train, val = split_records(rows, args.val_fraction, args.seed)
    import pyarrow as pa
    import pyarrow.parquet as pq
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, records in (("train", train), ("val", val)):
        pq.write_table(pa.Table.from_pylist(records), args.output_dir / f"{split}.parquet")
    print(json.dumps({"train": len(train), "val": len(val), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()

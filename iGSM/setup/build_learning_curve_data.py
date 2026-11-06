"""Rebuild the fixed iGSM 10100-row learning-curve dataset and nested train subsets.

Run in igsm-gpu with PYTHONHASHSEED=0. Existing identical completed stages are
checked and reused; conflicting files are never overwritten.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
IGSM = ROOT / "iGSM"
STEM = "igsm_ordered_10100_op52_gpt2_ge1024_20261104"
PARAMS = dict(max_op=52, max_edge=80, op=52, perm_level=6,
              detail_level=0, be_shortest=True)
SIZES = (10000, 5000, 2000, 1000, 500, 200, 100)
COUNT, TEST_COUNT, MIN_TOKENS, SEED_START, SPLIT_SEED = 10100, 100, 1024, 20000, 42
TOKENIZER = ID_GEN = FIX_SEED = TRUE_CORRECT = None


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def publish(path, payload):
    """Accept identical files on rerun, refuse to overwrite different contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing contents differ: {path}")
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(payload)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def publish_json(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def timeout(signum, frame):
    raise TimeoutError("Independent reconstruction exceeded 30 seconds")


def initialize_audit():
    global TOKENIZER, ID_GEN, FIX_SEED, TRUE_CORRECT
    os.chdir(IGSM)
    sys.path.insert(0, str(IGSM))
    import torch
    torch.set_num_threads(1)
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import tokenizer, fix_seed
    from tools.tools_test import true_correct
    TOKENIZER, ID_GEN, FIX_SEED, TRUE_CORRECT = tokenizer, IdGen, fix_seed, true_correct
    TOKENIZER.model_max_length = 10**9
    signal.signal(signal.SIGALRM, timeout)


def verify_row(row):
    signal.alarm(30)
    try:
        actual = len(TOKENIZER.encode(row["solution"], add_special_tokens=False, truncation=False))
        assert actual == row["solution_token_count"] and actual >= MIN_TOKENS, row["id"]
        assert "..." not in row["solution"], row["id"]
        FIX_SEED(row["seed"])
        sample = ID_GEN(**PARAMS)
        sample.gen_prob(list(range(23)), p_format="pq")
        assert sample.prob == row["problem"], row["id"]
        assert sample.sol == row["solution"], row["id"]
        assert str(sample.problem.ans) == row["answer"], row["id"]
        assert len(sample.prob_token) == row["problem_token_count"], row["id"]
        assert len(sample.ans_token) == row["answer_token_count"], row["id"]
        assert int(sample.problem.n_op) == row["n_op"] == 52, row["id"]
        correct, _, _ = TRUE_CORRECT(row["solution"], problem=sample.problem)
        assert correct, row["id"]
        return actual
    finally:
        signal.alarm(0)


def check_structure(rows):
    assert len(rows) == COUNT
    for field in ("id", "seed", "problem_sha256"):
        assert len({row[field] for row in rows}) == COUNT, f"Nonunique {field}"
    assert [row["seed"] for row in rows] == sorted(row["seed"] for row in rows)
    for index, row in enumerate(rows, 1):
        assert row["id"] == f"igsm-learning-curve-v1-{index:04d}"
        assert row["seed"] >= SEED_START
        assert row["generator_params"] == PARAMS and row["n_op"] == 52
        assert row["solution_validated"] is True
        assert row["answer"] in {str(i) for i in range(23)}
        assert row["solution_token_count"] >= MIN_TOKENS
        assert row["problem_sha256"] == hashlib.sha256(row["problem"].strip().encode()).hexdigest()


def stats(rows):
    lengths = [row["solution_token_count"] for row in rows]
    return dict(count=len(rows), min_solution_tokens=min(lengths),
                max_solution_tokens=max(lengths),
                mean_solution_tokens=statistics.mean(lengths),
                median_solution_tokens=statistics.median(lengths))


def independent_audit(pool_path, rows, workers):
    report_path = pool_path.with_suffix(".independent-audit.json")
    checksum = digest(pool_path)
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["dataset_sha256"] == checksum
        assert report["verified_count"] == COUNT and report["generator_params"] == PARAMS
        assert report["all_regenerated_text_matches"] and report["all_solutions_reverified"]
        return report
    lengths = []
    with mp.get_context("spawn").Pool(workers, initializer=initialize_audit) as pool:
        for index, length in enumerate(pool.imap(verify_row, rows, chunksize=4), 1):
            lengths.append(length)
            if index % 100 == 0 or index == COUNT:
                print(f"INDEPENDENT_CHECK {index}/{COUNT}", flush=True)
    report = dict(verified_count=COUNT, unique_questions=COUNT, unique_seeds=COUNT,
                  all_solution_lengths_ge1024=True, all_regenerated_text_matches=True,
                  all_solutions_reverified=True, no_placeholder_variables=True,
                  tokenizer="gpt2", truncation=False, generator_params=PARAMS,
                  dataset_sha256=checksum, workers=workers,
                  verified_at=datetime.now(timezone.utc).isoformat(), **stats(rows))
    publish_json(report_path, report)
    return report


def build_splits(data, pool_path, rows):
    ordered = list(rows)
    random.Random(SPLIT_SEED).shuffle(ordered)
    test, train = ordered[:TEST_COUNT], ordered[TEST_COUNT:]
    test_ids = {r["id"] for r in test}
    test_hashes = {r["problem_sha256"] for r in test}
    test_seeds = {r["seed"] for r in test}
    sets = [(data / "test" / "test_100.jsonl", test)]
    sets += [(data / "train" / f"train_{size}.jsonl", train[:size]) for size in SIZES]
    records = {}
    for path, selected in sets:
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")
        publish(path, payload)
        actual = read_rows(path)
        assert actual == selected
        if path.parent.name == "train":
            assert not test_ids.intersection(row["id"] for row in actual)
            assert not test_hashes.intersection(row["problem_sha256"] for row in actual)
            assert not test_seeds.intersection(row["seed"] for row in actual)
        records[path.relative_to(data).as_posix()] = dict(sha256=digest(path), **stats(actual))
    assert len(test) == 100 and len(train) == 10000
    assert len({r["problem_sha256"] for r in test + train}) == COUNT
    for size in SIZES:
        assert read_rows(data / "train" / f"train_{size}.jsonl") == train[:size]
    previous = data / "igsm_ordered_100_op52_gpt2_ge1024_20261103.jsonl"
    previous_check = None
    if previous.exists():
        old = read_rows(previous)
        previous_check = dict(file=previous.name, sha256=digest(previous),
                              shared_question_hashes=len({r["problem_sha256"] for r in old}
                                                         & {r["problem_sha256"] for r in rows}),
                              shared_seeds=len({r["seed"] for r in old} & {r["seed"] for r in rows}))
        assert previous_check["shared_question_hashes"] == previous_check["shared_seeds"] == 0
    generation = json.loads(pool_path.with_suffix(".audit.json").read_text())
    manifest_path = data / "learning_curve_manifest.json"
    stable = dict(schema_version=1, dataset_pool=pool_path.name,
                  dataset_pool_sha256=digest(pool_path), total_unique_examples=COUNT,
                  generator_params=PARAMS, minimum_solution_tokens=MIN_TOKENS,
                  tokenizer="gpt2", pythonhashseed=0, candidate_seed_start=SEED_START,
                  candidate_seed_end=generation["last_seed"],
                  processed_candidates=generation["processed"],
                  generation_workers=generation["workers"],
                  split_seed=SPLIT_SEED, test_count=TEST_COUNT,
                  train_sizes=list(SIZES), train_subsets_are_nested=True,
                  test_train_overlap_ids=0, test_train_overlap_question_hashes=0,
                  test_train_overlap_seeds=0,
                  split_method="random.Random(42).shuffle(pool rows); first 100 test, remaining 10000 train; train files are prefixes",
                  files=records, previous_100_dataset_overlap=previous_check,
                  generation_audit=pool_path.with_suffix(".audit.json").name,
                  independent_audit=pool_path.with_suffix(".independent-audit.json").name,
                  generator_script_sha256=generation["script_sha256"],
                  builder_script_sha256=digest(Path(__file__).resolve()),
                  source_commit=generation["source_commit"],
                  package_versions=generation["package_versions"],
                  full_dataset_statistics=stats(rows))
    # Completion time is intentionally absent: reruns reproduce the same manifest.
    publish_json(manifest_path, stable)
    return stable


def update_readme(data, manifest):
    begin, end = "<!-- LEARNING_CURVE_DATASET:BEGIN -->", "<!-- LEARNING_CURVE_DATASET:END -->"
    generation = json.loads((data / manifest["generation_audit"]).read_text())
    full = manifest["full_dataset_statistics"]
    table = "\n".join(f"| `{name}` | {info['count']} |"
                      for name, info in manifest["files"].items())
    text = f"""{begin}
# iGSM 训练数据量实验：10100 条数据

本批数据包含 10100 道唯一题目。固定留出 100 道 test，其余 10000 道组成训练池。
所有训练量实验使用同一份 test。各档训练集是嵌套子集：
`100 ⊂ 200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 5000 ⊂ 10000`。
每档都从同一固定随机顺序的训练池取前 N 条，不能把这些文件拼接后视为更多独立数据。

## 存储布局

| 文件（相对于 data/） | 条数 |
| --- | ---: |
{table}

完整源数据池：`{STEM}.jsonl`，保持候选 seed 的接收顺序，便于追溯和重新划分。
`learning_curve_manifest.json` 保存每个输出文件的 SHA256、条数、长度分布和划分规则。
每行 JSON 均保留 id、problem、solution、answer、seed、generator_params、长度、题目哈希和校验标记。
answer 为字符串形式的 0–22 整数。原始参考解答单独存储，不会向题目中拼入提示词。
当前统一提示词仍由 `exp/prompt.txt` 提供；本次不修改评测、SFT 或 RL 提示词。

## 固定生成条件

| 参数 | 取值 |
| --- | --- |
| 最终唯一题目数 | 10100 |
| max_op / op / max_edge | 52 / 52 / 80 |
| perm_level | 6，关闭题面打乱；保留生成器原排列，不保证条件严格按解题顺序出现 |
| detail_level / be_shortest | 0 / True |
| 模运算 / p_format / hash_bins | mod 23 / pq / 全部 0–22 哈希桶 |
| 参考解答长度门槛 | solution 正文 ≥1024 GPT-2 tokens，不计题目或单独 answer 字段 |
| tokenizer 参数 | add_special_tokens=False，truncation=False，无填充 |
| 候选 seed | 从 20000 递增；本次处理到 {generation['last_seed']} |
| PYTHONHASHSEED | 0，必须在启动 Python 前设置 |
| 生成并行数 | {generation['workers']} 个 CPU 进程；按 seed 顺序接收结果 |
| 筛选上限 | max_attempts=1000000；sample_timeout=30 秒；max_seconds=21600 秒 |
| id_prefix | igsm-learning-curve-v1 |
| 划分随机种子 | 42 |

仅接收长度达标且通过 `true_correct` 的解答，按去除题面两端空白后的 SHA256 去重。
生成阶段使用全部哈希桶；train/test 在生成之后划分。
使用新 seed 区间避开旧 100 题的 42–1803 范围；本机同时核对了与旧数据无题目哈希或 seed 重叠。

## 精确划分算法

1. 按候选 seed 递增生成并筛选，得到恰好 10100 条唯一记录。
2. 对完整记录列表执行 `random.Random(42).shuffle(rows)`。
3. 打乱后的前 100 条写入 `test/test_100.jsonl`；其余 10000 条作为唯一训练池。
4. 按训练池原顺序取前 N 条，分别写入各个 `train/train_N.jsonl`。
5. 重新读取全部输出文件，检查条数、内容、嵌套前缀关系，以及 test/train 的 id、seed、题目哈希零交集。

## 一条命令重新生成、审查和划分

使用已安装的 `igsm-gpu` 环境。机器没有 GPU 也可以运行；这是 CPU 生成任务。
如果数据丢失，保留仓库内的生成脚本和环境后执行：

```bash
source /root/sft-rl-experiment/iGSM/setup/activate.sh
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B /root/sft-rl-experiment/iGSM/setup/build_learning_curve_data.py \\
  --data-dir /root/sft-rl-experiment/data --workers 24
```

路径可按实际仓库位置调整。脚本串联生成、全量按 seed 独立重生成、划分与 README 更新。
若源数据池和独立审查报告已存在，先核对校验值，再复用完成的阶段；缺少的划分文件会重建。
已有同名但不同内容的数据文件会拒绝覆盖。
如果只丢失了源数据 JSONL，而生成审查报告仍在，脚本会在临时目录重建并核对原 SHA256，然后恢复缺失文件。
如果上次生成中断留下 `.partial.jsonl`，当前底层生成脚本不支持断点续接；
保留中断文件，并用 `--data-dir /root/sft-rl-experiment/data/rebuild_20261104` 指定新的空目录重新生成。
不需要保留已生成的 10100 条数据才能再次生成；需要保留脚本、固定参数和兼容环境。

如需只生成完整源数据池，等价底层命令为：

```bash
PYTHONHASHSEED=0 python -B /root/sft-rl-experiment/iGSM/setup/generate_hard_long_1030.py \\
  --repo /root/sft-rl-experiment/iGSM \\
  --output /root/sft-rl-experiment/data/{STEM}.jsonl \\
  --count 10100 --seed 20000 --workers 24 --min-solution-tokens 1024 \\
  --max-op 52 --max-edge 80 --op 52 --perm-level 6 \\
  --id-prefix igsm-learning-curve-v1 \\
  --max-attempts 1000000 --sample-timeout 30 --max-seconds 21600
```

需要重新做全量独立审查时，给统一脚本传入 `--recheck`，已有审查报告保持原样并与重审结果核对。
改变 CPU 并行数不会改变候选的接收顺序；超时、环境版本差异或生成器改动可能影响结果，应核对 manifest。
原始生成器将 difficulty 记为 hard，实际难度以本表及各行 generator_params 为准。

## 本次实际结果与校验

- 保留 {COUNT} 条，处理候选 {generation['processed']}；筛选计数：`{json.dumps(generation['rejection_counts'], ensure_ascii=False)}`。
- 全量参考解答长度：最短 {full['min_solution_tokens']}，最长 {full['max_solution_tokens']}，
  平均 {full['mean_solution_tokens']:.4f}，中位数 {full['median_solution_tokens']} GPT-2 tokens。
- 每条固定 52 次运算，答案均为 0–22，已逐条独立重生成并再次验证参考解答。
- train/test 的题目哈希、id、seed 均无交集，七档训练文件满足嵌套前缀关系。
- 完整数据池 SHA256：`{manifest['dataset_pool_sha256']}`。
- 生成环境：Python 3.10.19；torch=2.4.0+cu121，transformers=4.51.3，numpy=1.26.4，networkx=3.4.2。
- 源提交：`{manifest['source_commit']}`；生成和构建脚本 SHA256 见 manifest。
- 只创建数据与说明；没有启动模型训练。

{end}
"""
    path = data / "README.md"
    old = path.read_text() if path.exists() else ""
    if begin in old:
        start = old.index(begin)
        finish = old.index(end, start) + len(end)
        updated = old[:start] + text.strip() + old[finish:]
    else:
        if old.startswith("# iGSM 数据生成记录\n"):
            old = old.replace("# iGSM 数据生成记录", "# 历史记录：2026-11-03 的 100 题数据", 1)
        updated = text.strip() + "\n\n" + old
    temporary = path.with_name("README.md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path, default=ROOT / "data")
    cli.add_argument("--workers", type=int, default=24)
    cli.add_argument("--recheck", action="store_true", help="Repeat full independent reconstruction")
    args = cli.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        cli.error("Start Python with PYTHONHASHSEED=0 for reproducible generation")
    if args.workers < 1:
        cli.error("--workers must be positive")
    data = args.data_dir.resolve()
    data.mkdir(parents=True, exist_ok=True)
    pool_path = data / (STEM + ".jsonl")
    generation_audit = pool_path.with_suffix(".audit.json")
    if not pool_path.exists():
        command = [sys.executable, "-B", str(IGSM / "setup/generate_hard_long_1030.py"),
                   "--repo", str(IGSM), "--output", str(pool_path), "--count", str(COUNT),
                   "--seed", str(SEED_START), "--workers", str(args.workers),
                   "--min-solution-tokens", str(MIN_TOKENS), "--max-op", "52",
                   "--max-edge", "80", "--op", "52", "--perm-level", "6",
                   "--id-prefix", "igsm-learning-curve-v1", "--max-attempts", "1000000",
                   "--sample-timeout", "30", "--max-seconds", "21600"]
        if generation_audit.exists():
            # Git often retains provenance while ignoring JSONL data. Recreate a
            # missing pool in isolation and verify against the retained digest.
            previous = json.loads(generation_audit.read_text())
            assert previous["count"] == COUNT and previous["seed_start"] == SEED_START
            assert previous["generator_params"] == PARAMS
            with tempfile.TemporaryDirectory(prefix=".igsm-rebuild-", dir=data) as scratch:
                staged = Path(scratch) / pool_path.name
                command[command.index("--output") + 1] = str(staged)
                subprocess.run(command, check=True)
                assert digest(staged) == previous["dataset_sha256"], "Regenerated dataset SHA256 differs"
                publish(pool_path, staged.read_bytes())
                progress = pool_path.with_suffix(".progress.json")
                if not progress.exists():
                    publish(progress, staged.with_suffix(".progress.json").read_bytes())
        else:
            subprocess.run(command, check=True)
    generation = json.loads(generation_audit.read_text())
    assert generation["dataset_sha256"] == digest(pool_path)
    assert generation["generator_params"] == PARAMS
    assert generation["seed_start"] == SEED_START and generation["count"] == COUNT
    rows = read_rows(pool_path)
    check_structure(rows)
    if args.recheck and pool_path.with_suffix(".independent-audit.json").exists():
        with mp.get_context("spawn").Pool(args.workers, initializer=initialize_audit) as pool:
            for index, _ in enumerate(pool.imap(verify_row, rows, chunksize=4), 1):
                if index % 100 == 0 or index == COUNT:
                    print(f"INDEPENDENT_RECHECK {index}/{COUNT}", flush=True)
    independent_audit(pool_path, rows, args.workers)
    manifest = build_splits(data, pool_path, rows)
    update_readme(data, manifest)
    print("LEARNING_CURVE_COMPLETE", json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

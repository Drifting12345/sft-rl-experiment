<!-- LEARNING_CURVE_DATASET:BEGIN -->
# iGSM 训练数据量实验：10100 条数据

本批数据包含 10100 道唯一题目。固定留出 100 道 test，其余 10000 道组成训练池。
所有训练量实验使用同一份 test。各档训练集是嵌套子集：
`100 ⊂ 200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 5000 ⊂ 10000`。
每档都从同一固定随机顺序的训练池取前 N 条，不能把这些文件拼接后视为更多独立数据。

## 存储布局

| 文件（相对于 data/） | 条数 |
| --- | ---: |
| `test/test_100.jsonl` | 100 |
| `train/train_10000.jsonl` | 10000 |
| `train/train_5000.jsonl` | 5000 |
| `train/train_2000.jsonl` | 2000 |
| `train/train_1000.jsonl` | 1000 |
| `train/train_500.jsonl` | 500 |
| `train/train_200.jsonl` | 200 |
| `train/train_100.jsonl` | 100 |

完整源数据池：`igsm_ordered_10100_op52_gpt2_ge1024_20261104.jsonl`，保持候选 seed 的接收顺序，便于追溯和重新划分。
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
| 候选 seed | 从 20000 递增；本次处理到 205100 |
| PYTHONHASHSEED | 0，必须在启动 Python 前设置 |
| 生成并行数 | 24 个 CPU 进程；按 seed 顺序接收结果 |
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
python -B /root/sft-rl-experiment/iGSM/setup/build_learning_curve_data.py \
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
PYTHONHASHSEED=0 python -B /root/sft-rl-experiment/iGSM/setup/generate_hard_long_1030.py \
  --repo /root/sft-rl-experiment/iGSM \
  --output /root/sft-rl-experiment/data/igsm_ordered_10100_op52_gpt2_ge1024_20261104.jsonl \
  --count 10100 --seed 20000 --workers 24 --min-solution-tokens 1024 \
  --max-op 52 --max-edge 80 --op 52 --perm-level 6 \
  --id-prefix igsm-learning-curve-v1 \
  --max-attempts 1000000 --sample-timeout 30 --max-seconds 21600
```

需要重新做全量独立审查时，给统一脚本传入 `--recheck`，已有审查报告保持原样并与重审结果核对。
改变 CPU 并行数不会改变候选的接收顺序；超时、环境版本差异或生成器改动可能影响结果，应核对 manifest。
原始生成器将 difficulty 记为 hard，实际难度以本表及各行 generator_params 为准。

## 本次实际结果与校验

- 保留 10100 条，处理候选 185101；筛选计数：`{"too_short": 175001, "accepted": 10100}`。
- 全量参考解答长度：最短 1024，最长 1223，
  平均 1050.2321，中位数 1042.5 GPT-2 tokens。
- 每条固定 52 次运算，答案均为 0–22，已逐条独立重生成并再次验证参考解答。
- train/test 的题目哈希、id、seed 均无交集，七档训练文件满足嵌套前缀关系。
- 完整数据池 SHA256：`fdf556b40c739fba56eacc5495adeeb673ce0dba385e518b2eebe69cf845aab6`。
- 生成环境：Python 3.10.19；torch=2.4.0+cu121，transformers=4.51.3，numpy=1.26.4，networkx=3.4.2。
- 源提交：`c3126c001ea5d0261d63a2718fc419ea72ac9c7a`；生成和构建脚本 SHA256 见 manifest。
- 只创建数据与说明；没有启动模型训练。

<!-- LEARNING_CURVE_DATASET:END -->

# 历史记录：2026-11-03 的 100 题数据

数据文件：`igsm_ordered_100_op52_gpt2_ge1024_20261103.jsonl`。

本数据集包含 100 道英文 iGSM 题目。每题固定 52 次运算，关闭题面打乱，参考解答正文至少 1024 GPT-2 tokens。已完成逐题解答校验、去重、重新分词和按原 seed 独立重生成核对。

## 生成参数

| 参数 | 本次取值 | 含义 |
| --- | --- | --- |
| count | 100 | 最终保留的唯一题目数 |
| max_op | 52 | 运算次数上限 |
| op | 52 | 每题固定实际运算次数 |
| max_edge | 80 | 生成结构图的边数上限 |
| perm_level | 6 | rand_perm=none、perm=False，关闭题面打乱 |
| detail_level | 0 | 保留变量定义、公式和数值代入的详细参考解答 |
| be_shortest | True | 只展开回答问题所需的解答依赖 |
| mod | 23 | 加、减、乘结果均归约到 0–22 |
| p_format | pq | 条件描述后接问题 |
| hash_bins | list(range(23)) | 使用全部 0–22 哈希桶，不在生成阶段划分训练/测试 |
| min_solution_tokens | 1024 | 仅统计 solution 正文的 GPT-2 tokens |
| tokenizer | gpt2 | 不添加特殊 token，不截断，不填充 |
| seed_start | 42 | 候选 seed 从 42 逐个递增 |
| workers | 4 | CPU 工作进程数，按 seed 顺序收集结果 |
| PYTHONHASHSEED | 0 | 在启动 Python 前固定哈希随机种子 |
| sample_timeout | 30 秒 | 单个候选的生成时间限制 |
| max_seconds | 3600 秒 | 生成批次时间限制参数 |
| max_attempts | 100000 | 最多尝试的候选 seed 数 |

关闭打乱表示保留生成器原有排列，不保证每个条件都按解题顺序排列。数据中的 difficulty=hard 和 igsm-hard-long-* ID 沿用原脚本标签；实际配置以各行 generator_params 为准。

## 接收条件

1. solution 正文达到至少 1024 GPT-2 tokens；不包含题目、独立 answer 字段和训练特殊标记，也不是 Qwen tokenizer 的长度。
2. true_correct 校验参考解答的计算、变量引用及依赖关系，最终答案与生成器一致。
3. 题目哈希不重复；最终数量恰好为 100。
4. 发布前重新读取、重新分词检查；随后按每题 seed 和相同参数独立重生成，逐字核对 problem、solution 和 answer。
5. 输出及其伴随记录必须尚不存在，脚本拒绝覆盖已有运行。

## 实际生成结果

- 完成记录时间（Asia/Shanghai）：`2026-11-03T20:41:40.815198+08:00`。
- 处理候选：1762；保留：100；其余原因：`{"too_short": 1662}`。
- 候选 seed 范围：42–1803；保留题目的 seed 范围：66–1803。每题原始 seed 保存在数据行中。
- 参考解答长度：最小 1024，最大 1185，平均 1051.03，中位数 1042.0 GPT-2 tokens。
- 每题 n_op 均为 52；100 道题题目哈希唯一，所有答案均为 0–22。
- 当前评测脚本的 CPU dry-run 已通过，未启动 GPU 推理或训练。
- 数据 SHA256：`5e8a7d4c9480b3338610cf934dacf19861eebe6df3ee3bb30b111d2f8b52aecf`。

## 复现命令

运行前将输出文件名改为一个尚不存在的新文件名。

```bash
source /root/sft-rl-experiment/iGSM/setup/activate.sh
PYTHONHASHSEED=0 python -B /root/sft-rl-experiment/iGSM/setup/generate_hard_long_1030.py \
  --repo /root/sft-rl-experiment/iGSM \
  --output /root/sft-rl-experiment/data/igsm_ordered_100_op52_gpt2_ge1024_20261103.jsonl \
  --count 100 --seed 42 --workers 4 --min-solution-tokens 1024 \
  --max-op 52 --max-edge 80 --op 52 --perm-level 6 \
  --max-attempts 100000 --sample-timeout 30 --max-seconds 3600
```

独立校验命令（报告文件已存在时拒绝覆盖）：

```bash
PYTHONHASHSEED=0 python -B /root/sft-rl-experiment/iGSM/setup/audit_hard_long_1030.py \
  /root/sft-rl-experiment/data/igsm_ordered_100_op52_gpt2_ge1024_20261103.jsonl --perm-level 6
```

## 文件与来源

- `igsm_ordered_100_op52_gpt2_ge1024_20261103.jsonl`：正式 JSONL 数据，每行保留 problem、solution、answer、seed、generator_params、长度统计、哈希及校验标记。
- `igsm_ordered_100_op52_gpt2_ge1024_20261103.audit.json`：生成审查、长度分布、筛选计数、包版本和数据校验值。
- `igsm_ordered_100_op52_gpt2_ge1024_20261103.independent-audit.json`：独立重生成校验报告。
- `igsm_ordered_100_op52_gpt2_ge1024_20261103.progress.json`：最终生成进度记录。

生成环境：igsm-gpu，Python 3.10.19；torch=2.4.0+cu121, transformers=4.51.3, numpy=1.26.4, networkx=3.4.2。
根仓库提交：`c3126c001ea5d0261d63a2718fc419ea72ac9c7a`；当前生成和审查脚本含本地 --perm-level 参数改动。
生成脚本 SHA256：`f4ece852c0594a524e2370c442580141273e7ac07913aac473af52500e55d8d7`。
独立审查脚本 SHA256：`4fce2eeadeeddb5ea0531159e39695b459b04b4e5a2d7bb004cf06c00e1b5cbd`。

本次只生成并校验这 100 道题，没有执行训练/验证划分，也没有修改评测脚本默认指向的历史数据文件名。

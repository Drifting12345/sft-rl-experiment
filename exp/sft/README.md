# Qwen3-0.6B iGSM SFT

本目录提供基于原始 Qwen3-0.6B 的监督微调入口。已通过 CPU dry-run、真实 tokenizer 的标签/EOS/padding 检查和当前 Transformers API 检查；**尚未执行 GPU 训练**。本次不生成数据或启动训练；训练数据需另行准备并显式传入。实际脚本参数以 `python train_sft.py --help` 为准。

## 数据准备与拆分

训练数据必须通过 `--train-data` 显式提供，格式为原始 iGSM JSONL。每行保留：

| 字段 | 内容 |
| --- | --- |
| `problem` | 非空题面 |
| `solution` | iGSM 生成并校验的参考解答正文 |
| `answer` | 0–22 的整数或整数字符串 |
| `seed` | 该题实际生成种子，用于追溯和防泄漏检查 |

建议同时保留 `id`、`generator_params`、题目哈希等原始元数据。不要使用模型推理日志代替原始参考解答。

当前固定 100 题文件是保留的评测基准，**不得传给 `--train-data`**：

```text
/inspire/hdd/project/deepanalysis/guitao-25013/zfy/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl
```

训练入口默认将它作为 heldout 泄漏检查依据。训练集与这份基准不能有相同题面或相同 `seed`；仅改 ID、文件名或记录顺序不能形成新训练数据。

`--eval-data` 可选，用于显式提供独立验证集；验证集与训练集也不能重题或共享种子。建议从另外准备的新数据池预先划分 train/validation，并把固定 100 题保留给训练后的独立评测。拆分前按题面与种子去重，记录所用种子范围和生成参数；不要把同题的不同副本分到不同集合。

将来准备新训练数据时，可沿用 [iGSM 数据生成工具](../../iGSM/setup/README.md)，使用新输出名和不重叠种子范围。本批基准曾从候选种子 42–1624 中筛选；选择更大的新种子起点只能减少交叉风险，仍须执行题面与种子的泄漏检查。本入口不会替用户生成这些数据。

## 监督目标与长度

用户输入使用模 23 规则、当前题面及最终数字格式要求。assistant 目标由参考解答追加最后一行组成：

```python
assistant_text = row["solution"].rstrip() + f"\nanswer:{int(row['answer'])}"
```

最终前缀为小写 `answer:`，后面是实际整数。加、减、乘均使用模 23；目标不增加模型专用 thinking 内容，聊天模板使用 thinking 关闭模式。

默认复用上层 `prompt.txt` 和 `prompt_examples.json`，与当前评测输入保持一致；用 `--no-examples` 可关闭固定示例，后续评测也需使用相同输入方式。只对最后一个 assistant 的解答、数字答案和 EOS 计算 loss，题目、两对人工示例、空 thinking 前缀与 padding 都不参与监督。脚本逐题断言训练序列与评测的输入 token 前缀完全一致。

序列长度按 Qwen tokenizer 处理后的完整对话计算，包含输入与目标；超过 `--max-length` 或目标超过 `--max-response-tokens`（默认 2048）的样本报错，不截断长样本。数据生成时的“solution 至少 1024 GPT-2 tokens”与这里的 Qwen 序列长度不是同一口径。

## 默认配置

| 配置 | 值 |
| --- | --- |
| 微调方式 | 全参数训练 |
| 数值精度 | BF16 |
| Attention | SDPA |
| Gradient checkpointing | 开启 |
| Learning rate | `1e-5` |
| Epochs | 3 |
| 最大序列长度 | 8192 tokens，不截断 |
| 每卡 microbatch | 1 |
| Gradient accumulation | 8；单卡时有效 batch 为 8 |

这些是脚本实现采用的起始配置，尚未经实际 GPU 训练验收，不表示已证明训练收敛或评测成绩提升。

## 入口

使用已有独立 Conda 环境，避免混入 iGSM 数据生成环境的依赖：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp/sft
```

先执行 CPU dry-run，核对输入格式、题面/种子重叠、模板、assistant 标签范围及完整序列长度。它不会加载模型权重、创建输出目录或执行训练；通过不等于 GPU 训练通过。

```bash
bash run_sft.sh \
  --train-data /absolute/path/train.jsonl \
  --output-dir runs/qwen3-0.6b \
  --dry-run
```

下面仅为后续训练命令示例，本次不执行。将绝对路径替换为另行准备的训练数据和新的输出目录：

```bash
python train_sft.py \
  --train-data /absolute/path/train.jsonl \
  --output-dir /absolute/path/exp/sft/runs/qwen3-0.6b
```

需要训练期间的验证损失时，另行显式提供独立的 `--eval-data`。不要直接把当前固定 100 题当作训练集；保持训练、验证和最终评测的用途区分。

训练配置、模型和 checkpoint 等产物存放于所选 `--output-dir`。训练数据与运行输出不提交 Git；提交前确认根 `.gitignore` 覆盖数据文件及 `exp/sft/runs/`，只维护脚本、说明和必要配置。

`run_sft.sh` 会自动激活上述 Conda 环境并转发参数。每个 epoch 保存完整 checkpoint，保留最近两个；配置、数据/提示词/示例/源码 SHA256 和 token 长度记录保存在运行目录。最终可直接推理的完整模型及 tokenizer 导出到 `<output-dir>/final`。

续训需要相同数据、配置和原始模型路径，以及包含 optimizer/scheduler/RNG 的完整 checkpoint：

```bash
bash run_sft.sh \
  --train-data /absolute/path/train.jsonl \
  --output-dir runs/qwen3-0.6b \
  --resume-from-checkpoint runs/qwen3-0.6b/checkpoint-100
```

`--eval-data` 只计算独立验证集上的监督 loss，不是 pass@k。训练后可复用上层评测脚本，对固定 100 题做相同口径的独立评测：

```bash
python ../eval_qwen3_igsm.py \
  --model runs/qwen3-0.6b/final \
  --examples ../prompt_examples.json \
  --max-tokens 2048 --no-thinking \
  --output-dir ../runs/qwen3-0.6b-sft-eval
```

本次只编写和检查脚本。当前推理评测仍占用 GPU，训练需在之后单独启动。

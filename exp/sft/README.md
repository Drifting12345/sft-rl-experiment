# Qwen3-0.6B iGSM LoRA SFT

本目录提供基于原始 Qwen3-0.6B 的 LoRA 监督微调入口，复用既有 `verl-train` Conda 环境，使用 Hugging Face Trainer 和 PEFT 0.21.2，不新建环境。

支持单卡和 torchrun DDP。2026-11-04 正在执行五档独立基模实验，详见本文末尾的实际运行记录。

## 数据准备与拆分

`--train-data` 必须显式提供一份完整的新数据池，格式为原始 iGSM JSONL；脚本自动从中划分训练与验证分区。每行保留：

| 字段 | 内容 |
| --- | --- |
| `problem` | 非空题面 |
| `solution` | iGSM 生成并校验的参考解答正文 |
| `answer` | 0–22 的整数或整数字符串 |
| `seed` | 该题实际生成种子，用于分组、追溯和防泄漏检查 |

建议同时保留 `id`、`generator_params`、题目哈希等原始元数据。不要使用模型推理日志代替原始参考解答。

当前固定 100 题文件是保留的评测基准，**不得传给 `--train-data`**：

```text
/root/sft-rl-experiment/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl
```

`--heldout-data` 默认指向这份固定基准。完整的新数据池不能与 heldout 有相同题面或相同 `seed`；仅改 ID、文件名或记录顺序不能形成新数据。固定 100 题不参与梯度训练、训练/验证划分或每个 epoch 的最佳模型选择。

自动划分使用固定随机种子 **42**，先按原始 `seed` 分组，再将约 90% 的组用于训练、约 10% 用于验证。至少需要 **10 个独立 seed 组**；不足时拒绝自动划分。相同 seed 的记录始终进入同一分区，分区间没有种子重叠。准备数据时也应清理重复题面，不能将同题副本分到不同集合。

组大小不均时，按记录数计算的实际比例可能偏离 90/10。运行目录的 `split_manifest.json` 记录划分随机种子、分组和分区归属、样本 ID、题目哈希及实际数量，用于复核和恢复；每题原始生成 seed 与划分随机种子是不同概念。

将来准备新训练数据时，可沿用 [iGSM 数据生成工具](../../iGSM/setup/README.md)，使用新输出名和不重叠种子范围。本批基准曾从候选种子 42–1624 中筛选；选择更大的新种子起点只能减少交叉风险，仍须执行题面与种子的泄漏检查。本入口不会替用户生成这些数据。

## 监督目标与长度

用户输入使用模 23 规则、当前题面及最终数字格式要求。assistant 目标由参考解答追加最后一行组成：

```python
assistant_text = row["solution"].rstrip() + f"\nanswer: {int(row['answer'])}"
```

最终前缀为小写 `answer:`，后面是实际整数。加、减、乘均使用模 23；目标不增加模型专用 thinking 内容，聊天模板使用 thinking 关闭模式。

默认读取上层 `prompt.txt`，与评测和 RL 使用同一份完整模板，其中已内嵌一道 easy 示例。只对最后一个 assistant 的当前解答、数字答案和 EOS 计算 loss；整个 user 消息（含内嵌示例）、空 thinking 前缀与 padding 均不参与监督。脚本逐题断言训练序列与评测输入 token 前缀完全一致。

序列长度按 Qwen tokenizer 处理后的完整对话计算，包含输入与目标；超过 `--max-length` 或目标超过 `--max-response-tokens`（默认 2048）的样本报错，不截断长样本。数据生成时的“solution 至少 1024 GPT-2 tokens”与这里的 Qwen 序列长度不是同一口径。

## 默认 LoRA 配置

| 配置 | 值 |
| --- | --- |
| 微调方式 | LoRA |
| LoRA rank / alpha / dropout | `r=16` / `alpha=32` / `dropout=0.05` |
| LoRA target modules | `q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj` |
| 数值精度 | BF16 |
| Attention | SDPA |
| Gradient checkpointing | 开启 |
| Learning rate | `2e-4` |
| Epochs | 最多 10；可被验证 loss 连续升高早停提前结束 |
| 最大序列长度 | 8192 tokens，不截断 |
| assistant 目标及独立评测响应上限 | 2048 tokens，不截断监督目标 |
| 每卡 microbatch | 1 |
| Gradient accumulation | 8；单卡时有效 batch 为 8 |
| 训练/验证划分随机种子 | 42 |

这些是脚本实现采用的起始配置，尚未经实际 GPU 训练验收，不表示已证明训练收敛或评测成绩提升。

独立推理评测的实际日志已确认 vLLM 0.32、BF16 和 `FLASH_ATTN2`，top-k/top-p 采样使用 FlashInfer。SFT 训练使用 HF Trainer + PEFT 的 SDPA，属于另一条执行路径；不能把训练的 SDPA 设置当作独立推理的 attention 后端。

## 验证 loss、最佳模型与连续升高早停

HF Trainer 在每个 epoch 结束后，对自动划分的验证集计算 assistant-only、teacher-forced `eval_loss`。监督位置与训练一致：仅当前题目最后一条 assistant 的参考解答、数字答案和 EOS 参与 loss，主指令、题面、固定示例及 padding 均屏蔽。**验证 loss 越低越好。**

验证阶段不逐题生成答案，也不执行 greedy 生成或生成准确率回调。HF Trainer 先完成本轮验证并提供 `eval_loss`，再由 `early_stopping.py` 中的连续升高早停回调读取。验证 loss 用于训练控制；训练后的最终数字正确率仍由独立推理评测统计。

两种判断独立进行：

- **最佳 adapter**：按历史最低 `eval_loss` 保存，训练结束载入该最佳 adapter，并据此导出合并模型。
- **早停**：首次验证记录上一轮 loss，`counter=0`；之后与上一轮 `eval_loss` 比较。严格升高时 `counter += 1`；持平或降低均将 `counter` 清零。随后更新上一轮 loss。只有连续两轮严格升高、计数达到 2，才提前停止。

规则示例（不是实际 loss）：`0.30 → 0.32 → 0.33` 触发停止；`0.30 → 0.32 → 0.32 → 0.33` 不触发，因为持平时已清零，最后只有一次升高。loss 降低即使尚未低于历史最佳，也会清零升高计数。

checkpoint 保存上一轮验证 loss、`counter` 等回调状态，续训时恢复它们，不重新开始计算升高次数。最佳 checkpoint 的历史最低 loss 与连续升高计数分别保存，两者不混用。

## 入口

使用已有独立 Conda 环境，避免混入 iGSM 数据生成环境的依赖：

```bash
source /root/sft-rl-experiment/verl/setup/activate.sh
cd /root/sft-rl-experiment/exp/sft
```

先执行 CPU dry-run，核对新数据池、heldout 泄漏、按 seed 分组的划分、模板、assistant 标签范围及完整序列长度。dry-run 不启动 GPU 训练，通过也不等于 GPU 训练验收。

```bash
bash run_sft.sh \
  --train-data /absolute/path/new_pool.jsonl \
  --heldout-data /root/sft-rl-experiment/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl \
  --output-dir runs/qwen3-0.6b-lora \
  --dry-run
```

下面仅为后续训练命令示例，本次不执行。将绝对路径替换为另行准备的训练数据和新的输出目录：

```bash
python train_sft.py \
  --train-data /absolute/path/new_pool.jsonl \
  --heldout-data /root/sft-rl-experiment/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl \
  --output-dir /absolute/path/exp/sft/runs/qwen3-0.6b-lora
```

无需另外传入预先切好的验证集，脚本从完整新数据池自动划分。epoch 上限默认是 10，示例无需重复指定。保持训练池、内部验证分区和固定 100 题 heldout 的用途区分。

训练配置、模型和 checkpoint 等产物存放于所选 `--output-dir`。训练数据与运行输出不提交 Git；提交前确认根 `.gitignore` 覆盖数据文件及 `exp/sft/runs/`，只维护脚本、说明和必要配置。

`run_sft.sh` 自动激活上述 Conda 环境并转发参数。配置、数据/提示词/示例/源码 SHA256、token 长度和划分清单保存在运行目录。

| 输出 | 用途 |
| --- | --- |
| `checkpoint-*` | 完整续训状态：adapter、optimizer、scheduler、RNG、Trainer state 与早停回调状态；保留最佳和最新 checkpoint，两者相同时无需重复保留 |
| `best_adapter/` | 独立保存按历史最低验证 `eval_loss` 选择的 LoRA adapter；需要配合原始基础模型使用 |
| `final/` | 训练结束载入最佳 adapter 后，与基础模型合并导出的完整模型和 tokenizer，可直接供现有 vLLM 评测脚本加载 |

完整 checkpoint 的“完整”指续训状态完整，不要求重复保存冻结的基础模型权重。`best_adapter/` 不能替代完整 checkpoint 来恢复优化器和早停状态，也不能作为完整模型目录直接传给现有评测的 `--model`。

续训需要相同数据、配置和原始模型路径，以及包含 optimizer/scheduler/RNG 的完整 checkpoint：

```bash
bash run_sft.sh \
  --train-data /absolute/path/new_pool.jsonl \
  --heldout-data /root/sft-rl-experiment/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl \
  --output-dir runs/qwen3-0.6b-lora \
  --resume-from-checkpoint runs/qwen3-0.6b-lora/checkpoint-100
```

将示例中的 `checkpoint-100` 替换为实际保存的完整 checkpoint。续训保持新数据池、划分、主指令、示例、原始模型路径和训练配置一致。

训练后使用合并导出的 `final/`，复用上层 vLLM 评测脚本对固定 100 题做独立评测：

```bash
python ../eval_qwen3_igsm.py \
  --model runs/qwen3-0.6b-lora/final \
  --max-tokens 2048 --no-thinking \
  --output-dir ../runs/qwen3-0.6b-lora-eval
```

本次只编写和检查脚本，不自动生成数据、不启动 GPU SFT，也不中断已有基线评测。GPU 训练需在之后单独启动。

## 五档八卡实际运行（2026-11-04，v2）

顺序：100 → 10000 → 5000 → 500 → 200，每组训练后立即评测；每次从 Qwen3-0.6B 初始化新 LoRA，不跨组续训。
使用 data/train/train_N.jsonl；最终 test 固定为 data/test/test_100.jsonl 的100题。
内部按 seed=42 划分90%训练、10%验证。五组实际训练/验证数为90/10、9000/1000、4500/500、450/50、180/20。
torchrun --standalone --nproc_per_node=8，每卡 microbatch=1，梯度累积=1，有效batch=8。
其他默认训练参数保持一致。LoRA r16/alpha32/dropout0.05，学习率2e-4，最多10epoch；连续两次验证loss严格升高早停，以最低验证loss选择最佳adapter。
v2最佳adapter通过CPU tensor直接复制到各rank已有模型，避免PEFT按旧CPU device_map自动跨卡分配。
主rank写配置并导出完整final/；checkpoint保留各rank RNG状态。v1失败运行和checkpoint保留。
训练输出：exp/sft/runs/qwen3-0.6b-trainN-20261104-v2/。
评测输出：exp/runs/qwen3-0.6b-sft-trainN-test100-20261104-v2/。
统一评测thinking关闭，temperature0.6/top_p0.95/top_k20，seed42，每题最多50次、单次2048tokens；报告pass@1/3/8/50。
完成后汇总：exp/sft/runs/qwen3-0.6b-data-size-20261104-v2/README.md 与 comparison.json。
后台日志与状态：/root/sft-env-setup-20261104/sft-data-size-v2/。

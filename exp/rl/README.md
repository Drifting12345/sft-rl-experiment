# iGSM GRPO / DAPO

本目录使用当前精简版 verl 的 V1 同步训练器，对原始 Qwen3-0.6B 做全参数强化学习。默认基模为 `ROOT/models/Qwen3-0.6B`，其中 `ROOT` 是 zfy 根目录；不接入 SFT 的训练结果或 LoRA adapter。`--model` 仅接受 `ROOT/models` 内的原始模型目录。这里采用 FSDP、BF16 和梯度检查点，LoRA rank 为 0；SFT 目录中的 LoRA 设置不会影响 RL。

`run_rl.sh` 通过 `ROOT/verl/setup/activate.sh` 复用已有的 `verl-train` Conda 环境，不创建新环境。训练时自行启动本机单节点 Ray，使用 V1 sync trainer 和 vLLM rollout；vLLM 的异步请求接口不代表采用异步策略更新。当前没有这两个 8 卡配置的 GPU 训练实测结果，CPU dry-run 也不代表显存与训练吞吐验证通过。

## 数据与提示词

`--train-data` 必须指向用户另外提供的新 iGSM 原始 JSONL 数据池，不能使用现有正式 100 题评测集。每条至少包含 `problem` 和 `answer`；答案为模 23 下的整数 `0..22`。建议保留稳定的 `id` 和生成器 `seed`。RL 不需要 `solution`，即使文件里有解题过程，也不会把它或答案拼进题面。脚本只读取、检查并转换现有数据，不生成训练题。

数据以固定划分随机种子 42 自动拆为约 90% 训练、10% 验证。相同生成器 seed 的题必须在同一区；没有 seed 时按规范化题面分组。至少需要 10 个独立组，且训练分区题数不能小于训练 batch（默认 32）。组大小不均时，实际比例可能偏离 90/10。数据清单记录分区 ID、题面哈希、生成器 seed、文件 SHA256、token 长度和提示词哈希。

默认保护 `ROOT/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl`：新数据池与该正式 100 题不能有相同题面或生成器 seed。正式集只用于泄漏检查及后续独立评测，不参与训练和训练中的验证。`--heldout-data` 可补充受保护文件，第一份仍必须是正式 100 题文件。

训练与验证复用 `exp/prompt.txt` 和 `exp/prompt_examples.json`：模 23、简洁方程、末行小写 `answer:实际数字`；所有题使用同样两个人工 toy 示例。chat template 明确设置 `enable_thinking=False`。训练前会用真实 tokenizer 检查完整 few-shot 提示词的长度，超限即报错，不截断题面。

## 算法与默认参数

奖励只有 0 或 1。只有实际生成以 EOS 完整结束、最后一行符合现有 `parse_answer` 的规则、且数字等于标准答案时才得 1；错误答案、格式错误、未完整结束或长度截断均为 0。即使截断文本看起来已经包含正确数字，也不能得分。自定义 reward manager 在解码前检查末尾 EOS，避免把截断的 `answer:22` 误识别为 `answer:2`。没有长度奖励、格式加分、KL reward 或额外 shaping；DAPO 分组过滤使用同一个 canonical binary reward。

| 配置 | GRPO | DAPO（纯二值奖励版） |
| --- | --- | --- |
| 优势 | 标准组内相对优势，按组内标准差归一化 | 同样的组内相对优势 |
| PPO clip 下界 / 上界 | 0.2 / 0.2 | 0.2 / 0.28（Clip-Higher） |
| 动态过滤 | 关闭 | 过滤组内奖励无差异的组，并补采 |
| loss 聚合 | `seq-mean-token-mean` | `token-mean` |

DAPO 配置保留 Clip-Higher、动态过滤和 token 级 loss 聚合。为满足严格 0/1 奖励要求，未启用论文中的 soft overlong penalty；当前也没有额外的截断样本 loss mask。未完成的输出直接按二值规则得 0。

默认参数如下，可通过 `train_rl.py --help` 查看已提供的覆盖项：

| 参数 | 默认值与含义 |
| --- | --- |
| GPU / rollout TP | 单机 8 卡，TP=1，对应 8 个 rollout replicas |
| G / M / n | 每次更新 32 个 prompt groups；mini-batch 8 组；每题采样 8 条，共 256 条目标训练轨迹 |
| 每卡 micro-batch | 1 |
| 学习率 / 训练上限 | `1e-6` / 10 个 nominal epochs |
| prompt / response / context | 4096 / 2048 / 6144 tokens |
| 训练采样 | thinking 关闭，temperature=0.6，top_p=0.95，top_k=20 |
| 验证采样 | 每题 1 次，greedy，temperature=0，thinking 关闭，最多 2048 输出 tokens |
| 验证与保存间隔 | 默认 `floor(train_count / G)` 个优化步骤，即一个 nominal epoch；最后一步也验证和保存 |

这里的 epoch 按目标训练组数计步。DAPO 会过滤和补采，因此一个 nominal epoch 不等于原始题池恰好遍历一次。YAML 中 `data.gen_batch_size=8` 是 GRPO 的默认 fetch 大小；DAPO 初始化时被上游强制为 1，以便精确计算每次补采的 prompt 数，它不改变 G=32 的更新目标。

DAPO 每步最多提交 `max_generation_batches × G` 个 prompts，默认 `16 × 32 = 512`，包含初始批次与所有补采；每 prompt 采样 n=8 时最多请求 4096 条轨迹。默认还有 1800 秒的采样轮询 deadline。检查发生在提交和轮询边界，无法抢断已经阻塞的底层 RPC。预算耗尽或超时会明确报错，不伪造合格组、不悄悄退回未过滤 GRPO。这些补采限制只作用于 DAPO 训练，验证不做分组过滤。

## 两个硬件配置

| `--hardware` | 目标硬件 | vLLM GPU memory utilization | max_num_seqs | actor 参数 offload | optimizer offload |
| --- | --- | --- | --- | --- | --- |
| `4090-48g` | 单机 8 × RTX 4090 48GB | 0.45 | 32 | 开 | 关 |
| `3090-24g` | 单机 8 × RTX 3090 24GB | 0.30 | 16 | 开 | 开 |

实际训练前会检查可见 GPU 数量、型号与显存：4090 配置要求每卡至少 44 GiB，3090 配置要求至少 22 GiB。两个 profile 是待真实 8 卡机器验证的起始配置，不保证任意题长与机器软件组合都能容纳。默认 TP=1；调整 `--rollout-tp` 时，GPU 数及模型 attention heads 都必须能被 TP 整除。

## 先做 CPU dry-run

在开发机上设置路径。`POOL` 必须换成独立新数据池的绝对路径；不要指向正式 100 题。以下四条命令只进行数据、tokenizer、配置检查，不创建输出目录、不启动 Ray、不加载模型权重，也不运行 GPU 训练。

```bash
ROOT=/inspire/hdd/project/deepanalysis/guitao-25013/zfy
POOL=/absolute/path/new_igsm_pool.jsonl

bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 4090-48g \
  --train-data "$POOL" --output-dir "$ROOT/exp/rl/runs/grpo-4090-48g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm dapo --hardware 4090-48g \
  --train-data "$POOL" --output-dir "$ROOT/exp/rl/runs/dapo-4090-48g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 3090-24g \
  --train-data "$POOL" --output-dir "$ROOT/exp/rl/runs/grpo-3090-24g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm dapo --hardware 3090-24g \
  --train-data "$POOL" --output-dir "$ROOT/exp/rl/runs/dapo-3090-24g" --dry-run
```

检查通过后，在对应的真实 8 卡机器上选择一条命令，去掉 `--dry-run` 才会开始训练。首次训练要求输出目录不存在。dry-run 不检查 GPU 型号/显存，也不会验证已有 run 的完整恢复兼容性；这些检查在实际启动时执行。现有单卡机器可做 CPU 检查，不能据此声称已完成 8 卡验证。

`--gpus 1` 只用于另建输出目录的单卡调试，需要相应调整 `--train-batch-size`、`--mini-batch-size`、`--samples-per-prompt` 和运行预算；不是把八卡 profile 无条件等效为单卡配置。mini-batch 组数乘以 n 必须能被 actor GPU 数整除，G 必须能被 mini-batch 组数整除，n 至少为 2。当前入口只支持单节点。

## 验证、早停与恢复

RL 的质量指标是验证集平均二值奖励，等价于 greedy 单次作答准确率。优先读取 `val-core/igsm_rl/acc/mean@1`，缺少 acc 时读取同源 reward mean@1。这里不沿用 SFT 的验证 loss 早停：策略优化 loss 会随优势、采样和 clip 状态变化，不能直接当作作答质量分数。

默认比较相邻两轮验证分数：本轮严格低于上一轮，下降计数加一；持平或上升清零；连续两次严格下降才停止。例如 `0.30 → 0.28 → 0.27` 停止，而 `0.30 → 0.28 → 0.28 → 0.27` 最后只有一次连续下降，不触发停止。最佳 checkpoint 单独按历史最高验证奖励选择，持平保留原来的最佳版本。

每个完整 `checkpoints/global_step_N` 保存 actor 的 FSDP 模型分片、优化器及 extra 状态（调度器/RNG 等）、`data.pt` 数据迭代器状态，以及 `igsm_rl_state.json` 中的上一轮奖励、连续下降数、最佳分数和停止标志。当前是 sync 模式，不持久化异步 TransferQueue 快照；训练批次由上游循环清理，恢复读取数据迭代器状态。

运行目录中的 `state.json`、`latest_checkpoint.json`、`best_checkpoint.json` 是当前状态和 checkpoint 指针。仅保留完整的 best 和 latest；两者相同时只有一个目录。清理只处理受管理且有完整标记的 checkpoint，不删除半成品。若恢复训练再次需要保存到一个缺少完成标记的同名目录，会先将其安全改名为同 checkpoints 根目录下的 `.incomplete-global_step_N-随机后缀`，保留原文件，再保存新版本。完整 checkpoint 绝不覆盖。

上游在验证前保存 checkpoint。恢复时会先验证已加载的那一步，补上“保存后、验证前中断”的缺口；如果该步验证已经入档则不重复累计早停状态。触发停止后，当前步骤先完成清理和日志，再在下一个更新边界正常退出。恢复已满足停止条件的 checkpoint 不再训练。

恢复示例（按原运行的 algorithm、hardware、数据路径和所有自定义参数填写）：

```bash
RUN="$ROOT/exp/rl/runs/grpo-4090-48g"
bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 4090-48g \
  --train-data "$POOL" --output-dir "$RUN" \
  --resume "$RUN/checkpoints/global_step_N"
```

把 N 替换为该 run 指针中的实际步骤。checkpoint 必须直接位于同一个 run 的 `checkpoints` 下，并带完整状态。实际恢复会严格比较原 `run_config.json`：算法、硬件、模型、数据/提示词/示例、配置、源码 SHA256 和记录的环境版本必须一致，同时校验已准备 parquet 的哈希。不能通过更改 epoch、batch、seed 或脚本源码来接着同一条 run；新实验应使用新的输出目录。

## 导出后独立评测

训练 checkpoint 中的 actor 分片不能直接当成 Hugging Face 模型目录传给现有评测脚本。先读取 `best_checkpoint.json`，将其中的路径填为 `CHECKPOINT`，再使用当前 verl 自带的 FSDP merger 导出。以下参数名已按本仓库 `verl/model_merger` 实现核对：

```bash
source "$ROOT/verl/setup/activate.sh"
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT=/absolute/path/run/checkpoints/global_step_N
python -m verl.model_merger merge --backend fsdp \
  --local_dir "$CHECKPOINT/actor" --target_dir /absolute/path/run/merged_best
```

选择新的导出目录。导出的是完整模型及 tokenizer，随后可以把 `/absolute/path/run/merged_best` 传给 `exp/eval_qwen3_igsm.py --model`，并显式使用 `--examples "$ROOT/exp/prompt_examples.json" --no-thinking --max-tokens 2048`，对受保护的正式 100 题单独计算 pass@1/3/8/50。训练中的 greedy validation 与正式采样 pass@k 是不同评测设置，应分开记录。

实验目录会保存 `run_config.json`、`resolved_config.yaml`、源码快照、数据清单与完整性哈希。原始/转换数据、训练 checkpoint、导出权重和运行日志作为本地实验产物管理，不随源码提交 Git。本 README 提供运行方式，不表示训练或模型导出已经执行。

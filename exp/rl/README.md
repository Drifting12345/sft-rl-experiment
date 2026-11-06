# iGSM GRPO / DAPO

本目录使用当前精简版 verl 的 V1 同步训练器，对原始 Qwen3-0.6B 做 LoRA 强化学习，GRPO 和 DAPO 均启用 LoRA。默认基模为 `ROOT/models/Qwen3-0.6B`，其中 ROOT 是仓库根目录；只更新 LoRA 参数，冻结基模权重。LoRA 与 SFT 对齐：rank=16、alpha=32、dropout=0.05，目标模块为 q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj，bias=none。`--model` 接受 ROOT/models 内的原始模型目录。训练使用 FSDP、BF16 和梯度检查点。

`run_rl.sh` 通过 `ROOT/verl/setup/activate.sh` 复用已有的 `verl-train` Conda 环境，不创建新环境。训练时自行启动本机单节点 Ray，使用 V1 sync trainer 和 vLLM rollout；vLLM 的异步请求接口不代表采用异步策略更新。当前没有这两个 8 卡配置的 GPU 训练实测结果，CPU dry-run 也不代表显存与训练吞吐验证通过。

## 数据与提示词

`--train-data` 必须指向用户另外提供的新 iGSM 原始 JSONL 数据池，不能使用现有正式 100 题评测集。每条至少包含 `problem` 和 `answer`；答案为模 23 下的整数 `0..22`。建议保留稳定的 `id` 和生成器 `seed`。RL 不需要 `solution`，即使文件里有解题过程，也不会把它或答案拼进题面。脚本只读取、检查并转换现有数据，不生成训练题。

数据以固定划分随机种子 42 自动拆为约 90% 训练、10% 验证。相同生成器 seed 的题必须在同一区；没有 seed 时按规范化题面分组。至少需要 10 个独立组，且训练分区题数不能小于训练 batch（默认 32）。组大小不均时，实际比例可能偏离 90/10。数据清单记录分区 ID、题面哈希、生成器 seed、文件 SHA256、token 长度和提示词哈希。

默认保护 `ROOT/data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl`：新数据池与该正式 100 题不能有相同题面或生成器 seed。正式集只用于泄漏检查及后续独立评测，不参与训练和训练中的验证。`--heldout-data` 可补充受保护文件，第一份仍必须是正式 100 题文件。

训练、验证与独立评测共用 `exp/prompt.txt`：完整模板含模 23、差值方向、实体包含关系和固定 easy 示例，最后插入当前题面；只发送一条 user 消息，末行要求 `answer: N`。chat template 明确设置 `enable_thinking=False`。训练前使用真实 tokenizer 检查完整输入长度，超限即报错，不截断题面。

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
| 微调方式 | LoRA：只训练 adapter，冻结基模 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05，与 SFT 相同 |
| LoRA 目标模块 | q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj；bias=none |
| GPU / rollout TP | 单机 8 卡，TP=1，对应 8 个 rollout replicas |
| G / M / n | 每次更新 32 个 prompt groups；mini-batch 8 组；每题采样 8 条，共 256 条目标训练轨迹 |
| 每卡 micro-batch | 1 |
| 学习率 / 训练上限 | `1e-6` / 10 个 nominal epochs |
| prompt / response / context | 4096 / 2048 / 6144 tokens |
| 训练采样 | thinking 关闭，temperature=0.6，top_p=0.95，top_k=20 |
| 验证采样 | 每题 1 次，greedy，temperature=0，thinking 关闭，最多 2048 输出 tokens |
| 验证与保存间隔 | 默认 `floor(train_count / G)` 个优化步骤，即一个 nominal epoch；最后一步也验证和保存 |

GRPO 的一组表示同一道题的 8 个采样回答，由 `actor_rollout_ref.rollout.n=8` 控制；CLI 默认 `--samples-per-prompt 8`。32 组 prompt × 8 个回答 = 每次更新 256 条目标轨迹。`mini-batch=8` 是 prompt 组数，不是每题的回答数。

LoRA 参数位于 `config/base.yaml`；FSDP 使用 model.lora_rank/lora_alpha/target_modules 和 model.lora.dropout 构造 PEFT。rank、alpha、targets 的 flat/nested 配置通过插值保持一致；实际运行清单与源码快照记录 LoRA 配置及对应的框架实现 SHA256。

这里的 epoch 按目标训练组数计步。DAPO 会过滤和补采，因此一个 nominal epoch 不等于原始题池恰好遍历一次。YAML 中 `data.gen_batch_size=8` 是 GRPO 的默认 fetch 大小；DAPO 初始化时被上游强制为 1，以便精确计算每次补采的 prompt 数，它不改变 G=32 的更新目标。

DAPO 每步最多提交 `max_generation_batches × G` 个 prompts，默认 `16 × 32 = 512`，包含初始批次与所有补采；每 prompt 采样 n=8 时最多请求 4096 条轨迹。默认还有 1800 秒的采样轮询 deadline。检查发生在提交和轮询边界，无法抢断已经阻塞的底层 RPC。预算耗尽或超时会明确报错，不伪造合格组、不悄悄退回未过滤 GRPO。这些补采限制只作用于 DAPO 训练，验证不做分组过滤。

## 两个硬件配置

| `--hardware` | 目标硬件 | vLLM GPU memory utilization | max_num_seqs | actor 参数 offload | optimizer offload |
| --- | --- | --- | --- | --- | --- |
| `4090-48g` | 单机 8 × RTX 4090 48GB | 0.45 | 32 | 开 | 关 |
| `3090-24g` | 单机 8 × RTX 3090 24GB | 0.30 | 16 | 开 | 开 |

实际训练前会检查可见 GPU 数量、型号与显存：4090 配置要求每卡至少 44 GiB，3090 配置要求至少 22 GiB。两个 profile 是待真实 8 卡机器验证的起始配置，不保证任意题长与机器软件组合都能容纳。默认 TP=1；调整 `--rollout-tp` 时，GPU 数及模型 attention heads 都必须能被 TP 整除。

可选传入 `--rollout-execution cudagraph`，使用 vLLM 的 FULL_DECODE_ONLY CUDA Graph 加速解码；默认仍为 eager。该选项不改变 LoRA、每题回答数、采样参数、奖励或 DAPO 筛选规则。CUDA Graph 需要额外的初始化和少量显存，应先在目标机器验证兼容性。切换已有训练时必须保留完整 checkpoint 的模型、优化器、RNG 和数据迭代状态，并记录执行配置及源码的变更；不能绕过入口的原始 manifest 一致性检查后直接续跑。

## 先做 CPU dry-run

在开发机上设置路径。`POOL` 必须换成独立新数据池的绝对路径；不要指向正式 100 题。以下四条命令只进行数据、tokenizer、配置检查，不创建输出目录、不启动 Ray、不加载模型权重，也不运行 GPU 训练。

```bash
ROOT=/root/sft-rl-experiment
POOL=/absolute/path/new_igsm_pool.jsonl

bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 4090-48g \
  --train-data "$POOL" --heldout-data "$ROOT/data/test/test_100.jsonl" --output-dir "$ROOT/exp/rl/runs/grpo-4090-48g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm dapo --hardware 4090-48g \
  --train-data "$POOL" --heldout-data "$ROOT/data/test/test_100.jsonl" --output-dir "$ROOT/exp/rl/runs/dapo-4090-48g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 3090-24g \
  --train-data "$POOL" --heldout-data "$ROOT/data/test/test_100.jsonl" --output-dir "$ROOT/exp/rl/runs/grpo-3090-24g" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm dapo --hardware 3090-24g \
  --train-data "$POOL" --heldout-data "$ROOT/data/test/test_100.jsonl" --output-dir "$ROOT/exp/rl/runs/dapo-3090-24g" --dry-run
```

检查通过后，在对应的真实 8 卡机器上选择一条命令，去掉 `--dry-run` 才会开始训练。首次训练要求输出目录不存在。dry-run 不检查 GPU 型号/显存，也不会验证已有 run 的完整恢复兼容性；这些检查在实际启动时执行。现有单卡机器可做 CPU 检查，不能据此声称已完成 8 卡验证。

`--gpus 1` 只用于另建输出目录的单卡调试，需要相应调整 `--train-batch-size`、`--mini-batch-size`、`--samples-per-prompt` 和运行预算；不是把八卡 profile 无条件等效为单卡配置。mini-batch 组数乘以 n 必须能被 actor GPU 数整除，G 必须能被 mini-batch 组数整除，n 至少为 2。当前入口只支持单节点。

## 验证、早停与恢复

RL 的质量指标是验证集平均二值奖励，等价于 greedy 单次作答准确率。优先读取 `val-core/igsm_rl/acc/mean@1`，缺少 acc 时读取同源 reward mean@1。这里不沿用 SFT 的验证 loss 早停：策略优化 loss 会随优势、采样和 clip 状态变化，不能直接当作作答质量分数。

默认比较相邻两轮验证分数：本轮严格低于上一轮，下降计数加一；持平或上升清零；连续两次严格下降才停止。例如 `0.30 → 0.28 → 0.27` 停止，而 `0.30 → 0.28 → 0.28 → 0.27` 最后只有一次连续下降，不触发停止。最佳 checkpoint 单独按历史最高验证奖励选择，持平保留原来的最佳版本。

可用 `--save-every N` 独立设置 checkpoint 保存间隔；未传时沿用验证间隔。验证频率仍由 `--validate-every` 控制。对于可能被关机打断的运行，可设 `--save-every 1` 来减少未保存更新的损失；它不会改变奖励、每题回答数或策略更新参数。

每个完整 `checkpoints/global_step_N` 保存 actor 的 FSDP 模型分片、优化器及 extra 状态（调度器/RNG 等）、`data.pt` 数据迭代器状态，以及 `igsm_rl_state.json` 中的上一轮奖励、连续下降数、最佳分数和停止标志。当前是 sync 模式，不持久化异步 TransferQueue 快照；训练批次由上游循环清理，恢复读取数据迭代器状态。

当前 checkpoint 保留完整 FSDP 模型分片，因此可能包含冻结的基模权重；这些权重不会参与梯度更新。LoRA rank、alpha、dropout 随 `lora_train_meta.json` 保存，导出 adapter 时会恢复这些元数据。

运行目录中的 `state.json`、`latest_checkpoint.json`、`best_checkpoint.json` 是当前状态和 checkpoint 指针。仅保留完整的 best 和 latest；两者相同时只有一个目录。清理只处理受管理且有完整标记的 checkpoint，不删除半成品。若恢复训练再次需要保存到一个缺少完成标记的同名目录，会先将其安全改名为同 checkpoints 根目录下的 `.incomplete-global_step_N-随机后缀`，保留原文件，再保存新版本。完整 checkpoint 绝不覆盖。

上游在验证前保存 checkpoint。恢复时会先验证已加载的那一步，补上“保存后、验证前中断”的缺口；如果该步验证已经入档则不重复累计早停状态。触发停止后，当前步骤先完成清理和日志，再在下一个更新边界正常退出。恢复已满足停止条件的 checkpoint 不再训练。

恢复示例（按原运行的 algorithm、hardware、数据路径和所有自定义参数填写）：

```bash
RUN="$ROOT/exp/rl/runs/grpo-4090-48g"
bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 4090-48g \
  --train-data "$POOL" --heldout-data "$ROOT/data/test/test_100.jsonl" --output-dir "$RUN" \
  --resume "$RUN/checkpoints/global_step_N"
```

把 N 替换为该 run 指针中的实际步骤。checkpoint 必须直接位于同一个 run 的 `checkpoints` 下，并带完整状态。实际恢复会严格比较原 `run_config.json`：算法、硬件、模型、数据/提示词/示例、配置、源码 SHA256 和记录的环境版本必须一致，同时校验已准备 parquet 的哈希。不能通过更改 epoch、batch、seed 或脚本源码来接着同一条 run；新实验应使用新的输出目录。

## 导出后独立评测

先读取 best_checkpoint.json，选中最佳 checkpoint。FSDP merger 对 LoRA checkpoint 的输出是基模权重及 lora_adapter/，需要再应用并合并 adapter 后交给现有 vLLM 评测脚本。

```bash
source "$ROOT/verl/setup/activate.sh"
export PYTHONPATH="$ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT=/absolute/path/run/checkpoints/global_step_N
EXPORTED=/absolute/path/run/exported_base_and_adapter
MERGED=/absolute/path/run/merged_best
python -m verl.model_merger merge --backend fsdp \
  --local_dir "$CHECKPOINT/actor" --target_dir "$EXPORTED"

CUDA_VISIBLE_DEVICES="" python - "$EXPORTED" "$MERGED" <<'PY'
import json
from pathlib import Path
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

source, target = map(Path, sys.argv[1:])
adapter = source / "lora_adapter"
settings = json.loads((adapter / "adapter_config.json").read_text())
if settings["r"] <= 0 or settings["lora_alpha"] <= 0:
    raise ValueError("Invalid LoRA rank/alpha; inspect lora_train_meta.json")
target.mkdir(parents=True, exist_ok=False)
base = AutoModelForCausalLM.from_pretrained(
    source, local_files_only=True, trust_remote_code=False,
    dtype=torch.bfloat16, device_map={"": "cpu"})
model = PeftModel.from_pretrained(
    base, adapter, torch_device="cpu", device_map={"": "cpu"})
model = model.merge_and_unload(safe_merge=True)
model.config.use_cache = True
model.save_pretrained(target, safe_serialization=True)
AutoTokenizer.from_pretrained(
    source, local_files_only=True, trust_remote_code=False).save_pretrained(target)
print(target)
PY

python "$ROOT/exp/eval_qwen3_igsm.py" \
  --data "$ROOT/data/test/test_100.jsonl" --model "$MERGED" \
  --no-thinking --max-tokens 2048 --output-dir /absolute/path/run/test_eval
```

使用新的导出和评测目录。合并后的 MERGED 才是可直接评测的完整模型；单独的 lora_adapter/ 需要配合基模加载。训练期间 greedy validation 与最终采样 pass@k 分开记录。

实验目录会保存 `run_config.json`、`resolved_config.yaml`、源码快照、数据清单与完整性哈希。原始/转换数据、训练 checkpoint、导出权重和运行日志作为本地实验产物管理，不随源码提交 Git。本 README 提供运行方式，不表示训练或模型导出已经执行。


FP32 LoRA 修复：可训练的 LoRA 主参数与 Adam 的一阶、二阶状态保留 FP32；冻结基模仍为 BF16，FSDP MixedPrecision 继续使用 BF16 计算、FP32 梯度归约。LoRA-aware auto-wrap 将可训练的线性叶子与冻结基模分开，不能通过把 adapter 强制转成 BF16 来规避 flat-group 类型问题。FSDP merger 导出时保留 adapter 原有 dtype。原 BF16 checkpoint 保留历史值；将其加载为 FP32 不能恢复此前已舍入丢失的更新。更换精度后的新实验需要独立目录与配置记录，不与旧四档结果混为同一实验。

最佳模型选择：以历史最高内部验证准确率选 best；分数同分时保留最新 checkpoint。早停仍比较相邻验证：连续两次严格下降才停止，持平重置下降计数。train_100 的 90 条训练、batch32 对应每个名义 epoch 两步；100 步实验需同时设置 --epochs 50 --max-steps 100，避免先碰到 epoch 上限。

真实数据遍历模式：--epoch-unit data-pass --train-batch-size 1 --mini-batch-size 1 --samples-per-prompt 8 --epochs 10。每个候选题目组只生成一次；均匀0/1奖励组跳过Adam更新但计入已遍历题数，有差别的组立即完成一次更新并同步LoRA后再处理下一题。每90道训练题为一轮，轮末验证。checkpoint/global_step_N 中N在此模式表示候选题目组处理次数，实际优化次数单独保存在state.json的data_pass.optimizer_updates，恢复所需计数保存在data-pass-state/。

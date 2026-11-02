# sft-rl-experiment

开发机路径：`/inspire/hdd/project/deepanalysis/guitao-25013/zfy`。

版本维护仓库：[Drifting12345/sft-rl-experiment](https://github.com/Drifting12345/sft-rl-experiment)。

本项目包含 iGSM 数据生成、Qwen3 推理评测、LoRA SFT，以及从原始基模开始的 GRPO / DAPO 强化学习。固定 100 题的原始 Qwen3-0.6B baseline 已完成；新的 SFT/RL 实验入口已通过 CPU 数据、配置与状态逻辑检查，尚未进行正式 GPU 训练，两个 RL 八卡 profile 也尚未实测。

| 目录 | 用途 |
| --- | --- |
| `iGSM/` | iGSM 题目生成源码 |
| `verl/` | 精简版训练框架；当前 RL 实验提供 GRPO / DAPO 入口 |
| `iGSM/setup/` | 数据生成环境配方、激活入口、生成与独立审查脚本 |
| `verl/setup/` | 训练/推理环境激活入口、兼容版本配方与实际包快照 |
| `.conda/envs/igsm-gpu/` | 实际的数据生成 Conda 环境 |
| `.conda/envs/verl-train/` | 实际的训练与推理 Conda 环境 |
| `data/` | 正式 JSONL 数据集 |
| `models/` | 四个 Qwen3 模型及固定版本恢复工具 |
| `exp/` | 固定提示词、两对 toy 示例和独立 pass@k 推理评测 |
| `exp/sft/` | LoRA 监督微调、验证 loss 早停、adapter 与合并模型导出 |
| `exp/rl/` | GRPO / DAPO、二值答案奖励、验证奖励早停和 checkpoint 恢复 |
| `.cache/` | 下载及运行缓存 |

两个 `setup` 目录保存工具和环境记录；Python、PyTorch 等已安装软件位于 `.conda/envs/`。它们分别服务于数据生成和训练/推理，不应混装依赖。

数据生成入口：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM/setup/activate.sh
```

训练与推理入口：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
```

激活脚本会分别进入 `iGSM/` 和 `verl/`；执行实验前再进入对应实验目录。`run_sft.sh` 和 `run_rl.sh` 已负责激活训练环境。

| 任务 | 说明与入口 |
| --- | --- |
| 准备新题池 | [数据生成](iGSM/setup/README.md) |
| 恢复环境 | [训练/推理环境](verl/setup/README.md) |
| 恢复原始权重 | [模型恢复](models/README.md) |
| 独立 pass@1/3/8/50 评测 | [推理评测](exp/README.md) |
| 参考解答监督微调 | [SFT](exp/sft/README.md) |
| 原始基模强化学习 | [RL](exp/rl/README.md) |

训练与推理共用 `.conda/envs/verl-train`：Python 3.12.16、PyTorch 2.13.0+cu129、vLLM 0.32.0+cu129、transformers 5.16.1、TorchCodec 0.16.0+cu129、FFmpeg 8.1.2；SFT 使用 HF Trainer 与 PEFT 0.21.2。iGSM 继续使用独立 `.conda/envs/igsm-gpu`（Python 3.10.19、PyTorch 2.4.0+cu121），不与训练环境混装。环境配方和包快照见对应 setup 目录；迁移仍需核对平台、驱动和路径。

## 开始新实验

先恢复环境和原始模型，再另行准备新的 iGSM JSONL 题池。SFT 需要 `problem`、经校验的 `solution` 和 `answer`；RL 只需要题面与标准答案，不要求参考解答。两条入口都要求新题池独立于固定 100 题，检查题面和生成器 seed 重叠，并按 seed 分组自动划分约 90/10 的训练/验证集（划分随机种子 42，至少 10 个独立组）。固定 100 题不训练，也不用于每轮验证或选择最佳模型。

先进行 CPU dry-run；把 `POOL` 换成独立新题池的绝对路径：

```bash
ROOT=/inspire/hdd/project/deepanalysis/guitao-25013/zfy
POOL=/absolute/path/new_igsm_pool.jsonl

bash "$ROOT/exp/sft/run_sft.sh" --train-data "$POOL" \
  --output-dir "$ROOT/exp/sft/runs/qwen3-0.6b-lora" --dry-run

bash "$ROOT/exp/rl/run_rl.sh" --algorithm grpo --hardware 4090-48g \
  --train-data "$POOL" --output-dir "$ROOT/exp/rl/runs/grpo-4090-48g" --dry-run
```

dry-run 检查数据、泄漏、模板 token 长度和配置，不加载模型权重或启动 GPU 训练；RL 也不启动 Ray。通过后，在符合配置的机器上去掉 `--dry-run` 才开始训练。完整命令、另一种算法与硬件 profile、恢复方式见各实验 README。脚本不会自动生成新题池，也不会把当前 100 题当训练数据。

## SFT 与 RL 当前实现

两类训练都复用模 23 的主提示词、固定的两对 toy 示例以及关闭 thinking 的输入模板，独立推理评测最多生成 2048 tokens；不提供正式评测答案或历史纠错反馈。

| 项目 | LoRA SFT | GRPO / DAPO RL |
| --- | --- | --- |
| 起点 | 原始 `models/Qwen3-0.6B` | 原始 `models/Qwen3-0.6B`，不接 SFT 输出或 adapter |
| 更新方式 | LoRA，r=16 / alpha=32 / dropout=0.05 | 全参数 FSDP、BF16；SFT 的 LoRA 配置不影响 RL |
| 默认训练配置 | BF16、SDPA、梯度检查点、lr=`2e-4`，最多 10 epochs，microbatch=1、accumulation=8 | 单机 8 卡，lr=`1e-6`，最多 10 nominal epochs；G=32 prompts、mini-batch=8 组、每题 n=8、每卡 microbatch=1、rollout TP=1 |
| 长度 | 总序列 8192、assistant 目标最多 2048 tokens；超限报错，不截断 | prompt 4096 + response 2048 tokens；不截断题面 |
| 训练信号 | assistant-only loss：仅当前解答、最终 `answer:数字` 和 EOS | 完整 EOS 结束且最终答案数字正确得 1，其余为 0 |
| 验证与最佳模型 | 每 epoch 的 teacher-forced assistant-only `eval_loss`；历史最低为最佳 | 每 nominal epoch 的 greedy 单次作答平均二值奖励；历史最高为最佳 |
| 早停 | 与上一轮 loss 比较，连续两次严格升高才停；持平或降低清零 | 与上一轮平均奖励比较，连续两次严格下降才停；持平或上升清零 |
| 产物 | 完整续训 checkpoint、`best_adapter/`，以及合并最佳 adapter 后的完整模型 `final/` | 完整 actor 分片与续训状态，保留 best/latest checkpoint，可另行合并导出完整模型 |

SFT 的 LoRA 覆盖 attention 的 q/k/v/o 和 MLP 的 gate/up/down projection。`best_adapter/` 需配合原始基模使用；`final/` 是可直接交给现有 vLLM 评测的完整模型。SFT 验证不逐题生成答案，训练结束后的数字正确率另做独立评测。

GRPO 使用标准组内相对优势、0.2/0.2 clip 和序列均值聚合；DAPO 的纯二值奖励版本使用 0.2/0.28 Clip-Higher、动态分组过滤与补采、token 均值聚合。为保持用户要求的 0/1 奖励，未加 soft overlong shaping。DAPO 补采默认每步至多 `16 × G` 个 prompts、1800 秒轮询 deadline，耗尽报错，不伪造合格组；底层阻塞 RPC 无法被该轮询限制抢断。

RL 提供单机 `8 × 4090 48GB` 和 `8 × 3090 24GB` 两个 profile；nominal epoch 默认按 `floor(train_count / G)` 个更新步骤计，DAPO 过滤补采后不代表原题池恰好遍历一次。RL 用作答奖励控制训练，不能把随优势与 clip 改变的 policy loss 直接当成模型质量，也不沿用 SFT 的验证 loss 早停。

两类 checkpoint 都保存早停计数和完整恢复上下文。RL 还处理“checkpoint 保存后、验证前中断”的补评，已验证的同一步不重复计数。当前 SFT/RL 入口仅完成 CPU 检查，尚无正式训练、收敛或八卡性能结论；此前旧的单 GPU GRPO/PPO 各一步验收仅证明基础执行链路可用。

## 版本维护与源码来源

Git 操作统一在 `zfy` 根目录进行。`iGSM/` 和 `verl/` 作为普通源码目录纳入同一个仓库，后续修改从根目录提交：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
git status --short
git add -- .
git commit -m "Describe the experiment change"
git push
```

根 `.gitignore` 排除实际 Conda 环境、缓存、JSONL 数据、模型权重和实验运行输出；源码、setup 配方、提示词及模型恢复清单纳入版本维护。本批 100 道题的数据文件仅保留在开发机，不提交到仓库。

工作区只使用根目录 `zfy/.git`，`iGSM/` 和 `verl/` 不保留独立 Git 元数据。需要保留的原 `.gitignore`、`.gitattributes` 规则按各自目录的原作用域合并到根目录对应文件，由当前项目统一维护。

源码来源为 [facebookresearch/iGSM](https://github.com/facebookresearch/iGSM) 和 [verl-project/verl](https://github.com/verl-project/verl)。保留各源码目录中的 `LICENSE`、版权声明和来源信息；本项目的精简与实验改动由根仓库记录。

| 源码 | 导入前的上游基线提交 |
| --- | --- |
| iGSM | `a1ed1d04600add811beb08b58d9912ed30999642` |
| verl | `8e48d72de4766228c36623c90f9f5c49ee5b45f3` |

## 本批 100 道题的生成参数

数据文件：`data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl`，生成于 2026-10-30。每行保留英文题目 `problem`、参考解答 `solution`、最终整数答案 `answer`，以及该题的 `seed`、`generator_params` 和长度统计。

| 参数 | 本次取值与含义 |
| --- | --- |
| `count` | 100 道唯一题目 |
| `max_op` | 52，运算次数上限 |
| `max_edge` | 80，图中边数上限，不表示每题恰有 80 条边 |
| `op` | 52，固定每题实际 `n_op=52` |
| `perm_level` | 5，打乱题面描述顺序 |
| `detail_level` | 0，使用详细参考解答格式 |
| `be_shortest` | `True` |
| `mod` | 23，加、减、乘均按模 23 计算，答案在 0–22 |
| `p_format` | `pq` |
| `hash_bins` | `list(range(23))`，使用全部 0–22，不单独划分 train/test |
| `min_solution_tokens` | 1024，仅计算 `solution` 正文 |
| tokenizer | iGSM 自带 GPT-2 tokenizer；不添加特殊 token、不截断、不填充 |
| `seed` | 候选种子从 42 开始递增；每题实际种子记录在数据行中 |
| `workers` | 4 个进程，按候选种子顺序收集结果 |
| `PYTHONHASHSEED` | 0，启动 Python 前设置的环境变量 |

这是经调整的 **hard-long** 配置。原版 README 的 hard 示例为 `max_op=21`、`max_edge=28`，未固定 `op`；本批数据行的 `difficulty` 虽为 `hard`，具体难度以这里及每行的生成参数为准。

本次共处理 1583 个候选（种子 42–1624），丢弃 1483 个解答过短的候选，保留 100 道。参考解答长度为 **1024–1165 GPT-2 tokens**，平均 **1048.37**，中位数 **1042**；题目、单独答案字段及训练标记不计入该长度。这个统计不是 Qwen tokenizer 的长度，也不是评测输出上限。

生成时使用 iGSM 的 `true_correct` 校验解答、按题目哈希去重；交付前还按每题种子独立重生成复核。旧审核报告已按要求删除，正式 JSONL 保留不变。

- iGSM 源提交：`a1ed1d04600add811beb08b58d9912ed30999642`。
- 数据 SHA256：`ce09170ae40ffee9e02f5a6ae099a09322288eb72c5ae4458c4b66115a6329cd`。

同配置生成命令见 [数据生成说明](iGSM/setup/README.md)。重新生成应使用新输出名，并保持源码、依赖、分词器和哈希种子一致；运行超时等条件也可能影响候选选择。

## 已完成的原始基模 baseline

原始 Qwen3-0.6B 已完成正式 100 题评测，共 **4276 次尝试**。使用上下文 16384 tokens、每次最多生成 **2048 tokens**、关闭 thinking（`--no-thinking`），temperature=0.6、top_p=0.95、top_k=20。每题最多 50 次，首次答对即停止；pass@k 表示该题前 k 次内至少正确一次。运行日志已确认实际使用 vLLM 0.32、BF16、`FLASH_ATTN2`，top-k/top-p 采样使用 FlashInfer；这与 SFT 的 SDPA 训练路径分别记录。

| 指标 | 通过题数 / 100 | 通过率 |
| --- | --- | --- |
| pass@1 | 2 | 2% |
| pass@3 | 5 | 5% |
| pass@8 | 9 | 9% |
| pass@50 | 23 | 23% |

无效作答共 **1761 次**，其中长度截断 **753 次**；其余无效情况包括输出格式与未完整结束等。判分要求完整生成且最终答案数字正确，截断输出不因已有数字看似正确而通过。这些是本轮固定输入和采样配置的完成结果，不能据此推断 SFT/RL 已有提升。

`exp/prompt.txt` 使用简短主指令：全部计算按模 23 进行、使用简洁等式推导，末行用小写 `answer:` 紧接实际整数，随后附上当前题面。可选的 `--examples PATH` 在当前题目前加入固定的 user/assistant 示范；不传时仍只发送当前题目的单条 user 消息。

`exp/prompt_examples.json` 包含两对人工 toy 计算：`4 * 6 mod 23 = 1` 和 `18 + 10 mod 23 = 5`。带示例运行时显式传入 `--examples prompt_examples.json`（从 `exp` 目录解析），所有题和尝试复用同两对示例，不提供正式 100 题的参考答案或历史纠错反馈。选用示例的每轮保存示例快照及 SHA256，完整输入消息和 token IDs 记录在 `prompts.jsonl`；few-shot 轮与旧单 user 提示的结果分开统计。

这次正式 baseline 使用上述两对 toy 示例。few-shot 结果与旧单 user 提示的结果分开统计。逐次回答、日志和结果文件按根 `.gitignore` 留在开发机，仓库 README 仅记录这份静态完成结果。

## 查看后续运行

后续启动的信息从运行索引读取：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
cat exp/logs/current.json
```

索引中的 `log`、`status_file`、`output_dir` 分别指向本轮日志、退出状态文件和结果目录；实际参数以该目录的 `run_config.json` 为准，进展和成绩读取 `summary.json`。具体 PID、运行时间和即时状态只记录在运行文件中，不在 README 中逐次更新。

退出码为 0 且 `summary.complete=true` 表示本轮执行完成；还需结合格式错误数、截断数和原始回答解释成绩。退出状态尚未写入时，结合记录的主机、准确 PID 和日志确认是否仍在运行。`pass_at_k=null` 或未完成轮次的通过率下界不能当最终成绩。不同提示词、输出长度或 thinking 配置使用独立结果目录，不合并成绩。启动、查看和恢复方法见 [实验说明](exp/README.md)。

本工作区保留后续实验所需源码、环境配方、模型恢复工具、数据和实验结果；历史安装排错脚本、旧验证日志与 smoke checkpoint 已清理。数据、模型权重、实际环境、缓存和实验 runs 不提交 GitHub；版本维护包含代码、配置、提示词、环境配方和必要的静态实验说明。

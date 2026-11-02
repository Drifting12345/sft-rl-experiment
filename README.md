# sft-rl-experiment

开发机路径：`/inspire/hdd/project/deepanalysis/guitao-25013/zfy`。

版本维护仓库：[Drifting12345/sft-rl-experiment](https://github.com/Drifting12345/sft-rl-experiment)。

| 目录 | 用途 |
| --- | --- |
| `iGSM/` | iGSM 题目生成源码 |
| `verl/` | 面向 iGSM 文本任务的 GRPO/PPO 训练源码 |
| `iGSM/setup/` | 数据生成环境配方、激活入口、生成与独立审查脚本 |
| `verl/setup/` | 训练/推理环境激活入口、兼容版本配方与实际包快照 |
| `.conda/envs/igsm-gpu/` | 实际的数据生成 Conda 环境 |
| `.conda/envs/verl-train/` | 实际的训练与推理 Conda 环境 |
| `data/` | 正式 JSONL 数据集 |
| `models/` | 四个 Qwen3 模型及固定版本恢复工具 |
| `exp/` | 推理评测代码、提示词、运行日志与结果；`sft/` 提供监督微调入口 |
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

激活脚本会分别进入 `iGSM/` 和 `verl/`；执行实验前再进入 `exp/`。详细用法见 [数据生成](iGSM/setup/README.md)、[训练环境](verl/setup/README.md)、[推理评测](exp/README.md) 和 [模型恢复](models/README.md)。

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

## 评测配置与运行状态

默认使用原始 Qwen3-0.6B、每次最多生成 **2048 tokens**，并关闭模型专用 thinking 模式（`--no-thinking`）；每题最多 50 次、首次答对即停止，统计 pass@1/3/8/50。

`exp/prompt.txt` 使用简短主指令：全部计算按模 23 进行、使用简洁等式推导，末行用小写 `answer:` 紧接实际整数，随后附上当前题面。可选的 `--examples PATH` 在当前题目前加入固定的 user/assistant 示范；不传时仍只发送当前题目的单条 user 消息。

`exp/prompt_examples.json` 包含两对人工 toy 计算：`4 * 6 mod 23 = 1` 和 `18 + 10 mod 23 = 5`。带示例运行时显式传入 `--examples prompt_examples.json`（从 `exp` 目录解析），所有题和尝试复用同两对示例，不提供正式 100 题的参考答案或历史纠错反馈。选用示例的每轮保存示例快照及 SHA256，完整输入消息和 token IDs 记录在 `prompts.jsonl`；few-shot 轮与旧单 user 提示的结果分开统计。

最新一次启动的信息从运行索引读取：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
cat exp/logs/current.json
```

索引中的 `log`、`status_file`、`output_dir` 分别指向本轮日志、退出状态文件和结果目录；实际参数以该目录的 `run_config.json` 为准，进展和成绩读取 `summary.json`。具体 PID、运行时间和即时状态只记录在运行文件中，不在 README 中逐次更新。

退出码为 0 且 `summary.complete=true` 表示本轮执行完成；还需结合格式错误数、截断数和原始回答解释成绩。退出状态尚未写入时，结合记录的主机、准确 PID 和日志确认是否仍在运行。`pass_at_k=null` 或未完成轮次的通过率下界不能当最终成绩。不同提示词、输出长度或 thinking 配置使用独立结果目录，不合并成绩。启动、查看和恢复方法见 [实验说明](exp/README.md)。

本工作区保留后续实验所需源码、环境配方、模型恢复工具、数据和实验结果；历史安装排错脚本、旧验证日志与 smoke checkpoint 已清理。此前完成过单 GPU 的 GRPO/PPO 各一步运行验收，这不代表训练收敛或多卡验证。环境迁移仍需核对平台、驱动和路径，不能仅凭包清单保证完全复现。

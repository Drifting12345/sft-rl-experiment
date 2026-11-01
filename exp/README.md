# Qwen3-0.6B iGSM 推理评测

本目录使用已有 `verl-train` Conda 环境和原始 `../models/Qwen3-0.6B`，对 `../data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl` 的全部 100 道题进行推理评测。

| 路径 | 用途 |
| --- | --- |
| `eval_qwen3_igsm.py` | 采样、最终答案判分、提前停止和续跑 |
| `prompt.txt` | 已确认的模 23 提示词，末行要求 `answer:N` |
| `run_eval.sh` | 接受 `RUN_NAME` 和后续 CLI 参数，激活环境并记录最终退出码 |
| `logs/` | 日志、wrapper PID 索引和最终退出状态 |
| `runs/` | 每次评测的配置、实际输入、逐次回答和汇总 |

## 口径与参数

每题最多采样 50 次，每轮每道尚未答对的题只生成一次，首次正确后停止该题。记首次正确次数为 r，`pass@k = count(r <= k) / 100`，k 为 1、3、8、50；50 次均错则各项均失败。每次回答只包含相同原始题面和提示词，不带参考解答、答案或历史纠错反馈。

最终回答区必须以唯一的 `answer:N` 行结束，N 为 0–22 的整数；只比较最终数字，推理内容不参与判分，也不会对越界答案再取模。重复答案、未闭合思考、缺少最终答案、长度截断等记错。保留完整生成原文和解析原因。

默认参数为 **thinking 关闭**（`--no-thinking`）、temperature 0.6、top_p 0.95、top_k 20、seed 42；每次最多生成 **2048 tokens**，上下文上限 16384，batch 8、单 GPU、BF16、vLLM eager、GPU memory utilization 0.80。关闭专用 thinking 模式后，提示词仍要求逐步展示模运算。按实际 Qwen tokenizer 检查输入长度，不将数据集的 GPT-2 token 统计当作 Qwen 长度。

## 当前状态与历史轮

当前仅修改默认参数，没有启动新评测。旧轮 `qwen3-0.6b-igsm100-20261101T143815Z` 已停止，以下文件保留为历史记录：

- 历史任务索引：`logs/current.json`，状态标记为 `stopped`。
- 日志：`logs/qwen3-0.6b-igsm100-20261101T143815Z.log`。
- 最终退出码：`logs/qwen3-0.6b-igsm100-20261101T143815Z.status`。
- 结果：`runs/qwen3-0.6b-igsm100-20261101T143815Z/`。

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp
cat logs/current.json
tail -n 30 logs/qwen3-0.6b-igsm100-20261101T143815Z.log
cat runs/qwen3-0.6b-igsm100-20261101T143815Z/summary.json
```

旧轮使用 8192 tokens、thinking 开启，原脚本快照和原结果保留。其未完成汇总中的 `pass_at_k` 为 null，`current_pass_at_k_lower_bound` 只是部分结果。后续 2048/no-thinking 评测从新目录开始，不能合并两种配置的成绩；恢复旧轮需原脚本快照及原参数，不能直接使用已修改的默认脚本续跑。

## 启动与恢复

只检查全部数据和提示词长度，不创建推理 engine：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp
python -B eval_qwen3_igsm.py --max-tokens 2048 --no-thinking --dry-run
```

以下为以后手动启动新实验的示例，本次不执行。使用新的 `RUN_NAME`：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
run_name="qwen3-0.6b-igsm100-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p exp/logs
nohup bash exp/run_eval.sh "$run_name" --max-tokens 2048 --no-thinking > "exp/logs/$run_name.log" 2>&1 < /dev/null &
printf 'wrapper_pid=%s\n' "$!"
```

记录这次 PID、日志、status 和 `exp/runs/$run_name`；`current.json` 是当前任务索引，另启新任务时同步更新它。`run_eval.sh` 管理同名 `.status`，日志重定向由上述启动命令负责。

以后恢复已有的 **2048/no-thinking** 评测时，先确认对应进程已停止，并使用原脚本和全部原参数。下面的 `run_name` 沿用要恢复的那次新评测名称：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
: "${run_name:?请先设置要恢复的2048/no-thinking评测名称}"
nohup bash exp/run_eval.sh "$run_name" --max-tokens 2048 --no-thinking --resume >> "exp/logs/$run_name.log" 2>&1 < /dev/null &
printf 'wrapper_pid=%s\n' "$!"
```

恢复时以 `attempts.jsonl` 重建状态，已落盘的正确题不会再次生成。未返回或未落盘的 batch 可能重新计算，不重复计数。同一结果目录有进程锁；数据、提示词、脚本、包版本或参数变化会拒绝原地续跑。日志损坏或半行会报错并保留记录，不静默跳过。

每次结果保留 `run_config.json`、`prompt.txt`、`prompts.jsonl`、`attempts.jsonl` 和 `summary.json`。其中 `prompt.txt` 与 `prompts.jsonl` 不包含参考答案；逐次结果包含真值、预测、种子、完整回答、结束原因与 token 数，便于复核。

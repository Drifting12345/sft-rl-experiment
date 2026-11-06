# Qwen3-0.6B iGSM 推理评测

本目录使用已有 `verl-train` Conda 环境和原始 `../models/Qwen3-0.6B`，对 `../data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl` 的全部 100 道题进行推理评测。

| 路径 | 用途 |
| --- | --- |
| `eval_qwen3_igsm.py` | 采样、最终答案判分、提前停止和续跑 |
| `prompt.txt` | 完整默认模板：模 23、差值方向、包含关系、一道 easy 示例和待评测题 |
| `run_eval.sh` | 接受 `RUN_NAME` 和后续 CLI 参数，激活环境并记录最终退出码 |
| `logs/` | 日志、wrapper PID 索引和最终退出状态 |
| `runs/` | 每次评测的配置、实际输入、逐次回答和汇总 |
| `sft/` | 基于参考解答的 LoRA 微调、90/10 验证划分和验证 loss 早停 |
| `rl/` | 从原始基模开始的 verl GRPO/DAPO；严格 0/1 奖励、8 卡配置与验证奖励早停 |

监督微调入口见 [SFT 说明](sft/README.md)。它与评测共用包含 easy 示例的 `prompt.txt`，只监督当前目标解答、数字答案与 EOS；不会把正在评测的 100 道题直接用于训练。

## 口径与参数

每题最多采样 50 次，每轮每道尚未答对的题只生成一次，首次正确后停止该题。记首次正确次数为 r，`pass@k = count(r <= k) / 100`，k 为 1、3、8、50；50 次均错则各项均失败。每次尝试重新使用相同题面、主指令及模板内的固定 easy 示例，不带正式题目的参考解答、真实答案或历史纠错反馈。

最终回答区末行由小写前缀 `answer:` 紧接实际的 0–22 整数组成；只比较最终数字，推理内容不参与判分，也不会对越界答案再取模。字母占位符、重复答案、未闭合思考、缺少最终答案、长度截断等记错。保留完整生成原文和解析原因。

`prompt.txt` 是完整的单条 user 消息模板，明确模 23、`Difference of A and B` 的减法方向和实体包含关系，内嵌一道 Dragon/Griffin easy 题及解答，最后通过 `Now solve: {problem}` 插入当前题面。最终行使用 `answer: N`，其中 N 为 0–22 的整数。关闭模型专用 thinking 模式不会禁止输出可见解答。

默认直接渲染 `prompt.txt`。完整指令、一道 easy 示例和解答均在同一模板内，每个待评测题只发送一条 user 消息；不加载或拼接独立聊天示例。示例参考解答已通过生成器校验，不包含正式题目的答案或历史纠错反馈。


默认参数为 **thinking 关闭**（`--no-thinking`）、temperature 0.6、top_p 0.95、top_k 20、seed 42；每次最多生成 **2048 tokens**，上下文上限 16384，batch 8、单 GPU、BF16、vLLM eager、GPU memory utilization 0.80。按实际 Qwen tokenizer 检查包含内嵌示例的完整输入长度，不将数据集的 GPT-2 token 统计当作 Qwen 长度。

## 查看运行状态

`logs/current.json` 索引最近一次启动的任务，保存名称、主机、PID、日志路径 `log`、退出状态文件 `status_file` 和结果目录 `output_dir`。每次启动更新索引，无需为具体运行修改 README。

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp
python -B - <<'PY'
from pathlib import Path
import json
import subprocess

info = json.loads(Path("logs/current.json").read_text())
print(json.dumps(info, ensure_ascii=False, indent=2))
status = Path(info["status_file"])
print("exit_status:", status.read_text().strip() if status.is_file() else "未写入")
output = Path(info["output_dir"])
config_path = output / "run_config.json"
if config_path.is_file():
    config = json.loads(config_path.read_text())
    keys = ["max_tokens", "thinking", "max_attempts", "pass_k", "batch_size",
            "examples", "examples_sha256"]
    print("config:", json.dumps({key: config.get(key) for key in keys}, ensure_ascii=False))
summary_path = output / "summary.json"
if summary_path.is_file():
    summary = json.loads(summary_path.read_text())
    keys = ["complete", "num_questions", "solved", "total_attempts", "pass_at_k",
            "current_pass_at_k_lower_bound"]
    print("summary:", json.dumps({key: summary.get(key) for key in keys}, ensure_ascii=False))
if Path(info["log"]).is_file():
    subprocess.run(["tail", "-n", "30", info["log"]], check=True)
PY
```

- `.status` 中的退出码为 0 且 `summary.complete=true`，表示本轮完整评测成功结束。
- 非零退出码需结合停止原因和日志，区分人为中止与执行失败；不能当作完整成绩。
- 退出状态文件尚不存在时，还需在记录的同一主机上核对准确 PID、命令行和日志，不能仅据文件缺失认定运行中。
- `pass_at_k=null` 表示尚未完成；`current_pass_at_k_lower_bound` 只是当前通过率下界。

正常退出仅表示执行完成；同时查看 `invalid_answer_attempts`、`truncated_attempts` 与原始回答，区分格式失败、截断和最终数字算错。

每轮独立保存实际配置、主指令、选用的示例、结果及必要的源代码快照。不同主指令、示例、token 上限或 thinking 设置不合并成绩；恢复旧轮需该轮的原脚本、原提示词与示例文件及全部原参数，不能直接套用已改变的默认值。

## 启动与恢复

只检查全部数据和提示词长度，不创建推理 engine：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp
python -B eval_qwen3_igsm.py --max-tokens 2048 --no-thinking --dry-run
```

新评测默认包含模板内的示例，使用新的 `RUN_NAME`：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
run_name="qwen3-0.6b-igsm100-fewshot-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p exp/logs
nohup bash exp/run_eval.sh "$run_name" --max-tokens 2048 --no-thinking > "exp/logs/$run_name.log" 2>&1 < /dev/null &
printf 'wrapper_pid=%s\n' "$!"
```

记录这次 PID、日志、status 和 `exp/runs/$run_name`；`current.json` 是当前任务索引，另启新任务时同步更新它。`run_eval.sh` 管理同名 `.status`，日志重定向由上述启动命令负责。

恢复使用当前内嵌示例模板的 **2048/no-thinking** 评测时，先确认对应进程已停止，并使用原脚本、主指令、示例文件和全部原参数。下面的 `run_name` 沿用要恢复的那次评测名称：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy
: "${run_name:?请先设置要恢复的2048/no-thinking few-shot评测名称}"
nohup bash exp/run_eval.sh "$run_name" --max-tokens 2048 --no-thinking --resume >> "exp/logs/$run_name.log" 2>&1 < /dev/null &
printf 'wrapper_pid=%s\n' "$!"
```

恢复时以 `attempts.jsonl` 重建状态，已落盘的正确题不会再次生成。未返回或未落盘的 batch 可能重新计算，不重复计数。同一结果目录有进程锁；数据、主指令、示例、脚本、包版本或参数变化会拒绝原地续跑。恢复历史上使用独立聊天示例的旧运行时，仍须提供原轮的脚本、提示词、示例文件和参数；新模板不能直接用于续跑旧结果。日志损坏或半行会报错并保留记录，不静默跳过。

每次结果保留 `run_config.json`、`prompt.txt`、`prompts.jsonl`、`attempts.jsonl` 和 `summary.json`。配置记录完整模板的 SHA256；实际消息和聊天模板处理后的 token IDs 保存在 `prompts.jsonl`。逐次结果包含正式题目的真值、预测、种子、完整回答、结束原因和 token 数，便于复核。

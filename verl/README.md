# verl：iGSM 文本强化学习精简版

上游：https://github.com/verl-project/verl.git

基线提交：`8e48d72de4766228c36623c90f9f5c49ee5b45f3`

范围：同步 GRPO / PPO，单轮文本，FSDP / FSDP2 训练，vLLM 生成，规则奖励。

## 目录和阅读顺序

1. `examples/igsm/`：数据转换、配置、GRPO/PPO 启动脚本。
2. `verl/trainer/main_ppo.py`：Hydra 入口和 Ray 初始化。
3. `verl/trainer/ppo/v1/trainer_sync.py`、`trainer_base.py`：采样、优势、更新、验证、保存。
4. `verl/trainer/ppo/core_algos.py`：GRPO 组内优势、PPO GAE 和裁剪损失。
5. `verl/workers/engine/fsdp/`：actor、reference、critic 的前向和反向。
6. `verl/workers/rollout/vllm_rollout/`：生成和权重同步。
7. `verl/utils/reward_score/igsm.py`：整数答案奖励。
8. `verl/model_merger/`：将 FSDP checkpoint 导出为 Hugging Face 模型。

保留 `single_controller`、`protocol.py`、`checkpoint_engine`、`experimental/agent_loop` 和 `reward_loop`，因为同步训练仍依赖这些组件。
上游共享文件中的部分数据结构、关闭的可选分支和其他优势函数仍保留；它们与训练主链路共用文件，未重写核心算法。
不支持 SFT、蒸馏训练、多轮工具 agent、视觉任务、异步 trainer、Megatron、SGLang、TensorRT、NPU 或 ROCm。
完整删除清单和验证边界见 `SLIMMING_REPORT.md`、`slimming-manifest.json`。

## 环境状态

按用户要求，训练使用独立 Conda 环境：
`/inspire/hdd/project/deepanalysis/guitao-25013/zfy/.conda/envs/verl-train`。
已有 `igsm-gpu` 数据生成环境（torch 2.4 / CUDA 12.1）独立保留，不往其中安装本仓库。

激活方式：

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
```

当前已安装 Python 3.12.16、torch 2.13.0+cu129、torchvision 0.28.0、torchaudio 2.11.0、
vLLM 0.32.0+cu129、TorchCodec 0.16.0+cu129、FFmpeg 8.1.2、transformers 5.16.1、
TRL 0.27.0、TransferQueue 0.1.11 和 Ray 2.59.0。
兼容组合由 `setup/requirements-cu129.txt` 记录，环境入口和实际包快照说明见 [setup/README.md](setup/README.md)。
上游 `fsdp` / `vllm` extras 仍指向 CUDA 13.0，不要用它们覆盖当前 cu129 配置。

此前已通过 GPU/SDPA 反向、NCCL/FSDP 优化器、critic 反向、vLLM 实际生成，以及单 GPU GRPO/PPO 各一步执行与 checkpoint 保存验收。历史验收使用本地 Qwen3-0.6B、6 条 iGSM 样本（train 4 / val 2）、batch 2、响应上限 16，实际为 FSDP1/BF16/CPU offload、actor/critic SDPA 与 vLLM eager。GRPO 的损失和梯度为 0，只证明执行与保存链路可用，不证明有效权重更新、训练收敛、多卡、FSDP2 或完整任务质量。

旧安装排错脚本、验证状态、日志、汇总和 smoke checkpoint 已清理；上述结论属于历史验收记录。后续恢复先检查现有环境和保留的包快照，不重新执行已清理的旧任务。工作区目录、数据生成参数和统一版本维护说明见 [根 README](../README.md)。

## 准备 iGSM 数据

输入 JSONL 每行须有 `problem`（或 `question`）和整数 `answer`；`solution` 不会放进 prompt。
按题目去重并确定性划分训练/验证集，冲突答案和明确标为未验证的样本会报错。

```bash
python examples/igsm/prepare_data.py \
  --input ../data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl \
  --output-dir ../datasets/igsm-rl
```

输出目录必须不存在。`train.parquet` / `val.parquet` 包含 verl 的 prompt、reward_model、data_source、extra_info 字段。
问题提示要求模型最后输出 `#### 整数答案`；奖励是最终整数完全匹配，不校验推理步骤。

## 训练入口

```bash
# GRPO：每题采样 4 条，无 critic
MODEL_PATH=../models/Qwen3-0.6B bash examples/igsm/run_grpo.sh

# PPO：GAE + critic，每题采样 1 条
MODEL_PATH=../models/Qwen3-0.6B bash examples/igsm/run_ppo.sh

# 指定数据，或覆盖 Hydra 参数
DATA_DIR=/path/to/parquet MODEL_PATH=/path/to/model \
  bash examples/igsm/run_grpo.sh trainer.total_training_steps=1

# 只打印配置；仍要求先安装完整运行依赖
bash examples/igsm/run_grpo.sh --cfg job
```

默认使用单机单 GPU、小批次和 CPU offload，actor/critic 使用 SDPA，vLLM 使用 eager，保存到 `../checkpoints/igsm-grpo` 或 `igsm-ppo`。
实际显存还取决于模型、长度和采样数；4096 是响应上限，长答案需要调整 max_response_length、max_model_len 和 token budget。
不同实验应覆盖 `trainer.default_local_dir`，避免默认自动恢复同名目录中的旧训练。

## 检查

```bash
python -B tests/igsm/test_igsm.py
python -B tests/igsm/check_structure.py
bash -n examples/igsm/train_common.sh examples/igsm/run_grpo.sh examples/igsm/run_ppo.sh
git diff --stat -- .
```

当前源码由父工作区统一 Git 维护；上方基线提交用于记录上游来源，恢复文件时以父仓库实际提交为准。保留 Apache-2.0 LICENSE 和保留代码中的版权声明。

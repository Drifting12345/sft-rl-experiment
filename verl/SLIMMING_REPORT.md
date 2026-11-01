# 精简记录

- 上游：https://github.com/verl-project/verl.git
- 上游基线提交：`8e48d72de4766228c36623c90f9f5c49ee5b45f3`，用于记录来源；当前源码由 zfy 根仓库统一维护。
- 范围：iGSM 单轮文本 GRPO / PPO，V1 同步 trainer，FSDP/FSDP2 + vLLM，规则奖励。
- 原始受版本控制文件：1303。
- 保留上游文件：236；删除：1067。
- 新增任务入口和检查：11 个文件（含本报告和清单）。

上述文件数和清单记录首次精简，不包含后续环境配置目录迁入等改动，也不是当前父工作区的总文件数。

删除内容包括部署镜像/CI、上游大部分文档与示例、研究 recipe、SFT 入口、独立异步训练器、其他训练/生成后端、其他数据集奖励和无关测试。
`AGENTS.md` 和它引用的编辑指南、LICENSE 原样保留。

共享模块有显式和动态依赖：`protocol`、Ray controller、checkpoint、agent/reward loop、tokenizer、profiling、config 数据结构仍是主链路的一部分。
这些共享文件中仍可能有未启用的视觉、蒸馏或其他算法分支；为避免重写上游核心，本次不逐行切除它们。
不再提供已移除后端的注册；入口对非 GRPO/PPO、非同步、非 FSDP/vLLM、蒸馏/奖励模型配置明确报错。

`unchanged_core_sha256` 记录了 7 个核心训练/算法文件的原始内容散列；核心算法、trainer 主循环、FSDP 实现和 vLLM 服务代码未改写。
替换 iGSM 默认奖励、缩小注册表和工具导出，去掉仅服务于被删后端的工具函数；精简打包依赖和配置文件。

## 验证边界

首次精简阶段检查覆盖语法、内部模块引用与注册字符串、核心文件散列、iGSM 奖励与数据逻辑、两种 Hydra 配置组合、shell 语法和 Parquet 数据往返。该阶段尚未配置完整训练环境，也未完成完整 `main_ppo` 导入或 GPU 强化学习训练。

2026-10-30 远端精简验证通过：9 项奖励/数据测试，212 个 Python 文件语法与内部引用，7 个核心文件散列，两套 Hydra 配置组合，3 个 Bash 脚本语法，git diff --check，以及 5 条训练/1 条验证样本的 Parquet 往返；当时退出码为 0。

后续独立 Conda 环境 `verl-train` 已配置 Python 3.12.16、torch 2.13.0+cu129 和 vLLM 0.32.0+cu129，并通过 GPU/SDPA 反向、NCCL/FSDP 优化器、critic 反向、实际生成，以及 GRPO/PPO 各一步执行与 checkpoint 保存验收。使用本地 Qwen3-0.6B、6 条 iGSM 样本（train 4 / val 2）、batch 2、响应上限 16，实际为单 GPU FSDP1/BF16/CPU offload、actor/critic SDPA 和 vLLM eager。

本次 GRPO 损失与梯度均为 0，只验证执行与保存链路，不能据此声称有效权重更新或训练收敛；多卡、FSDP2 和完整任务质量也未在该轮验收。上述均为历史结果，原验证状态、日志、汇总及 smoke checkpoint 已按要求清理，不再提供它们作为当前文件入口。当前环境配置和包快照说明见 [setup/README.md](setup/README.md)。

## 后续维护

在 zfy 根目录检查父仓库实际提交和目标文件改动：

```bash
git log --oneline -- verl/path/to/file
git diff -- verl/path/to/file
```

需要恢复时选择父仓库中确实存在的提交，并使用以 zfy 为根的 `verl/...` 路径。上游基线提交仅记录来源，不能假定该对象存在于新父仓库；原子仓库 Git 元数据已按用户要求移除。

首次精简的完整保留/删除列表见 `slimming-manifest.json`。当前工作区布局和统一版本维护说明见 [根 README](../README.md)。

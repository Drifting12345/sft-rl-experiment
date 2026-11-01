# verl 训练与推理环境

本目录位于 `verl/setup/`，保存环境入口和版本记录。实际 Conda 环境为 `../../.conda/envs/verl-train`，verl 训练源码位于上一级目录 `..`；推理评测代码为 `../../exp`。

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/verl/setup/activate.sh
```

脚本会激活环境、设置项目缓存和 Python 搜索路径，并进入 `verl/`。运行推理评测时随后执行：

```bash
cd /inspire/hdd/project/deepanalysis/guitao-25013/zfy/exp
```

已配置组合为 Python 3.12.16、PyTorch 2.13.0+cu129、vLLM 0.32.0+cu129、transformers 5.16.1、TorchCodec 0.16.0+cu129、FFmpeg 8.1.2。

| 文件 | 用途 |
| --- | --- |
| `activate.sh` | 激活实际 Conda 环境；依赖个人父目录的 `miniconda3` |
| `requirements-cu129.txt` | 兼容的 CUDA 12.9 组合与官方 wheel 地址/校验值 |
| `requirements-installed.txt` | 实际 Python 包与 editable 安装快照 |
| `conda-explicit.txt` | 已安装 Conda 原生包的精确清单，包括 Python/FFmpeg 等依赖 |
| `installed-packages.json` | 便于检查的 Python 包版本快照 |

环境已完成单 GPU 运行验收，包括 GPU 生成以及 GRPO/PPO 各一步执行和 checkpoint 保存。此次 GRPO 验收梯度为零，只证明执行链路可用；没有据此验证收敛、多卡或 FSDP2。旧安装排错脚本、验证日志和 smoke checkpoint 已清理，不再作为当前文件入口。

以后先检查现有环境，再决定是否修复或重建。迁移时需分别恢复 Conda 原生依赖和 pip 包，并核对 Linux 架构、GPU 驱动、路径及 `verl` 的 editable 安装位置；这些记录不保证新机器一条命令完全复现。沿用 cu129 组合，避免使用上游默认 CUDA 13 extras 覆盖当前环境。

`igsm-gpu` 继续负责数据生成，不把这两套依赖混装。正式推理评测使用本环境，运行期间保持环境、模型及评测输入不变，操作方法见 [实验说明](../../exp/README.md)。

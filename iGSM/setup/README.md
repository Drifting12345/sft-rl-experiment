# iGSM 数据生成工具

本目录位于 `iGSM/setup/`，保存数据生成所需的配方、激活入口和可复查的生成脚本。实际 Conda 环境位于 `../../.conda/envs/igsm-gpu`，iGSM 源码位于上一级目录 `..`。

| 文件 | 用途 |
| --- | --- |
| `activate.sh` | 激活 `igsm-gpu`、设置项目缓存并进入 iGSM 源码目录 |
| `environment.yml` | Conda/Python 与 pip 依赖配方 |
| `requirements-core.txt` | 核心 Python 依赖版本 |
| `packages-gpu.json` | 已配置环境的实际包快照 |
| `generate_hard_long_1030.py` | 生成题目、按解答长度筛选、验证答案并保存生成记录 |
| `audit_hard_long_1030.py` | 对 100 条指定参数数据重新分词、按种子重生成并独立核对 |

环境配置为 Python 3.10.19、PyTorch 2.4.0+cu121；训练与推理继续使用独立的 `verl-train`。

## 使用

```bash
source /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM/setup/activate.sh
```

下面示例生成一份新的数据集，输出及其伴随记录必须尚不存在：

```bash
PYTHONHASHSEED=0 python -B /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM/setup/generate_hard_long_1030.py \
  --repo /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM \
  --output /inspire/hdd/project/deepanalysis/guitao-25013/zfy/data/igsm_hard_long_100_new.jsonl \
  --count 100 --seed 42 --workers 4 --min-solution-tokens 1024 \
  --max-op 52 --max-edge 80 --op 52
```

```bash
PYTHONHASHSEED=0 python -B /inspire/hdd/project/deepanalysis/guitao-25013/zfy/iGSM/setup/audit_hard_long_1030.py \
  /inspire/hdd/project/deepanalysis/guitao-25013/zfy/data/igsm_hard_long_100_new.jsonl
```

独立审查脚本对应 `max_op=52`、`max_edge=80`、`op=52` 的 100 题数据，会新建审查报告，已有报告时拒绝覆盖。生成脚本使用 GPT-2 tokenizer 计算 **solution 字段**长度，1024 token 的门槛不包含题目和独立答案字段，也不是 Qwen tokenizer 的长度。该参数组合属于调高难度后的 hard-long 配置，应与原版 hard 参数区分。

生成脚本内固定 `perm_level=5`、`detail_level=0`、`be_shortest=True`、`p_format="pq"`，hash bins 为全部 `0..22`，数值计算使用模 23。`seed=42` 是候选种子起点，每题实际种子单独保存；`max_edge=80` 是上限。`PYTHONHASHSEED=0` 需要在启动 Python 前设置，脚本不会自行设置。完整参数与本批长度统计见 [根目录说明](../../README.md)。

已交付的数据为 `../../data/igsm_hard_long_100_op52_gpt2_ge1024_20261030.jsonl`；原文保留，旧审查记录已按要求清理。评测配置指向这份数据，生成新实验数据时另取文件名。

恢复环境时结合配方、包快照和实际平台重新核实；激活脚本依赖个人父目录下的 `miniconda3`。两个生成/审查脚本直接使用 iGSM 源码，不依赖已清理的旧 smoke 验证脚本。

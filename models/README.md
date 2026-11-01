# Qwen3 模型与固定版本恢复

本目录保留 Qwen3-0.6B、Qwen3-1.7B、Qwen3-4B、Qwen3-8B 的完整模型目录，包括权重、tokenizer 和配置。当前正式评测使用原始 `Qwen3-0.6B/`。

| 文件 | 用途 |
| --- | --- |
| `manifest.json` | 四个模型的固定快照清单，记录 Revision、文件大小和 SHA256 |
| `restore.py` | 按清单恢复指定模型、断点下载或校验已有文件 |
| `requirements-download.txt` | 下载工具依赖 `requests==2.32.5` |

固定清单的 SHA256 为：

```text
607913fbc0cc085ade8c34d810124c9307abe0ab08e5726c3d567650d8a2c0b9
```

工具以此清单恢复同一版本，不查询最新版。清单变化会拒绝执行，已有模型文件不符时保留现场并报错。

## 使用

在带有所需依赖的 Linux Python 环境中，按需校验已有模型：

```bash
python /inspire/hdd/project/deepanalysis/guitao-25013/zfy/models/restore.py \
  --models Qwen3-0.6B --verify-only
```

需要在新位置恢复时，先为该 Python 环境配置 `requirements-download.txt` 的依赖，再指定目标目录，例如：

```bash
python /inspire/hdd/project/deepanalysis/guitao-25013/zfy/models/restore.py \
  --output /your/new/model-directory --models Qwen3-0.6B --workers 8
```

可用 `--models` 选择一个或多个上述模型；省略时恢复全部四个。`--cache-dir` 可以提供含 `Qwen3-*` 子目录的已有模型来源，复制后仍校验文件哈希。恢复目标还需具备网络、写入权限和足够空间；模型恢复不等于 Conda/推理环境重建。

旧 `.downloads/` 临时目录已清理；工具再次运行时会在目标目录重建下载锁、分块和校验状态。模型文件仍在，清理临时状态不要求重新下载完整权重。当前运行期间不要修改或移走评测使用的模型目录。

根目录 `.gitignore` 排除模型权重和临时下载文件，保留本说明、清单、恢复脚本及依赖文件，方便版本控制源码和恢复依据。

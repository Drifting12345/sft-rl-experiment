# RL 样本学习轨迹

直接用浏览器打开 `index.html`。这是单文件页面，无需安装依赖、编译或连接外网；所有数据在当前浏览器处理，不会上传。

## 导入实验数据

支持 CSV、JSONL / NDJSON、JSON。可一次选中多份文件，例如每轮保存一份 JSONL，再一起导入。样本数自动识别，不固定为 1000。

每条记录是**同一个样本在某轮训练完成后的独立评估**：

```json
{"sample_id":"sample_0001","epoch":2,"eval_correct":40,"eval_total":256,"update_count":1}
```

| 字段 | 要求 |
| --- | --- |
| `sample_id` | 稳定的样本 ID；跨轮次保持不变；CSV 中保留为字符串 |
| `epoch` | 非负整数；0 是训练前评估，1 是第一轮结束后的评估 |
| `eval_correct` | 独立评估中答对的次数，非负整数 |
| `eval_total` | 独立评估总采样数，正整数，不小于 `eval_correct` |
| `update_count` | 该训练轮中，这道题产生有效奖励学习信号的 rollout 组数；非负整数；初始记录填 0 |

每题每轮只生成一组训练 rollout 时，`update_count` 为 0 或 1；若一轮内多次使用同一题，则累计该轮有效组数。**不要填写整个模型的 optimizer step 数，也不要用独立评估正确数推断这个字段。** 对二元奖励、组内相对优势的标准 GRPO，有对有错的一组通常有奖励信号，全对或全错的一组通常没有。实际应按训练器采用的奖励、advantage 和过滤规则统计；不包含 KL、熵项、优化器动量等。

训练记录的逻辑示例（不是对现有训练器的补丁）：

```python
import json

# epoch=0 先独立评估全部样本，update_count=0。
# 每轮开始将逐题计数清零；每个 rollout 组按照真实训练逻辑判断
# 是否产生有效奖励信号，若有则令 effective_groups[sample_id] += 1。
# 该轮结束后，用独立评估采样写出以下记录。
with open(f"trajectory_epoch_{epoch:03d}.jsonl", "w", encoding="utf-8") as out:
    for sample_id in sample_ids:
        correct, total = eval_counts[sample_id]
        record = dict(
            sample_id=str(sample_id), epoch=epoch,
            eval_correct=correct, eval_total=total,
            update_count=effective_groups.get(sample_id, 0),
        )
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
```

JSON 支持记录数组，或带预期样本数的对象：

```json
{
  "expected_samples": 1000,
  "records": []
}
```

`expected_samples` 可选；填写后会检查实际 ID 数量是否一致。所有样本必须有 `epoch=0`；重复的 `(sample_id, epoch)` 会拒绝导入，不会覆盖或偷偷平均。缺少其他轮次记录会显示斜纹与 `?`，不会填成零。导入失败时保留已载入的数据。CSV 支持 UTF-8 BOM、带逗号/引号/换行的引号字段。

`example.csv` 是两题三次评估的**格式示例，不是真实实验结果**。页面内也能下载 CSV / JSON 模板，并按指定样本数和训练轮数载入模拟数据。

## 图表

- 横轴是训练前及各轮末评估，纵轴是固定样本顺序。
- 初始档位为 `0%`、`(0%,5%]`、`(5%,10%]`、…、`(95%,100%]`，一共 21 档。零命中单独列出；其余每档 5%。先按初始档位，再按初始成功率、ID 排序。后续不会重新排序。
- 格子颜色也采用固定的 5% 档位，表示单次成功率 `eval_correct / eval_total`；圆点表示 `update_count > 0`，没有点表示 `update_count = 0`。不设置“未采样”类别。
- 右侧是全局末轮与初始成功率之差，单位为百分点；该样本缺少全局末轮评估时显示“缺失”。
- 按初始档位筛选、搜索 ID、点击格子或使用下拉菜单定位样本；方向键也可在热图区切换样本与轮次。
- 下方曲线显示精确成功率和逐点 95% Wilson 区间，柱图显示有效组数；缺失处不连接曲线。表格保留全部精确计数。
- 单题 pass@k 使用 `1 - C(n-c,k)/C(n,k)` 的组合无偏估计；点击某个轮次的格子、曲线区域或详情表格可切换轮次。要求整数 `1 <= k <= n`，不做超预算外推。

初始分档只依据有限次采样估计，不表示严格能力边界。有效信号出现与成功率上升的先后关系，也不能单独证明跨题迁移的因果关系。评估应固定 prompt、采样设置、长度和判分方式。

## 规模与导出

热图按可见区域绘制，画布尺寸不随样本总数增长。1000 样本无需分页，纵向滚动查看；训练轮数较多时横向滚动。初始分组标签和表头固定。

- **PNG**：当前可见热图区域。
- **SVG**：当前筛选的全部样本和轮次，适合继续编辑或排版；完整 SVG 限制为 30 万格，可按初始档位分批导出。
- **CSV**：当前筛选的完整原始记录，可再次导入，缺失记录不伪造。

导入文件总量上限为 256 MiB，样本数乘轮次数最多 500 万格。模拟生成最多 100 万条记录。浏览器刷新后需要重新导入数据。

## 可选 HTTP 访问

页面本身可直接打开，无需服务。如需通过 SSH 端口转发访问，在远端执行：

```bash
cd /root/sft-rl-experiment/picture
python3 -m http.server 8767 --bind 127.0.0.1
```

本机另开终端：

```bash
ssh -p 15000 -L 8767:127.0.0.1:8767 root@10.176.50.17
```

浏览器打开 `http://127.0.0.1:8767/`。服务仅绑定远端回环地址。本次交付不自动启动远端长期服务。

## 验证

如安装了 Node.js，可运行 `node test-data-core.cjs`，验证分档边界、CSV / JSONL / JSON、输入校验、导出重载、Wilson 区间和 pass@k。测试直接读取页面中的数据处理代码，无生产依赖。

## 论文紧凑图

`plot_compact.py` 接受与 HTML 完全相同的 CSV / JSONL / JSON。默认使用**单张双色热图**：

- 蓝色：本轮 `update_count > 0`，产生过有效奖励学习信号。
- 橙色：本轮 `update_count = 0`，没有有效奖励学习信号。
- 两个色系各自按成功率从浅到深，仍然是 5% 一档；零命中独立为 0% 档。
- 初始评估列用灰度表示成功率，不赋予训练更新状态；缺失格用灰色斜纹。
- 每个样本是一条细线，不标逐题 ID；按初始分组固定排序，不聚合或抽样。
- 不使用“未采样”类别。这里的更新状态仅指记录的有效奖励信号。

依赖可安装到你用于画图的 Python 环境：

```bash
python -m pip install -r requirements-plot.txt
```

真实实验示例（样本数自动识别，可选检查是否为 1000）：

```bash
python plot_compact.py trajectory_epoch_*.jsonl \
  --output figures/learning-trajectories \
  --expected-samples 1000
```

默认导出 7.2 × 4.7 英寸、600 DPI PNG 和 SVG，支持 `--formats png svg pdf`。图中不自动添加真实实验标题，适合由论文 caption 解释。`--width`、`--height`、`--dpi` 可调，`--initial-bin 1` 可只画初始 (0%,5%] 组；省略时包含所有样本。

预览图可重现：

```bash
python plot_compact.py --demo-samples 1000 --demo-epochs 12 \
  --output previews/compact-bivariate
```

这类预览图的标题明确标注 **SIMULATED DATA**。`previews/compact-bivariate.png` / `.svg` 是 1000 样本、12 轮的模拟效果图，不是真实实验结果。旧 `split` 和黑色短线 `overlay` 方式保留为可选 `--layout`，默认 `bivariate` 是同一张图里的双色编码。

远程系统默认 `python3` 当次检查未安装 numpy / matplotlib；脚本和预览文件已交付，未改动现有训练环境。HTML 无需这两个依赖。

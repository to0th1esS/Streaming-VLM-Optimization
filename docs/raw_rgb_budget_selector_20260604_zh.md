# Raw-RGB Budget Selector 实验记录（2026-06-04）

## 1. 实验目的

上一轮 `budget_topk（预算 Top-K）` 证明了一件重要事情：

```text
语义增量选择可以比 periodic sampling（周期/均匀采样）保留更少帧，并降低相对 dense baseline（密集基线）的部分损失。
```

但它也暴露了核心瓶颈：

```text
selection overhead（选择开销）太高。
```

原因是当前 `vit_embedding（ViT 嵌入）` 选择信号需要先对窗口内所有帧运行 `vision embedding（视觉嵌入）`，再做 `novelty ranking（新颖性排序）`。这会吃掉原本通过少写入帧获得的速度收益。

本轮实验的目的不是追求最终质量，而是验证一个速度上界问题：

```text
如果 selection signal（选择信号）足够便宜，semantic admission（语义准入）能否显著快于 periodic sampling（周期采样）？
```

因此新增 `raw_rgb（原始 RGB）` 低成本签名作为选择源。

## 2. 方法设计

新增参数：

```text
semantic_selection_feature_source = raw_rgb
```

其执行逻辑：

```text
输入视频帧
  -> 对每帧做 4x4 adaptive average pooling（自适应平均池化）
  -> 得到低维 RGB signature（RGB 签名）
  -> 在 budget window（预算窗口）内做 novelty ranking（新颖性排序）
  -> 只保留 Top-K 帧
  -> 仅对保留帧运行 ViT encoder（ViT 编码器）和 LLM context write（大模型上下文写入）
```

和 `vit_embedding（ViT 嵌入）` 版本的关键区别：

| 版本 | 选择信号来源 | 选择前是否运行 ViT | 预期优势 | 预期风险 |
|---|---|---:|---|---|
| `vit_embedding（ViT 嵌入）` | ViT patch embedding（图像块嵌入） | 是 | 语义信号更强 | 延迟高 |
| `raw_rgb（原始 RGB）` | 低分辨率颜色/布局签名 | 否 | 延迟极低 | 语义不足 |

## 3. 实验设置

数据与模型：

- 数据集：`RVS-Movie（电影剧情视频问答子集）`
- 采样率：`1.0 fps（每秒 1 帧）`
- 模型：`LLaVA-OV-7B（视觉语言模型）`
- 对照：
  - `dense baseline（密集基线）`
  - `periodic sampling（周期/均匀采样）`
  - `budget_topk + vit_embedding（预算 Top-K + ViT 嵌入）`
  - `budget_topk + raw_rgb（预算 Top-K + 原始 RGB）`

远程环境：

- 代码提交：`af954f2`
- 模型目录：`/home/mllm/models`
- 数据目录：`/home/mllm/datasets`

## 4. 主结果

### 4.1 与 dense baseline（密集基线）对比

| 方法 | budget window（预算窗口） | kept / input（保留/输入帧） | token reduction（令牌减少） | encode time（编码时间） | speedup vs dense（相对密集加速） | token-F1（词重叠 F1） | W/T/L vs dense（胜/平/负） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dense（密集）` | - | 4067 / 4067 | 0.00% | 373.5553s | 1.00x | 0.0551 | 0 / 24 / 0 |
| `periodic（周期采样）` | - | 99 / 4067 | 97.57% | 27.5793s | 13.54x | 0.0410 | 2 / 17 / 5 |
| `vit_embedding（ViT 嵌入）` | 96 | 85 / 4067 | 97.91% | 29.8811s | 12.50x | 0.0552 | 1 / 21 / 2 |
| `raw_rgb（原始 RGB）` | 96 | 85 / 4067 | 97.91% | 4.6555s | 80.24x | 0.0343 | 0 / 20 / 4 |
| `raw_rgb（原始 RGB）` | 80 | 94 / 4067 | 97.69% | 5.6758s | 65.82x | 0.0297 | 0 / 19 / 5 |

### 4.2 与 periodic sampling（周期采样）对比

| 方法 | kept frames（保留帧数） | encode time（编码时间） | speedup vs periodic（相对周期采样加速） | token-F1（词重叠 F1） | W/T/L vs periodic（胜/平/负） |
|---|---:|---:|---:|---:|---:|
| `periodic（周期采样）` | 99 | 27.5793s | 1.00x | 0.0410 | - |
| `raw_rgb bw96（原始 RGB，窗口 96）` | 85 | 4.6555s | 5.92x | 0.0343 | 1 / 21 / 2 |
| `raw_rgb bw80（原始 RGB，窗口 80）` | 94 | 5.6758s | 4.86x | 0.0297 | 1 / 20 / 3 |

## 5. 关键现象

### 5.1 速度上界被打开

`raw_rgb（原始 RGB）` 版本把选择前的 ViT 计算完全移除，因此真实编码时间从 `27.5793s` 降到 `4.6555s`。

这说明：

```text
semantic admission（语义准入）本身不是慢，慢的是当前 selection signal（选择信号）的提取方式。
```

换句话说，我们之前遇到的速度瓶颈不是方法方向错误，而是选择器信号太贵。

### 5.2 质量明显下降

`raw_rgb（原始 RGB）` 的 `token-F1（词重叠 F1）` 从周期采样的 `0.0410` 降到 `0.0343`，且相对 dense baseline（密集基线）的 `W/T/L（胜/平/负）` 变成 `0 / 20 / 4`。

这说明：

```text
仅靠颜色和低分辨率布局变化，不足以稳定代表语义变化。
```

尤其在电影剧情视频中，重要信息可能来自：

- 人物身份；
- 细粒度动作；
- 物体出现/消失；
- 场景语义转换；
- 字幕或局部区域。

这些都不一定能被 `4x4 raw RGB signature（4x4 原始 RGB 签名）` 捕捉。

### 5.3 当前结果形成了明确的 Pareto 边界

当前三类方法形成了一个清晰的 `Pareto frontier（帕累托边界）`：

| 方法类型 | 速度 | 质量 | 研究含义 |
|---|---|---|---|
| `periodic（周期采样）` | 中等 | 稳定 | 强基线，不可忽视 |
| `vit_embedding selector（ViT 嵌入选择器）` | 慢 | 较好 | 语义信号有效，但开销过高 |
| `raw_rgb selector（原始 RGB 选择器）` | 极快 | 较弱 | 速度空间很大，但信号太粗 |

因此下一步不应继续在单一选择源上硬调参数，而应设计：

```text
middle-cost semantic selector（中等开销语义选择器）
```

## 6. 对论文方法设计的启发

这轮实验可以被整理成一个重要 `insight（洞察）`：

```text
Streaming VLM（流式视觉语言模型）中的帧准入不应被设计成“是否跳过帧”的简单阈值问题，
而应被设计成“在固定上下文预算下，以最小感知成本找到语义增量最大帧”的预算分配问题。
```

对应到论文表达：

- 不强调我们试过很多工程版本；
- 强调我们发现了 `selection quality-cost dilemma（选择质量-成本矛盾）`；
- 最终方法应自然解决这个矛盾。

这个矛盾可以表述为：

```text
High-level semantic signals（高层语义信号） are accurate but expensive.
Low-level visual signals（低层视觉信号） are cheap but unreliable.
```

最终方法需要把二者结合，而不是二选一。

## 7. 下一步方法方向

建议进入 `two-stage selector（两阶段选择器）`：

```text
Stage 1（第一阶段）：
  用 raw_rgb（原始 RGB）或 motion signature（运动签名）做廉价 prefilter（预筛选），
  每个 budget window（预算窗口）先选出 M 个候选帧。

Stage 2（第二阶段）：
  只对候选帧提取 vit_embedding（ViT 嵌入）或 shallow ViT feature（浅层 ViT 特征），
  再从候选帧里选最终 K 个写入视觉上下文。
```

目标：

```text
接近 raw_rgb（原始 RGB）的速度，同时接近 vit_embedding（ViT 嵌入）的选择质量。
```

建议第一版参数：

- `budget_window_size = 96`
- `budget_keep_per_window = 1`
- `candidate_multiplier = 4`
- 每个窗口先用 `raw_rgb（原始 RGB）` 预筛出 `4K` 个候选；
- 再用 `vit_embedding（ViT 嵌入）` 从候选中选 `K` 个最终帧。

## 8. 阶段性结论

本轮不是最终方法，但它非常关键：

```text
它把问题从“语义选择有没有意义”推进到“如何以足够低的成本获得足够强的语义选择信号”。
```

这比继续讨论 `feature cosine（特征余弦相似度）` 或单纯追求密集特征还原更贴近最终目标：

```text
把 dense visual stream（密集视觉流）转化为 sparse semantic stream（稀疏语义流），同时减少 ViT 计算和 LLM KV cache（大模型键值缓存）写入。
```


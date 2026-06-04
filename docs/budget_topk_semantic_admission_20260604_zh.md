# Budget-TopK Semantic Admission 实验记录（2026-06-04）

## 1. 实验目的

前一轮严格 `0.5/1.0 fps（帧率）` 实验说明，固定阈值 `semantic gate（语义门控）` 没有稳定胜过 `periodic sampling（周期/均匀采样）`。

核心问题是：固定阈值无法控制保留帧数，导致 semantic（语义方法）有时比 periodic（周期方法）保留更多帧。

本轮目标：

```text
让 semantic admission（语义准入）具备 budget control（预算控制）
```

也就是：

- 每个 `budget window（预算窗口）` 最多保留 K 个语义新颖帧；
- 保留帧数应低于或等于 periodic baseline（周期基线）；
- 再比较 QA 表现和真实延迟。

## 2. 新增方法

新增策略：

```text
semantic_selection_policy = budget_topk
```

含义：

```text
在每个 streaming window（流式窗口）中：
  1. 用轻量 ViT embedding（ViT 嵌入）计算每帧 semantic novelty（语义新颖性）
  2. 每个 budget window（预算窗口）选择 Top-K 新颖帧
  3. 强制保留 recency frames（最近帧）
  4. 只把被选中的帧写入 LLM visual context（大模型视觉上下文）
```

新增参数：

- `semantic_selection_policy`
- `semantic_budget_window_size`
- `semantic_budget_keep_per_window`

默认仍为旧策略 `threshold（阈值策略）`，因此旧实验不受影响。

## 3. 本地与远程验证

本地验证：

- `py_compile（Python 语法检查）` 通过；
- `budget_topk（预算 Top-K）` smoke test（冒烟测试）通过；
- 小例子中每 3 帧选择 1 个 novelty（新颖性）最高帧，输出符合预期。

远程验证：

- 远程 HEAD：`7f213bf`
- 模型：`llava_ov_7b`
- 数据集：`RVS-Movie`
- fps：`1.0 fps（每秒 1 帧）`
- 对照：
  - dense baseline（密集基线）
  - periodic baseline（周期/均匀采样基线）

## 4. 主结果：RVS-Movie 1fps

已有 baseline：

| 方法 | kept / input | token reduction | speedup vs dense | token-F1 | W/T/L vs dense |
|---|---:|---:|---:|---:|---:|
| dense | 4067 / 4067 | 0.00% | 1.00x | 0.0551 | 0 / 24 / 0 |
| periodic | 99 / 4067 | 97.57% | 13.54x | 0.0410 | 2 / 17 / 5 |
| threshold semantic | 146 / 4067 | 96.41% | 11.60x | 0.0464 | 2 / 19 / 3 |

Budget-TopK 结果：

| 方法 | budget window | kept / input | token reduction | speedup vs dense | token-F1 | W/T/L vs dense |
|---|---:|---:|---:|---:|---:|---:|
| budget_topk | 80 | 94 / 4067 | 97.69% | 10.91x | 0.0438 | 2 / 19 / 3 |
| budget_topk | 96 | 85 / 4067 | 97.91% | 12.50x | 0.0552 | 1 / 21 / 2 |
| budget_topk | 128 | 75 / 4067 | 98.16% | 11.44x | 0.0368 | 1 / 19 / 4 |

## 5. 与 periodic baseline 对比

最有价值的是 `budget_window=96`：

| 方法 | kept frames | token-F1 | W/T/L vs dense | speedup |
|---|---:|---:|---:|---:|
| periodic | 99 | 0.0410 | 2 / 17 / 5 | 13.54x |
| budget_topk 96 | 85 | 0.0552 | 1 / 21 / 2 | 12.50x |

观察：

1. `budget_topk 96（预算 Top-K，窗口 96）` 保留帧数更少：`85 < 99`。
2. token-F1 更高：`0.0552 > 0.0410`。
3. loss 更少：`2 < 5`。
4. 但真实 speedup 低于 periodic：`12.50x < 13.54x`。

这说明：

> selection signal（选择信号）开始有效，但 selection overhead（选择开销）还没有优化。

换句话说，方法方向不再是“没有意义”，而是：

```text
语义选择质量优于均匀采样，但当前实现的轻量预判还不够轻。
```

## 6. Judge 结果

对 `budget_topk 96` 跑 Qwen2.5-VL judge（模型裁判）：

| 方法 | valid / total | dense acc | sparse acc | dense-only | sparse-only | better / same / worse |
|---|---:|---:|---:|---:|---:|---:|
| periodic | 20 / 24 | 20.0% | 40.0% | 0 | 4 | 4 / 4 / 12 |
| budget_topk 96 | 22 / 24 | 18.2% | 27.3% | 0 | 2 | 2 / 4 / 16 |

Judge 不支持 budget_topk 96 全面优于 periodic。

因此当前结论必须谨慎：

- 字符串/overlap 指标支持 budget_topk 96；
- judge 指标仍然 periodic 更强；
- 说明 RVS-Movie 问题可能仍然不够适合验证语义选择，或者当前 novelty（新颖性）信号还不够强。

## 7. 关键发现

本轮最重要的正向信号：

```text
budget_topk 96 can keep fewer frames than periodic and reduce dense-side loss.
budget_topk 96 能比周期采样保留更少帧，并减少相对 dense 的 loss。
```

这回答了前面的质疑：

> 增量语义选择理论上应该比均匀采样保留更少帧。

现在初步做到了：

```text
85 frames vs 99 frames
```

但还没解决：

```text
latency overhead（延迟开销）
judge quality（模型裁判质量）
```

## 8. 为什么延迟仍慢于 periodic

periodic sampling（周期采样）几乎没有选择开销：

```text
if frame_idx % interval == 0: keep
```

budget_topk（预算 Top-K）需要：

1. 对窗口内所有帧做 ViT embedding（ViT 嵌入）；
2. 计算 novelty（新颖性）；
3. 排序选 Top-K；
4. 再编码保留帧。

当前实现是正确性优先版本，不是 optimized implementation（优化实现）。

下一步需要把 selection signal（选择信号）做得更便宜：

- 用 patch embedding（图像 patch 嵌入）而不是完整视觉特征；
- 用低分辨率 thumbnail（缩略帧）；
- 用 motion/color histogram（运动/颜色直方图）预筛；
- 或把 novelty 计算和 ViT 第一层复用合并。

## 9. 下一步

下一步不应继续盲扫窗口大小，而应做两件事：

### 9.1 优化 selection overhead（选择开销）

目标：

```text
budget_topk latency <= periodic latency
预算 Top-K 延迟不高于周期采样
```

否则即使帧数更少，也会输给 periodic。

### 9.2 接入 OVO-Bench / StreamingBench

RVS-Movie judge 对 periodic 仍然友好，说明这个数据可能无法充分体现 semantic selection（语义选择）的优势。

需要在更强流式数据集上验证：

- `OVO-Bench（在线视频理解基准）`
- `StreamingBench（流式视频理解基准）`

尤其要找：

- backward tracing（向后追溯）
- real-time perception（实时感知）
- event transition（事件转移）
- long-context QA（长上下文问答）

这些任务更可能体现 semantic selection（语义选择）优于 uniform sampling（均匀采样）。

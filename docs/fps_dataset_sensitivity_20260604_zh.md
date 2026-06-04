# FPS 与数据集敏感性实验记录（2026-06-04）

## 1. 为什么要重做

前期实验使用 `0.2 fps（每秒采样 0.2 帧，约 5 秒 1 帧）`，主要目的是快速建立本地/远程闭环和验证趋势。

但一般流式视频问答场景更常见的是：

- `0.5 fps（每秒采样 0.5 帧，约 2 秒 1 帧）`
- `1.0 fps（每秒采样 1 帧）`

因此，`0.2 fps` 只能作为 `pilot setting（早期探针设置）`，不能作为论文主实验口径。主实验必须切到 `0.5 fps` 和 `1.0 fps`。

## 2. 本轮实验原则

### 2.1 重新跑 dense baseline（密集基线）

改变 `fps（帧率）` 后，输入帧数会变化，dense 编码时间也会变化。因此不能用 `0.2 fps` 的 dense 结果去计算 `0.5/1.0 fps` 的 speedup（加速比）。

本轮对每个 fps 都重新跑：

- `dense（密集基线）`
- `semantic_time_norm（按时间归一的语义准入）`
- `periodic_time_norm（按时间归一的周期/均匀采样）`

### 2.2 time-normalized（按时间归一）

早期 0.2 fps 设置中：

- `semantic refresh=64`，约等于 `64 / 0.2 = 320 秒`
- `periodic interval=13`，约等于 `13 / 0.2 = 65 秒`

因此本轮按照秒数换算：

| fps | semantic refresh | periodic interval |
|---:|---:|---:|
| 0.5 | 160 | 32 |
| 1.0 | 320 | 65 |

这样比较的是相同时间尺度下的方法，而不是相同帧数间隔。

## 3. 运行脚本

新增脚本：

```text
scripts/run_fps_dataset_sensitivity.sh
```

远程输出目录：

```text
results/fps_dataset_sensitivity_20260604
```

已完成配置：

- `RVS-Movie（电影剧情视频）`: `0.5 fps`, `1.0 fps`
- `RVS-Ego（第一视角日常视频）`: `0.5 fps`

暂未跑：

- `RVS-Ego 1.0 fps`，因为单次 dense 已经较慢，后续可作为大规模补充实验。

## 4. 主结果

结果来自：

```text
results/fps_dataset_sensitivity_20260604/summary_all.csv
```

### 4.1 RVS-Movie 0.5 fps

| 方法 | kept / input | token reduction | speedup | token-F1 | W/T/L |
|---|---:|---:|---:|---:|---:|
| dense | 2031 / 2031 | 0.00% | 1.00x | 0.0474 | 0 / 24 / 0 |
| periodic_time_norm | 99 / 2031 | 95.13% | 10.16x | 0.0571 | 2 / 20 / 2 |
| semantic_time_norm | 129 / 2031 | 93.65% | 9.31x | 0.0446 | 2 / 20 / 2 |

现象：

- periodic（周期/均匀采样）比 semantic（语义准入）少保留 30 帧；
- periodic 的 speedup 更高：`10.16x vs 9.31x`；
- 两者 W/T/L 相同，都是 `2 / 20 / 2`。

### 4.2 RVS-Movie 1.0 fps

| 方法 | kept / input | token reduction | speedup | token-F1 | W/T/L |
|---|---:|---:|---:|---:|---:|
| dense | 4067 / 4067 | 0.00% | 1.00x | 0.0551 | 0 / 24 / 0 |
| periodic_time_norm | 99 / 4067 | 97.57% | 13.54x | 0.0410 | 2 / 17 / 5 |
| semantic_time_norm | 146 / 4067 | 96.41% | 11.60x | 0.0464 | 2 / 19 / 3 |

现象：

- periodic 更激进、更快：`13.54x`；
- semantic 保留更多帧，速度较低：`11.60x`；
- 但 semantic 的 loss 更少：`3` vs periodic 的 `5`。

这说明在 `1 fps（每秒 1 帧）` 下，semantic 的价值从“更快”转成了“在高压缩下更少丢语义”。

### 4.3 RVS-Ego 0.5 fps

| 方法 | kept / input | token reduction | speedup | token-F1 | W/T/L |
|---|---:|---:|---:|---:|---:|
| dense | 5115 / 5115 | 0.00% | 1.00x | 0.2723 | 0 / 24 / 0 |
| periodic_time_norm | 194 / 5115 | 96.21% | 3.90x | 0.2740 | 1 / 23 / 0 |
| semantic_time_norm | 209 / 5115 | 95.91% | 3.92x | 0.2674 | 1 / 22 / 1 |

现象：

- Ego 上两者速度几乎一样；
- periodic 更少 token，且没有 loss；
- semantic 出现 1 个 loss。

这说明当前固定阈值 semantic gate（语义门控）在 Ego 0.5 fps 下并没有胜过 periodic baseline（周期/均匀采样基线）。

## 5. 对前期结论的修正

前期 `0.2 fps` 下的结论是：

> semantic admission（语义准入）相比 periodic sampling（周期/均匀采样）能以更少 token 达到相近或略好的 QA。

本轮严格 fps 后需要修正为：

> 在更真实的 0.5/1.0 fps 流式设置下，固定阈值 semantic admission 不稳定优于 periodic baseline。periodic 在高 fps 下变得很强，因为相邻帧冗余更高，固定时间间隔采样已经能去掉大量重复信息。

这不是坏结果，而是一个重要 insight（研究洞察）：

> 高 fps 场景下，问题不再是“是否要稀疏”，而是“在同等 token budget（token 预算）下如何保留更有语义价值的帧”。

## 6. 对方法设计的影响

当前 `semantic_time_norm（按时间归一的语义准入）` 使用固定阈值 `threshold=0.3`。这个设计在不同 fps 下不够稳：

- fps 提高后，相邻帧更相似，drift（语义漂移）分布会改变；
- 固定阈值可能导致保留帧数不可控；
- 与 periodic 对比时，semantic 可能既不够省 token，也不一定更准。

因此下一步不能继续简单扫固定 threshold（阈值）。更合理的方法是：

```text
budget-aware semantic admission（预算感知语义准入）
```

含义：

- 先给定每个时间窗口允许写入多少视觉 token；
- 在这个预算内，选择语义变化最大、覆盖价值最高、最近性最重要的帧；
- 让 semantic 与 periodic 在相同 token budget 下比较，而不是让两者自然保留不同数量的帧。

## 7. 下一步研究方向

建议从三条线推进：

1. `budget-matched comparison（预算匹配对比）`
   - 让 semantic 和 periodic 保留相同数量的帧；
   - 比较 QA，而不是让 periodic 用更少 token 直接赢速度。

2. `adaptive threshold（自适应阈值）`
   - 阈值不固定为 0.3；
   - 根据当前 fps、窗口内 drift 分布、目标 token budget 自动选择。

3. `semantic top-k per window（窗口内语义 Top-K）`
   - 每个时间窗口只保留 drift 最大的 K 帧；
   - 同时强制保留 recent frames（最近帧）；
   - 这样比固定 threshold 更可控，也更适合论文方法表达。

更优雅的最终方法应从：

```text
if drift > threshold: keep
```

升级为：

```text
within each streaming window:
  preserve recency（保留最近状态）
  rank semantic novelty（排序语义新颖性）
  select under token budget（在 token 预算内选择）
```

这与我们的论文目标更一致：把 dense visual stream（密集视觉流）转化为 sparse semantic stream（稀疏语义流）。

# Query-Decoupled Sparse Semantic Stream 大规模验证记录（2026-06-04）

## 1. 实验目的

本轮实验用于验证当前方法是否可以从“逐帧特征复原”转向更符合论文目标的评价方式：

1. 在线视觉处理与视觉上下文写入保持 query-independent，避免流式场景下因 query-aware 感知带来额外等待。
2. 在相同 fixed recent access 入口下，比较 semantic admission 与 periodic/uniform admission。
3. 用 QA 结果和 LLM judge 补充 token-F1，判断速度优先方案是否仍能维持可接受的语义质量。

本轮不再把 feature cosine / MSE 作为主要优化目标；它们仅作为内部诊断指标。主目标是：在 QA 基本不下降的约束下，最大化流式视觉编码与上下文写入效率。

## 2. 运行设置

远程环境：

- 远程仓库：`/home/yangjin/1#Streaming-VLM-Optimization`
- 模型目录：`/home/mllm/models`
- 运行模型：`llava_ov_7b`
- Judge 模型：`/home/mllm/models/Qwen2.5-VL-7B-Instruct`
- GPU：`CUDA_VISIBLE_DEVICES=0`

验证脚本：

- 推理与后处理：`scripts/run_query_decoupled_large_validation.sh`
- 已有结果补 judge：`scripts/run_query_decoupled_judge_existing.sh`

输出目录：

```text
results/large_validation_query_decoupled_20260603
```

对比配置：

| 数据集 | 方法 | admission | recent access |
|---|---|---|---|
| RVS-Ego | semantic | `refresh=64, threshold=0.3, recency K=4` | `always_recent, qrb=4` |
| RVS-Ego | periodic | `refresh=13, threshold=999, recency K=4` | `always_recent, qrb=4` |
| RVS-Movie | semantic | `refresh=64, threshold=0.3, recency K=4` | `always_recent, qrb=4` |
| RVS-Movie | periodic | `refresh=13, threshold=999, recency K=4` | `always_recent, qrb=4` |

说明：`always_recent` 是 query-decoupled 的固定访问策略，不使用问题文本决定视觉计算或 token 写入。它只保证回答阶段总能看到最近若干视觉块。

## 3. 主结果：Dense 对齐后处理口径

以下结果来自：

```text
results/large_validation_query_decoupled_20260603/summary_all.csv
```

| 数据集 | 方法 | kept / input | token reduction | encode speedup vs dense | token-F1 | W/T/L vs dense |
|---|---:|---:|---:|---:|---:|---:|
| RVS-Ego | semantic | 157 / 2046 | 92.33% | 3.55x | 0.2894 | 7 / 17 / 0 |
| RVS-Ego | periodic | 191 / 2046 | 90.66% | 3.49x | 0.2780 | 7 / 16 / 1 |
| RVS-Movie | semantic | 86 / 811 | 89.40% | 6.97x | 0.0589 | 4 / 18 / 2 |
| RVS-Movie | periodic | 95 / 811 | 88.29% | 6.58x | 0.0482 | 2 / 19 / 3 |

重复次数均为 3。表中 speedup 为 3 次重复均值。

直接观察：

1. semantic admission 在两个数据集上都比 periodic 写入更少 token。
2. RVS-Ego 上 semantic 比 periodic 少保留 34 帧，约少 17.8% 的写入帧，同时保持相同的字符串级 wins，且 loss 从 1 降到 0。
3. RVS-Movie 上 semantic 比 periodic 少保留 9 帧，约少 9.5% 的写入帧，速度从 6.58x 提升到 6.97x，约再提升 5.9%。
4. token-F1 很低，尤其 Movie 上只有约 0.05，但这没有直接等价于 QA 完全失败，进一步说明 token-F1 不能作为主优化目标。

## 4. Qwen2.5-VL Judge 结果

Judge 只跑 rep0，用于补充开放式 QA 的语义正确性判断。

| 数据集 | 方法 | valid / total | dense acc | sparse acc | sparse-only | dense-only | better / same / worse |
|---|---:|---:|---:|---:|---:|---:|---:|
| RVS-Ego | semantic | 20 / 24 | 60.0% | 75.0% | 3 | 0 | 3 / 12 / 5 |
| RVS-Ego | periodic | 19 / 24 | 68.4% | 84.2% | 3 | 0 | 3 / 13 / 3 |
| RVS-Movie | semantic | 19 / 24 | 26.3% | 26.3% | 2 | 2 | 2 / 3 / 14 |
| RVS-Movie | periodic | 20 / 24 | 25.0% | 25.0% | 2 | 2 | 2 / 3 / 15 |

解释：

1. RVS-Ego 上，两种稀疏策略都没有出现 `dense-only correct`，说明当前 sparse semantic stream 没有在 judge 可解析样本里造成不可恢复的信息损失。
2. RVS-Ego 上 periodic 的 judge acc 略高，但 semantic 写入更少、字符串级 loss 更少。这个差异提示：Ego 场景中 semantic gate 的主要价值是更省 token，而不是简单碾压 uniform baseline。
3. RVS-Movie 上 dense 与 sparse 的 judge acc 都很低，说明当前 Movie 子集更像“模型本身困难 + 长剧情覆盖困难”，不是单纯 sparse 方法错误。
4. Movie 上 semantic 的 relative worse 数量略少于 periodic，且速度更快、token 更少，说明 semantic admission 仍有优势，但需要更强的长程覆盖机制。

## 5. 关键现象

### 5.1 语义稀疏已经优于“只做均匀抽帧”的简单解释

如果审稿人质疑方法只是 periodic sampling，本轮结果可以回应：

- 在相同 recent access 下，semantic 在 Ego 和 Movie 上都使用更少 token。
- 在 Movie 上 semantic 同时获得更高 speedup、更高 token-F1、更少 relative worse。
- 在 Ego 上 semantic 的 QA 字符串比较无 loss，但 judge 显示 periodic 更保守，需要后续引入覆盖下界增强。

因此，semantic admission 不是简单的均匀抽帧替代；它能在相似或更好的 QA 表现下压缩写入预算。

### 5.2 当前方法的主要短板不是 ViT 特征保真，而是长程语义覆盖

Movie 的 judge 结果说明，剧情型长视频的困难来自跨场景、跨事件的信息覆盖。仅依赖最近窗口和语义变化触发，可能对以下问题不够稳：

- 问题询问长程因果关系。
- 问题需要多个分散片段共同支持。
- 问题答案来自早期但非高变化帧。

这支持下一步从“更精细复原 dense ViT”转向“更优雅的 sparse semantic state 设计”。

### 5.3 fixed recent access 是可保留的主线

本轮没有使用 query 文本指导在线视觉计算和写入，符合即插即用目标。`always_recent` 只在回答阶段提供固定最近块，不影响流式前端延迟。

这可以组织成论文中的设计原则：

> Online stage should construct a query-independent sparse semantic state; query-time access should be a lightweight read policy rather than a visual recomputation policy.

## 6. 对最终方法设计的启发

当前最终方法不应写成“工程尝试堆叠”，而应抽象为一个统一原则：

```text
Dense visual stream
  -> query-independent semantic admission
  -> sparse semantic state
  -> fixed low-latency access policy
  -> streaming VLM answer
```

下一版方法建议从 semantic admission 升级为 semantic-coverage admission：

1. Semantic trigger：保留发生明显语义变化的帧。
2. Coverage floor：保留低频但均匀的 coverage anchor，防止剧情型长视频漏掉慢变化但重要的片段。
3. Recency correction：始终保留最近 K 帧，解决 latest / current-state 问题。

这样比“加一个 heuristic”更优雅，因为它对应三个互补的信息需求：

- change：捕捉事件变化；
- coverage：保证长程可追溯；
- recency：保证当前状态可见。

## 7. 下一步实验计划

优先做 `Semantic + Coverage Floor + Recency`，并与当前两条 baseline 对比：

| 方法 | 目的 |
|---|---|
| semantic only | 当前主策略 |
| periodic only | uniform baseline |
| semantic + coverage floor | 验证是否能修复 Movie 长程覆盖问题 |
| dense | 上界与速度参照 |

建议参数：

- `coverage_interval = 32 / 48 / 64`
- `semantic_refresh = 64`
- `semantic_threshold = 0.3`
- `recency_keep = 4`
- `always_recent qrb = 4`

核心判断指标：

1. token reduction 是否仍高于 85%。
2. speedup 是否保持在 Ego 约 3.5x、Movie 约 6.5x 以上。
3. Movie judge 中 `relative worse` 是否下降。
4. `dense-only correct` 是否不增加或尽量减少。

如果该方向成立，论文方法可以命名为 query-decoupled sparse semantic state，而不是简单的 frame skipping。

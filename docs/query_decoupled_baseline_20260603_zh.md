# Query-decoupled Access 与 Periodic Baseline 实验记录（2026-06-03）

## 1. 实验动机

我们讨论后明确了一个重要原则：

```text
流式视觉计算和视觉 token/cache 写入应该尽量 query-independent。
```

原因是：

1. 流式场景中未来 query 通常未知。
2. 如果 query 到来后才指导视觉感知，会引入额外视觉编码延迟。
3. query-dependent perception 会削弱即插即用和泛化性。

因此，本轮要确认：

1. 当前实现是否已经违反了这个原则；
2. 如果去掉 query 内容判断，只使用固定 recent access，性能是否还能保持；
3. 与一个审稿人容易想到的 `periodic/uniform admission` baseline 相比，semantic gate 是否仍有优势。

## 2. 方法边界澄清

当前系统分成两阶段：

```text
Online streaming stage:
    query-independent semantic gate
    query-independent sparse visual writing
    query-independent current-state buffer

Answer-time stage:
    lightweight state access / retrieval
```

也就是说，query 没有参与：

```text
哪些帧被 ViT 编码
哪些视觉 token 被写入 KV/cache
```

query 只参与回答阶段“访问哪些已写入视觉块”。为了让这个边界更清楚，本轮新增：

```text
--query_retrieval_policy internal
--query_retrieval_policy latest_recent
--query_retrieval_policy always_recent
```

其中：

- `latest_recent`：只有 latest/current/setting 类问题访问 recent blocks。
- `always_recent`：所有问题都固定访问最近 N 个视觉块，不依赖 query 内容。

`always_recent` 是更保守、更即插即用的 query-decoupled baseline。

## 3. 实验设置

模型和数据：

```text
Model: llava_ov_7b
Dataset: RVS-Ego open QA subset
Sample FPS: 0.2
Videos / QA: 8 / 24
Dense baseline: r1/t0
```

对比配置：

| 配置 | Admission | Access |
|---|---|---|
| Semantic + latest recent | r64/t0.3 + recency K=4 | latest/current 问题取最近 4 blocks |
| Semantic + always recent | r64/t0.3 + recency K=4 | 所有问题固定取最近 4 blocks |
| Periodic + always recent | refresh=13, threshold=999 + recency K=4 | 所有问题固定取最近 4 blocks |

Periodic baseline 说明：

```text
refresh=13, threshold=999
```

这会近似形成 uniform/periodic admission：基本不靠 drift keep，只按固定间隔和 recency 写入。选择 `13` 是为了让保留帧数接近 semantic 方法的预算。

## 4. 主要结果

全量 8 videos / 24 QA 口径，与 dense repeat0 对比：

| Method | Kept Frames | Token Reduction | Encode Sec | Speedup vs Dense | Mean Token-F1 | W/T/L vs Dense | Route |
|---|---:|---:|---:|---:|---:|---:|---|
| Semantic + latest recent | 157 / 2046 | 92.33% | 51.63 | 3.59x | 0.3076 | 10 / 14 / 0 | 6 latest_recent + 18 internal |
| Semantic + always recent | 157 / 2046 | 92.33% | 51.73 | 3.58x | 0.2894 | 7 / 17 / 0 | 24 always_recent |
| Periodic + always recent | 191 / 2046 | 90.66% | 53.37 | 3.47x | 0.2780 | 7 / 16 / 1 | 24 always_recent |

关键样例 `000005 latest clip`：

| Method | Prediction |
|---|---|
| Semantic + latest recent | The setting portrayed in the latest clip is a kitchen. |
| Semantic + always recent | The setting portrayed in the latest clip is a kitchen. |
| Periodic + always recent | The setting portrayed in the latest clip is a kitchen. |

## 5. LLM Judge 结果

Qwen2.5-VL-7B judge，dense vs method：

| Method | Valid | Dense Correct | Method Correct | Dense-only | Method-only | Both Correct | Both Wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semantic + always recent | 20 / 24 | 60.0% | 75.0% | 0 | 3 | 12 | 5 |
| Periodic + always recent | 19 / 24 | 68.4% | 84.2% | 0 | 3 | 13 | 3 |

解释：

1. 两个 fixed recent access 配置在 judge 上都不差，说明固定 recent access 是一个可行的 query-decoupled answer-time strategy。
2. Periodic baseline 的 judge 分数在这个小集上不低，但它使用了更多写入帧，token reduction 更低，token-F1 也更低。
3. 当前 RVS-Ego 24 题规模偏小，judge 结果不能单独证明 semantic admission 显著优于 periodic admission；但效率和 token-F1 已经显示 semantic gate 更优。

Semantic vs Periodic，同样 `always_recent qrb=4`：

```text
Semantic kept frames: 157 / 2046
Periodic kept frames: 191 / 2046
Semantic token reduction: 92.33%
Periodic token reduction: 90.66%
Semantic mean token-F1: 0.2894
Periodic mean token-F1: 0.2780
Semantic vs Periodic W/T/L: 2 / 21 / 1
Semantic speedup over Periodic encode: 1.03x
```

## 6. 对用户担心的回答

用户担心：

> 如果需要 query 来指导感知过程的计算，会带来延迟，并削弱泛化和即插即用。

本轮结论：

```text
这个担心没有在当前核心方法里发生。
```

原因：

1. 视觉计算仍由 semantic gate 决定，和 query 内容无关。
2. 视觉 token/cache 写入仍由 semantic gate + rolling recency 决定，和 query 内容无关。
3. `always_recent` 证明，即使回答阶段也不使用 query 内容分类，只固定访问最近 blocks，也能修复关键 latest stale case。

因此，最终论文可以把主方法定义为：

```text
Query-decoupled Sparse Semantic State Construction
```

而把 query-aware 部分降级为：

```text
optional query-time access policy
```

更稳妥的主线是：

```text
Online stage:
    query-independent semantic memory admission
    query-independent rolling current-state buffer

Answer stage:
    fixed recent access as plug-and-play default
    query-aware access as optional upper-bound / adaptive variant
```

## 7. 对最终方法优雅性的影响

这轮实验建议我们调整最终表述：

不要把主方法命名为：

```text
Query-aware perception
```

而应命名为：

```text
Query-decoupled Semantic Stream State
```

或：

```text
Sparse Semantic State for Streaming VLMs
```

核心贡献可以写成：

> We construct a query-independent sparse visual state online, consisting of a long-term semantic memory and a rolling current-state buffer. At answer time, the model accesses this compact state with either fixed recent access or lightweight query-time routing, without re-encoding video frames.

这样既保留了 qrb=4 的实验价值，也避免审稿人攻击“query-guided perception 不适合 streaming”。

## 8. Baseline 方向

本轮已经加入第一个可用 baseline：

```text
Periodic/uniform admission + same recent access
```

但要在论文中更有说服力，还需要扩展：

1. Same-budget periodic baseline：更精确匹配 kept frames。
2. Random keep baseline：同预算随机保留，多 seed。
3. Periodic + internal retrieval。
4. Semantic gate + fixed access vs semantic gate + adaptive access。
5. RVS-Movie 上重复同样对比。

当前最有价值的下一步：

```text
在 RVS-Movie 上验证 Semantic + always_recent qrb4 和 Periodic + always_recent qrb4。
```

如果 RVS-Movie 也显示 semantic gate 用更少 token 达到更好 QA/latency，就可以更明确地把 periodic/uniform 作为主 baseline。

## 9. 当前结论

当前设计可以被整理成一个更稳的最终框架：

```text
Query-independent online state construction:
    Sparse semantic memory
    Rolling current-state buffer

Plug-and-play answer-time access:
    Fixed recent access as default
    Optional query-time routing as adaptive enhancement
```

这个框架同时回应：

- 如何避免 query-dependent perception latency；
- 如何保持泛化和即插即用；
- 如何利用 query-aware 实验发现而不让它成为方法主干风险；
- 如何引入 periodic/uniform baseline 来证明 semantic admission 的必要性。

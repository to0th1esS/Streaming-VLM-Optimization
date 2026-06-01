# ViT + REKV 联合流式缓存设计记录

## 目标判断

当前 Turbo-ViT 实验已经证明：

1. ViT 前端存在可利用的跨帧冗余；
2. dual-anchor semantic stability 能比 rolling-only 更稳；
3. token-level reuse 比 frame-level routing 更接近最终方法；
4. segment-aware reuse 能提升低分位保真；
5. 但仅靠 ViT 侧 gather/scatter 稀疏执行，速度收益已经接近当前原型边界。

因此，下一阶段应该开始做联合设计。

但不建议立刻接入完整大模型 benchmark 做大闭环。更稳的顺序是：

```text
先做轻量后端接口验证，
再接入真实 VLM/REKV 推理。
```

原因：

- 当前问题已经从“是否有视觉冗余”转为“视觉冗余如何影响语言侧缓存写入和检索”；
- 如果直接接完整 QA benchmark，误差来源会混在 prompt、采样、模型能力、视频理解难度中；
- 顶会论文需要方法设计清晰，而不是把 ViT trick 和 LLM cache trick 堆在一起。

## 与 REKV 的关系

仓库中的 REKV 代码已经包含流式 KV 管理逻辑：

- `model/attention/kv_cache_manager.py`
  - 维护 local window；
  - 将历史 KV 分块 offload；
  - 用代表向量做 block retrieval；
  - 支持 top-k/chunk_size 检索；
  - 支持异步 CPU/GPU cache 管理。

- `video_qa/rekv_stream_vqa.py`
  - 按时间窗口逐步编码视频；
  - 在每次 QA 前只编码新增视频片段；
  - 后端通过 `encode_video` 累积视觉 token 到语言 KV cache。

这和 Turbo-ViT 的 natural interface 是：

```text
Turbo-ViT 输出每帧/每 segment 的 semantic stability 与 reuse decision；
REKV 根据这些视觉稳定性信号决定哪些视觉 token 写入、保留、检索或跳过。
```

## 为什么现在需要联合设计

仅优化 ViT 编码速度有三个边界：

1. **视觉 selector 已经不再是主瓶颈**
   - v5/v7 中 token selector 只有约 0.2 ms；
   - 继续优化 selector 对总速度帮助很小。

2. **稀疏 ViT 执行受 gather/scatter 限制**
   - token/segment 选择可以改善质量；
   - 但如果后端仍是离散 gather/scatter，GPU 速度收益有限。

3. **流式 VLM 的最终开销不只在 ViT**
   - 视频 token 最终还要进入 LLM prefilling/cache；
   - 如果稳定视觉 token 仍反复写入语言 KV，前端复用收益会被后端吞掉。

所以最终方法应该从：

```text
只做 ViT 编码复用
```

推进为：

```text
视觉语义稳定性驱动的 ViT-LLM 联合缓存复用
```

## 推荐论文方法主线

建议最终方法不要命名成多个工程模块，而收敛为：

```text
Dual-Anchor Semantic Stability Guided Streaming VLM
```

核心抽象：

```text
每个视觉 token/segment 都有一个由 rolling anchor 和 long anchor 共同定义的 semantic stability。
这个 stability 同时决定：
1. ViT 中是否重算；
2. 视觉 token 是否写入 LLM cache；
3. 已有视觉 KV 是否复用；
4. 历史视觉 segment 是否需要被 REKV 检索。
```

这样设计更符合 AAAI/AI 顶会口味：

- 一个核心概念贯穿前后端；
- 不是“ViT 加速 + REKV 加速”的拼接；
- 能解释为什么 rolling-only 不安全；
- 能解释为什么固定 keyframe refresh 不够；
- 能解释为什么 segment 比孤立 token 更适合流式视频。

## 最小联合验证闭环

下一步不要直接跑完整大 QA。

建议先做三个小实验：

### 实验 A：视觉 token 写入量模拟

输入：

- Turbo-ViT v5/v7 的每帧 decision；
- dynamic ratio；
- segment count；
- semantic stability。

输出：

- dense 视觉 token 写入量；
- Turbo-ViT token 写入量；
- segment-aware token 写入量；
- 理论 LLM prefill token reduction。

目的：

```text
证明前端稳定性不仅能减少 ViT 计算，
还能减少进入后端 cache 的视觉 token 流量。
```

### 实验 B：REKV block retrieval 与视觉 stability 对齐

输入：

- 每帧/segment 的 semantic stability；
- REKV block retrieval 选择结果。

分析：

- 高稳定视觉 segment 是否更少被检索；
- 低稳定 segment 是否更常被检索；
- rolling/long anchor stability 是否能预测 REKV 需要保留的块。

目的：

```text
把 Turbo-ViT 的视觉稳定性和 REKV 的缓存检索建立联系。
```

### 实验 C：小规模真实 VLM QA sanity check

只在 A/B 之后做。

设置：

- 小视频数；
- 小问题数；
- greedy decoding；
- 比较 dense visual input、Turbo-ViT visual input、Turbo-ViT + cache policy。

记录：

- ViT latency；
- LLM prefill/cache latency；
- total latency；
- generated answer exact/string similarity；
- visual token cache size；
- REKV retrieved block count。

目的：

```text
证明联合设计不会只优化前端，而能影响端到端流式 VLM。
```

## 下一步代码建议

先实现一个轻量模块：

```text
experiments/turbovit_v1/eval/cache_policy_sim.py
```

功能：

- 读取 v5/v7 的 `v*_latency.csv` 与 `decision_summary.csv`；
- 根据 frame decision 和 dynamic ratio 估计视觉 token 写入量；
- 对 segment-aware 方法，额外使用 segment count / segment expansion ratio；
- 输出 csv/json：
  - dense_visual_tokens;
  - rewritten_visual_tokens;
  - reused_visual_tokens;
  - estimated_llm_prefill_reduction;
  - estimated_kv_cache_reduction;

再实现：

```text
experiments/turbovit_v1/scripts/run_cache_policy_sim.py
```

这样我们可以先在本地/远程快速验证联合设计的上界，再决定是否接真实 VLM。

## 当前结论

可以开始接入后端大模型方向，但方式应谨慎：

```text
先做 REKV-style cache policy simulation，
再做真实 VLM 小规模 QA，
最后再做大规模 benchmark。
```

这会让论文路线更自然：

```text
不是先提出一个复杂系统，
而是从前端语义稳定性出发，
逐步推出视觉 segment 复用与语言 KV cache 管理的统一机制。
```

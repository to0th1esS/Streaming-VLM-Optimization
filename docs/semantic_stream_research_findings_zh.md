# 流式语义流实验发现与下一步研究方向

更新日期：2026-06-02

本文档用于把目前已经完成的 Turbo-ViT / Semantic Stream 实验，从“科学研究发现”的角度重新梳理。它不是逐次实验流水账，而是把已有尝试压缩为可指导下一阶段方法设计的证据链：我们最初想验证什么，实验揭示了什么，哪些假设被修正，最终方法应该往哪里收敛。

---

## 1. 当前研究目标的重新表述

最初目标是：在流式视频 VLM 中利用 ViT 编码阶段的帧间冗余，通过“参考帧全量编码 + 后续帧选择性更新”减少视觉前端计算。

经过多轮实验后，目标需要提升一层：

> 不再以逐帧视觉特征完全复现 dense ViT 为核心目标，而是以 QA 性能基本不下降为约束，把密集视觉流转化为稀疏语义流，从而同时减少视觉计算、视觉 token 写入、上下文/cache 压力和后端检索负担。

这个修正非常关键。因为流式视频 VLM 的瓶颈不只在 ViT encoder，也在长时间视频带来的视觉 token 写入、prefill、KV cache 和 retrieval/memory 管理。真正有论文价值的方向不是“只让 ViT 快一点”，而是提出一个统一的语义稀疏化原则，让前端计算和后端上下文管理共享同一个稳定性判断。

---

## 2. 初始假设与被实验修正的地方

### 2.1 初始假设

早期 Turbo-ViT-v1 假设如下：

1. 相邻视频帧在 ViT hidden states / key projections 上存在大量冗余。
2. 可以在每层判断当前 token 与 reference token 的相似度。
3. 对动态 token 重新计算 attention + MLP，对静态 token 复用 reference hidden state。
4. 这样可以显著降低 ViT 编码成本。

这个假设在“语义冗余存在”这一点上是正确的，但在“逐层逐 token 稀疏重算是主要速度来源”这一点上被真实模型实验修正了。

### 2.2 关键修正

目前实验给出的修正是：

1. 冗余确实存在，但 GPU 上非连续 token gather / scatter / 混合 K/V 构造会吞掉大部分理论收益。
2. 严格追求 feature cosine / MSE 会过早否定很多 QA 上可接受甚至更有效的速度优先策略。
3. 对流式 VLM 而言，更重要的是保留语义事件，而不是保留每一帧的视觉特征细节。
4. 最有效的速度来源目前不是 token-level sparse recomputation，而是低成本 anchor-conditioned semantic routing：判断当前帧是否带来新的语义事件，若没有则跳过 ViT 或跳过视觉 token 写入。

---

## 3. 实验发现一：视频帧间冗余真实存在，但逐 token 重算不是理想核心

早期 dense baseline、layer-wise hook、synthetic/TinyViT 和 CLIP/ViT 实验都确认：视频相邻帧、尤其同一镜头内的相邻帧，在 ViT 中间特征和输出特征上存在可观冗余。这支撑了“流式视频不应被当成独立图像序列处理”的基本动机。

但当我们把 per-layer / per-token sparse update 放到真实 ViT 上后，速度表现不理想。

代表性结果：

| 方法 | 配置 | 加速 | 特征保真度 | 现象 |
|---|---:|---:|---:|---|
| v7 segment reuse | ratio 0.80->1.00 | 1.000x | mean cosine 0.994970 | 质量高，但几乎无速度收益 |
| v8 layer-KV reuse | 同配置 | 0.995x | mean cosine 0.994917 | 理论合理，但无实际加速 |
| v8 scatter mode | 多种配置 | 0.891x-0.929x | 有下降 | scatter / mixed cache 开销明显 |

在真实 LLaVA-OneVision/SigLIP 迁移中，这个问题更明显：

| 方法 | 配置 | 加速 | mean cosine | 现象 |
|---|---:|---:|---:|---|
| v7 保守稀疏 | 接近 dense | 0.789x | 0.9993 | 保真但更慢 |
| v7 forced sparse | 强制稀疏 | 0.518x | 0.596 | 既慢又漂移大 |
| v7 skip/sparse mixed | 连续 skip/sparse | 0.606x | 0.6997 | 稀疏帧反而比 dense 帧慢 |

结论：

> 逐层逐 token 动态判别可以作为分析工具，但不适合作为当前最终系统的主速度来源。

这不是否定 ViT 内部冗余，而是说明在真实 GPU 推理框架中，细粒度稀疏计算必须面对 kernel 形状、memory movement、scatter 写回、attention cache 混合等工程代价。若没有专门稀疏 kernel 或 block-sparse kernel，直接用 PyTorch 做 token-level sparse update 很难释放理论收益。

论文层面的 insight 应该表述为：

> Temporal redundancy is abundant, but exploiting it at a too fine granularity creates routing and memory-movement overhead that cancels the benefit. Therefore, streaming VLM acceleration should route semantic segments/events rather than recompute arbitrary individual tokens at every layer.

---

## 4. 实验发现二：feature cosine / MSE 不是最终目标，QA-first 才是正确约束

早期实验中，我们把 dense ViT feature cosine / MSE 看得过重。这带来了一个偏差：只要某个方法让最终视觉特征偏离 dense，就倾向于认为它不可用。

后续 QA 实验表明，这个判断过于保守。

### 4.1 BBB tiny QA 结果

在 Big Buck Bunny 小规模 QA 上：

| 方法 | encoded frames | token reduction | encode time | QA |
|---|---:|---:|---:|---:|
| Dense 0.5B, 1fps | 16/16 | 0% | 约 0.572s | 3/3 |
| Semantic write gate, N=4,t=0.01 | 7/16 | 56.25% | 主要减少写入 | 3/3 |
| Semantic write gate, N=8,t=0.03 | 6/16 | 62.5% | 主要减少写入 | 3/3 |
| Semantic compute+write gate, 7B | 5/16 | 68.75% | 约 0.503s | 3/3 |

7B dense 当前约 0.771s，semantic compute+write gate 约 0.503s，对应约 1.53x 加速，latency reduction 约 34.8%，QA 仍保持 3/3。

这说明：即使视觉特征并不逐帧完全复原，只要语义事件被保留，QA 可以不下降。

### 4.2 BBB hard QA 结果

我们构造了更细粒度的 60s 事件型 hard QA，共 8 个问题：

| 方法 | kept frames | token reduction | encode time | speedup | QA |
|---|---:|---:|---:|---:|---:|
| Dense 7B | 56/56 | 0% | 2.832s | 1.00x | 7/8 |
| r8,t0.03 | 12/56 | 78.6% | 1.296s | 2.19x | 7/8 |
| r16,t0.1 | 8/56 | 85.7% | 0.943s | 3.00x | 7/8 |
| r64,t0.3 | 6/56 | 89.3% | 0.822s | 3.45x | 8/8 |

这里出现了非常重要的现象：更激进的语义稀疏化不仅没有必然降低 QA，某些配置还可能让回答更聚焦。单视频结果不能过度声称“提升精度”，但它足以推翻“feature fidelity 越高越好”的单一评价标准。

研究结论：

> Dense visual feature reconstruction is an over-constrained objective for streaming QA. A better objective is semantic event preservation under a QA-performance constraint.

---

## 5. 实验发现三：低成本 AnchorGate 是当前最强速度来源

在真实 LLaVA-OneVision/SigLIP 上，纯 skip/AnchorGate 明显优于 per-token sparse recomputation。

代表性 v9 AnchorGate 结果：

| threshold | dense/skip | mean cosine | speedup | latency reduction |
|---:|---:|---:|---:|---:|
| 0.030 | 4/28 | 0.757 | 3.987x | 74.9% |
| 0.010 | 12/20 | 0.8935 | 2.150x | 约 53.5% |
| 0.005 | 16/16 | 0.9333 | 1.645x | 约 39.2% |

其中 embedding / gate cost 只有毫秒级甚至亚毫秒级，远低于完整 ViT 编码。它说明对流式视频而言，最有价值的决策不是“这一层哪些 token 要重算”，而是更高层次的：

1. 当前帧是否包含新的语义事件？
2. 如果没有，是否可以直接复用 anchor 表征？
3. 如果有，是否需要刷新 anchor 或写入新的视觉 token？
4. 写入多少 token 才足够支撑后续 QA？

这使方法主线从 Turbo-ViT-v1 自然演进为 Semantic Stream：

> 用低成本语义稳定性判别，把 dense visual frame stream 转换为 sparse semantic event stream。

---

## 6. 实验发现四：计算加速与选择性 token 保留应该协同，而不是分开做

目前最重要的系统设计原则是：同一个 semantic gate 同时控制前端计算和后端写入。

### 6.1 计算侧

对稳定帧：

1. 不运行完整 ViT。
2. 复用最近 anchor / reference 的视觉特征。
3. 只付出低成本 embedding signature 或浅层 probe 的判断开销。

对动态帧：

1. 执行完整 ViT。
2. 更新 anchor。
3. 把该帧作为新的语义事件写入后端上下文。

### 6.2 写入侧

对稳定帧：

1. 不重复写入视觉 token。
2. 减少 LLM prefill token 数。
3. 减少 KV cache 增长。
4. 降低 ReKV / retrieval memory 的冗余候选。

对动态帧：

1. 写入关键视觉 token 或整帧视觉表征。
2. 给后端保留真正代表事件变化的视觉上下文。

因此，方法不是“ViT 加速模块 + cache 压缩模块”的简单堆叠，而可以被设计为一个统一框架：

> Anchor-conditioned Semantic Stream Routing.

它把视觉计算、视觉 token 写入、上下文缓存、后端检索都绑定到同一个语义稳定性变量上。这种统一性更符合顶会论文需要的方法美感。

---

## 7. 实验发现五：真实长视频更能放大方法收益

BBB 是 sanity check，RVS-Movie 才开始触摸真实流式视频的困难度。

当前 RVS-Movie 数据状态：

| 数据集 | 可用视频 | 问题数 | 状态 |
|---|---:|---:|---|
| RVS-Movie | 8/8 | 24 | 已下载、解压、链接 |
| RVS-Ego | 0/8 | 24 | 仍在下载 |
| BBB hard QA | 1/1 | 8 | 已可用 |

RVS-Movie 8-video subset，sample_fps=0.2：

| 方法 | mean token-F1 | total encode | kept frames | token reduction | speedup |
|---|---:|---:|---:|---:|---:|
| Dense 7B | 0.0540 | 62.32s | 811/811 | 0% | 1.00x |
| Semantic r16,t0.1 | 0.0694 | 11.18s | 119/811 | 85.3% | 5.57x |
| Semantic r64,t0.3 | 0.0504 | 8.16s | 56/811 | 93.1% | 7.64x |

这个结果有两层含义。

第一，长视频使收益显著放大。相比 BBB 的 3.00x-3.45x，RVS-Movie 已经出现 5.57x-7.64x 的视觉编码侧加速。这符合我们的核心判断：流式场景越长，冗余帧越多，语义流稀疏化越有价值。

第二，当前 token-F1 只是粗糙代理指标。Dense 本身在 RVS-Movie 上也很弱，说明任务难度、采样率、模型能力和评价方式都会影响结论。不能用 token-F1 直接作为最终 paper 的 QA 指标，需要引入更强的 LLM judge、官方 evaluator 或更结构化的 QA 评测规则。

但从方向判断上，RVS-Movie 的结果已经足够说明：

> 真正值得继续推进的是长流式视频上的 semantic event sparsification，而不是短片段上的 feature reconstruction。

---

## 8. 与 STC 工作的关系和超越点

STC 类工作报告了明显的端到端收益，例如已有记录中提到：

| 工作/设置 | ViT encoding latency | LLM prefilling latency |
|---|---:|---:|
| STC-Cacher | 下降约 24.5% | 下降约 45.3% |

我们当前结果和 STC 的关系如下：

1. 在纯 ViT speed-first 设置下，v5 已经能达到 1.364x speedup，即 latency reduction 约 26.7%，与 STC 的 ViT 侧收益同量级。
2. 在真实 OneVision/SigLIP 上，AnchorGate 可以达到 2x-4x 的视觉侧加速，但 feature fidelity 会下降。
3. 在 QA-first 设置下，BBB hard QA 达到 3.00x-3.45x，RVS-Movie 达到 5.57x-7.64x 的视觉编码侧加速，并伴随 85%-93% 视觉 token reduction。

我们的潜在超越点不应只是“某个 latency 数字更高”，而是：

1. 从 frame/token compression 提升到 semantic stream construction。
2. 同一个 gate 同时作用于视觉计算和后端 token/cache 写入。
3. 用 QA 约束而不是 dense feature reconstruction 约束方法。
4. 在更长、更真实的流式视频中展示大幅收益。
5. 通过实验解释为什么过细粒度的 token sparse update 不是最优路径，从而引出更优雅的 anchor-conditioned routing。

---

## 9. 当前方法主线的收敛版本

当前最合理的 paper 方法雏形如下。

### 9.1 核心对象

把输入视频看作 dense visual frame stream：

```text
F_1, F_2, ..., F_T
```

方法输出 sparse semantic event stream：

```text
E_1, E_2, ..., E_K,  K << T
```

每个事件 E_i 对应一个被保留的 anchor frame / semantic keyframe，以及必要的视觉 token。

### 9.2 核心决策

对每个新帧，计算低成本 signature，并与当前 anchor 比较：

```text
drift_t = D(signature(F_t), signature(anchor))
```

若 drift 小于阈值：

```text
skip ViT compute
skip visual token write
reuse anchor representation
```

若 drift 大于阈值或达到强制刷新间隔：

```text
run ViT
write visual tokens
update anchor
```

### 9.3 当前默认配置

基于现有证据：

| 配置 | 定位 |
|---|---|
| r16,t0.1 | 当前稳定默认；RVS-Movie 5.57x，token reduction 85.3%，token-F1 高于 dense proxy |
| r64,t0.3 | 速度上界；RVS-Movie 7.64x，token reduction 93.1%，质量风险更高 |
| r8,t0.03 | 保守 QA sanity 配置；BBB 2.19x，QA 不下降 |

### 9.4 暂不作为主线的部分

以下方向不应作为下一阶段主线，除非有新的失败案例迫使我们回到它们：

1. 每层逐 token sparse recomputation。
2. 高保真 feature cosine 最大化。
3. 复杂的 ViT 内部 mixed K/V cache 构造。
4. 在没有强 QA evaluator 前过度优化 open-ended token-F1。

---

## 10. 当前证据强度与不足

### 10.1 已经较强的证据

1. 流式视频存在大量语义冗余。
2. 逐 token 稀疏重算在现有实现下不如高层语义路由有效。
3. QA-first 目标比 feature reconstruction 更符合流式 VLM。
4. Semantic compute+write gate 可以同时减少 ViT 编码和视觉 token 写入。
5. 长视频真实数据能显著放大收益。

### 10.2 仍然薄弱的证据

1. BBB hard QA 只有单视频，不能支撑最终论文结论。
2. RVS-Movie 当前只用了 token-overlap F1，评价较粗糙。
3. Dense baseline 在 RVS-Movie 上质量也弱，需要更合适的模型、采样率或评价方式。
4. RVS-Ego / StreamingBench / OVO-Bench 尚未形成完整结果。
5. ReKV/cache memory savings 还停留在设计层和 token reduction proxy，缺少真实 cache 统计。

---

## 11. 下一步方向判断

### 11.1 第一优先级：补强 QA 评价，而不是继续追 feature cosine

下一步最需要的是把“QA 基本不下降”变成可信证据。

建议执行：

1. 等 RVS-Ego 下载完成，完成 asset link 和可用性检查。
2. 在 RVS-Movie 上对 dense、r16,t0.1、r64,t0.3 做 repeats。
3. 引入更强评价：
   - 规则型 evaluator 用于可结构化问题；
   - LLM judge 用于 open-ended QA；
   - 若 benchmark 有官方 evaluator，优先接入。
4. 输出每个配置的：
   - QA score；
   - encode latency；
   - visual token reduction；
   - kept frame ratio；
   - 每视频 breakdown。

判据：

> 若 r16,t0.1 在更强 QA evaluator 下与 dense 持平或轻微下降，但维持 5x 左右视觉侧加速和 80%+ token reduction，则可作为主方法默认配置。

### 11.2 第二优先级：把 ReKV/cache savings 做成真实指标

当前我们已经证明视觉 token 写入减少，但还需要证明这会转化为后端收益。

建议记录：

1. 输入 LLM 的 visual token 数。
2. prefill latency。
3. KV cache size。
4. ReKV 检索候选数。
5. memory read/write 次数。
6. 问题时刻可访问的有效视觉事件数。

判据：

> 如果 semantic stream 同时减少 prefill latency 和 KV/cache footprint，而 QA 不下降，就能与 STC/REKV 类工作形成直接对比。

### 11.3 第三优先级：只有在单 anchor 失败时再引入 dual/rolling correction

dual-anchor 或 rolling-anchor correction 是合理方向，但不应过早复杂化。它应该由失败案例自然引出。

适合引入 dual/rolling 的条件：

1. 长视频中单 anchor 因 stale 导致事件漏检。
2. RVS-Ego 这种第一视角剧烈运动视频出现明显误 skip。
3. r16,t0.1 QA 明显低于 dense，而 r8,t0.03 又太慢。
4. drift 曲线显示 frame-level gate 对缓慢变化不敏感。

若触发这些条件，可以设计：

1. short-term rolling anchor：捕捉最近局部变化；
2. long-term stable anchor：保持语义背景；
3. correction trigger：当二者分歧超过阈值时执行 ViT refresh。

论文表达上不要把它写成工程补丁，而要写成：

> Streaming video contains both slow semantic evolution and abrupt event changes; a single anchor cannot model both, so we maintain complementary anchors for stability and novelty.

### 11.4 第四优先级：重新考虑 ViT 内部层间特性，但只作为轻量 routing probe

目前不建议回到每层 token 判别。但可以利用 ViT 层间特性减少判断成本：

1. 只在 patch embedding 或浅层 probe 计算 semantic drift。
2. 每 N 帧做一次更强判别，中间帧只做 cheaper gate。
3. 对不同视频段自适应调整 refresh interval。
4. 在必要时只做 block-level correction，而不是 token-level scatter。

这个方向可以作为最终方法的轻量增强，而不是主干。

---

## 12. 论文叙事建议

最终论文不需要展示所有中间尝试。中间尝试应被转化为 insight。

建议叙事：

1. 流式 VLM 的核心浪费不是单帧视觉理解，而是把密集帧序列当作密集语义序列处理。
2. 我们首先观察到 ViT 特征具有跨帧冗余，但细粒度 token sparse update 在真实 GPU 上不释放收益。
3. 进一步发现，dense feature fidelity 不是 streaming QA 的必要条件；保留语义事件才是关键。
4. 因此提出 Semantic Stream Routing：用 anchor-conditioned stability gate 把 dense visual stream 转化为 sparse semantic event stream。
5. 该语义流同时控制视觉计算和视觉 token/cache 写入，从而获得端到端收益。
6. 在长流式视频上，方法大幅减少视觉计算和上下文写入，同时保持 QA 性能。

可以避免的表述：

1. “我们尝试了很多工程技巧，最后发现 skip 快。”
2. “我们为了加速牺牲了特征保真度。”
3. “我们只是 STC 的另一个压缩模块。”

更好的表述：

1. “Streaming video should be represented as sparse semantic events rather than dense visual frames.”
2. “Feature reconstruction is unnecessarily strict for streaming QA; semantic event preservation is the right constraint.”
3. “A unified semantic gate couples visual computation with memory/cache writing.”

---

## 13. 当前最明确的下一步实验清单

短期应按下面顺序推进：

1. 检查 RVS-Ego 下载是否完成，完成 link 和 asset check。
2. 对 RVS-Movie 跑 repeats：
   - dense；
   - r16,t0.1；
   - r64,t0.3；
   - 可选 r8,t0.03。
3. 增加 LLM judge 或官方 evaluator，减少 token-F1 的评价偏差。
4. 输出 QA-latency-token 三维 trade-off 表。
5. 加入 prefill / KV cache / ReKV memory 统计。
6. 若 RVS-Ego 或长视频出现单 anchor 漏事件，再开始 dual-anchor / rolling correction。

当前方向性结论：

> 我们已经不应继续把主要资源放在“更高 feature cosine 的 ViT 稀疏重算”上。下一阶段应转向“QA-constrained Sparse Semantic Stream”：用语义事件级 gating 同时驱动视觉计算跳过与视觉 token/cache 写入压缩，并在更真实的流式 QA benchmark 上验证端到端收益。


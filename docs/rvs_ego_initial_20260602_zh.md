# RVS-Ego 初始实验记录

日期：2026-06-02

本文档记录 RVS-Ego 真实第一视角流式 QA 小子集的资产补齐、链接检查和第一轮 dense / semantic stream 对比实验。它的主要作用是判断第一视角视频是否会快速暴露单 anchor semantic gate 的失败模式，并为后续是否引入 dual-anchor / rolling-anchor correction 提供依据。

---

## 1. 实验目的

RVS-Movie 的重复实验已经证明，在电影场景视频中，Semantic Stream 可以稳定减少视觉编码和视觉 token 写入。但 MovieNet 视频通常镜头更稳定、语义结构更接近叙事片段。

RVS-Ego 使用 Ego4D 第一视角视频，具有更强的手部操作、视角抖动、局部物体变化和连续动作流程。因此本轮实验重点回答：

1. RVS-Ego 数据资产是否已经可用；
2. 第一视角视频是否让 dense baseline 本身更难；
3. `refresh=16, threshold=0.1` 是否仍能保持较强速度收益；
4. `refresh=64, threshold=0.3` 是否会在第一视角场景中明显漏事件；
5. 是否已经有必要开始 dual-anchor / rolling-anchor correction。

---

## 2. 数据资产状态

RVS-Ego 三个分片已全部补齐：

| file | bytes | status |
| --- | ---: | --- |
| ego4d_frames_online.partaa | 9663676416 | complete |
| ego4d_frames_online.partab | 9663676416 | complete |
| ego4d_frames_online.partac | 9119200328 | complete |

解压结果：

```text
/home/mllm/datasets/vstream_qa/frames/rvs_ego/.extract_complete
/home/mllm/datasets/vstream_qa/frames/rvs_ego/ego4d_frames/
```

链接检查：

| dataset | videos | questions | available videos | missing |
| --- | ---: | ---: | ---: | ---: |
| RVS-Ego subset | 8 | 24 | 8/8 | 0 |
| RVS-Movie subset | 8 | 24 | 8/8 | 0 |
| BBB hard QA | 1 | 8 | 1/1 | 0 |

RVS-Ego 链接入口：

```text
data/rvs/ego/ego4d_oe.json
data/rvs/ego/videos/000000.mp4
...
data/rvs/ego/videos/000007.mp4
```

这里的 `.mp4` 是兼容现有 annotation 的软链接目录入口，内部实际是 Ego4D frame images。

---

## 3. 实验设置

```text
server: remote-docker
GPU: NVIDIA A100-SXM4-80GB
model: LLaVA-OneVision-Qwen2-7B
dataset: RVS-Ego subset
videos: 8
questions: 24
sample_fps: 0.2
repeats: 1
annotation: data/rvs/ego/ego4d_oe.json
remote result dir: results/rvs_ego_initial_20260602/
```

运行配置：

| config | refresh interval | skip threshold | compute gate | 定位 |
| --- | ---: | ---: | --- | --- |
| Dense-equivalent | 1 | 0.0 | true | 每帧强制 refresh，作为 dense 视觉编码基线 |
| Semantic r16/t0.1 | 16 | 0.1 | true | 当前稳定默认配置 |
| Semantic r64/t0.3 | 64 | 0.3 | true | 速度优先配置 |

---

## 4. 主要结果

Open-QA overlap evaluator，完整聚合口径：

| method | mean token-F1 | total encode | kept frames | token reduction | speedup | latency reduction | rule proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense-equivalent | 0.2273 | 185.55s | 2046/2046 | 0.0% | 1.00x | 0.0% | 18/24 |
| Semantic r16/t0.1 | 0.2529 | 58.44s | 318/2046 | 84.5% | 3.17x | 68.5% | 18/24 |
| Semantic r64/t0.3 | 0.2489 | 50.22s | 127/2046 | 93.8% | 3.69x | 72.9% | 19/24 |

`run_semantic_stream_sweep.py` final-row 口径：

| method | final-row input frames | final-row kept frames | token reduction | encode |
| --- | ---: | ---: | ---: | ---: |
| Dense-equivalent | 492 | 492 | 0.0% | 45.55s |
| Semantic r16/t0.1 | 492 | 81 | 83.5% | 15.06s |
| Semantic r64/t0.3 | 492 | 32 | 93.5% | 11.29s |

和 RVS-Movie 一样，本文主分析采用 open-QA evaluator 的完整聚合口径，因为它覆盖全部 24 个 QA 样本和 8 个视频。

---

## 5. 现象分析

### 5.1 RVS-Ego 明显比 RVS-Movie 更重

在同样 `sample_fps=0.2` 下：

| dataset | dense input frames | dense total encode |
| --- | ---: | ---: |
| RVS-Movie | 811 | 65.55s |
| RVS-Ego | 2046 | 185.55s |

Ego 的输入帧数和 dense 编码时间都明显更高，说明第一视角视频对流式 VLM 的视觉处理压力更大。这正是 Sparse Semantic Stream 应该发挥价值的场景。

### 5.2 单 anchor 没有立即崩坏

我们担心第一视角剧烈运动会导致单 anchor gate 漏掉动作变化。但初始结果没有看到明显崩坏：

1. r16/t0.1 的 token-F1 proxy 高于 dense；
2. r64/t0.3 的 token-F1 proxy 也高于 dense；
3. rule proxy 中 r64/t0.3 为 19/24，高于 dense 的 18/24；
4. 两个 semantic 配置都大幅减少视觉 token 写入。

这不能说明 semantic 配置真的比 dense 更准，因为当前 evaluator 仍然粗糙；但它至少说明在这个小子集上，单 anchor 没有出现肉眼可见的任务崩坏。

### 5.3 Ego 上速度收益低于 Movie，但仍然显著

RVS-Movie repeats：

```text
r16/t0.1: 5.86x
r64/t0.3: 8.59x
```

RVS-Ego initial：

```text
r16/t0.1: 3.17x
r64/t0.3: 3.69x
```

Ego 的 speedup 较低，可能原因包括：

1. 第一视角动作变化更频繁，gate 保留更多帧；
2. Ego 视频更长，部分视频的加载和帧处理开销更重；
3. r64/t0.3 虽然只保留 127/2046 帧，但每个被保留的视频段可能更长或视觉编码耗时更不均匀；
4. 当前实现仍有 frame loading、signature、I/O 和 LLM wrapper 开销，并非纯 ViT kernel benchmark。

这反而提供了一个有价值的论文切入点：不同视频类型的语义变化率不同，Semantic Stream 应该能够自适应事件密度。

### 5.4 r16 仍是默认稳定配置，r64 是速度优先配置

在 RVS-Ego 上，r16/t0.1 保留 318/2046 帧，token reduction 84.5%，加速 3.17x。这个配置质量 proxy 稳定，仍然适合作为默认方法配置。

r64/t0.3 保留 127/2046 帧，token reduction 93.8%，加速 3.69x。它比 r16 更激进，但在当前 proxy 下没有明显掉点，适合继续作为 speed mode。

### 5.5 暂时还不需要立刻引入 dual-anchor

dual-anchor / rolling-anchor correction 的动机是解决单 anchor stale 或漏事件。如果 RVS-Ego 初始实验显示 r16/r64 大幅掉点，就应该马上进入 dual-anchor。

但当前结果是：

```text
semantic QA proxy >= dense QA proxy
token reduction > 84%
speedup > 3x
```

因此现在更合理的下一步不是马上增加方法复杂度，而是：

1. 先做 RVS-Ego repeats；
2. 引入更强 QA evaluator；
3. 定位具体失败样本；
4. 只有在明确发现单 anchor 漏事件时，再用 dual-anchor / rolling-anchor correction 做针对性修复。

---

## 6. 对研究方向的影响

本轮 RVS-Ego 初始结果进一步支持当前主线：

```text
Dense visual streams should be converted into sparse semantic event streams.
The objective is QA-constrained semantic preservation, not dense feature reconstruction.
```

更具体地说：

1. RVS-Ego 证明该方向不只适用于 MovieNet 叙事视频，也能初步迁移到 Ego4D 第一视角视频。
2. 第一视角视频的收益低于 Movie，但仍然明显，说明事件密度会影响最优 gate 策略。
3. 未来方法可以引入 adaptive threshold / adaptive refresh，但必须由错误分析驱动，而不是提前堆复杂模块。
4. 统一控制 ViT compute 和 visual token writing 的设计仍然成立。

---

## 7. 下一步

1. 对 RVS-Ego 跑 3 次 repeats，确认 r16/t0.1 和 r64/t0.3 的稳定性。
2. 把 RVS-Movie 与 RVS-Ego 的结果合并成一个 QA-latency-token trade-off 表。
3. 接入 LLM judge 或官方 evaluator，替代 token-F1 proxy。
4. 增加 per-video breakdown，定位哪些视频/问题受 aggressive skip 影响。
5. 在失败样本上分析 drift 决策序列，判断是否需要 dual-anchor 或 rolling correction。
6. 开始记录 ReKV/cache 真实指标，把 token reduction 转化为 prefill latency、KV cache memory 和 retrieval cost 证据。


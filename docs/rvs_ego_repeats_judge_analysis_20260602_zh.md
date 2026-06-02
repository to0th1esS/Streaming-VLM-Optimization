# RVS-Ego Repeats + LLM Judge + Failure Analysis

日期：2026-06-02

本文档记录 RVS-Ego 小子集的 3 次重复实验、Qwen2.5-VL-7B judge 精度评估，以及第一版按问题类型的 failure analysis。它是 `docs/rvs_ego_initial_20260602_zh.md` 和 `docs/llm_judge_results_20260602_zh.md` 的后续验证。

---

## 1. 实验目的

前一轮 RVS-Ego 初始实验只跑了单次，结果显示 Semantic Stream 在第一视角视频上没有明显崩坏。但单次结果不足以支撑方向判断。

本轮目标：

1. 对 RVS-Ego 的 dense、r16/t0.1、r64/t0.3 各跑 3 次 repeats；
2. 用完整 2046 帧聚合口径重新计算视觉编码速度和 token reduction；
3. 用 Qwen2.5-VL-7B text-only judge 复核 QA correctness；
4. 按问题类型做 failure analysis；
5. 判断是否已经有必要启动 dual-anchor / rolling-anchor correction。

---

## 2. 实验设置

```text
server: remote-docker
GPU: NVIDIA A100-SXM4-80GB
model: LLaVA-OneVision-Qwen2-7B
dataset: RVS-Ego subset
videos: 8
questions: 24
sample_fps: 0.2
repeats: 3
annotation: data/rvs/ego/ego4d_oe.json
result dir: results/rvs_ego_repeats_20260602/
```

配置：

| config | refresh interval | threshold | 定位 |
| --- | ---: | ---: | --- |
| Dense-equivalent | 1 | 0.0 | 每帧 refresh，作为 dense 视觉流基线 |
| Semantic r16/t0.1 | 16 | 0.1 | 当前稳定默认配置 |
| Semantic r64/t0.3 | 64 | 0.3 | 速度优先配置 |

---

## 3. Repeats 速度与 token 结果

完整 open-QA evaluator 聚合口径，覆盖 2046 个输入帧：

| method | repeats | mean token-F1 | total encode mean | kept frames | token reduction | speedup | latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense-equivalent | 3 | 0.2273 | 185.41s | 2046/2046 | 0.0% | 1.00x | 0.0% |
| Semantic r16/t0.1 | 3 | 0.2529 | 63.72s | 318/2046 | 84.5% | 2.91x | 65.6% |
| Semantic r64/t0.3 | 3 | 0.2489 | 50.46s | 127/2046 | 93.8% | 3.67x | 72.8% |

逐次 total encode：

| method | run 0 | run 1 | run 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| Dense-equivalent | 185.41s | 186.57s | 184.26s | 185.41s |
| Semantic r16/t0.1 | 63.29s | 64.55s | 63.31s | 63.72s |
| Semantic r64/t0.3 | 52.92s | 47.62s | 50.83s | 50.46s |

`run_semantic_stream_sweep.py` final-row 口径：

| method | input frames | kept frames | token reduction | encode mean |
| --- | ---: | ---: | ---: | ---: |
| Dense-equivalent | 492 | 492 | 0.0% | 45.58s |
| Semantic r16/t0.1 | 492 | 81 | 83.5% | 15.94s |
| Semantic r64/t0.3 | 492 | 32 | 93.5% | 12.27s |

注意：最终论文表格应优先使用完整 open-QA evaluator 口径，因为它覆盖全部 24 个 QA 样本和 8 个视频的累计帧。

---

## 4. LLM Judge 精度结果

使用：

```text
judge: /home/mllm/models/Qwen2.5-VL-7B-Instruct
mode: text-only judge
input: dense prediction vs sparse prediction vs reference answer
```

Repeat 0 的 dense vs semantic 逐题对齐结果：

| comparison | dense correct | sparse correct | correctness W/T/L | token-F1 W/T/L |
| --- | ---: | ---: | ---: | ---: |
| Ego dense vs r16/t0.1 | 54.2% | 62.5% | 2 / 22 / 0 | 3 / 21 / 0 |
| Ego dense vs r64/t0.3 | 54.2% | 58.3% | 2 / 21 / 1 | 4 / 19 / 1 |

这里的 correctness W/T/L 定义为：

```text
win  = sparse correct, dense wrong
loss = dense correct, sparse wrong
tie  = both correct or both wrong
```

观察：

1. r16/t0.1 没有 correctness loss，2 个 sparse-only correct。
2. r64/t0.3 有 1 个 correctness loss，2 个 sparse-only correct。
3. r64 的 judged correctness 仍略高于 dense，但已经出现速度优先配置的风险信号。

---

## 5. 问题类型分析

新增脚本：

```text
scripts/summarize_judge_categories.py
```

它基于 question 文本做 heuristic category：

```text
action
object
scene_latest
yes_no
other
```

这是分析工具，不是官方问题类型。

### r16/t0.1

| category | samples | dense acc | sparse acc | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| action | 6 | 33.3% | 50.0% | 1 / 5 / 0 |
| object | 2 | 50.0% | 50.0% | 0 / 2 / 0 |
| other | 6 | 33.3% | 33.3% | 0 / 6 / 0 |
| scene_latest | 6 | 66.7% | 83.3% | 1 / 5 / 0 |
| yes_no | 4 | 100.0% | 100.0% | 0 / 4 / 0 |

r16 没有任何 category 上出现 loss。它继续支持“默认稳定配置”的定位。

### r64/t0.3

| category | samples | dense acc | sparse acc | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| action | 6 | 50.0% | 50.0% | 0 / 6 / 0 |
| object | 2 | 50.0% | 50.0% | 0 / 2 / 0 |
| other | 6 | 33.3% | 50.0% | 1 / 5 / 0 |
| scene_latest | 6 | 66.7% | 66.7% | 1 / 4 / 1 |
| yes_no | 4 | 75.0% | 75.0% | 0 / 4 / 0 |

r64 的唯一 loss 出现在 `scene_latest`。

---

## 6. Failure Case

r64/t0.3 的 correctness loss：

```text
Category: scene_latest
Video: 000005
Question: What setting is portrayed in the latest clip?
GT: A kitchen setting.
Dense: The latest clip shows a kitchen setting with a focus on the person's hands and the kitchen environment.
Sparse r64: The latest clip shows a kitchen setting with a Christmas tree in the background.
```

这个错误很有价值，因为它不是完全错过大语义场景，而是在 aggressive skip 下给 latest-frame query 引入了无依据细节。

它提示：

1. r64 的主要风险可能不是 action 全局理解，而是 recency-sensitive detail；
2. latest-frame / current-state query 可能需要更强的短期视觉锚点；
3. 如果后续大规模实验中类似错误增多，可以自然引出 rolling-anchor correction。

---

## 7. 方向判断

### 7.1 r16/t0.1 继续作为默认配置

r16/t0.1 在 RVS-Ego repeats 中：

```text
speedup: 2.91x
token reduction: 84.5%
judge sparse correct: 62.5%
correctness W/T/L: 2 / 22 / 0
```

它没有 correctness loss，说明作为默认主方法配置是稳的。

### 7.2 r64/t0.3 作为速度优先配置

r64/t0.3：

```text
speedup: 3.67x
token reduction: 93.8%
judge sparse correct: 58.3%
correctness W/T/L: 2 / 21 / 1
```

它提供更强 token/cache 压缩，但出现一个 latest-scene hallucination。它适合作为 speed mode，而不是默认稳态配置。

### 7.3 是否现在启动 dual-anchor

当前还不建议把完整 dual-anchor 作为主方法立刻加入，因为：

1. r16 没有 loss；
2. r64 只有 1 个 loss；
3. dense 本身也有大量错误；
4. 当前 judge 仍是 text-only local proxy。

但现在已经有了第一个真实动机，可以做一个更小、更优雅的下一步：

```text
Recency-aware anchor correction:
maintain a short-term rolling anchor for latest-frame/current-state queries,
while keeping the main semantic anchor for long-range event stream compression.
```

这比直接堆完整 dual-anchor 更自然。它可以从 insight 出发：

> Streaming QA has two semantic needs: long-range event memory and short-term current-state fidelity.

---

## 8. 下一步实验建议

1. 在 r64/t0.3 上加入轻量 recency anchor：
   - 始终保留最近一个局部 anchor；
   - 或对 latest/current-state 类问题追加最近保留帧；
   - 或对最后 K 帧设置更低 skip threshold。
2. 不改变 r16 默认配置，先只在 r64 speed mode 上测试 correction。
3. 评价指标：
   - speedup；
   - token reduction；
   - Qwen2.5-VL judge correctness；
   - scene_latest category loss 是否消失；
   - 是否牺牲太多速度。
4. 若 recency correction 只用很少额外 token 就消除 latest hallucination，则可以作为最终方法的优雅组成部分。


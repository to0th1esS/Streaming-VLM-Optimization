# RVS-Movie 3 次重复实验记录

日期：2026-06-02

本文档记录 RVS-Movie 真实流式 QA 小子集上的 3 次重复实验。它是 `docs/turbovit_v1_experiment_log_zh.md` 中 RVS-Movie 单次实验的后续复核，目标是判断 Semantic Stream 在真实长视频上的速度收益是否稳定，以及当前默认配置和速度上界配置是否值得继续作为论文方法候选。

---

## 1. 实验目的

上一轮 RVS-Movie 单次实验显示：

1. Semantic Stream 在真实长视频上能显著减少视觉编码时间；
2. `refresh=16, threshold=0.1` 看起来是稳定默认配置；
3. `refresh=64, threshold=0.3` 看起来是速度上界配置；
4. token-overlap F1 粗评估下，r16 配置甚至高于 dense proxy，r64 配置略低但没有明显崩坏。

但单次实验不能排除测量波动。因此本轮做 3 次重复实验，重点回答：

1. dense-equivalent baseline 的视觉编码时间是否稳定；
2. r16/t0.1 的速度收益和 token reduction 是否稳定；
3. r64/t0.3 是否能作为速度优先上界；
4. QA proxy 是否出现明显退化；
5. 结果是否进一步支持“QA-constrained sparse semantic stream”这一主线。

---

## 2. 实验设置

```text
server: remote-docker
GPU: NVIDIA A100-SXM4-80GB
model: LLaVA-OneVision-Qwen2-7B
dataset: RVS-Movie subset
videos: 8
questions: 24
sample_fps: 0.2
repeats: 3
annotation: data/rvs/movie/movienet_oe.json
remote result dir: results/rvs_movie_repeats_20260602/
```

运行配置：

| config | refresh interval | skip threshold | compute gate | 定位 |
| --- | ---: | ---: | --- | --- |
| Dense-equivalent | 1 | 0.0 | true | 每帧强制 refresh，作为 dense 视觉编码基线 |
| Semantic r16/t0.1 | 16 | 0.1 | true | 当前稳定默认配置 |
| Semantic r64/t0.3 | 64 | 0.3 | true | 速度上界配置 |

---

## 3. 评价口径

本轮有两个统计口径，需要明确区分。

第一，`run_semantic_stream_sweep.py` 的 `aggregate_summary.csv` 使用每次运行最后一行的 cumulative stats，当前显示 `input_frames=172`。这个口径适合观察脚本内部单次 summary 是否稳定，但不能代表全部视频累计帧数。

第二，`scripts/evaluate_open_qa_overlap.py` 会按全部 24 个 QA 样本和 8 个视频聚合，当前显示 `semantic_input_frames=811`。这个口径更适合与上一轮 RVS-Movie 单次实验对齐。

因此，本文主表采用 open-QA evaluator 的 811 帧全视频聚合口径；同时保留 rule-based pass count 作为 sanity check。

注意：token-overlap F1 只是自动化粗评估，不能作为最终论文 QA 指标。RVS-Movie 的答案经常是开放式语义描述，token-F1 会低估同义改写。后续必须接入 LLM judge 或官方 evaluator。

---

## 4. 主要结果

Open-QA overlap evaluator，3 次重复均值：

| method | mean token-F1 | total encode mean | kept frames | token reduction | speedup | latency reduction | rule proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense-equivalent | 0.0537 | 65.55s | 811/811 | 0.0% | 1.00x | 0.0% | 20/24 |
| Semantic r16/t0.1 | 0.0694 | 11.19s | 119/811 | 85.3% | 5.86x | 82.9% | 21/24 |
| Semantic r64/t0.3 | 0.0504 | 7.63s | 56/811 | 93.1% | 8.59x | 88.4% | 21/24 |

Per-run visual encode time：

| method | run 0 | run 1 | run 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| Dense-equivalent | 65.37s | 65.34s | 65.95s | 65.55s |
| Semantic r16/t0.1 | 11.30s | 11.62s | 10.66s | 11.19s |
| Semantic r64/t0.3 | 7.31s | 7.98s | 7.59s | 7.63s |

`run_semantic_stream_sweep.py` final-row 口径也显示相同趋势：

| method | final-row kept frames | final-row token reduction | final-row encode mean |
| --- | ---: | ---: | ---: |
| Dense-equivalent | 172/172 | 0.0% | 14.58s |
| Semantic r16/t0.1 | 18/172 | 89.5% | 2.01s |
| Semantic r64/t0.3 | 5/172 | 97.1% | 1.22s |

---

## 5. 现象分析

### 5.1 速度收益稳定

3 次重复中，dense 视觉编码总时间约为 65.55s，r16/t0.1 约为 11.19s，r64/t0.3 约为 7.63s。重复间波动很小，说明加速不是偶然的 warmup 或单次测量偏差。

按完整聚合口径：

```text
r16/t0.1 speedup = 5.86x
r64/t0.3 speedup = 8.59x
```

这已经明显超过早期只优化 ViT feature reuse 时的 20%-30% 级收益，也超过 STC 类工作中单独 ViT encoding latency reduction 约 24.5% 的量级。这里的提升来自更高层的语义流稀疏化，而不是逐 token 重算。

### 5.2 r16/t0.1 更适合作为默认主方法配置

r16/t0.1 保留 119/811 帧，减少 85.3% 视觉 token 写入，视觉编码加速 5.86x。token-F1 proxy 为 0.0694，高于 dense-equivalent 的 0.0537；rule proxy 为 21/24，也高于 dense 的 20/24。

这不能直接声称方法提升 QA 精度，因为 token-F1 和 rule proxy 都不够强。但它说明该配置没有出现明显任务崩坏，并且在开放式 QA proxy 下保持了可接受行为。

因此，r16/t0.1 当前应作为默认稳定配置。

### 5.3 r64/t0.3 是明确的速度上界

r64/t0.3 只保留 56/811 帧，视觉 token 写入减少 93.1%，视觉编码加速 8.59x。它的 token-F1 为 0.0504，略低于 dense-equivalent 的 0.0537，但 rule proxy 仍为 21/24。

这说明 r64/t0.3 是很有价值的速度优先配置：它可能不适合作为默认主结果，但适合作为 upper-bound 或 speed mode，用于探索 QA 可接受边界。

### 5.4 Feature fidelity 不是当前最该优化的目标

如果按早期 feature cosine/MSE 标准，r64/t0.3 很可能会被提前否定，因为它跳过了大量视觉帧，必然偏离 dense feature stream。

但从 QA proxy 看，r64/t0.3 仍保留了不少问答可用语义。这进一步支持我们对目标函数的修正：

```text
目标不是复原 dense ViT feature stream，
而是在 QA 性能约束下构造稀疏语义事件流。
```

### 5.5 长视频场景放大收益

BBB hard QA 上 speedup 约为 3.00x-3.45x；RVS-Movie repeats 上稳定达到 5.86x-8.59x。

这说明随着视频长度、采样帧数和视觉 token 数增加，Semantic Stream 的收益会被放大。这个现象非常符合 streaming VLM 的论文定位：越是真实长流式场景，密集帧处理越浪费，语义流稀疏化越重要。

---

## 6. 对论文方法设计的影响

这轮重复实验强化了一个核心判断：

```text
The method should not be presented as a ViT feature approximation method.
It should be presented as a QA-constrained semantic event stream construction method.
```

最终方法应强调：

1. dense frame stream 中并非每一帧都产生新的语义事件；
2. anchor-conditioned semantic gate 判断哪些帧需要进入语义流；
3. 同一个 gate 同时减少 ViT compute 和 visual token/cache writing；
4. 在长流式 QA 中，这种统一路由比逐 token 重算更直接、更稳定、更有系统收益。

这能把之前的工程尝试转化为论文 insight：

| 工程观察 | 论文 insight |
| --- | --- |
| 每层 token 判别开销大 | 过细粒度 routing 会被 memory movement 吞掉收益 |
| feature cosine 约束过严 | streaming QA 需要语义事件保留，而非 dense feature reconstruction |
| skip/AnchorGate 速度显著 | 高层语义路由比 token sparse recomputation 更适合真实 GPU 推理 |
| RVS-Movie 收益大于 BBB | 长流式视频会放大语义稀疏流的系统收益 |

---

## 7. 当前方向性结论

本轮 3 次重复使 RVS-Movie 结果从“单次观察”提升为“稳定趋势”：

```text
r16/t0.1: 5.86x visual encoding speedup, 85.3% token reduction, QA proxy not worse than dense.
r64/t0.3: 8.59x visual encoding speedup, 93.1% token reduction, speed upper bound with quality risk.
```

因此：

1. `r16/t0.1` 应作为当前默认稳定配置；
2. `r64/t0.3` 应作为速度优先配置；
3. 下一步不要继续主攻 feature reconstruction；
4. 应优先补强 QA evaluator 和 cache/retrieval 指标；
5. dual-anchor / rolling-anchor correction 不应现在凭直觉加入，应等待 RVS-Ego 或更强 evaluator 暴露单 anchor 漏事件问题后再引入。

---

## 8. RVS-Ego 下载状态

本轮同时继续尝试补齐 RVS-Ego：

```text
partaa: complete, 9663676416 bytes
partab: complete, 9663676416 bytes
partac: failed during retry, expected 9119200328 bytes, downloaded 1746780687 bytes before consistency failure
```

失败原因是网络下载一致性校验失败。由于 `scripts/download_vstream_assets.py` 原先在导入 `huggingface_hub` 后才设置 `HF_ENDPOINT`，mirror endpoint 可能没有稳定生效；本轮已修正为在 `download_file()` 中先设置 `HF_ENDPOINT` 再导入 `hf_hub_download`。

后续需要重新拉取 `partac`，再解压并链接 RVS-Ego。

---

## 9. 下一步

1. 重新下载 RVS-Ego `partac`，完成解压和 `data/rvs/ego/videos` 链接。
2. 在 RVS-Ego 上复现 dense、r16/t0.1、r64/t0.3 三组结果，判断第一视角剧烈运动是否触发单 anchor 失败。
3. 接入 LLM judge 或官方 evaluator，替换当前 token-F1 proxy。
4. 增加 ReKV/cache 真实统计，包括 visual token count、prefill latency、KV cache block、retrieval candidates 和 memory read/write。
5. 若 RVS-Ego 证明 r16/t0.1 发生漏事件，再开始 dual-anchor / rolling-anchor correction，而不是提前把方法复杂化。


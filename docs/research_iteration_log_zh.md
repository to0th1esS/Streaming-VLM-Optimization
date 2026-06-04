# Streaming VLM 稀疏语义流研究主日志

> 本文件从 2026-06-04 起作为唯一追加型中文主记录。后续每次代码改动、实验、结论修正，都优先追加到这里；其他单独文档只作为原始证据或图表补充，不再默认新建实验记录文件。

## 0. 记录规范

每次新增记录必须包含：

1. `日期 / 提交号 / 远程结果路径`；
2. `实验目的（为什么做）`；
3. `方法改动范围（改了哪些文件，是否影响主模型路径）`；
4. `实验设置（模型、数据集、fps（帧率）、预算、基线）`；
5. `核心指标（保留帧、token reduction（令牌减少）、encode time（编码时间）、speedup（加速）、QA 指标）`；
6. `现象解释（支持了什么，否定了什么）`；
7. `下一步判断（继续、回退、合并、废弃）`。

术语要求：英文术语首次出现或关键位置必须带中文解释，例如 `periodic sampling（周期/均匀采样）`、`budget_topk（预算 Top-K）`、`hybrid selector（混合选择器）`。

## 1. 最初设计目的回看

原始目标不是单纯让 ViT 输出特征尽量接近 dense ViT（密集 ViT），而是：

```text
把 dense visual stream（密集视觉流）转化为 sparse semantic stream（稀疏语义流），
在 QA（问答）性能基本不下降的约束下，最大化流式视觉处理和上下文写入效率。
```

这个目标包含两层收益：

1. **计算收益**：少跑 ViT encoder（ViT 编码器）或少跑高成本视觉处理；
2. **缓存/上下文收益**：少写入 LLM visual context（大模型视觉上下文）和 KV cache（键值缓存）。

因此当前阶段不应继续把 `feature cosine（特征余弦相似度）` 或 `MSE（均方误差）` 当作唯一目标。它们只用于诊断视觉表示漂移，不是最终优化指标。

## 2. 当前方法边界

当前方法只动了 **ViT 前端编码与视觉 token 写入入口**，没有改 LLM（大语言模型）内部结构，也没有改主干模型权重。

### 2.1 核心文件

| 文件 | 作用 | 当前改动性质 |
|---|---|---|
| `model/vit_patch.py` | ViT 编码路径 patch（补丁）入口 | 核心方法入口；负责是否启用稀疏更新、语义门控、两阶段选择 |
| `model/vision_accelerator/semantic_stream.py` | `SemanticStreamGate（语义流门控器）` | 核心选择策略；负责 threshold（阈值）、budget_topk（预算 Top-K）、候选帧二阶段选择和统计 |
| `video_qa/base.py` | 单次 VideoQA（视频问答）运行入口参数 | 参数透传，不是方法本体 |
| `video_qa/run_eval.py` | 数据集级评估入口 | 参数透传，不是方法本体 |
| `scripts/run_semantic_stream_sweep.py` | 小规模参数扫描脚本 | 实验工具；记录和聚合指标 |

### 2.2 代码边界详解

#### `model/vit_patch.py`

关键位置：

- `vit_patch_hf()`：注册所有 ViT 稀疏与语义流参数。
- `vit_output_postprocess（ViT 输出后处理）`：保留了最早要求的输出后处理入口。
- `_encode_video_window_with_semantic_compute_gate()`：当前最核心的运行路径。
- `_raw_rgb_signatures()`：低成本 RGB 签名。
- `_raw_rgb_candidate_indices()`：`hybrid selector（混合选择器）` 的第一阶段候选筛选。

当前有三种选择源：

```text
semantic_selection_feature_source = vit_embedding
semantic_selection_feature_source = raw_rgb
semantic_selection_feature_source = hybrid
```

含义：

| 选择源 | 中文解释 | 是否先跑 ViT embedding | 作用 |
|---|---|---:|---|
| `vit_embedding` | ViT 嵌入选择 | 是 | 语义较强，开销较高 |
| `raw_rgb` | 原始 RGB 选择 | 否 | 极快，但语义较弱 |
| `hybrid` | 混合两阶段选择 | 只对候选帧跑 | 折中速度与语义质量 |

#### `model/vision_accelerator/semantic_stream.py`

核心类：

```text
SemanticStreamGate（语义流门控器）
```

当前支持：

- `threshold（阈值策略）`：基于 drift（漂移）决定是否保留；
- `budget_topk（预算 Top-K）`：每个 budget window（预算窗口）只保留 Top-K 个语义新颖帧；
- `recency_keep（最近帧保留）`：QA 前强制保留最近帧；
- `coverage_keep（覆盖保底）`：周期性保留帧，防止长期漏看；
- `candidate_signatures（候选签名）`：给 `hybrid（混合）` 使用，只对候选帧做第二阶段判别，但统计完整输入帧。

#### 参数入口文件

`video_qa/base.py`、`video_qa/run_eval.py` 和 `scripts/run_semantic_stream_sweep.py` 只负责暴露参数：

```text
semantic_selection_policy
semantic_selection_feature_source
semantic_candidate_multiplier
semantic_budget_window_size
semantic_budget_keep_per_window
```

这些文件不是方法设计本体，只是为了实验可复现。

## 3. 当前方法是否合理优美

### 3.1 合理之处

当前 `hybrid selector（混合选择器）` 的设计动机是清晰的：

```text
高层语义信号质量高但成本高；
低层视觉信号成本低但语义弱；
所以用低层信号先缩小候选集合，再用高层信号做少量复核。
```

这个逻辑比“不断调 threshold（阈值）”更像论文方法，也更容易写成统一原则：

```text
Quality-cost aware semantic admission
质量-成本感知的语义准入
```

它和最初目标一致：

- 减少进入 ViT encoder（ViT 编码器）的帧；
- 减少写入 LLM context（大模型上下文）的视觉 token（视觉令牌）；
- 保持 query-independent（查询无关），避免等待用户 query（查询）后再决定感知；
- 能作为即插即用前端模块。

### 3.2 不够优美的风险

当前代码层面已经开始显得复杂，主要风险是：

1. `SemanticStreamGate（语义流门控器）` 同时承担了策略、状态、统计，职责偏重；
2. `vit_patch.py` 中 `_encode_video_window_with_semantic_compute_gate()` 同时处理三种选择源，后续再加方法会变臃肿；
3. `recency（最近帧保留）`、`coverage（覆盖保底）`、`budget_topk（预算 Top-K）`、`hybrid（混合）` 容易被审稿人看成工程堆叠。

### 3.3 当前是否需要回退

暂时不建议回退 `hybrid（混合）`，因为它已经给出第一组同时优于 `periodic sampling（周期采样）` 的正向结果：

```text
RVS-Movie 1fps:
periodic: 99 / 4067 frames, 27.579s, token-F1 0.0410
hybrid:   85 / 4067 frames,  9.597s, token-F1 0.0503
```

但需要在论文方法上收束表达：

```text
最终论文不要呈现 threshold、coverage、raw_rgb、vit_embedding、hybrid 的所有工程尝试；
只呈现最终方法为“预算化两阶段语义准入”。
```

中间尝试只作为内部 ablation（消融）和 insight（洞察）来源。

## 4. 当前推荐的最终方法抽象

建议把最终方法收束成一个简洁框架：

```text
Budgeted Two-Stage Semantic Admission
预算化两阶段语义准入
```

层次：

```text
Dense video stream（密集视频流）
  -> Cheap proposal（低成本候选生成）
       raw_rgb / motion / shallow signal（原始 RGB / 运动 / 浅层信号）
  -> Semantic verification（语义复核）
       candidate-only ViT embedding（只对候选帧提取 ViT 嵌入）
  -> Budgeted admission（预算准入）
       每个窗口只保留 K 个新颖帧
  -> Sparse semantic stream（稀疏语义流）
       写入 LLM visual context（大模型视觉上下文）
```

这个设计比当前代码名字更优雅，后续可以逐步把实现重构成这四层。

## 5. 时间线整理

### 2026-06-02：从 ViT 稀疏更新转向 QA 优先验证

相关记录：

- `docs/turbovit_v1_experiment_log_zh.md`
- `docs/rvs_movie_repeats_20260602_zh.md`
- `docs/rvs_ego_initial_20260602_zh.md`
- `docs/qa_accuracy_metrics_20260602_zh.md`
- `docs/llm_judge_results_20260602_zh.md`
- `docs/rvs_ego_repeats_judge_analysis_20260602_zh.md`

关键结论：

1. 只看 `feature cosine（特征余弦）` 和 `MSE（均方误差）` 会过早否定速度优先方案；
2. QA（问答）表现需要用 `token-F1（词重叠 F1）`、rule proxy（规则代理）和 LLM judge（大模型裁判）共同粗筛；
3. RVS-Movie（电影视频）和 RVS-Ego（第一视角视频）都能用于早期验证，但 open-ended QA（开放式问答）的自动指标偏弱。

### 2026-06-03：语义流、最近帧修正与 query-aware 风险

相关记录：

- `docs/semantic_stream_research_findings_zh.md`
- `docs/semantic_recency_correction_20260603_zh.md`
- `docs/query_aware_recent_retrieval_20260603_zh.md`
- `docs/query_decoupled_baseline_20260603_zh.md`
- `docs/design_route_summary_20260603_zh.md`
- `docs/joint_vit_rekv_design_zh.md`

关键结论：

1. `recency correction（最近帧修正）` 能减少流式 QA 对最新状态丢失的问题；
2. query-aware retrieval（查询感知检索）有收益，但用户明确指出最终前端计算与存储应 query-independent（查询无关）；
3. 因此后续方法应避免依赖 query（查询）指导视觉感知，否则会损害流式实时性和即插即用性；
4. 研究方向从“还原 dense feature（密集特征）”转向“生成 sparse semantic stream（稀疏语义流）”。

### 2026-06-04 上午：严格 fps 与周期采样强基线

相关记录：

- `docs/fps_dataset_sensitivity_20260604_zh.md`
- `docs/research_pivot_vit_reuse_streaming_bench_20260604_zh.md`
- `docs/streaming_vqa_benchmark_plan_zh.md`

关键结果：

```text
RVS-Movie 1fps:
periodic: 99 / 4067 frames, 27.579s, 13.54x, token-F1 0.0410
semantic threshold: 146 / 4067 frames, 32.207s, 11.60x, token-F1 0.0464
```

关键结论：

1. 在真实流式设置 `0.5 fps（每秒 0.5 帧）` 和 `1 fps（每秒 1 帧）` 下，`periodic sampling（周期采样）` 是很强的 baseline（基线）；
2. 固定阈值 `semantic gate（语义门控）` 不能稳定胜过周期采样；
3. 需要从 threshold（阈值）转向 budget control（预算控制）。

### 2026-06-04 中午：budget_topk（预算 Top-K）

相关记录：

- `docs/budget_topk_semantic_admission_20260604_zh.md`

关键结果：

```text
RVS-Movie 1fps:
budget_topk bw96: 85 / 4067 frames, 29.881s, 12.50x, token-F1 0.0552
periodic:          99 / 4067 frames, 27.579s, 13.54x, token-F1 0.0410
```

关键结论：

1. 预算化选择能比周期采样保留更少帧；
2. `vit_embedding（ViT 嵌入）` 信号质量较好；
3. 但选择开销高，导致速度没有赢过周期采样；
4. 形成 `selection quality-cost dilemma（选择质量-成本矛盾）`。

### 2026-06-04 下午：raw_rgb（原始 RGB）低成本选择

相关记录：

- `docs/raw_rgb_budget_selector_20260604_zh.md`

关键结果：

```text
RVS-Movie 1fps:
raw_rgb bw96: 85 / 4067 frames, 4.656s, 80.24x, token-F1 0.0343
```

关键结论：

1. 低成本选择可以极大打开速度上界；
2. 但原始 RGB 信号语义不足，质量明显弱于 `vit_embedding（ViT 嵌入）`；
3. 这直接支持两阶段设计，而不是继续单独调 raw_rgb。

### 2026-06-04 晚上：hybrid selector（混合选择器）

相关记录：

- `docs/hybrid_selector_probe_20260604_zh.md`

相关提交：

- `5cf5a2c feat: add hybrid semantic selector`
- `2be0aec docs: record hybrid selector probe`

关键结果：

```text
RVS-Movie 1fps:
dense:     4067 / 4067 frames, 373.555s,  1.00x, token-F1 0.0551
periodic:    99 / 4067 frames,  27.579s, 13.54x, token-F1 0.0410
raw_rgb:     85 / 4067 frames,   4.656s, 80.24x, token-F1 0.0343
vit_embed:   85 / 4067 frames,  29.881s, 12.50x, token-F1 0.0552
hybrid:      85 / 4067 frames,   9.597s, 38.92x, token-F1 0.0503
```

对比 `periodic sampling（周期采样）`：

```text
保留帧：99 -> 85
编码时间：27.579s -> 9.597s
相对周期采样加速：2.87x
token-F1：0.0410 -> 0.0503
W/T/L：2 / 20 / 2
```

阶段性结论：

`hybrid selector（混合选择器）` 是当前最合理的主线候选。它不是简单工程堆叠，而是由“语义质量”和“选择成本”的矛盾推导出来。

## 6. 当前风险清单

1. `token-F1（词重叠 F1）` 仍然是弱指标，必须补 Qwen2.5-VL judge（模型裁判）或官方 evaluator（评测器）；
2. RVS-Movie（电影视频）样本少，必须在 RVS-Ego（第一视角视频）和更流式的 OVO-Bench / StreamingBench 上验证；
3. 当前实现中统计与选择耦合，需要后续重构以提升代码优雅性；
4. `raw_rgb（原始 RGB）` 预筛可能在细粒度动作、字幕、局部物体变化上漏召回，需要扫 `candidate_multiplier（候选倍数）`。

## 7. 下一步计划

优先级从高到低：

1. 扫 `candidate_multiplier（候选倍数）= 2, 4, 8`，验证速度-质量曲线；
2. 在 RVS-Ego（第一视角视频）上复验 hybrid；
3. 对 hybrid vs periodic 跑 Qwen2.5-VL judge（模型裁判）；
4. 下载/接入 OVO-Bench（在线视频理解基准）或 StreamingBench（流式视频理解基准）；
5. 代码重构：把 `proposal（候选生成）`、`verification（语义复核）`、`admission（预算准入）`、`accounting（统计记账）` 分成更清晰的模块。

## 8. 设计收束原则

后续每次改动都必须回答：

```text
它是否让 dense visual stream（密集视觉流）更可靠地转化为 sparse semantic stream（稀疏语义流）？
它是否同时改善计算成本和上下文写入成本？
它是否保持 query-independent（查询无关）？
它是否能被解释为一个统一方法，而不是临时工程补丁？
```

不能满足这些问题的改动，默认只作为 ablation（消融）或负结果记录，不进入最终方法主线。


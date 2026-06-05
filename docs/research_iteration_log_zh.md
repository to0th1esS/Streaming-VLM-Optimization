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

## 9. 2026-06-04 深夜：候选倍数扫描与新数据集资产

### 9.1 实验目的

本轮目标是触碰当前 `hybrid selector（混合选择器）` 的边界，而不是只验证一个好看的单点。

需要回答：

1. `candidate_multiplier（候选倍数）` 太小时是否漏掉关键语义？
2. `candidate_multiplier（候选倍数）` 太大时是否引入更多噪声和选择开销？
3. 当前 RVS-Movie / RVS-Ego 是否过于简单，导致方法边界没有暴露？
4. 能否尽快准备更强流式基准数据，如 `StreamingBench（流式视频理解基准）` 和 `OVO-Bench（在线视频理解基准）`？

### 9.2 方法改动范围

本轮没有改核心模型方法，只新增/修正数据工具：

- `scripts/check_streaming_benchmark_targets.py`
  - 修正远程 `/home/mllm/datasets/streamingbench` 和 `/home/mllm/datasets/ovo_bench` 的检测路径；
- `scripts/download_streaming_benchmarks.sh`
  - 固化 StreamingBench / OVO-Bench 的下载命令；
  - 默认使用 `HF_ENDPOINT=https://hf-mirror.com`，因为服务器直连 `huggingface.co` 超时。

核心方法文件未变：

- `model/vit_patch.py`
- `model/vision_accelerator/semantic_stream.py`

### 9.3 RVS-Movie：candidate_multiplier 扫描

设置：

```text
dataset（数据集）= RVS-Movie
fps（帧率）= 1.0
semantic_selection_feature_source（语义选择特征源）= hybrid
budget_window_size（预算窗口）= 96
budget_keep_per_window（每窗口保留数）= 1
recency_keep_frames（最近帧保留）= 4
query_retrieval_policy（查询检索策略）= always_recent
latest_retrieval_blocks（最近检索块数）= 4
```

结果：

| 方法 | kept / input（保留/输入帧） | encode time（编码时间） | speedup vs dense（相对密集加速） | speedup vs periodic（相对周期加速） | token-F1（词重叠 F1） | W/T/L vs dense（胜/平/负） | W/T/L vs periodic（胜/平/负） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hybrid cm2（混合，候选倍数 2）` | 85 / 4067 | 7.140s | 52.32x | 3.86x | 0.0586 | 2 / 19 / 3 | 2 / 21 / 1 |
| `hybrid cm4（混合，候选倍数 4）` | 85 / 4067 | 9.597s | 38.92x | 2.87x | 0.0503 | 1 / 19 / 4 | 2 / 20 / 2 |
| `hybrid cm8（混合，候选倍数 8）` | 85 / 4067 | 9.248s | 40.39x | 2.98x | 0.0396 | 0 / 21 / 3 | 2 / 20 / 2 |

现象：

1. `cm2（候选倍数 2）` 在 RVS-Movie 上反而最好；
2. `cm8（候选倍数 8）` 没有带来更好 QA proxy（问答代理指标），反而 token-F1 更低；
3. 这说明 RVS-Movie 的 24 QA 自动指标可能不足以稳定暴露“候选召回不足”的边界；
4. 也说明当前方法不能只在 Movie 上判断，需要更复杂数据集和 LLM judge（大模型裁判）。

阶段判断：

```text
cm2 是当前速度-质量最优默认候选，但不能据此认定候选越少越好。
```

### 9.4 RVS-Ego：第一视角复验

设置：

```text
dataset（数据集）= RVS-Ego
fps（帧率）= 0.5
semantic_selection_feature_source（语义选择特征源）= hybrid
candidate_multiplier（候选倍数）= 2
budget_window_size（预算窗口）= 96
budget_keep_per_window（每窗口保留数）= 1
```

结果：

| 方法 | kept / input（保留/输入帧） | token reduction（令牌减少） | encode time（编码时间） | speedup vs dense（相对密集加速） | token-F1（词重叠 F1） | W/T/L vs dense（胜/平/负） |
|---|---:|---:|---:|---:|---:|---:|
| `dense（密集）` | 5115 / 5115 | 0.00% | 475.824s | 1.00x | 0.2723 | 0 / 24 / 0 |
| `periodic（周期采样）` | 194 / 5115 | 96.21% | 122.057s | 3.90x | 0.2740 | 1 / 23 / 0 |
| `hybrid cm2（混合，候选倍数 2）` | 96 / 5115 | 98.12% | 35.302s | 13.48x | 0.2767 | 1 / 22 / 1 |

与周期采样对比：

```text
保留帧：194 -> 96
编码时间：122.057s -> 35.302s
相对周期采样加速：3.46x
token-F1：0.2740 -> 0.2767
W/T/L：2 / 21 / 1
```

现象：

1. hybrid 在第一视角数据上也明显减少保留帧；
2. 相对 periodic（周期采样）同时拿到更少写入、更快编码和略高 token-F1；
3. 但 `QA=18/24` 和 token-F1 仍是弱指标，需要 LLM judge（大模型裁判）验证。

阶段判断：

```text
hybrid cm2 不只是 Movie 特例，已在 Ego 上复现效率优势。
```

### 9.5 StreamingBench 资产状态

下载位置：

```text
/home/mllm/datasets/streamingbench
```

已完成：

- `README.md`
- `StreamingBench/*.csv`

CSV 规模：

```text
Real_Time_Visual_Understanding.csv: 2500 lines
Omni_Source_Understanding.csv: 1000 lines
Contextual_Understanding.csv: 500 lines
Sequential_Question_Answering.csv: 250 lines
Proactive_Output.csv: 250 lines
Proactive_Output_50.csv: 50 lines
```

媒体包状态：

- 尝试下载 `Real-Time Visual Understanding_1-50.zip` 和 `Sequential Question Answering_1-25.zip`；
- 直连 `huggingface.co` 超时；
- 使用 `hf-mirror.com` 后仍在大文件下载阶段超时；
- 已终止卡住进程，未得到完整 zip。

阶段判断：

```text
StreamingBench 标注已可用于分析数据结构和设计 adapter（适配器），但视频媒体仍需分批下载或人工辅助下载。
```

### 9.6 OVO-Bench 资产状态

下载位置：

```text
/home/mllm/datasets/ovo_bench
```

状态：

- 使用 `HF_ENDPOINT=https://hf-mirror.com` 下载；
- 下载进程已结束；
- 当前目录大小约 `179GB`；
- 已有 `README.md`、`.gitattributes`、`chunked_videos.tar.part*` 和 `src_videos.tar.part*` 分片。

已观察到的文件：

```text
chunked_videos.tar.partaa ... partag
chunked_videos.tar.partai ... partao
src_videos.tar.partaa ... partae
```

注意：

- 顶层列表中暂未看到 `chunked_videos.tar.partah`；
- 需要下一步用 Hugging Face manifest（文件清单）或重新运行下载命令做完整性校验；
- 暂不解压，避免在 `/home` 已使用 97% 的情况下产生额外空间压力。

OVO-Bench 任务特性：

- `Backward Tracing（向后追溯）`
- `Real-Time Visual Perception（实时视觉感知）`
- `Forward Active Responding（前向主动响应）`

这些任务比 RVS 更适合检验：

```text
稀疏语义流是否会漏掉关键事件；
固定预算选择是否能处理延迟回答和未来信息；
query-independent（查询无关）的前端选择是否足够泛化。
```

### 9.7 下一步

1. 对 `hybrid cm2（混合，候选倍数 2）` 跑 Qwen2.5-VL judge（模型裁判），优先比较 RVS-Ego 和 RVS-Movie 的 hybrid vs periodic；
2. 对 OVO-Bench 做完整性校验，不急于解压；
3. 解析 StreamingBench CSV，写 adapter（适配器）前先确认 video id、timestamp（时间戳）、question type（问题类型）字段；
4. 选择 StreamingBench 的 `Real-Time Visual Understanding（实时视觉理解）` 和 `Sequential Question Answering（顺序问答）` 作为首批边界任务；
5. 后续方法层面重点检查 `raw_rgb prefilter（原始 RGB 预筛）` 是否漏掉细粒度动作、字幕、局部物体变化。

## 10. 2026-06-05：OVO-Bench 接入与周期基线公平性修正

### 10.1 本轮实验目的

本轮不继续在 RVS 简单数据上调参，而是解决两个会直接影响论文可信度的问题：

1. 接入更困难的 OVO-Bench（在线视频理解基准），用官方任务准确率替代过度依赖 `feature cosine（特征余弦相似度）` 和 `token-F1（词重叠 F1）`；
2. 重新审计 `periodic sampling（周期/均匀采样）` 基线，确认它是否真的在 ViT 之前跳过未选帧。

最终要回答：

```text
hybrid selector（混合选择器）的收益是否来自更好的语义选择，
而不是来自一个实现成本偏高的周期基线？
```

### 10.2 OVO-Bench 官方协议梳理

本地读取官方仓库 `JoeLeelyf/OVO-Bench` 的 `ovo_bench_new.json`：

```text
source items（源条目）= 1640

Backward Tracing（向后追溯）：
EPM=297, ASI=148, HLD=186

Real-Time Visual Perception（实时视觉感知）：
OCR=149, ACR=109, ATR=116, STU=178, FPD=101, OJR=184

Forward Active Responding（前向主动响应）：
REC=82, SSR=42, CRR=48
```

三类任务的输出形式：

- 向后追溯和实时感知：multiple choice（多项选择），输出选项字母；
- REC：repetition counting（重复动作计数），输出单个数字；
- SSR / CRR：yes-no decision（是/否判断）。

重要协议限制：

```text
官方 offline protocol（离线协议）为每个查询提供截止到查询时刻的视频片段，
适合验证“观察到当前时刻时，稀疏语义流是否保留了回答所需信息”；
但它不是一个跨多个查询持续不清空状态的完整在线会话。
```

因此 OVO-Bench 可以作为当前方法的强任务质量验证，但论文最终仍需要补充真正连续多查询的流式协议。

### 10.3 新增实验基础设施

新增文件：

- `scripts/prepare_ovo_bench_subset.py`
  - 将官方标注转换为现有 ReKV streaming VQA（流式视频问答）输入格式；
  - 保留 `task（任务）`、`group（任务组）`、`official_id（官方编号）`、`query_index（查询序号）`；
  - 支持按任务限制源条目数和每个源条目的查询数。
- `scripts/evaluate_ovo_bench.py`
  - 输出 official-compatible accuracy（官方兼容准确率）；
  - 同时输出 strict accuracy（严格规范化准确率），避免宽松字符串匹配虚高。
- `scripts/check_ovo_bench_assets.py`
  - 检查 5 个源视频分片和 15 个预切片分片；
  - 检查标注、已解压目录和子集所需视频。
- `scripts/extract_ovo_bench_subset.sh`
  - 通过分片管道只提取子集需要的视频；
  - 不生成额外的 144GB 合并 tar（归档文件）。
- `scripts/run_ovo_bench_validation.sh`
  - 一键运行 dense（密集）、periodic（周期）、hybrid cm2（混合，候选倍数 2）；
  - 明确关闭 `query-aware retrieval（查询感知检索）`，保持前端选择查询无关。
- `scripts/summarize_ovo_bench_validation.py`
  - 汇总准确率、编码时间、相对密集加速、帧保留和视觉令牌缩减。

`video_qa/rekv_stream_vqa.py` 仅增加基准元数据透传，没有改变问答或缓存逻辑。

### 10.4 本地适配验证

从官方 1640 个源条目中执行最小覆盖抽样：

```text
每个任务抽 1 个源条目；
前向任务每个源条目最多取 2 个查询；
最终得到 15 个查询：
backward（向后追溯）=3
realtime（实时感知）=6
forward（前向响应）=6
```

本地 Conda 环境结果：

```text
unittest（单元测试）：6 / 6 passed
Python compile（语法编译）：passed
原有 ViT sparse patch certification（ViT 稀疏补丁认证）：passed
Shell scripts（Shell 脚本）：LF 换行，无 CRLF 污染
```

本机 RTX 5070 可被 `nvidia-smi` 识别，但既有 GPU Conda 环境加载 PyTorch 时出现：

```text
WinError 1455：页面文件太小，无法加载 nvperf_host.dll
```

因此本轮模型侧逻辑使用 CPU Conda 环境做确定性验证；真实模型实验继续放到远程 A100。

### 10.5 关键发现：旧 periodic 并非真正低成本周期采样

代码审计发现，旧配置：

```text
selection_policy=threshold
skip_threshold=999
refresh_interval=N
```

虽然最终保留帧近似周期采样，但执行顺序是：

```text
所有帧 -> 图像预处理 -> patch embedding（图像块嵌入）
-> 再根据 refresh interval（刷新间隔）决定保留
```

这意味着旧 periodic 仍为所有帧支付了至少 patch embedding 和逐帧判别成本，不是通常意义上“按固定时间索引先采样、再送入 ViT”的强基线。

对旧实验结论的影响必须分开处理：

1. periodic 的保留帧数量和 QA 质量对比仍有参考价值；
2. hybrid 相对 periodic 的编码时间加速被高估，不能直接进入论文主表；
3. hybrid 相对 dense 的加速结果不因该问题自动失效，但需要在同一新代码版本重跑；
4. 所有后续主要结果必须加入真正的 index-level periodic sampling（索引级周期采样）。

### 10.6 周期基线修正

新增 `selection_policy=periodic`：

```text
输入帧
-> 按全局帧序号判断 frame_idx % interval == 0
-> 可选保留最近 recency frames（最近帧）
-> 只对选中帧执行图像预处理、patch embedding、ViT 和上下文写入
```

该策略不计算 RGB 差异、ViT 嵌入或语义相似度，因此它是低成本、查询无关、可复现的公平基线。

单元测试覆盖：

- 单批次周期索引正确性；
- 跨批次全局帧序号连续性；
- 最近帧保护；
- 统计中的输入帧、保留帧和写入 token（令牌）一致性。

### 10.7 远程资产与运行状态

远程路径：

```text
代码：/home/yangjin/1#Streaming-VLM-Optimization
模型：/home/mllm/models
数据：/home/mllm/datasets/ovo_bench
```

2026-06-05 当前检查：

```text
远程代码 commit（提交）= b8d6e78
工作区干净
OVO-Bench 总大小约 179GB
src_videos parts（源视频分片）= 5 / 5
chunked_videos parts（预切片分片）= 14 / 15
唯一缺失：chunked_videos.tar.partah
磁盘剩余约 1.7TB
GPU：8 x NVIDIA A100 80GB
```

官方标注已补到：

```text
/home/mllm/datasets/ovo_bench/ovo_bench_new.json
```

缺失的 `partah` 已启动单文件续补下载。完整后才能进行分片管道解压和真实视频验证。

### 10.8 当前方向性结论

本轮没有否定 hybrid 主线，而是提高了证据标准：

```text
最终方法仍是“廉价候选生成 + 少量语义复核 + 固定预算准入”；
真正周期采样只作为强基线；
OVO-Bench 官方任务准确率成为质量约束；
feature cosine / MSE 退回诊断指标，不再作为主要优化目标。
```

下一轮必须依次完成：

1. 补齐 `chunked_videos.tar.partah`；
2. 只解压 15 条本地适配子集需要的视频；
3. 先用 0.5B 模型做端到端冒烟验证；
4. 再用 7B 模型在相同子集运行 dense / true periodic / hybrid cm2；
5. 比较 official accuracy（官方准确率）、strict accuracy（严格准确率）、编码时间、写入帧数和视觉 token 缩减；
6. 若 hybrid 不能稳定优于 true periodic，则优先研究细粒度事件召回，而不是继续堆叠规则。

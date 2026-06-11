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

补充执行路径：

- 镜像 CLI（下载客户端）仅产生锁文件、没有字节增长，已切换为支持断点续传的直接下载；
- 直接下载已确认产生实际字节增长，但按当前速度仍需较长时间；
- 因 5 个源视频分片完整，新增 `extract_ovo_bench_source_subset.sh` 和
  `chunk_ovo_bench_source_subset.py`，先选择性提取 12 个源视频并生成 15 个查询片段；
- 官方 `utils/chunk_videos.py` 存在先跳过非 forward task（前向任务）的控制流问题，
  不能直接用于本轮三类任务子集。

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

### 10.9 OVO-Bench 端到端冒烟与 0.5B 子集结果

远程结果路径：

```text
results/ovo_bench/smoke_0p5b_rec1
results/ovo_bench/subset15_0p5b
```

REC 单查询冒烟实验验证了：

- 0.5B 模型加载；
- dense / true periodic / hybrid 三条路径；
- GPU 视频编码；
- CSV 元数据记录；
- OVO 官方兼容评测；
- 统一结果汇总。

单查询中，0.5B 输出 `One` 而不是官方要求的数字 `1`，因此官方计数得分为 0。
这是小模型的格式遵循问题，不是评测器错误。

15 查询、12 类任务结果：

| 方法 | kept / input（保留/输入帧） | encode time（编码时间） | speedup vs dense（相对密集加速） | official macro accuracy（官方宏平均准确率） |
|---|---:|---:|---:|---:|
| dense（密集） | 2380 / 2380 | 106.723s | 1.00x | 38.89% |
| periodic96（96 秒周期） | 93 / 2380 | 3.874s | 27.55x | 38.89% |
| hybrid cm2（修正预算前） | 108 / 2380 | 9.550s | 11.17x | 38.89% |

阶段结论：

```text
0.5B 模型不足以稳定体现语义选帧差异；
真正周期基线明显比旧周期实现更强；
必须使用 7B 模型判断内容选择价值。
```

### 10.10 7B、15 查询初步结果

远程结果路径：

```text
results/ovo_bench/subset15_7b
```

| 方法 | kept / input | token reduction（令牌减少） | encode time | speedup vs dense | official macro accuracy |
|---|---:|---:|---:|---:|---:|
| dense | 2380 / 2380 | 0.00% | 175.307s | 1.00x | 72.22% |
| periodic96 | 93 / 2380 | 96.09% | 4.639s | 37.79x | 72.22% |
| hybrid cm2（修正预算前） | 108 / 2380 | 95.46% | 10.358s | 16.92x | 77.78% |

逐样本差异：

- hybrid 相对 periodic：1 胜、0 负；
- 唯一胜例为 ATR 任务，问题要求识别短暂出现目标的颜色。

该结果提出了“语义变化驱动选帧可能优于均匀覆盖”的候选洞见，但样本过少，不能直接形成论文结论。

### 10.11 7B、30 查询边界复验

远程结果路径：

```text
results/ovo_bench/subset30_7b
```

数据规模：

```text
source items（源条目）=24
queries（查询）=30
input frames（输入帧）=6034
backward / realtime / forward = 6 / 12 / 12 queries
```

首次运行结果：

| 方法 | kept / input | encode time | speedup vs dense | official macro accuracy | query correct（查询正确数） |
|---|---:|---:|---:|---:|---:|
| dense | 6034 / 6034 | 583.270s | 1.00x | 69.44% | 21 / 30 |
| periodic96 | 201 / 6034 | 13.836s | 42.16x | 69.44% | 21 / 30 |
| hybrid cm2（修正预算前） | 230 / 6034 | 25.545s | 22.83x | 75.00% | 23 / 30 |

hybrid 相对 periodic 的逐样本结果：

```text
wins（胜）=2
losses（负）=0
```

两条胜例均为 ATR：

1. `What is the color of the dog?`
2. `What is the color of the fire I'm collecting?`

这表明当前 hybrid 的收益集中在短暂属性证据，而不是所有任务普遍提升。

### 10.12 预算公平性漏洞与修正

进一步检查发现，旧 `budget_topk（预算 Top-K）` 在每个视频首窗口中：

```text
保留 reference frame（参考帧）
+ 再保留 1 个 budget frame（预算帧）
```

因此 `budget_keep_per_window=1` 在首窗口实际保留 2 帧。30 个视频中多出的 29 帧几乎完全由该问题解释。

修正原则：

- reference / refresh / coverage（参考/刷新/覆盖）帧占用窗口预算；
- recency（最近帧保护）作为显式流式保障，不占语义预算；
- 每个窗口的内容选择只能使用剩余预算。

相关提交：

```text
91a2350 fix: enforce semantic window budget
```

本地验证：

```text
unittest（单元测试）=7 / 7 passed
ViT sparse patch certification（ViT 稀疏补丁认证）=passed
```

### 10.13 修正后的公平对比

| 方法 | kept / input | token reduction | encode time | speedup vs dense | official macro accuracy |
|---|---:|---:|---:|---:|---:|
| dense | 6034 / 6034 | 0.00% | 583.270s | 1.00x | 69.44% |
| periodic96 | 201 / 6034 | 96.67% | 13.836s | 42.16x | 69.44% |
| hybrid cm2 budget-fixed（预算修正） | 201 / 6034 | 96.67% | 25.491s | 22.88x | 72.22% |
| periodic65（预算近似匹配） | 228 / 6034 | 96.22% | 14.714s | 39.64x | 75.00% |
| raw RGB budget-fixed | 201 / 6034 | 96.67% | 25.406s | 22.96x | 69.44% |

公平结论：

1. 完全相同的 201 帧写入预算下，hybrid 比 periodic96 高 `2.78` 个百分点；
2. hybrid 的编码时间是 periodic96 的 `1.84x`，候选选择成本仍然过高；
3. periodic65 多写入 27 帧后达到 `75.00%`，并且仍比 hybrid 快；
4. raw RGB 在相同预算下没有超过 periodic96，说明纯像素变化不足以稳定识别关键语义；
5. raw RGB 与 hybrid 耗时接近，说明当前全分辨率 raw signature（原始像素签名）计算本身也很重，
   “廉价预筛”在工程上尚未真正廉价。

当前方法不能宣称全面优于均匀采样。更准确的表述是：

```text
在严格写入预算下，内容自适应选择可以提高任务准确率；
但现有候选生成与语义复核开销使其尚未形成完整 Pareto 优势。
```

### 10.14 新的核心研究问题

当前瓶颈已从“能否找到语义帧”收敛为：

```text
如何在不增加帧预算、不过度计算所有候选特征的条件下，
同时获得周期覆盖和事件敏感性？
```

后续最终方法应围绕一个统一设计，而不是继续叠加规则：

1. `coverage（时间覆盖）` 提供稳定的长时上下文下界；
2. `novelty（事件新颖性）` 在固定预算内重分配保留位置，而不是额外加帧；
3. 低成本 proposal（候选生成）必须在低分辨率或解码阶段完成；
4. semantic verification（语义复核）只处理极少数候选，并需要与 ViT 稀疏层计算共享特征；
5. 所有 forced keep（强制保留）必须进入统一预算记账。

这可以凝练为后续候选方法：

```text
Budget-Neutral Coverage-Novelty Allocation
（预算中性的覆盖-新颖性分配）
```

该方法的论文动机不是“工程上试了很多策略”，而是由两个可复现实验事实推出：

```text
Fact 1：纯周期覆盖速度极强，但会漏掉短暂属性事件；
Fact 2：语义选择能找回事件，但当前复核成本破坏端到端效率。
```

### 10.15 资产最终状态

2026-06-05 最终检查：

```text
src_videos parts（源视频分片）=5 / 5
chunked_videos parts（预切片分片）=15 / 15
30-query subset videos（30 查询子集视频）=30 / 30
missing（缺失）=0
```

`chunked_videos.tar.partah` 已通过断点续传补齐到 `10GB`。

## 11. 低成本候选生成与候选倍率消融（2026-06-05）

### 11.1 实验目的

上一轮公平实验说明：

```text
内容自适应选帧在相同 201 帧写入预算下比 periodic96 高 2.78 个百分点，
但旧 hybrid cm2 的编码时间是 periodic96 的 1.84x。
```

本轮不再增加选择规则，而是回答两个可证伪问题：

1. 旧方法的主要开销是否来自全分辨率 `raw signature（原始像素签名）`？
2. 每个时间窗生成两个候选是否必要，还是一个廉价候选已经足够？

控制变量保持一致：

- 模型：LLaVA-OneVision-Qwen2-7B；
- 数据：OVO-Bench 30 条查询、6034 个输入帧；
- 流式采样率：1 FPS（每秒一帧）；
- 时间窗：96 帧；
- 写入预算：每窗 1 帧，总计 201 帧；
- 查询无关：视觉选帧阶段不读取问题文本；
- 最终约束：OVO-Bench 官方三组宏平均准确率，而不是逐帧特征还原误差。

### 11.2 实现与回退点

提交：

```text
0c7ab75 feat: add low-cost semantic proposal profiling
```

回退标签：

```text
research/ovo-fair-baseline-20260605
```

新增能力：

1. `grid_sample（网格采样）`：在图像转浮点和完整预处理前，只读取固定网格像素构造低维签名；
2. `avg_pool（全图平均池化）`：保留旧实现作为可复现实验对照；
3. 分阶段计时：
   - `proposal（候选生成）`；
   - `preprocess（图像预处理）`；
   - `embedding（图像块嵌入）`；
   - `verification（语义复核）`；
   - `vit_encoder（视觉编码器）`；
   - `context_write（上下文写入）`；
4. 记录候选帧数和真正执行预处理的帧数。

本地验证：

```text
unittest（单元测试）=9 / 9 passed
ViT sparse patch certification（ViT 稀疏补丁认证）=passed
```

512 帧、224x224 CPU 微基准：

| 签名方式 | 中位耗时 |
|---|---:|
| avg_pool（全图平均池化） | 0.033752s |
| grid_sample（网格采样） | 0.000426s |

`grid_sample` 的签名计算约快 `79x`。

### 11.3 15 条查询的等价性验证

远程结果：

```text
results/ovo_bench/low_cost15_7b/hybrid_avg
results/ovo_bench/low_cost15_7b/hybrid_grid
```

两种实现均保留 93 / 2380 帧，候选数均为 141，准确率均为 `77.78%`。

| 方法 | proposal | 总编码时间 | 准确率 |
|---|---:|---:|---:|
| avg_pool hybrid cm2 | 5.429s | 11.943s | 77.78% |
| grid_sample hybrid cm2 | 0.036s | 7.682s | 77.78% |

低成本签名使候选生成约快 `151x`，总编码时间下降 `35.7%`，且没有改变候选数量、写入预算和问答结果。

### 11.4 30 条查询的 cm2 复验

远程结果：

```text
results/ovo_bench/low_cost30_7b/hybrid_grid
results/ovo_bench/low_cost30_7b/periodic96
```

| 方法 | kept / input | candidates | encode time | official accuracy |
|---|---:|---:|---:|---:|
| periodic96（周期均匀采样） | 201 / 6034 | 0 | 12.013s | 69.44% |
| grid hybrid cm2 | 201 / 6034 | 310 | 13.620s | 72.22% |

相对旧 `avg_pool hybrid cm2` 的 25.491s，低成本 cm2 降至 13.620s，下降 `46.6%`。
在完全相同写入预算下，准确率优势仍为 `+2.78` 个百分点，而额外编码时间缩小到约 `13.4%`。

分阶段结果显示，新的主要边际开销不再是 `proposal（候选生成）`：

```text
proposal = 0.095s
verification = 0.239s
preprocess = 4.556s
```

cm2 需要预处理 310 个候选，而 periodic96 只预处理最终保留的 201 帧。因此，瓶颈已经转移为候选帧的完整图像预处理。

### 11.5 candidate multiplier（候选倍率）消融

`cm1` 表示每个写入名额只生成一个内容候选；其余配置不变。远程结果：

```text
results/ovo_bench/low_cost30_7b/hybrid_grid_cm1
results/ovo_bench/low_cost30_7b/hybrid_grid_cm1_r2
results/ovo_bench/low_cost30_7b/hybrid_grid_cm1_r3
results/ovo_bench/low_cost30_7b/periodic96_r2
results/ovo_bench/low_cost30_7b/periodic96_r3
```

三次同卡重复：

| 方法 | kept / input | candidates | encode time（均值 ± 标准差） | official accuracy |
|---|---:|---:|---:|---:|
| periodic96 | 201 / 6034 | 0 | 10.684 ± 1.153s | 69.44% |
| grid hybrid cm1 | 201 / 6034 | 230 | 10.672 ± 0.133s | 72.22% |

准确率在三次重复中完全一致。两种方法的编码时间均值只差 `0.012s`，应判定为速度持平，
不能据此声称 cm1 稳定快于周期采样。

cm1 的平均分阶段耗时：

| 阶段 | periodic96 | grid hybrid cm1 | 差值 |
|---|---:|---:|---:|
| proposal（候选生成） | 0.010s | 0.063s | +0.053s |
| preprocess（图像预处理） | 2.919s | 3.302s | +0.384s |
| embedding（图像块嵌入） | 0.139s | 0.145s | +0.006s |
| verification（语义复核） | 0.000s | 0.099s | +0.099s |
| vit_encoder（视觉编码器） | 1.885s | 1.517s | -0.368s |
| context_write（上下文写入） | 5.550s | 5.317s | -0.233s |

后两项差异主要来自 GPU 运行波动和不同被选帧内容，不能解释为算法性加速。可信结论是：

```text
cm1 将内容自适应选择的额外候选成本压缩到了端到端计时噪声范围，
同时保留了相同预算下的 QA 准确率优势。
```

相对旧 `avg_pool hybrid cm2`，cm1 平均编码时间下降 `58.1%`。
若仅用历史 dense 结果 583.270s 作近似参照，cm1 的视觉编码加速约为 `54.7x`；
该数字不是同版本重复计时，只能作为阶段性参考，不能直接进入论文主表。

### 11.6 科学发现与方法收敛

本轮形成三个连续证据：

1. 全分辨率像素统计不是方法所必需，低分辨率网格足以生成当前有效候选；
2. 第二候选没有带来额外问答收益，说明当前收益来自预算位置的内容重分配，而不是大量语义复核；
3. 在固定写入预算下，单候选内容分配可以达到周期采样的速度，同时找回周期采样遗漏的短暂属性证据。

因此，最终方法应继续收敛为一个统一操作：

```text
每个时间窗拥有固定的一个视觉写入名额；
低成本 novelty（新颖性）信号决定该名额落在哪一帧；
只有被提议的极少数帧进入视觉编码与上下文写入。
```

这比“周期采样再叠加若干补帧规则”更简洁，也满足：

- `budget-neutral（预算中性）`：不增加写入帧数；
- `query-independent（查询无关）`：选帧不依赖问题；
- `streaming-compatible（兼容流式）`：只使用当前与历史状态；
- `plug-and-play（即插即用）`：不训练、不修改语言模型；
- `compute-storage co-design（计算与存储协同设计）`：同一决策同时减少 ViT 计算和上下文写入。

### 11.7 当前证据边界与下一步

当前 30 条查询中，优势仍主要来自 ATR（短暂属性识别）任务，样本量不足以证明普遍泛化。
下一步不能继续在同一小子集微调规则，而应：

1. 扩大 OVO-Bench 查询规模，优先增加短暂事件、状态变化、OCR 和动作边界样本；
2. 固定比较 `periodic96 / grid hybrid cm1 / grid hybrid cm2`，验证 cm1 的质量优势是否稳定；
3. 至少进行 3 次独立计时，并将视觉编码与完整生成时延分开报告；
4. 统计选中帧相对周期位置的位移和任务类别收益，验证“预算内重分配”这一论文机制；
5. 在更大样本复验前，不把全局 `semantic_candidate_multiplier` 默认值改为 1，避免探索结果被误当成最终方法。

## 12. 时长分层边界实验与显著性门控替换（2026-06-05）

### 12.1 实验目的

上一轮 30 条查询中，`grid hybrid cm1（网格单候选混合方法）` 在相同 201 帧预算下：

```text
QA accuracy（问答准确率）=72.22%
periodic96（96 帧周期采样）=69.44%
encode time（编码时间）基本持平
```

但该结果主要来自一条 ATR（短暂属性识别）胜例，且原子集按官方标注文件顺序取前若干条，
可能存在源视频顺序、时长和前向查询时间点偏差。

本轮回答四个问题：

1. 在更长、更均衡的视频上，纯新颖性选帧能否稳定超过周期采样？
2. `cm2（双候选）` 是否比 `cm1（单候选）` 提供额外质量收益？
3. 纯像素变化失效时，具体选择了什么帧？
4. 能否在不增加预算和预处理帧数的前提下，只对可信事件替换周期槽位？

### 12.2 时长分层数据构造

提交：

```text
894bdde feat: add stratified OVO benchmark subsets
d8ed354 feat: add disjoint OVO validation subsets
```

新增确定性子集策略：

- `duration_stratified（按视频时长分层）`：每个任务覆盖短、中、长源视频；
- `time_stratified（按查询时间点分层）`：前向任务覆盖早、中、晚查询；
- `source fold（源视频互斥折）`：按时长排序后的源视频可拆成互斥池；
- `exclude subset（排除已有子集）`：留出集显式排除开发集全部官方源视频编号。

默认仍为旧 `head（按原顺序取样）`，因此历史实验可复现。

本地验证：

```text
OVO adapter + semantic stream unittest（单元测试）=16 / 16 passed
ViT sparse patch certification（ViT 稀疏补丁认证）=passed
bash syntax check（Shell 语法检查）=passed
```

### 12.3 开发集规模与资产

开发集：

```text
data/ovo_bench/ovo_rekv_stratified90.json
```

规模：

```text
source videos（源视频）=72
queries（查询）=90
backward / realtime / forward（回溯/实时/前向）=18 / 36 / 36
queries per task（每任务查询）=6；REC/SSR/CRR 各 12
query time range（查询时间范围）=0.03s 至 1785s
input frames at 1 FPS（1 FPS 输入帧）=28670
```

资产准备没有扫描 142.75 GiB 的完整切片归档，而是：

1. 从 43.16 GiB 源视频归档选择性提取 72 个源视频；
2. 根据每条查询的结束时间生成 90 个查询片段；
3. 对 90 个片段执行实际首尾帧解码验证。

### 12.4 极短视频边界修复

首轮 90 条实验在第 37 个样本停止：

```text
task（任务）=FPD
official id（官方编号）=1117
query time（查询时间）=0.03s
error（错误）=cannot convert float NaN to integer
```

原因：

- 0.03 秒码流复制只生成了不可稳定解码的极短视频；
- `Decord（视频解码库）` 返回非有限平均帧率；
- 原时间索引使用向下取整，正时间但小于一秒的查询可能编码 0 帧。

修复提交：

```text
8832b61 fix: handle short OVO video boundaries
5249013 fix: use portable OVO clip fallback
```

修复内容：

1. 极短片段至少生成 1 秒可解码内容；
2. 正时间查询使用向上取整，至少看到第一帧；
3. 非有限 FPS（帧率）使用安全回退；
4. 资产检查新增 `decode validation（实际解码验证）`；
5. FFmpeg 回退编码使用环境内置 `mpeg4`，不依赖缺失的 `libx264`。

最终开发集资产：

```text
available videos（可用视频）=90 / 90
decode failures（解码失败）=0
```

### 12.5 90 条开发集：周期、纯新颖性与双候选

远程结果：

```text
results/ovo_bench/stratified90_7b_r2/periodic96
results/ovo_bench/stratified90_7b_r2/hybrid_grid_cm1
results/ovo_bench/stratified90_7b_r2/hybrid_grid_cm2
```

所有方法：

- 模型：LLaVA-OneVision-Qwen2-7B；
- 输入：1 FPS；
- 窗口：96 帧；
- 写入预算：每窗 1 帧，加相同的最近帧保护；
- 查询无关；
- 28670 个输入帧；
- 697 个写入帧；
- 视觉 token reduction（视觉令牌减少）约 `97.57%`。

| 方法 | candidates（候选帧） | encode time | official accuracy |
|---|---:|---:|---:|
| periodic96 | 0 | 33.820s | 53.70% |
| grid hybrid cm1 | 774 | 35.398s | 52.78% |
| grid hybrid cm2 | 1112 | 39.709s | 52.78% |

逐题差异：

```text
cm1 vs periodic96：
wins（胜）=1
losses（负）=2

cm2 vs periodic96：
wins（胜）=1
losses（负）=2

cm1 vs cm2 prediction differences（预测差异）=0
```

具体变化：

| 样本 | 任务 | 结果 | 现象 |
|---|---|---|---|
| ovo-995 | ATR | win | 找回紫色火焰这一短暂属性 |
| ovo-909 | OJR | loss | 丢失盘子上物品这一稳定状态 |
| ovo-1472-0 | CRR | loss | 错误判断当前信息已经足够回答 |

结论：

1. 纯新颖性不是周期采样的稳定替代；
2. 它偏向短暂突变，但会破坏稳定状态和最近可回答性；
3. cm2 没有带来任何预测收益，却比 cm1 多预处理 338 帧；
4. 增加候选数量不是后续方向。

### 12.6 原始候选信号审计

远程分析：

```text
results/ovo_bench/stratified90_analysis/raw_window_audit.csv
results/ovo_bench/stratified90_analysis/raw_window_audit.json
results/ovo_bench/stratified90_analysis/contact_sheets
```

352 个窗口统计：

| 指标 | 中位数 | 90 分位 | 最大值 |
|---|---:|---:|---:|
| novelty offset（新颖性候选相对周期位置偏移） | 33 帧 | 84 帧 | 95 帧 |
| adjacent selected gap（相邻选择帧间隔） | 96 帧 | 151 帧 | 183 帧 |

可视化观察：

- 游戏视频中大量选择黑帧、菜单打开和转场；
- 电影视频中大量选择片头、字幕页和场景切换；
- 做饭视频中选择人物或镜头切换，而不一定选择问题所需稳定物体状态。

这说明 4×4 RGB（红绿蓝像素）最大变化同时混合了：

```text
真实短暂事件
场景切换
菜单和界面变化
黑帧与解码边界
相机运动
```

因此，“变化最大”不等价于“语义价值最大”。

### 12.7 方法推导：显著性门控槽位替换

提交：

```text
434bbae feat: add saliency-gated slot replacement
```

方法原则：

```text
每个 96 帧窗口仍只有一个基础写入槽位；
周期位置是默认 coverage anchor（覆盖锚点）；
计算窗口内相邻 RGB 网格变化；
只有最大变化相对窗口背景足够突出时，才用事件候选替换周期位置；
否则保持周期位置。
```

标准化显著性：

```text
z = (maximum_delta - window_mean) / (window_std + epsilon)
```

该设计满足：

- `budget-neutral（预算中性）`：不增加写入帧；
- `query-independent（查询无关）`：不读取问题；
- `no-regret default（无悔默认）`：证据不足时退回周期覆盖；
- `single-candidate（单候选）`：不恢复 cm2 开销；
- `plug-and-play（即插即用）`：不训练、不修改语言模型。

### 12.8 开发集阈值扫描

远程结果：

```text
results/ovo_bench/stratified90_saliency_7b/saliency_z3p5
results/ovo_bench/stratified90_saliency_7b/saliency_z4p0
results/ovo_bench/stratified90_saliency_7b/saliency_z4p5
```

| 方法 | kept / input | preprocessed | encode time | accuracy | 相对 periodic |
|---|---:|---:|---:|---:|---:|
| periodic96 | 697 / 28670 | 697 | 33.820s | 53.70% | - |
| saliency z=3.5 | 697 / 28670 | 697 | 35.727s | 53.70% | 1 胜 1 负 |
| saliency z=4.0 | 697 / 28670 | 697 | 33.130s | 54.63% | 1 胜 0 负 |
| saliency z=4.5 | 697 / 28670 | 697 | 34.311s | 54.63% | 1 胜 0 负 |

`z=4.0` 和 `z=4.5`：

- 保留 ovo-995 的 ATR 胜例；
- 消除 ovo-909 的 OJR 负例；
- 消除 ovo-1472-0 的 CRR 负例；
- 候选帧、预处理帧和写入帧均为 697，与周期采样完全相同。

单次计时中 z=4.0 略快于周期采样，但差异处于 GPU 波动范围，不能宣称算法性加速。

### 12.9 零重叠 held-out（留出）验证

> 2026-06-05 数据审计修正：本小节当时只按 `official_id（官方条目编号）` 排除，
> 后续发现开发集与本留出集仍共享 11 个 `original_video（原始视频）`。
> 因此本小节结果只能视为探索性 `official-item-disjoint（官方条目不重叠）` 证据，
> 不能作为视频级泛化结论。真正 `video-disjoint（原始视频不重叠）` 的 v2 结果见第 13 节。

留出集：

```text
data/ovo_bench/ovo_rekv_heldout90.json
```

构造约束：

```text
development source videos（开发集源视频）=72
held-out source videos（留出集源视频）=72
source overlap（源视频交集）=0
queries（查询）=90
query time range（查询时间范围）=4.0s 至 1623s
input frames at 1 FPS（1 FPS 输入帧）=27800
decode failures（解码失败）=0
```

远程结果：

```text
results/ovo_bench/heldout90_7b/periodic96
results/ovo_bench/heldout90_7b/hybrid_grid_cm1
results/ovo_bench/heldout90_7b/saliency_z4p0
```

| 方法 | kept / input | preprocessed | encode time | accuracy |
|---|---:|---:|---:|---:|
| periodic96 | 699 / 27800 | 699 | 34.457s | 57.41% |
| pure novelty cm1（纯新颖性单候选） | 697 / 27800 | 784 | 36.461s | 57.41% |
| saliency z=4.0 | 699 / 27800 | 699 | 33.863s | 57.41% |

逐题比较：

```text
pure novelty cm1 vs periodic96：
wins（胜）=2
losses（负）=3

saliency z=4.0 vs periodic96：
wins（胜）=0
losses（负）=0
prediction differences（预测差异）=0
```

留出集说明：

1. 纯新颖性仍会改变任务行为，虽然宏平均分碰巧相同；
2. z=4.0 没有过拟合出负迁移，能够稳定回退到周期行为；
3. 但 z=4.0 在留出集没有获得新的事件收益。

### 12.10 180 条合并结果

开发集和留出集共 180 条查询、144 个互斥源视频：

| 方法 | kept / input | preprocessed | encode time | official accuracy |
|---|---:|---:|---:|---:|
| periodic96 | 1396 / 56470 | 1396 | 68.277s | 55.56% |
| pure novelty cm1 | 1394 / 56470 | 1558 | 71.859s | 55.09% |
| saliency z=4.0 | 1396 / 56470 | 1396 | 66.993s | 56.02% |

合并任务变化：

```text
periodic ATR accuracy（周期 ATR 准确率）=66.67%
saliency ATR accuracy（门控 ATR 准确率）=75.00%
其他任务准确率与 periodic96 相同
```

因此当前可支持的最强结论是：

```text
显著性门控能够在与周期采样相同的视觉计算和写入预算下，
避免纯新颖性造成的负迁移，并在 180 条查询中找回一条短暂属性证据。
```

不能支持的结论：

```text
不能宣称已经普遍优于周期采样；
不能把单次略低的编码时间解释为算法性加速；
不能证明 4×4 RGB 显著性能够泛化识别语义事件；
不能仅凭一条 ATR 增益形成论文主结果。
```

### 12.11 方向性结论

本轮把研究问题进一步收敛：

```text
coverage（覆盖）不是需要被删除的冗余，而是稳定任务质量的先验；
novelty（新颖性）不应无条件替代覆盖，只能作为高置信度的预算重分配信号；
低层像素显著性可以过滤明显无事件窗口，但无法稳定判断语义价值。
```

最终方法不应继续调 RGB 阈值，而应升级为：

```text
Confidence-Calibrated Semantic Slot Reallocation
（置信度校准的语义槽位重分配）
```

建议结构：

1. 周期槽位提供确定性的时间覆盖下界；
2. 低成本像素变化只做 event proposal（事件提议），不直接决定写入；
3. 使用与稀疏 ViT 共享的浅层语义特征做 semantic confidence（语义置信度）；
4. 只有外观显著性和语义置信度同时成立时，才替换周期槽位；
5. 替换始终发生在固定总预算内，同时控制 ViT 计算和上下文写入。

下一步实验不再扫描更多 RGB 阈值，而是验证：

```text
浅层 ViT 语义置信度能否区分“真实事件”与“黑帧/菜单/场景切换”，
并在不恢复全量 ViT 计算的情况下扩大 ATR、ASI、SSR 等事件任务收益。
```

## 13. 2026-06-05：稳定事件提议、视频级留出修正与配对语义验证

### 13.1 本轮实验目的

本轮围绕三个问题展开：

1. 原始 RGB（红绿蓝像素）候选是否存在实现退化，导致黑帧被错误识别为最大变化？
2. 浅层 ViT（视觉 Transformer）语义是否能在固定一个写入槽位内，可靠地决定周期帧和事件帧二选一？
3. 前一轮留出集是否真的实现了原始视频级隔离？

对应代码节点：

```text
525b51f feat: add stable proposal and shallow ViT probe
ac6254f feat: add paired semantic slot verification
2cdd59b feat: track paired semantic decisions
c0e349f fix: isolate OVO subsets by original video
bfb5209 fix: preserve the initial semantic anchor
50b91d1 feat: analyze patch-level semantic changes
```

本地验证：

```text
unit tests（单元测试）=26/26 passed
ViT sparse patch certification（ViT 稀疏补丁认证）=passed
bash syntax check（脚本语法检查）=passed
```

### 13.2 原始 RGB 签名的零向量退化

旧 `grid_sample（网格采样）` 签名对纯黑帧产生全零向量。
PyTorch 的 cosine similarity（余弦相似度）对两个零向量返回 0，
因此两个完全相同的黑帧会被错误解释为：

```text
similarity（相似度）=0
drift（变化量）=1
```

这解释了候选可视化中大量黑帧、菜单边界和转场被选中的现象。

修复：

```text
grid_sample_stable（稳定网格采样）
= 网格 RGB 签名 + 常数维度 + 归一化
```

修复后，相同黑帧的余弦相似度为 1。
旧 `grid_sample` 保留不变，用于复现实验。

### 13.3 浅层 ViT 探针

新增：

```text
scripts/analyze_shallow_vit_candidates.py
```

探针提取：

- embedding layer（嵌入层）；
- ViT 第 1、3、6 层；
- 周期帧与事件帧的全局余弦相似度；
- 事件帧与前后邻帧的持续性；
- patch token（图像块词元）对应位置的相似度分布；
- 最低 10% token 相似度；
- 低于 0.90、0.95、0.99 的 token 比例。

第一轮探针曾显示：部分有效候选在 embedding 层已与周期帧明显不同。
但完整 QA 验证证明，全局相似度不能作为通用事件价值判据。

### 13.4 配对语义槽位验证

实现 `saliency_paired（显著性配对）`：

```text
每个窗口保留 periodic frame（周期帧）作为默认候选；
RGB 显著性只提出一个 event frame（事件帧）；
只计算两个候选的 ViT embedding（嵌入）；
若两者全局相似度低于阈值，则事件帧替换周期帧；
最终完整 ViT 编码和上下文写入仍只有一帧。
```

8 个旧方法变化视频的小规模阈值实验：

| similarity threshold（相似度阈值） | QA 正确数 | semantic reallocations（语义重分配次数） |
|---:|---:|---:|
| 0.60 | 5/8 | 1 |
| 0.80 | 6/8 | 4 |
| 0.95 | 6/8 | 5 |

该结果一度支持 0.80，但完整开发集给出反证：

| 方法 | accuracy（准确率） | wins（胜） | losses（负） |
|---|---:|---:|---:|
| periodic96（96 帧周期采样） | 53.70% | - | - |
| paired 0.80，允许首窗口替换 | 51.85% | 0 | 1 |

唯一负例是 ovo-120 / EPM。
它发生在第一个窗口，事件帧替换了初始化覆盖帧。

加入首锚点约束：

```text
首窗口只建立 reference anchor（参考锚点）；
首窗口不允许事件候选替换；
后续窗口才允许槽位重分配。
```

修正后完整开发集：

| 指标 | paired anchor（锚点配对） |
|---|---:|
| accuracy | 53.70% |
| wins / losses | 0 / 0 |
| prediction differences（预测差异） | 0 |
| reallocated frames（重分配帧） | 18 |
| candidate / preprocessed frames（候选/预处理帧） | 792 |
| fully encoded / written frames（完整编码/写入帧） | 697 |

结论：

```text
首锚点不可替换是有效且可泛化的覆盖约束；
全局 ViT 相似度配对可以过滤风险，但也过滤了有用的细微事件；
它不应作为最终主方法的硬门控。
```

### 13.5 旧留出集的数据泄漏审计

旧开发集和旧留出集统计：

| 子集 | queries（查询） | official items（官方条目） | unique original videos（唯一原始视频） |
|---|---:|---:|---:|
| development（开发） | 90 | 72 | 68 |
| old held-out（旧留出） | 90 | 72 | 68 |

虽然 `official_id` 交集为 0，但原始视频交集为 11：

```text
AutoEvalMetaData: 5
Ego4D: 3
MovieNet: 2
YouTube Games: 1
```

典型例子：

```text
ovo-995 和 ovo-999 的 official_id 不同，
但 original_video 都是：
YouTube_Games/PLJ3VIGhVd3r8Int6IZT_v3S_BzG9RVfiG&index=2.mp4
```

修复 `scripts/prepare_ovo_bench_subset.py`：

```text
exclude official_id（排除官方条目编号）
AND
exclude original_video（排除原始视频路径）
```

### 13.6 真正 video-disjoint 的 held-out v2

新子集：

```text
data/ovo_bench/ovo_rekv_heldout90_v2.json
```

构造与资产检查：

| 指标 | 数值 |
|---|---:|
| queries | 90 |
| official items | 72 |
| unique original videos | 67 |
| development official-id overlap | 0 |
| development original-video overlap | 0 |
| available clips（可用切片） | 90/90 |
| input frames at 1 FPS（1 FPS 输入帧） | 27586 |

缺失的 38 个切片从 OVO-Bench 官方分卷归档中选择性解压，最终无缺失资产。

### 13.7 video-disjoint v2 主结果

配置：

```text
model（模型）= LLaVA-OneVision-Qwen2-7B
sample rate（采样率）= 1 FPS
window（窗口）= 96 frames
recency（最近帧保留）= 4
query-aware（查询感知）= disabled
z threshold（显著性阈值）= 3.5 / 4.0
```

结果：

| 方法 | accuracy | wins | losses | encode time | kept / input |
|---|---:|---:|---:|---:|---:|
| periodic96 | 51.85% | - | - | 34.521s | 695 / 27586 |
| stable saliency z=3.5（稳定显著性） | 53.70% | 2 | 1 | 35.712s | 691 / 27586 |
| stable saliency z=4.0 | 53.70% | 2 | 0 | 34.377s | 691 / 27586 |
| paired anchor z=3.5, s=0.8（锚点配对） | 53.70% | 1 | 0 | 35.282s | 695 / 27586 |

`z=4.0` 的两个独立胜例：

| video | task | periodic | stable z=4.0 |
|---|---|---|---|
| ovo-991 | OJR | wrong | correct |
| ovo-1506-0 | CRR | wrong | correct |

没有 QA 负例。

### 13.8 开发集 + video-disjoint v2 的 180 条联合结果

联合路径：

```text
results/ovo_bench/video_disjoint180_summary
```

| 方法 | official accuracy | encode time | preprocessed | fully encoded / written |
|---|---:|---:|---:|---:|
| periodic96 | 52.78% | 68.342s | 1392 | 1392 |
| stable saliency z=4.0 | **54.17%** | 67.897s | 1392 | 1388 |
| paired anchor | 53.70% | 70.175s | 1570 | 1392 |

稳定显著性相对周期采样：

```text
absolute accuracy gain（绝对准确率提升）= +1.39 percentage points
relative accuracy gain（相对准确率提升）= +2.63%
error reduction（错误率相对下降）= 2.94%
```

分组变化：

| group（任务组） | periodic | stable z=4.0 | change |
|---|---:|---:|---:|
| backward（回溯） | 44.44% | 44.44% | 0 |
| realtime（实时） | 66.67% | 69.44% | +2.78 pp |
| forward（前向） | 47.22% | 48.61% | +1.39 pp |

三个无负迁移胜例覆盖：

```text
ATR：瞬时属性识别
OJR：在线判断
CRR：当前可回答性判断
```

单次联合编码时间只下降 0.65%，处于 GPU 波动范围。
因此本轮不能宣称新的算法性加速，只能宣称：

```text
在几乎相同的视觉计算和上下文写入预算下，
video-disjoint 评测准确率提高 1.39 个百分点，且未观察到负迁移。
```

### 13.9 patch token 分布分析

全局帧签名会掩盖 token 级变化：

- ovo-1506-0 的有效事件在第 6 层约 10% token 低于 0.90，相对局部；
- ovo-991 的有效事件在第 6 层约 88% token 低于 0.90，属于大范围变化；
- ovo-120 的首窗口负例约 91% token 低于 0.90，同样属于大范围变化。

因此：

```text
局部变化比例不能单独区分有用事件和有害转场；
全局相似度也不能单独区分；
事件是否对任意未来 QA 有价值，本质上不能由单个通用硬阈值完全决定。
```

但 patch 分析仍给出重要方法启示：

```text
token 变化分布更适合决定“计算哪些 token”，
而不是决定“整帧是否值得保留”。
```

### 13.10 当前方法结论

当前最优且最简洁的版本可以概括为：

```text
Anchor-Preserved Saliency Slot Reallocation
（锚点保持的显著性槽位重分配）
```

结构：

1. `stable RGB signature（稳定 RGB 签名）` 修复黑帧零向量退化；
2. 首窗口建立不可替换的 coverage anchor（覆盖锚点）；
3. 每个后续 96 帧窗口默认保留周期槽位；
4. 只有标准化显著性 `z>=4.0` 时，事件候选替换周期槽位；
5. 总帧预算、完整 ViT 编码帧数和上下文写入预算基本不变；
6. 选择过程完全 query-independent（查询无关）。

这一设计比“全局语义差异越大越重要”更符合当前证据。

### 13.11 下一步研究方向

下一轮不继续堆叠帧级硬门控，而做正交联合：

```text
temporal saliency（时间显著性）
决定“在哪个时间位置取帧”；

patch-level semantic drift（图像块级语义变化）
决定“该帧内部哪些 token 需要重计算”；

fixed semantic slot budget（固定语义槽位预算）
决定“写入多少视觉上下文”。
```

优先实验：

1. 在 `stable saliency z=4.0` 上重新启用 ViT 层内稀疏更新；
2. 用 patch 变化分布控制 dynamic token ratio（动态 token 比例），不再用它否决整帧；
3. 测量 attention / MLP / gather-scatter / context write 的真实时间分解；
4. 在 7B 和更大模型上复验端到端速度；
5. 保持 video-disjoint v2 作为当前最小可信泛化集，并继续扩展更难流式 benchmark（基准）。

## 14. 从层内不规则稀疏到固定语义带宽：计算、写入与缓存的联合边界

### 14.1 本轮实验目的

本轮不再把 feature cosine / MSE（特征余弦相似度 / 均方误差）作为主要淘汰标准，而是回答三个端到端问题：

1. 已保留帧内部的 ViT layer sparse update（ViT 层内稀疏更新）能否带来真实 GPU 加速；
2. 在 QA 基本不下降的约束下，减少每帧写入视觉 token 是否能降低 ReKV cache（检索式键值缓存）；
3. token 选择开销与 context write（上下文写入）收益能否形成端到端正收益。

统一设置：

- 模型：LLaVA-OneVision-Qwen2-7B；
- 数据：OVO-Bench 12 任务小规模边界集，每个任务一个中等时长样本；
- 输入：1 FPS；
- 帧级方法：stable saliency z=4.0（稳定显著性），96 帧窗口，每窗口 1 个预算槽，末端保留 4 帧；
- query-independent（查询无关），不使用问题指导视觉计算或存储；
- 正式计时前在同一进程加入一个短视频 warm-up（预热）样本，结果中排除该样本。

### 14.2 ViT 层内 token 稀疏更新复验

首先修复旧实现的流式状态问题：显著性路径原来会批量编码所有保留帧，没有逐帧推进 `InferenceContext`，因此后续帧可能缺少正确参考缓存。本轮改为：

```text
保留帧按时间顺序逐帧进入 ViT
-> reference frame（参考帧）完整计算
-> intermediate frame（中间帧）按 token 比例更新
-> 记录 dense / sparse 帧数、计划更新 token 数和分阶段时延
```

12 样本结果：

| 方法 | official QA | encode time | ViT encoder | 说明 |
|---|---:|---:|---:|---|
| batch dense（批量密集） | 61.11% | 4.255s | 0.636s | 吞吐上界，不是公平顺序对照 |
| sequential dense（逐帧密集） | 61.11% | 4.684s | 1.081s | 公平顺序对照 |
| sparse i=2, r=0.25 | 61.11% | 5.353s | 1.432s | 计划更新 65.16% token |
| sparse i=4, r=0.25 | 61.11% | 5.246s | 1.431s | 计划更新 47.30% token |
| sparse i=4, r=0.50 | 61.11% | 5.271s | 1.534s | 更高更新比例 |

相对公平的逐帧密集对照：

- `i=2, r=0.25` 端到端慢 14.3%；
- `i=4, r=0.25` 端到端慢 12.0%；
- `i=4, r=0.50` 端到端慢 12.5%；
- `i=4, r=0.25` 的 ViT encoder 本身慢 32.4%。

结论：

```text
当前 gather/scatter（收集/写回）式不规则 token 稀疏
虽然减少了计划计算 token 数，却没有减少真实 GPU 时间。
该结果不是“帧间冗余不存在”，而是“不规则稀疏没有映射成硬件友好的算子”。
```

因此旧层内稀疏路径保留为研究入口，不再作为当前主方法扩展。

### 14.3 端到端瓶颈重新定位

在 video-disjoint 180 样本 stable saliency z=4.0 结果上汇总：

| 分量 | 时间 | 总编码占比 |
|---|---:|---:|
| raw proposal（原始帧提议） | 1.050s | 1.5% |
| preprocess（预处理） | 18.396s | 27.1% |
| embedding（嵌入） | 0.423s | 0.6% |
| verification（验证） | 0.441s | 0.6% |
| ViT encoder（视觉编码器） | 10.252s | 15.1% |
| context write（上下文写入） | 35.239s | **51.9%** |

保留 1388 / 56256 帧。由此可得：

1. 帧级选择已经将 ViT 占比压到 15.1%；
2. 即使 ViT 再理想减少 50%，端到端理论收益也只有约 7.5%；
3. 当前更主要的瓶颈已经转移到视觉 token 写入语言模型和 KV cache。

这验证了研究目标必须从“只加速 ViT”扩展为：

```text
Dense visual stream（密集视觉流）
-> sparse frame stream（稀疏帧流）
-> fixed-bandwidth semantic token stream（固定带宽语义 token 流）
```

### 14.4 固定预算视觉 token 压缩

新增 `FixedBudgetTokenReducer`（固定预算 token 缩减器），每个保留帧固定输出相同数量 token，以保持 ReKV 块对齐。

第一版策略：

- `coverage tokens`（覆盖 token）：均匀覆盖空间位置；
- `innovation tokens`（变化 token）：按相对上一保留帧的 token 余弦漂移选取；
- 默认 25% 预算用于覆盖，75% 用于变化；
- 第一帧只做均匀覆盖；
- 完全 query-independent（查询无关）。

扫描 96 / 128 / 160 token 后，128 token 是当前最稳定边界：

| 方法 | token reduction | official QA | encode | context write |
|---|---:|---:|---:|---:|
| none-196（不压缩） | 0% | 55.56% | 4.281s | 2.064s |
| innovation-96 | 51.02% | 55.56% | 4.648s | 2.093s |
| innovation-128 | **34.69%** | 55.56% | 4.300s | **1.993s** |
| innovation-160 | 18.37% | 61.11% | 4.578s | 2.188s |

128 token 与同轮 196 token 的逐样本 QA 判定一致，仅 REC 任务出现 `1` 与 `One` 的答案格式差异，二者官方分数都为 0。

160 token 在一个样本上变好，但 12 样本规模不足以把该变化解释为方法收益。

### 14.5 KV cache 实际存储验证

旧 `calc_memory_usage()` 只统计已经 offload（卸载）到 CPU 的历史块。当前短视频未超过 `n_local`，因此原指标恒为 0，不能代表实际缓存。

本轮新增缓存观测，分别统计：

- GPU local KV（局部键值缓存）；
- GPU global remainder（全局剩余缓存）；
- GPU preallocated retrieval buffers（预分配检索缓冲区）；
- CPU offloaded blocks（已卸载块）；
- logical cache tokens（逻辑缓存 token 数）。

独立复验：

| 方法 | official QA | mean cache | max cache | mean logical tokens | context write | total encode |
|---|---:|---:|---:|---:|---:|---:|
| none-196 | 55.56% | 2.321 GB | 2.433 GB | 1385 | 2.168s | 4.325s |
| innovation-128 | 55.56% | **1.517 GB** | **1.591 GB** | 909 | **2.041s** | 4.407s |

定量结论：

```text
平均 KV cache 减少：34.61%
峰值 KV cache 减少：34.61%
逻辑视觉 token 减少：34.37%
context write 加速：5.82%
端到端编码：慢 1.89%
QA：同轮不下降
```

因此当前可以正式宣称：

> 固定 128 token 的语义带宽在小规模 OVO-Bench 上保持 QA，不仅减少逻辑 token，还将实际 ReKV 缓存降低 34.61%，并降低上下文写入时间。

当前不能宣称：

> 已获得稳定端到端加速。

### 14.6 选择开销定位

为定位 128 token 端到端仍未加速的原因，进行了三类对照。

#### A. ViT native selection（ViT 原生空间选择）

将变化评分从 3584 维投影空间移到 1152 维 ViT 原生空间，但为了对齐 196 个输出位置，需要额外空间池化。

结果：

- QA 与投影空间完全一致；
- ViT 阶段反而慢约 3.1%；
- 端到端差异约 0.23%，属于噪声范围。

结论：额外池化抵消了低维相似度收益，该路径不进入主方法。

#### B. low-dimensional temporal sketch（低维时间变化草图）

不新增池化，直接从 3584 维投影 token 中均匀抽取 64 / 128 个通道计算漂移。

64 维草图将选择相关的额外 ViT 时间从约 0.132s 降到约 0.107s，但仍不能完全消除 `top-k + gather` 开销。

#### C. uniform-128（均匀 128 token）控制

该方法不做相似度和 top-k，仅保留均匀空间位置，用于测量索引和写回的最低成本。

同轮结果：

| 方法 | official QA | ViT encoder | context write | total encode | cache reduction |
|---|---:|---:|---:|---:|---:|
| none-196 | 55.56% | 0.733s | 2.063s | 4.377s | 0% |
| uniform-128 | 61.11% | 0.791s | 2.023s | 4.375s | 34.69% |
| sketch64-128 | 61.11% | 0.784s | 2.041s | 4.245s | 34.69% |

不能直接把 sketch64 的单次 3.0% 总时间下降当作算法加速，因为分解显示主要差异来自 proposal / preprocess 波动：

- none-196：proposal + preprocess = 1.472s；
- sketch64-128：proposal + preprocess = 1.329s；
- 与 token 选择无关。

真正受方法影响的 ViT 阶段仍慢约 7%，context write 只快约 1%--2%。

### 14.7 当前方向性结论

本轮形成三个可用于论文方法设计的 insight（研究洞察）：

#### Insight 1：冗余必须转化为结构化计算

逐 token 不规则 gather/scatter 在 A100 上不能自动转化为速度。层内稀疏和输出 top-k 都重复证明：

```text
理论 FLOPs 减少 != 实际 GPU latency 减少
```

最终方法需要 block-structured sparsity（块结构稀疏）、规则张量形状或专用 kernel（内核），不能依赖 Python/PyTorch 级不规则索引。

#### Insight 2：时间稀疏与语义带宽应联合设计

帧级 stable saliency 决定“何时写入”，固定 token 预算决定“每次写入多少”。两者共同将密集视觉流转为受控语义流，并已经得到实际缓存证据。

#### Insight 3：QA 约束允许明显偏离 dense feature

128 token 减少 34.69% 的视觉写入和缓存，同轮 QA 没有下降。这支持新的目标函数：

```text
maximize streaming efficiency（最大化流式效率）
subject to QA degradation <= epsilon（QA 退化不超过容忍阈值）
```

而不是逐帧还原 dense ViT 特征。

### 14.8 下一步研究计划

下一轮不继续堆叠帧级阈值，重点转向硬件友好的 structured semantic bandwidth（结构化语义带宽）：

1. 用规则空间块或规则池化替代逐 token top-k，验证能否同时减少投影、写入和缓存；
2. 保持固定输出块大小，使 ReKV 无需处理 ragged shape（不规则形状）；
3. 在更难、更长的 OVO-Bench / StreamingBench 子集上比较：
   - none-196；
   - uniform / structured pooling（均匀或结构化池化）；
   - semantic block allocation（语义块分配）；
4. 至少 3 次重复计时，报告 median（中位数）和标准差；
5. QA 以 official accuracy（官方准确率）为主，feature cosine 只作为诊断指标；
6. 长视频必须实际触发 CPU offload，分别验证 GPU 峰值、CPU cache 和端到端检索时延。

当前主线可概括为：

```text
Anchor-preserved temporal saliency
（锚点保持的时间显著性）
        +
Fixed semantic bandwidth
（固定语义带宽）
        +
Hardware-aligned structured compression
（硬件对齐的结构化压缩）
```

前两部分已有 QA 与缓存证据，第三部分是下一轮获得稳定速度收益的核心。

### 14.9 远端复现注意事项

本轮确认服务器存在两个代码目录：

- 当前研究同步目录：`/home/yangjin/1#Streaming-VLM-Optimization`；
- 历史旧目录：`/home/Streaming-VLM-Optimization`，停留在旧提交并包含大量未提交改动。

后续实验只使用前者，不对历史目录执行 pull、reset 或清理。

正确启动方式：

```bash
/root/miniconda3/bin/python -m video_qa.rekv_stream_vqa
```

关键复现参数：

```text
semantic_refresh_interval=96
semantic_budget_window_size=96
semantic_budget_keep_per_window=1
semantic_recency_keep_frames=4
semantic_raw_signature_mode=grid_sample_stable
semantic_raw_proposal_policy=saliency_gated
semantic_saliency_z_threshold=4.0
```

其中 `semantic_refresh_interval=4` 属于早期 ViT 默认值，会错误地把长视频近似每 4 帧保留一次，不能用于当前 stable saliency 实验。

## 15. 结构化语义网格：从缓存压缩推进到编码加速

### 15.1 本轮目的

上一轮的固定 128 token 方案已经证明：

- QA 可以容忍约 35% 的视觉 token 压缩；
- 实际 KV cache 可以同步下降；
- 但逐 token `top-k + gather`（前 K 选择 + 不规则收集）会抵消速度收益。

本轮目标是验证一个更硬件友好的设计：

```text
不再选择离散 token
-> 将规则二维视觉网格直接压缩为更小的规则网格
-> 保持固定形状、连续内存和 ReKV 块对齐
```

同时根据论文指标需求，将时延统一分成三级：

1. `model encoding latency`（模型编码时延）：patch embedding + ViT + projector / pooling；
2. `visual encoding latency`（视觉编码时延）：预处理 + 模型编码；
3. `stream ingestion latency`（流式摄取时延）：帧选择 + 视觉编码 + context write。

端到端系统时延继续保留，但不再掩盖视觉编码本身的收益。

### 15.2 Structured Grid Token Reducer

新增 `StructuredGridTokenReducer`（结构化网格 token 缩减器）。

输入为规则 ViT token 网格：

```text
27 x 27 = 729 native ViT tokens
```

候选输出：

```text
12 x 12 = 144 tokens
11 x 11 = 121 tokens
10 x 10 = 100 tokens
```

与旧 `coverage + innovation`（覆盖 + 变化）方案相比：

| 属性 | 旧逐 token 选择 | 结构化网格 |
|---|---|---|
| 选择方式 | cosine + top-k | 二维规则池化 |
| 内存访问 | 不连续 gather | 连续规则张量 |
| 输出形状 | 固定数量但位置离散 | 固定方形网格 |
| query-aware | 否 | 否 |
| 训练参数 | 无 | 无 |
| ReKV 对齐 | 支持 | 支持 |

该设计仍保持 query-independent（查询无关），适合流式场景在问题到达之前持续运行。

### 15.3 Post-projector 初筛：只减少存储，不能减少编码

第一版将规则池化放在 projector（投影器）之后：

```text
ViT 729 tokens
-> projector 仍处理 729 tokens
-> structured pooling
-> 144 / 121 / 100 tokens
```

12 样本 OVO-Bench 初筛：

| 方法 | official QA | visual encoding | context write | total encode | mean cache |
|---|---:|---:|---:|---:|---:|
| none-196 | 55.56% | 2.252s | 2.320s | 5.142s | 2.321 GB |
| post-pool-144 | 61.11% | 2.310s | 2.445s | 5.029s | 1.707 GB |
| post-pool-121 | 61.11% | 2.560s | 2.292s | 5.158s | 1.435 GB |
| post-pool-100 | 61.11% | 2.507s | 2.116s | 4.976s | 1.198 GB |

三个压缩点均没有 QA 负迁移；共同变化为：

- `ovo-991` 从错误 B 变为正确 D；
- `ovo-1575-7` 只发生错误答案格式变化，官方分数不变。

但视觉编码没有加速，原因明确：

```text
projector 仍对 729 个 token 做完整计算，
规则池化只减少后续写入与缓存。
```

因此 post-projector pooling（投影后池化）只作为消融，不进入当前主方法。

### 15.4 Pre-projector 结构化压缩

将同一个规则池化前移：

```text
ViT 27 x 27 native grid
-> structured pool to 11 x 11
-> projector only processes 121 tokens
-> ReKV writes 121 tokens
```

这个顺序同时作用于：

1. projector 计算；
2. LLM context write；
3. KV cache 存储。

相比逐 token 稀疏，它不依赖不规则索引，也不要求专用 sparse kernel（稀疏内核）。

### 15.5 12 样本系统初筛

121 token 与 196 token 同进程、同数据顺序比较：

| 指标 | none-196 | pre-pool-121 | 相对变化 |
|---|---:|---:|---:|
| official QA | 55.56% | 61.11% | +5.56 pp |
| visual encoding | 2.468s | 2.506s | -1.52% |
| context write | 3.105s | 2.614s | **+15.83%** |
| stream ingestion | 5.726s | 5.243s | **+8.43%** |
| total encode | 5.938s | 5.449s | **+8.25%** |
| mean KV cache | 2.321 GB | 1.435 GB | **-38.18%** |

此处 `+` 表示速度提升，`-` 表示速度下降。

QA 没有观察到负迁移，仍只有 `ovo-991` 的正向变化。

该结果目前只运行一次，且服务器有外部任务，因此：

- 写入、缓存和 QA 方向可信；
- 8% 左右的系统加速只能视为 preliminary result（初步结果）；
- 不能直接作为最终论文主表数字。

### 15.6 为什么进程级视觉编码计时波动

对 4 个视频做三次 GPU 交换重复时，完整 model encoding 出现较大波动：

- 前两次 pre-pool-121 明显更快；
- 第三次反向；
- 理论上输入完全相同的 vision backbone 也出现 25% 以上变化。

这说明服务器并发任务、GPU 频率和执行次序污染了跨进程比较。

因此增加两个同进程 CUDA event（CUDA 事件）基准：

1. 固定同一批 ViT feature，测 post-ViT projection；
2. 固定同一模型，直接测 dense frames 与 semantic frames 的视觉编码。

### 15.7 Post-ViT projection 微基准

设置：

- 同一模型进程；
- 同一视频的 8 帧；
- 同一个 `selected_video_feature`；
- 每轮 10 次 warm-up，50 次正式重复；
- 共 5 轮，dense / structured 交替执行顺序；
- GPU 3 有外部模型常驻，但测试期间计算利用率为 0。

121 token：

```text
dense projection median:      37.14 ms
structured projection median:  9.71 ms
speedup:                       3.82x
latency reduction:            73.85%
```

100 token：

```text
dense projection median:      37.13 ms
structured projection median:  9.70 ms
speedup:                       3.83x
latency reduction:            73.87%
```

100 与 121 token 几乎同速，说明当前 projector 已触到该 batch shape（批形状）下的 kernel efficiency floor（内核效率下限）。

因此 121 token 更合理：

```text
与 100 token 同速，但保留 21% 更多视觉 token。
```

### 15.8 Dense visual stream 与 semantic visual stream

为了直接回答“编码加速有多少”，新增纯视觉编码基准，不进入 LLM，不写 KV cache。

4 个 OVO 视频：

| video | dense frames | semantic frames |
|---|---:|---:|
| ovo-1303 | 167 | 6 |
| ovo-1472-0 | 696 | 12 |
| ovo-379 | 324 | 8 |
| ovo-991 | 260 | 7 |
| total | 1447 | 33 |

帧减少比例：

```text
97.72%
```

三次运行：

| run | temporal sparse speedup | temporal + structured speedup |
|---|---:|---:|
| 1 | 40.64x | 50.15x |
| 2 | 37.43x | 36.89x |
| 3 | 44.19x | 54.57x |
| median | **40.64x** | **50.15x** |

第二次运行中一个长视频受到外部 GPU 任务恢复干扰，导致 structured 路径异常偏慢。使用三次中位数后：

```text
temporal saliency only:
40.64x visual encoding speedup

temporal saliency + 121 structured grid:
50.15x visual encoding speedup

structured grid on the same semantic frames:
about 1.23x additional speedup
about 18.96% additional latency reduction
```

输出 token 总量：

```text
dense:                     283,612
temporal sparse-196:         6,468
temporal + structured-121:   3,993
```

相对 dense visual stream，联合方法写入前的视觉 token 数减少：

```text
98.59%
```

### 15.9 当前可宣称与不可宣称

当前可以宣称：

1. 帧级 query-independent temporal saliency（查询无关时间显著性）在该 4 视频基准上带来 40.64x 视觉编码加速；
2. pre-projector structured grid（投影前结构化网格）将 post-ViT projection 加速 3.82x；
3. 两者联合的视觉编码中位数加速为 50.15x；
4. 121 token 在 12 样本 OVO-Bench 上没有 QA 下降；
5. 实际 ReKV cache 降低 38.18%，context write 初筛快 15.83%。

当前不能宣称：

1. 50.15x 是完整 VLM 端到端加速；
2. 8.25% 系统编码加速已经稳定；
3. 12 样本上的 QA 上升具有统计显著性；
4. 当前方法已经在完整 OVO-Bench 或 StreamingBench 上超过已有工作。

### 15.10 方法设计的当前形式

当前主方法可以收敛为两级统一预算分配：

```text
Level 1: Temporal slot allocation
时间槽分配

Anchor-preserved saliency
决定“哪些时刻值得进入视觉语义流”

Level 2: Spatial bandwidth allocation
空间带宽分配

Pre-projector structured semantic grid
决定“每个被保留时刻用多少规则视觉槽表示”
```

两级都满足：

- query-independent（查询无关）；
- training-free（无需训练）；
- fixed-shape（固定形状）；
- streaming-compatible（兼容流式处理）；
- compute-storage co-design（计算与存储联合设计）。

相比“多个工程阈值堆叠”，该结构更适合论文表达：

> 将密集视频流映射为受时间槽预算和空间带宽预算共同约束的规则语义流。

### 15.11 下一步

1. 在服务器低负载窗口重复 12 样本系统计时至少 3 次；
2. 将 121 token 扩展到 video-disjoint 180，验证 QA 不下降和缓存收益；
3. 在更难、更长的视频上触发 CPU offload，验证长期缓存；
4. 增加 dense visual encoding baseline 到正式实验表；
5. 对比 hierarchical token compression（分层 token 压缩）基线：
   - 相同 QA 约束；
   - 视觉编码；
   - context write；
   - cache memory；
   - 完整端到端；
6. 暂不重新引入层内不规则 token sparse，除非改成 block-structured kernel（块结构内核）。

## 16. 121-token 结构化语义网格：低负载复验与 video-disjoint 180 验证

### 16.1 实验目的

本轮验证两个问题：

1. `pre-projector structured grid`（投影前结构化网格）能否在更大、视频不重叠的数据上维持 QA；
2. 编码、上下文写入和 KV cache（键值缓存）收益能否在低负载和 GPU 交换后复现。

本轮不改变时间选择器，只比较每个保留帧的空间带宽：

```text
共同时间策略：
stable saliency + recency anchors
稳定显著性 + 近期锚点

对照：
196 tokens / retained frame

方法：
11 x 11 = 121 structured tokens / retained frame
```

因此两条路径的输入帧、保留帧和预处理帧完全相同，差异只来自空间 token 压缩。

### 16.2 代码、模型与数据配置

- 日期：2026-06-09；
- 代码提交：`507bb9a`；
- 模型：`LLaVA-OneVision-Qwen2-7B`；
- 模型位置：远程 `/home/mllm/models` 对应本地模型注册项；
- 采样率：`1 FPS`；
- QA 最大生成长度：16 tokens；
- 时间窗口：96 帧；
- 每窗口显著性预算：1 帧；
- 近期锚点：4 帧；
- 显著性阈值：`z=4.0`；
- 层内不规则 token sparse（稀疏更新）：关闭；
- query-aware（查询感知）检索：关闭；
- 开发集：`ovo_rekv_stratified90.json`；
- 真正视频不重叠保留集：`ovo_rekv_heldout90_v2.json`；
- 合计：180 个问题，135 个原视频；
- 远程结果目录：
  `results/ovo_bench/prepool121_video_disjoint180_20260609`。

每条运行在正式 90 样本前加入同一个预热样本，汇总时剔除，避免首样本 CUDA 初始化污染。

### 16.3 12 样本低负载三次配对复验

三次实验使用相同样本和相同数据顺序，服务器 GPU 计算利用率在启动前为 0。
下表中的正值表示方法更快：

| 指标 | 第 1 次 | 第 2 次 | 第 3 次 | 中位数 |
|---|---:|---:|---:|---:|
| model encoding（模型编码） | +1.90% | +9.30% | +2.89% | **+2.89%** |
| visual encoding（视觉编码） | -1.30% | +4.60% | +2.37% | **+2.37%** |
| context write（上下文写入） | -2.02% | -3.33% | +3.83% | **-2.02%** |
| stream ingestion（流式摄入） | -1.66% | +0.17% | +3.96% | **+0.17%** |
| total encode（总编码） | -1.04% | +1.52% | +4.76% | **+1.52%** |
| mean KV cache（平均键值缓存） | +38.18% | +38.18% | +38.18% | **+38.18%** |

三次 QA 均为：

```text
196 tokens: 55.56%
121 tokens: 61.11%
```

三次都只有 `ovo-991` 从错误变为正确，没有负向翻转。由于样本过少，该 QA 上升不具有统计意义。

低负载复验修正了上一轮单次实验的过强表述：

- 38.18% 缓存下降是确定性结构收益；
- 小样本系统总编码收益约为 1%--5%，不是稳定的 8%；
- context write 在 12 样本上噪声较大，不能单次宣称 15.83%。

### 16.4 video-disjoint 180 的 QA 结果

首轮开发集与保留集各自同时运行 196-token 和 121-token 路径。

| 数据划分 | 196-token QA | 121-token QA | 差值 |
|---|---:|---:|---:|
| 开发 90 | 53.70% | 59.26% | +5.56 pp |
| 视频不重叠保留 90 | 51.85% | 50.00% | -1.85 pp |
| 合并 180 | 52.78% | 54.63% | +1.85 pp |

逐样本配对分析：

| 数据划分 | 错误变正确 | 正确变错误 | McNemar p 值 |
|---|---:|---:|---:|
| 开发 90 | 9 | 4 | 0.267 |
| 保留 90 | 8 | 9 | 1.000 |
| 合并 180 | 17 | 13 | 0.585 |

`McNemar test`（麦克尼马尔配对检验）用于判断同一批问题上两种方法的正确/错误翻转是否显著。
所有 `p` 值都远大于 0.05，因此当前证据支持：

> 121-token 结构化网格与 196-token 对照在当前 180 样本上统计等价，未观察到显著 QA 退化，也不能宣称显著提升。

这与研究目标一致：空间压缩首先满足“QA 基本不下降”的约束，再最大化计算和存储效率。

### 16.5 保留集 GPU 交换复验

为了排除 GPU 位置差异，保留 90 在 GPU 2/3 上运行两次，并在第二次交换两种方法：

```text
第 1 次：196 -> GPU 2，121 -> GPU 3
第 2 次：196 -> GPU 3，121 -> GPU 2
```

QA 在两次运行中完全一致：

```text
196 tokens: 51.85%
121 tokens: 50.00%
```

编码与存储结果：

| 指标 | 第 1 次收益 | GPU 交换后收益 | 两次中点 |
|---|---:|---:|---:|
| model encoding（模型编码） | +11.01% | +11.87% | **+11.44%** |
| visual encoding（视觉编码） | +3.72% | +0.05% | **+1.89%** |
| context write（上下文写入） | +13.58% | +4.60% | **+9.09%** |
| stream ingestion（流式摄入） | +8.94% | +2.45% | **+5.70%** |
| total encode（总编码） | +8.24% | +1.60% | **+4.92%** |
| mean KV cache（平均键值缓存） | +38.18% | +38.18% | **+38.18%** |

其中最稳定的速度证据是：

```text
model encoding reduction:
11.01% / 11.87%

mean KV cache reduction:
38.18% / 38.18%
```

视觉编码和总编码仍受视频预处理、H.264 解码和 context write 波动影响，因此论文中应报告多次运行的均值/标准差或中位数，而不是挑选最好一次。

### 16.6 与直接视觉编码微基准的关系

本轮系统结果与上一节微基准并不矛盾：

- temporal saliency（时间显著性）减少进入 ViT 的帧数，4 视频微基准中位数为 40.64x；
- 121-token structured grid（结构化网格）在相同语义帧上进一步减少 projector（投影器）和后续写入；
- 单独 post-ViT projection（ViT 后投影）微基准为 3.82x；
- 完整保留集系统中，模型编码稳定减少约 11%；
- 完整总编码收益被预处理、视频解码和其他固定成本稀释到约 2%--8%。

因此论文应同时报告两类指标：

1. `visual/model encoding speedup`（视觉/模型编码加速），证明方法确实减少视觉计算；
2. `end-to-end streaming speedup`（端到端流式加速），反映系统固定成本后的实际收益。

只报告端到端会掩盖编码收益，只报告微基准则会夸大真实系统收益。

### 16.7 当前可宣称结论

1. 121-token 结构化网格将每个保留帧的视觉 token 减少 38.27%；
2. 实际平均 KV cache 稳定减少 38.18%；
3. 视频不重叠 180 上未出现统计显著的 QA 下降；
4. 保留集 GPU 交换后，模型编码稳定加速 11.01%--11.87%；
5. 完整总编码加速为 1.60%--8.24%，仍需更多重复实验给出置信区间；
6. 当前方法仍保持 query-independent、training-free、fixed-shape 和 streaming-compatible。

### 16.8 下一步判断

1. 将保留 90 的 GPU 交换复验扩展到至少 3 次，报告均值、标准差和 95% 置信区间；
2. 对 121/144/196 token 做同一保留集的质量-速度-缓存曲线，确认 121 是否为 Pareto 点；
3. 在 OVO-Bench 更完整规模和 StreamingBench 上复验 QA 等价性；
4. 增加 Hierarchical Token Compression（分层 token 压缩）同模型、同 FPS、同 QA 约束对照；
5. 长视频触发 CPU offload（中央处理器卸载），验证缓存压缩能否转化为更长上下文容量；
6. 保留“时间槽分配 + 空间带宽分配”的两级统一方法，不重新堆叠 query-aware 或不规则 token 选择模块。

## 17. 结构化空间带宽 Pareto 验证：121 / 144 / 196 token

### 17.1 实验目的

上一轮确认 121-token 结构化网格能够显著降低缓存，并在 video-disjoint 180 上保持统计等价的 QA。
本轮进一步回答：

1. 121 是否压缩过强；
2. 144 是否能以较小存储代价换取更稳定的 QA；
3. 速度收益是否来自特定 GPU，而不是 token 预算；
4. 哪个预算是真正的 Pareto（帕累托最优）工作点。

### 17.2 连接与实验环境核验

实验开始前，`remote-docker=127.0.0.1:22` 一度返回不同 SSH 主机指纹。
没有关闭严格主机校验或直接接受未知指纹，而是依次核验：

- 本机 22 端口监听进程；
- Windows OpenSSH 主机指纹；
- Tabby 中继连接；
- Git 远端可达性；
- 远端 hostname、GPU 与仓库提交。

最终确认实验目标仍为：

```text
bj9-llm-g8a100-node00.alicn.idc.xiaomi.com
Linux
commit: dedc23e
```

该过程避免了将实验误运行在本机 Windows SSH 服务上的风险。

### 17.3 公平对照设置

三种空间预算：

```text
196 tokens: 原始对照
144 tokens: 12 x 12 structured grid
121 tokens: 11 x 11 structured grid
```

其他设置保持完全一致：

- OVO-Bench 真正视频不重叠保留集 90；
- `1 FPS`；
- 96 帧时间窗口；
- 每窗口 1 个显著性帧；
- 4 个近期锚点；
- `z=4.0`；
- query-independent（查询无关）；
- layer sparse（层内稀疏）关闭；
- 每次运行前使用同一个预热样本，统计时剔除。

执行两轮 GPU 轮换：

```text
Round 1:
196 -> GPU 1
144 -> GPU 2
121 -> GPU 3

Round 2:
121 -> GPU 1
196 -> GPU 2
144 -> GPU 3
```

结果目录：

```text
results/ovo_bench/structured_grid_pareto_heldout90_20260609
results/ovo_bench/structured_grid_pareto_heldout90_20260609_round2
results/ovo_bench/structured_grid_pareto_heldout90_20260609_summary
```

汇总目录包含：

- `pareto_summary.csv`；
- `pareto_summary.json`；
- `pareto_curve.png`。

### 17.4 QA 结果

两轮中每种配置的预测完全一致，说明当前推理路径具有确定性。

| 每保留帧 token | official QA | 正确样本数 |
|---:|---:|---:|
| 196 | 51.85% | 50 / 90 |
| 144 | 49.07% | 47 / 90 |
| 121 | 50.00% | 49 / 90 |

相对 196 的逐样本翻转：

| 对比 | 错误变正确 | 正确变错误 | McNemar p 值 |
|---|---:|---:|---:|
| 196 -> 144 | 7 | 10 | 0.629 |
| 196 -> 121 | 8 | 9 | 1.000 |

144 与 121：

```text
144 -> 121:
5 个错误变正确
3 个正确变错误
p = 0.727
```

所有差异均不显著，但结果明确否定了“token 越多，QA 必然越高”的简单假设。

### 17.5 两轮内部计时均值

| token | model encoding | visual encoding | context write | stream ingestion | total encode |
|---:|---:|---:|---:|---:|---:|
| 196 | 6.005s | 15.867s | 18.062s | 35.170s | 36.105s |
| 144 | 5.311s | 15.209s | 17.459s | 33.966s | 34.758s |
| 121 | 5.458s | 15.122s | 16.707s | 33.009s | 33.734s |

相对 196 的加速：

| token | model encoding | visual encoding | context write | stream ingestion | total encode |
|---:|---:|---:|---:|---:|---:|
| 144 | **11.56%** | 4.15% | 3.34% | 3.42% | 3.73% |
| 121 | 9.12% | **4.70%** | **7.50%** | **6.14%** | **6.57%** |

144 在狭义模型编码阶段更快，但 121 在视觉编码、上下文写入、流式摄入和总编码上更快。

### 17.6 缓存结果

| token | 平均 KV cache | 相对 196 |
|---:|---:|---:|
| 196 | 2.337 GB | 0 |
| 144 | 1.718 GB | -26.47% |
| 121 | 1.445 GB | **-38.18%** |

缓存下降在两轮 GPU 轮换中完全一致，属于由固定 token 形状决定的确定性收益。

### 17.7 核心 Insight 1：更多 token 不保证更高语义质量

观察：

```text
QA:
196 > 121 > 144
```

如果质量只由 token 数量决定，应满足：

```text
196 >= 144 >= 121
```

实际结果不满足该单调关系。

原因判断：

- 规则池化不仅改变 token 数量，也改变空间采样网格；
- `12 x 12` 与 `11 x 11` 的 receptive field（感受野）和对齐位置不同；
- 某些目标、文字或动作边界可能在一种网格下被更好聚合；
- 视觉语义质量由“保留多少”与“如何组织空间支持域”共同决定。

该 Insight 推翻：

> token 数是空间质量的唯一控制量。

导出的研究方向：

> 空间压缩应被表述为 semantic bandwidth allocation（语义带宽分配），而不是简单 token 截断。

### 17.8 核心 Insight 2：模型局部最优不等于系统最优

观察：

- 144 的 model encoding 加速为 11.56%，优于 121 的 9.12%；
- 但 121 的 total encode 加速为 6.57%，优于 144 的 3.73%；
- 121 的缓存下降也从 26.47% 扩大到 38.18%。

原因判断：

- GPU kernel（图形处理器内核）存在离散效率区间；
- projector 的局部耗时不直接等于上下文写入和缓存管理的整体收益；
- 更少的输出 token 会继续降低后端写入和 KV cache，即使局部 projector 没有线性加速。

该 Insight 推翻：

> 选择局部模型编码最快的预算，就能得到最快系统。

导出的论文原则：

> 工作点必须依据 compute-storage co-design（计算与存储联合设计）的系统 Pareto 前沿选择。

### 17.9 核心 Insight 3：121 是当前系统级 Pareto 点

以三个核心目标判断：

1. QA 越高越好；
2. total encode 越低越好；
3. KV cache 越低越好。

121 相比 144：

- QA：50.00% > 49.07%；
- total encode：33.734s < 34.758s；
- cache：1.445 GB < 1.718 GB。

因此 121 在三个系统目标上同时优于 144，144 被 121 支配，不属于当前系统级 Pareto 前沿。

当前方法默认工作点固定为：

```text
11 x 11 structured semantic grid
= 121 tokens per retained frame
```

这不是经验性选择最小 token，而是由质量、计算和存储三目标共同导出的工作点。

### 17.10 核心 Insight 4：墙钟时延与方法时延必须分离

首轮墙钟：

```text
196: 33m58s
144: 34m32s
121: 34m30s
```

第二轮墙钟：

```text
121: 33m59s
196: 34m39s
144: 34m25s
```

墙钟排序随 GPU 和视频解码波动改变，而内部模型计时与缓存趋势更稳定。

原因：

- H.264 视频解码；
- 文件系统 I/O；
- CPU 预处理；
- GPU 调度与频率；
- 外部共享服务器任务。

因此论文实验必须分层报告：

1. synchronized model timing（同步模型计时）；
2. stream ingestion timing（流式摄入计时）；
3. wall-clock end-to-end timing（墙钟端到端计时）。

不能用单次墙钟结果替代方法加速结论。

### 17.11 当前结论与下一步

当前结论：

```text
Temporal slot allocation:
anchor-preserved saliency

Spatial bandwidth allocation:
11 x 11 structured semantic grid

Default bandwidth:
121 tokens / retained frame
```

下一步不继续微调 121/144 之间的预算，而应转向更有论文价值的验证：

1. 在 StreamingBench 和更完整 OVO-Bench 上验证 121 的 QA 泛化；
2. 增加 Hierarchical Token Compression（分层 token 压缩）公平基线；
3. 测试长视频 CPU offload（中央处理器卸载）和最大可支持上下文长度；
4. 将视觉编码加速、上下文写入和缓存容量放入同一系统主表；
5. 分析 11 x 11 网格为何优于 12 x 12，判断是否能形成“空间支持域对齐”而非预算搜索的更一般理论。

## 18. 实时流边界修正、系统收益归因与 QA 失效定位

### 18.1 实验目的

本轮不再笼统使用“端到端”描述所有墙钟时间，而是回答三个更严格的问题：

1. 对实时采集输入的视频流，哪些阶段属于本方法的系统边界；
2. Insight 2 中“局部编码最快不等于系统最快”到底由什么组件造成；
3. 121-token 方案的约 50% QA 处于什么绝对水平，损失集中在哪里，以及下一版算法应如何修正。

代码起点：

```text
05cca7c feat: 明确评测指标并增加带宽组件分析
```

### 18.2 科研问题边界

研究对象固定为：

```text
已解码 RGB 帧 x_t 按真实时间顺序到达
-> 查询无关帧选择
-> 图像预处理与 ViT 编码
-> 空间语义带宽压缩
-> 多模态投影
-> 视觉上下文写入与 ReKV/KV cache 更新
-> 查询到达后的检索与回答
```

以下时间不属于方法系统边界：

- 摄像头采集耗时；
- 网络传输耗时；
- 视频文件读取；
- H.264 等离线视频解码；
- 等待下一帧按真实时间轴到达的自然间隔。

原因是本论文优化的是“帧到达后的在线模型计算和上下文写入”，不是视频存储或编解码系统。
OVO-Bench 的视频文件读取仅用于把离线 benchmark 适配成帧流，不得进入论文主加速比。

### 18.3 指标定义

后续实验固定使用以下定义：

| 指标 | 中文含义 | 包含阶段 | 不包含阶段 | 论文用途 |
|---|---|---|---|---|
| `model_encoding` | 模型内部视觉编码 | patch embedding + ViT encoder | 图像预处理、帧选择、投影后写入 | 分析 ViT 内部计算 |
| `visual_encoding` | 视觉编码 | 图像预处理 + model encoding | 帧选择、上下文写入 | 分析视觉前端 |
| `stream_ingestion` | 已打点流式摄入 | 帧选择 + visual encoding + context write | 未单独打点的 Python 开销 | 组件归因 |
| `online_video_processing` | 在线视频处理 | `encode_video` 内全部同步计算和运行时开销 | 文件读取、等待帧到达、初始化提示、QA | **论文主效率指标** |
| `online_model_pipeline` | 在线模型流水线 | 初始化提示 + online video processing + QA | 文件读取、等待帧到达 | 系统辅助指标 |
| `observed_stream_duration` | 输入流时长 | 到达帧数 / 采样 FPS | 程序运行时间 | 实时能力归一化 |
| `realtime_compute_ratio` | 实时计算比 | online video processing / 输入流时长 | 离线读取 | 小于 1 表示能跟上实时流 |
| `online_processing_fps` | 在线处理吞吐 | 到达帧数 / online video processing | 离线读取 | 最大摄入能力 |
| `official_three_group_average` | OVO 官方三组宏平均 | 任务内准确率、组内宏平均、三组等权平均 | 样本数加权微平均 | QA 主指标 |
| `kv_cache_memory` | 上下文缓存占用 | QA 前 ReKV/KV cache 实际字节数 | 模型权重 | 存储主指标 |

历史字段 `total_encode_video_sec` 保留用于兼容旧脚本，但其准确含义就是
`online_video_processing`，后续正文不再使用容易误解的 `total encode` 名称。

`full_pipeline` 包含离线视频文件读取，仅作为 benchmark 数据适配器诊断项，
明确禁止用于论文主加速比。

代码已增加机器可读的：

```text
metric_definitions
paper_reporting_policy
realtime_metrics
```

本地验证：

```text
39 tests passed
```

### 18.4 核心 Insight 2 的精确定位

结果目录：

```text
results/ovo_bench/structured_bandwidth_component_profile_20260611/profile.json
```

设置：

- 单进程；
- 同一张 A100；
- 固定 8 帧 ViT 特征；
- warmup 10 次；
- 每个操作重复 50 次；
- 共 6 轮；
- 每轮轮换操作顺序，降低 GPU 升降频和缓存热度偏差。

CUDA event（CUDA 事件）中位时延：

| 组件 | 196 token | 144 token | 121 token |
|---|---:|---:|---:|
| projector only（仅投影器） | 2.096 ms | 0.243 ms | **0.198 ms** |
| structured pool（结构池化） | - | 2.112 ms | **1.863 ms** |
| pool + projector | 17.251 ms | 4.983 ms | **4.801 ms** |

相对 196：

```text
144 token visual tail speedup = 3.46x
121 token visual tail speedup = 3.59x
```

这证明先前“144 的 model encoding 比 121 更快”不是预算的真实因果结果。
ViT backbone（视觉主干）在三种预算下理论上完全相同，但跨进程计时分别出现
4.39 / 4.70 / 4.85 秒，差异来自 GPU、进程和视频内容波动。

真正受 token 预算控制的组件中，121 始终快于 144：

```text
projector:        0.198 ms < 0.243 ms
pool + projector: 4.801 ms < 4.983 ms
context write:   16.707 s  < 17.459 s
```

对 196 -> 121 的系统收益做因果闭合：

```text
结构化视觉尾部预计节省 = 1.082 s
上下文写入实际节省     = 1.355 s
两者合计预计节省       = 2.436 s
在线处理实测节省       = 2.371 s
```

预计值是实测值的 102.8%，误差约 2.8%。
按预计收益拆分：

```text
视觉尾部贡献约 44.4%
上下文写入贡献约 55.6%
```

**核心 Insight 2：**

> 121 的系统优势不是“投影器偶然更快”，而是固定语义带宽同时减少视觉尾部计算与语言模型上下文写入；二者共同解释了几乎全部在线收益。

进一步的优化优先级因此明确为：

1. 继续降低每个保留帧的语义写入带宽；
2. 在固定带宽内提高细节保真度；
3. 不再把时间主要花在微调已经很小的 projector kernel（投影器内核）。

### 18.5 121-token 的绝对 QA 水平与 SOTA 对照

必须区分评测协议，不能把不同任务子集直接混成一个排行榜。

#### 12 任务三组宏平均

| 方法 | 模型/协议 | QA |
|---|---|---:|
| 当前时间稀疏 + 196 token | LLaVA-OneVision-7B，video-disjoint 90 | 51.85% |
| 当前时间稀疏 + 121 token | LLaVA-OneVision-7B，video-disjoint 90 | 50.00% |
| LLaVA-OneVision | OVO-Bench 官方完整集 | 52.74% |
| StreamForest | OVO-Bench 公开结果 | 55.60% |
| Gemini-1.5-Pro | OVO-Bench 官方完整集 | 63.00% |
| Human（人类） | OVO-Bench 官方完整集 | 92.81% |

121 相对同协议 196 的质量保持率：

```text
50.00 / 51.85 = 96.4%
绝对下降 = 1.85 个百分点
```

该差异在当前 90 样本上 McNemar 检验 `p=1.000`，未达到统计显著。
但“未显著”不等于已经证明等价，仍需要更大样本。

#### 9 任务在线子集

SimpleStream 和 OASIS 使用 Real-Time Visual Perception（实时视觉感知）
与 Backward Tracing（向后追溯）两个类别，不包含 3 个 Forward（前向）任务。

按相同两类别宏平均投影：

| 方法 | 模型 | 9 任务 QA |
|---|---|---:|
| 当前 196 token | LLaVA-OneVision-7B | 54.17% |
| 当前 121 token | LLaVA-OneVision-7B | 50.00% |
| 官方/公开 LLaVA-OneVision | 7B | 53.85% |
| HERMES | Qwen2.5-VL-7B | 59.20% |
| SimpleStream | Qwen2.5-VL-7B，4 帧 | 65.13% |
| OASIS | Qwen3-VL-8B | 67.68% |
| SimpleStream | Qwen3-VL-8B | 67.70% |
| SimpleStream | Qwen3-VL-32B，8 帧 | 74.09% |

来源：

- OVO-Bench CVPR 2025：<https://openaccess.thecvf.com/content/CVPR2025/html/Niu_OVO-Bench_How_Far_is_Your_Video-LLMs_from_Real-World_Online_Video_CVPR_2025_paper.html>
- OASIS NeurIPS 2025：<https://proceedings.neurips.cc/paper_files/paper/2025/hash/71e983f4e703f67c01d2d1c7c2429d24-Abstract-Conference.html>
- SimpleStream 2026：<https://arxiv.org/html/2604.02317v1>

协议结论：

1. 当前 196-token 保留集结果与公开 LLaVA-OneVision-7B 的绝对水平接近，说明评测没有明显失真；
2. 121-token 在完整 12 任务宏平均上只下降 1.85 个百分点；
3. 但在 9 任务投影上下降 4.17 个百分点，说明 Forward 任务的收益掩盖了部分向后追溯和实时感知损失；
4. 当前方法不要求达到精度 SOTA，但必须报告与 SOTA 的差距，并证明作为效率插件不会额外限制更强主干。

因此论文质量主张应写成：

> 在同一主干、同一协议下报告质量保持和效率提升；SOTA 仅用于说明绝对任务难度和主干上限，不把不同模型规模的精度差异归因于压缩方法。

### 18.6 QA 失效定位

新增：

```text
scripts/analyze_bandwidth_qa_failures.py
```

结果：

```text
results/ovo_bench/structured_grid_pareto_heldout90_20260609_analysis/
```

逐样本翻转：

```text
positive flip（错误变正确）= 8
negative flip（正确变错误）= 9
```

注意：失效分析中的逐样本微平均为 55.56% -> 54.44%，
仅用于定位样本；论文 QA 主指标仍是 51.85% -> 50.00% 的三组宏平均。

负翻转集中在：

```text
HLD: -2
FPD: -2
SSR: -2
OCR: -1
CRR: -1
REC: -1
```

正翻转集中在：

```text
CRR: +3
REC: +2
ACR: +1
ASI: +1
OJR: +1
```

负翻转与正翻转的统计对比：

| cohort（样本组） | 平均到达帧 | 平均保留帧 | 近期锚点占比 |
|---|---:|---:|---:|
| negative flip | 165.9 | 6.44 | 64.25% |
| positive flip | 162.9 | 6.25 | 65.24% |

二者非常接近，因此当前证据不支持：

- 视频越长越容易负迁移；
- 近期锚点占比过高是主要原因；
- 单纯增加刷新频率就能解决空间压缩损失。

逐问题检查显示：

- OCR 文本块、细粒度操作准备、教程步骤状态等任务更依赖局部高频细节；
- 动作计数和部分可回答性判断在池化后反而改善，说明规则池化可能抑制冗余噪声；
- EPM 在 196 和 121 下均为 0，属于当前主干或流式记忆能力上限，不能归因于空间压缩。

**核心 Insight 5：**

> 规则空间池化同时产生“全局语义去噪”和“局部细节损失”；固定预算的关键不是继续搜索 10x10、11x11、12x12，而是显式分离低频全局语义与高频细节残差。

### 18.7 下一版算法方向：固定带宽的全局基底 + 细节残差

下一版不增加查询感知，不引入任务标签，也不扩大 121-token 缓存预算。

候选结构：

```text
196 个 ViT 原生 token
        |
        +-> 10 x 10 structured base（结构化全局基底）= 100 token
        |
        +-> reconstruction residual（重建残差）选择 = 21 token
        |
        +-> 固定 121-token semantic packet（语义包）
```

设计依据：

1. 100 个规则网格 token 保证完整空间覆盖和低频场景语义；
2. 21 个残差 token 保留最难被规则池化重建的局部区域；
3. 总输出仍为 121，ReKV block size（缓存块大小）不变；
4. 选择完全由视觉特征自身决定，保持 query-independent（查询无关）和即插即用；
5. 同一固定预算内同时服务全局动作/场景语义与 OCR/状态/边界细节。

残差分数优先采用：

```text
native token
与
structured base 上采样回原网格后的特征
之间的局部重建误差
```

它比堆叠文字检测器、光流、目标检测器更统一，也更符合顶会方法的简洁性。

论文方法逻辑可压缩为一个原则：

> 将密集视觉流编码为固定带宽语义包：时间上只保留语义事件，空间上用全局基底承载稳定语义、用稀疏残差承载不可预测细节。

### 18.8 正在执行的边界实验

远端主机重新核验：

```text
hostname: bj9-llm-g8a100-node00.alicn.idc.xiaomi.com
commit: 05cca7c
```

实验 A：100-token 固定网格边界

```text
GPU 7
1 FPS
video-disjoint heldout 90 + 1 warmup
10 x 10 structured grid
results/ovo_bench/structured_grid_budget100_heldout90_20260611/
```

目的：

- 判断 121 是否已经是固定网格的质量边界；
- 判断继续压低上下文写入是否仍有 QA 可接受空间；
- 为“100 base + 21 residual”提供固定预算对照。

实验 B：真正 Dense-1FPS 基线

```text
GPU 0
1 FPS
每个到达帧均执行完整视觉编码和上下文写入
results/ovo_bench/realtime_dense_heldout90_20260611/
```

目的：

- 得到完整方法相对逐帧密集视觉流的在线处理加速；
- 将时间稀疏收益、空间带宽收益和联合收益分开报告；
- 避免用 121 vs 196 的 6.57% 增量收益代表整套方法。

### 18.9 全局基底 + 细节残差最小实现

新增策略：

```text
vit_output_token_policy = structured_residual
vit_output_token_budget = 121
vit_output_base_tokens = 100
```

涉及文件：

```text
model/vision_accelerator/token_reducer.py
model/vit_patch.py
video_qa/base.py
video_qa/run_eval.py
tests/test_token_reducer.py
```

实现保证：

1. 输入 ViT 原生规则网格首先双线性压缩为 10 x 10 全局基底；
2. 将基底上采样回原网格，计算逐位置特征重建误差；
3. 选取误差最大的 21 个原生 token 作为细节残差；
4. 拼接后严格输出 121 token；
5. ReKV 块长度、缓存预算和每帧写入预算与当前 11 x 11 对照完全相同；
6. 全过程不读取 query、任务类型或标注。

本地验证：

```text
41 tests passed
Python compile passed
```

完整 QA 前的准入条件：

1. 同进程 A100 微基准中，残差评分开销不能抵消 pre-projector（投影前）压缩收益；
2. 固定 121 token 下，相比 11 x 11 网格优先修复 HLD / FPD / OCR / SSR 负翻转；
3. 不允许依靠增加缓存或 query-aware（查询感知）选择获得精度。

### 18.10 时间选择配置审计与第 17 节口径更正

对历史结果逐答案和运行统计进行复核后确认：

```text
structured_grid_pareto 的 none196
与
heldout90_v2 的 periodic96

90 / 90 个预测答案完全一致
保留帧均为 695
QA 均为 51.85%
```

而早期 `stable_z4p0`：

```text
保留帧 = 691
candidate frames（候选帧）> 0
QA = 53.70%
```

根因是两条路径的 `semantic_selection_feature_source（语义选择特征源）` 不同：

- 第 17 节空间 Pareto 实际使用 `raw_rgb（仅原始像素）` 或周期等价路径；
- 53.70% 的稳定显著性使用 `hybrid（原始像素提名 + ViT embedding 验证）` 两阶段路径。

因此第 17 节中“空间 Pareto 已叠加稳定显著性时间选择”的表述不准确。
该节的 196 / 144 / 121 仍是有效的**周期时间选择下空间预算对照**，
但不能代表当前最佳时间选择器上的空间结果。

**核心 Insight 6：**

> 时间选择策略和空间压缩策略必须形成二维公平实验矩阵；不能只凭保留帧数接近，就假设两种时间选择产生了相同语义流。

已停止三条错误配置的早期运行，并按以下统一配置重启：

```text
semantic_selection_feature_source = hybrid
semantic_candidate_multiplier = 1
semantic_raw_proposal_policy = saliency_gated
semantic_saliency_z_threshold = 4.0
```

正确公平对照目录：

```text
results/ovo_bench/stable_hybrid_spatial_heldout90_20260611/none196_gpu3
results/ovo_bench/stable_hybrid_spatial_heldout90_20260611/grid121_gpu2
results/ovo_bench/stable_hybrid_spatial_heldout90_20260611/residual100plus21_gpu1
```

### 18.11 periodic 时间选择下的 100-token 边界结果

结果目录：

```text
results/ovo_bench/structured_grid_budget100_heldout90_20260611/
```

正式统计自动排除 1 个 warmup（预热）样本，保留 90 个评测样本。

主结果：

| 指标 | periodic-196 | periodic-121 | periodic-100 |
|---|---:|---:|---:|
| OVO 三组宏平均 | 51.85% | 50.00% | **51.85%** |
| 保留帧 | 695 | 695 | 695 |
| 每保留帧 token | 196 | 121 | 100 |
| 平均 KV cache | 2.337 GB | 1.445 GB | **1.195 GB** |
| 相对 196 缓存下降 | 0 | 38.18% | **48.87%** |

periodic-100 的任务组：

```text
backward（向后追溯）= 38.89%
realtime（实时感知）= 63.89%
forward（前向响应）= 52.78%
```

periodic-196 的任务组：

```text
backward = 33.33%
realtime = 75.00%
forward = 47.22%
```

因此虽然两者总体宏平均都为 51.85%，100-token 相对 196：

```text
positive flips = 12
negative flips = 13
worst-group drop = realtime -11.11 percentage points
```

主要负迁移：

```text
OCR: -3
ATR: -2
FPD: -2
HLD: -1
STU: -1
```

主要增益：

```text
ACR: +2
OJR: +2
SSR: +2
EPM: +1
ASI: +1
```

**核心 Insight 7：**

> 总体宏平均持平不等于语义能力保持；空间压缩可能在任务组之间重新分配能力，必须增加 worst-group drop（最差任务组下降）约束。

因此 100-token 规则网格暂时不能作为最终工作点。
它是更强的效率边界和残差方法的必要对照：

- 全局语义和缓存效率很强；
- 实时细节感知损失明显；
- 正好需要额外的细节残差通道补偿。

本轮并行运行下的探索性时间：

```text
online video processing = 30.375 s
context write = 14.759 s
online processing FPS = 908.3
realtime compute ratio = 0.00110
```

相对单轮 periodic-196：

```text
context write 下降 16.39%
online video processing 下降 13.29%
```

但当前多条实验共享 CPU 和存储，以上墙钟变化只作探索，
正式论文时延必须在工作点确定后顺序执行并进行 GPU 轮换复验。

### 18.12 全局基底 + 细节残差的 GPU 准入结果

结果：

```text
results/ovo_bench/structured_residual_component_profile_20260611/
results/ovo_bench/structured_grid100_component_profile_20260611/
```

同进程、同 GPU、固定 8 帧 ViT 特征的视觉尾部中位时延：

| 空间策略 | 8 帧 pool + projector |
|---|---:|
| 196 dense tail | 17.240 ms |
| 100 regular grid | **4.752 ms** |
| 121 regular grid | 4.794 ms |
| 100 base + 21 residual | 7.242 ms |

结论：

1. 100 与 121 规则网格的视觉尾部几乎同速，差值仅 0.042 ms / 8 帧；
2. 100 的主要额外系统收益来自更少的上下文写入和缓存，而不是 projector 再次加速；
3. 残差策略比规则 121 多 2.448 ms / 8 帧；
4. 按 695 个保留帧折算，残差评分预计只增加约 0.213 秒；
5. 残差策略相对 196 密集视觉尾部仍有约 2.38x 加速。

因此残差策略通过完整 QA 准入：

> 允许用约 0.21 秒在线计算换取细节任务恢复，同时保持固定 121-token 缓存预算。

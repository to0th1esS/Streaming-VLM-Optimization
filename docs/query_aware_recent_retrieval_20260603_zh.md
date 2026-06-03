# Query-aware Recent Retrieval 实验记录（2026-06-03）

## 1. 实验动机

上一轮 `semantic_recency_keep_frames` 说明，前端只补写最近帧可以提高 QA overlap，但不能稳定解决 `latest/current-state` 类问题的旧视觉干扰。典型现象是：

```text
Question: What setting is portrayed in the latest clip?
GT: A kitchen setting.
K=4 recency: The latest clip portrays a kitchen setting with a Christmas tree in the background.
```

这说明问题不只是“最后几帧有没有被 ViT 编码”，还包括：

> 已经写入 LLM KV cache 的旧视觉 token，在当前问题阶段是否仍被等权参与回答。

因此，本轮验证从前端 sparse write 扩展到后端 query-time routing：

```text
长期语义记忆负责历史信息；
短期 recent blocks 负责 latest/current-state；
问题到来时按问题类型选择使用哪部分视觉 KV。
```

## 2. 方法设计

新增开关：

```text
--enable_query_aware_retrieval
--latest_retrieval_blocks
--latest_query_terms
```

当前是最小可验证版本：

1. 前端仍使用上一轮最优候选：
   - `refresh_interval = 64`
   - `skip_threshold = 0.3`
   - `semantic_recency_keep_frames = 4`
2. 如果问题命中 latest/current/setting/where 等触发词：
   - 不走 ReKV 默认内部 retrieval；
   - 显式传入最近 N 个视觉块 `retrieved_indices`；
   - 只让最近视觉块参与问题 prefill。
3. 如果问题未命中：
   - 保持 ReKV 默认内部 retrieval。

直观结构：

```text
Streaming frames
    |
    v
Semantic Stream Gate
    |-- keep semantic drift / refresh frames --> Long-term semantic memory
    |-- keep recent K frames before QA -------> Short-term recent buffer
                                                |
Question arrives ------------------------------|
    |
    |-- latest/current query --> retrieve recent N visual blocks
    |-- other query -----------> internal ReKV retrieval
    v
VLM answer
```

这个版本没有改底层 attention，也没有重写 cache manager，而是利用 ReKV 已有的 external retrieval 接口完成后端路由验证。

## 3. 实验设置

远程服务器：

```text
Host: remote-docker
Repo: /home/yangjin/1#Streaming-VLM-Optimization
Model root: /home/mllm/models
GPU: CUDA_VISIBLE_DEVICES=0
```

模型与数据：

```text
Model: llava_ov_7b
Dataset: RVS-Ego open QA subset
Sample FPS: 0.2
Videos / QA: 8 / 24
Dense baseline: r1/t0
Front-end sparse setting: r64/t0.3 + recency K=4
Back-end query routing: latest_recent blocks = 4 / 8 / 16
```

命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python scripts/run_semantic_stream_sweep.py \
  --model llava_ov_7b \
  --anno-path data/rvs/ego/ego4d_oe.json \
  --output-dir results/rvs_ego_query_routing_20260603/r64_t0p3_recency4_qrb4 \
  --sample-fps 0.2 \
  --retrieve-size 64 \
  --refresh-intervals 64 \
  --thresholds 0.3 \
  --compute-gates true \
  --repeats 1 \
  --debug false \
  --semantic-recency-keep-frames 4 \
  --enable-query-aware-retrieval true \
  --latest-retrieval-blocks 4
```

## 4. 单次结果

与 dense repeat0 对比，全量 8 videos / 24 QA 口径：

| Method | Latest Blocks | Kept Frames | Token Reduction | Encode Sec | Speedup | Mean Token-F1 | W/T/L vs Dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| Recency only | all/internal | 157 / 2046 | 92.33% | 53.07 | 3.49x | 0.2937 | 9 / 15 / 0 |
| Query routing | 4 | 157 / 2046 | 92.33% | 51.63 | 3.59x | 0.3076 | 10 / 14 / 0 |
| Query routing | 8 | 157 / 2046 | 92.33% | 52.13 | 3.56x | 0.3001 | 8 / 16 / 0 |
| Query routing | 16 | 157 / 2046 | 92.33% | 48.73 | 3.80x | 0.2937 | 9 / 15 / 0 |

注意：query routing 不改变前端写入量，因此 token/cache 写入压缩保持 92.33%。收益来自问题阶段减少旧视觉块参与 latest/current 类问题。

## 5. 关键样例

`000005 What setting is portrayed in the latest clip?`

| Method | Prediction |
|---|---|
| Recency only K=4 | The latest clip portrays a kitchen setting with a Christmas tree in the background. |
| Query routing qrb=4 | The setting portrayed in the latest clip is a kitchen. |
| Query routing qrb=8 | The setting portrayed in the latest clip is a kitchen. |
| Query routing qrb=16 | The latest clip portrays a kitchen setting with a Christmas tree in the background. |

这里的现象非常关键：

1. qrb=4/8 修复了旧视觉干扰。
2. qrb=16 又退化回旧干扰。
3. 因此，latest/current 问题不需要“更多视觉上下文”，而需要“更干净的当前状态上下文”。

这可以作为后续论文方法的核心 insight。

## 6. LLM Judge 结果

对最优候选 qrb=4 使用 Qwen2.5-VL-7B-Instruct judge：

```text
valid judgments: 21 / 24
dense correct rate: 61.9%
sparse correct rate: 66.7%
better / same / worse: 1 / 13 / 7
dense-only correct: 0
sparse-only correct: 1
both correct: 13
both wrong: 7
```

按问题类型：

| Category | Dense Acc | Query Routing Acc | Wins | Losses |
|---|---:|---:|---:|---:|
| action | 33.3% | 33.3% | 0 | 0 |
| object | 50.0% | 50.0% | 0 | 0 |
| other | 33.3% | 33.3% | 0 | 0 |
| scene_latest | 66.7% | 83.3% | 1 | 0 |
| yes_no | 100.0% | 100.0% | 0 | 0 |

对比上一轮 recency-only K=4：

```text
Recency-only K=4: scene_latest dense/sparse = 66.7% / 66.7%, losses = 1
Query routing qrb=4: scene_latest dense/sparse = 66.7% / 83.3%, losses = 0
```

## 7. Repeat3 稳定性

qrb=4 repeat3，全量 8 videos / 24 QA 口径：

| Repeat | Kept Frames | Token Reduction | Encode Sec | Speedup | Mean Token-F1 | W/T/L vs Dense |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 157 / 2046 | 92.33% | 49.25 | 3.76x | 0.3076 | 10 / 14 / 0 |
| 1 | 157 / 2046 | 92.33% | 49.22 | 3.79x | 0.3076 | 10 / 14 / 0 |
| 2 | 157 / 2046 | 92.33% | 49.54 | 3.72x | 0.3076 | 10 / 14 / 0 |

这说明 qrb=4 的结果不是单次随机波动。

## 8. 阶段性结论

本轮实验把方法从“只减少 ViT 编码和视觉写入”推进到“流式视觉语义状态管理”：

```text
Front-end: sparse semantic writing
Back-end: query-aware visual state routing
```

当前最强的候选配置：

```text
r64 / t0.3
semantic_recency_keep_frames = 4
enable_query_aware_retrieval = true
latest_retrieval_blocks = 4
```

它在 RVS-Ego repeat3 上达到：

```text
ViT/visual encode speedup: about 3.7x
visual token/cache write reduction: 92.33%
mean token-F1: 0.3076
W/T/L vs dense: 10 / 14 / 0
Qwen judge sparse correctness: 66.7% vs dense 61.9%
scene_latest: 83.3% vs dense 66.7%
```

## 9. 对最终论文方法的启发

这组实验可以抽象成一个更优雅的方法，而不是工程堆叠：

> Streaming video VLMs should maintain a sparse semantic memory and a compact current-state buffer, then route each query to the appropriate temporal state.

可以命名为：

```text
Query-aware Semantic Stream Routing
```

核心思想：

1. 不把 dense video stream 全部写入 VLM。
2. 用 semantic gate 写入长期语义变化。
3. 用短 recent buffer 保护当前状态。
4. 在问题阶段按 query intent 选择历史语义 memory 或当前状态 buffer。

下一步建议：

1. 在 RVS-Movie 上验证 qrb=4 是否同样有效。
2. 把 latest/current 路由从关键词规则升级为轻量 query classifier。
3. 对 long-horizon/action/object 问题设计不同的 semantic memory routing，而不是只处理 latest。
4. 进一步记录 QA 阶段 retrieval blocks 数量和生成耗时，量化后端 cache 选择的收益。

## 10. 结果文件

```text
results/rvs_ego_query_routing_20260603/r64_t0p3_recency4_qrb4
results/rvs_ego_query_routing_20260603/r64_t0p3_recency4_qrb8
results/rvs_ego_query_routing_20260603/r64_t0p3_recency4_qrb16
results/rvs_ego_query_routing_20260603/r64_t0p3_recency4_qrb4_repeats3
```

关键分析文件：

```text
analysis/overlap.json
analysis/dense_vs_query_routing_compare.json
analysis/dense_vs_query_routing_judge_qwen25vl7b.json
analysis/category_summary/category_summary_all.json
```

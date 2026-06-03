# 语义流 Recency Correction 实验记录（2026-06-03）

## 1. 实验目的

前一轮 RVS-Ego 实验表明，速度优先的语义流视觉写入策略已经可以在 QA 指标基本不下降的前提下，把视觉写入量压到约 6% 左右，并带来 3x 以上端到端视觉编码加速。但 r64/t0.3 的激进设置在 `scene_latest` 类问题上出现了一个代表性失败：

```text
Question: What setting is portrayed in the latest clip?
GT: A kitchen setting.
Dense: The latest clip shows a kitchen setting with a focus on the person's hands and the kitchen environment.
Sparse: The latest clip shows a kitchen setting with a Christmas tree in the background.
```

这个失败提示：我们的目标不应继续追求逐帧视觉特征还原，而应围绕“流式 QA 需要的语义状态”设计视觉写入策略。具体到本轮实验，需要回答：

1. 最近帧是否需要被强制写入，才能保护 current-state / latest 类问题。
2. 强制写入最近 K 帧是否会显著破坏速度与 token/cache 压缩。
3. 如果简单增加最近帧仍不能修复 stale 信息，下一步是否应该转向视觉写入后的上下文/缓存管理。

## 2. 方法改动

新增 `semantic_recency_keep_frames` 参数，默认值为 0，保持旧实验完全不变。

当每个 streaming QA 问题到来前，`ReKVStreamVQA` 会知道本次问题窗口需要新编码的帧范围 `[encode_start_idx, encode_end_idx)`。如果 `semantic_recency_keep_frames = K`，则窗口末尾最近 K 帧会被标记为 recency window。

在 `SemanticStreamGate` 中：

- 原本会因为 `reference`、`refresh` 或 `drift_keep` 被保留的帧，仍按原逻辑保留并更新 anchor。
- 原本会被 `skip` 的窗口末尾最近帧，被改为 `recency_keep` 并写入视觉 token。
- 默认 `semantic_recency_updates_anchor = false`，即 recency 只补充当前 QA 所需视觉上下文，不改变主语义门控的 anchor 轨迹。

这使 recency correction 成为一个独立的最小 correction，而不是重写原门控策略。

涉及代码：

- `model/vision_accelerator/semantic_stream.py`
- `model/vit_patch.py`
- `video_qa/base.py`
- `video_qa/rekv_stream_vqa.py`
- `video_qa/run_eval.py`
- `scripts/run_semantic_stream_sweep.py`

## 3. 验证设置

远程服务器：

```text
Host: remote-docker
Repo: /home/yangjin/1#Streaming-VLM-Optimization
Model root: /home/mllm/models
Dataset root: /home/mllm/datasets
GPU: CUDA_VISIBLE_DEVICES=0
```

模型与数据：

```text
Model: llava_ov_7b
Dataset: RVS-Ego open QA subset
Sample FPS: 0.2
Samples: 8 videos / 24 QA
Dense baseline: r1/t0
Sparse base: r64/t0.3, compute gate on, ViT layer sparse off
```

主要命令模板：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python scripts/run_semantic_stream_sweep.py \
  --model llava_ov_7b \
  --anno-path data/rvs/ego/ego4d_oe.json \
  --output-dir results/rvs_ego_recency_20260603/r64_t0p3_recencyK \
  --sample-fps 0.2 \
  --retrieve-size 64 \
  --refresh-intervals 64 \
  --thresholds 0.3 \
  --compute-gates true \
  --repeats 1 \
  --debug false \
  --semantic-recency-keep-frames K
```

## 4. 结果总表

同一 repeat0 口径，与 dense repeat0 对比：

| Recency K | Kept Frames | Token Reduction | Encode Sec | Speedup vs Dense | Mean Token-F1 | Token W/T/L vs Dense |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 127 / 2046 | 93.79% | 52.92 | 3.50x | 0.2489 | 4 / 19 / 1 |
| 1 | 134 / 2046 | 93.45% | 52.91 | 3.50x | 0.2401 | 4 / 19 / 1 |
| 2 | 141 / 2046 | 93.11% | 51.58 | 3.59x | 0.2701 | 5 / 19 / 0 |
| 4 | 157 / 2046 | 92.33% | 53.07 | 3.49x | 0.2937 | 9 / 15 / 0 |
| 8 | 186 / 2046 | 90.91% | 54.18 | 3.42x | 0.2781 | 6 / 17 / 1 |

Qwen2.5-VL-7B judge（dense vs K=4）：

```text
valid judgments: 20 / 24
dense correct rate: 65.0%
sparse correct rate: 65.0%
better / same / worse: 1 / 12 / 7
dense-only correct: 1
sparse-only correct: 1
both correct: 12
both wrong: 6
```

按问题类型：

| Category | Dense Acc | Sparse K=4 Acc | Wins | Losses |
|---|---:|---:|---:|---:|
| action | 33.3% | 33.3% | 0 | 0 |
| object | 50.0% | 50.0% | 0 | 0 |
| other | 33.3% | 33.3% | 0 | 0 |
| scene_latest | 66.7% | 66.7% | 1 | 1 |
| yes_no | 100.0% | 100.0% | 0 | 0 |

## 5. 关键现象

### 5.1 Recency K=2/4 是低成本有效 correction

K=2 和 K=4 相比 K=0 只额外写入 14/30 帧级别的视觉 token，整体 token reduction 仍保持在 93.1% / 92.3%。这说明“在问题到来前补写短最近窗口”对速度和 cache 写入量的破坏很小。

K=4 在 token overlap 上最强：

```text
mean token-F1: 0.2937
token W/T/L vs dense: 9 / 15 / 0
speedup: 3.49x
token reduction: 92.33%
```

这支持一个论文级 insight：

> 高效流式视觉推理不需要均匀保留帧，而需要把视觉预算分配给“语义变化帧”和“问题到来前的状态校正窗口”。

### 5.2 继续增大 K 不一定更好

K=8 写入帧数增加到 186/2046，token reduction 降到 90.91%，但 mean token-F1 从 K=4 的 0.2937 下降到 0.2781，且 token W/T/L 重新出现 1 个 loss。

这说明 recency correction 不是越大越好。更大的最近窗口会引入更多冗余视觉上下文，可能增加回答时的干扰，而不是单调提升 QA。

### 5.3 `latest` 失败没有被短窗口彻底修复

关键样例 `000005`：

| K | Prediction |
|---:|---|
| 0 | The latest clip shows a kitchen setting with a Christmas tree in the background. |
| 1 | The latest clip portrays a kitchen setting with a Christmas tree in the background. |
| 2 | The latest clip portrays a kitchen setting with a Christmas tree in the background. |
| 4 | The latest clip portrays a kitchen setting with a Christmas tree in the background. |
| 8 | The latest clip portrays a kitchen setting with a wooden cutting board, a large metal bowl, and various kitchen utensils. |

K=8 去掉了 `Christmas tree`，但没有稳定优于 K=4，说明问题不是简单的“缺最后一帧”。更可能是：

1. 旧视觉 token 已经写入 LLM KV cache，问题到来时仍会干扰回答。
2. ViT 侧 sparse 写入解决的是视觉编码与写入数量，但还没有控制“哪些已写入视觉状态应该在当前问题中被优先使用”。
3. latest/current-state 问题需要一个后端上下文选择机制，而不仅是前端补写更多帧。

## 6. 阶段性结论

本轮 recency correction 不是最终方法，但它提供了重要方向判断。

可以保留的有效设计：

- 语义变化帧作为长期 sparse memory。
- 问题到来前短 recency window 作为当前状态校正。
- K=4 可作为当前默认候选，因为它在本轮实验中达到较好的速度/QA 平衡。

不应继续深入的方向：

- 只通过继续增大最近帧数量修复 latest 问题。
- 只看 feature cosine / MSE 或 token-F1 来否定速度优先方案。
- 把所有问题都当成同一种视觉需求处理。

更有论文设计感的下一步：

> Dense video stream should be converted into a sparse semantic stream with two complementary states: a long-horizon semantic memory and a short-horizon query-time state correction buffer. The final answer should be produced from a controlled mixture of these states rather than from all previously written visual tokens.

## 7. 下一步建议

下一步不建议继续盲目调 K，而应开始实现“前端语义流 + 后端上下文选择”的联合设计。

优先级 1：按问题类型/时态进行视觉状态路由。

- 如果问题包含 `latest`、`current`、`now`、`setting`，提高 recency buffer 的检索/保留权重。
- 如果问题是 long-horizon activity / event summary，优先使用 semantic memory。
- 这可以让方法从工程 heuristic 上升为“query-aware temporal state routing”。

优先级 2：控制已写入视觉 KV 的参与范围。

- 当前 sparse ViT 只减少写入量，但没有显式压低旧视觉 token 对 latest 问题的影响。
- 应考虑类似 ReKV 的 streaming cache 管理，把视觉 token 分为 semantic memory / recency buffer 两类，并在问题阶段动态选择。

优先级 3：保留 K=4 作为默认 correction，再做 repeat3。

- 先对 K=4 做 RVS-Ego repeat3，确认 judge correctness 稳定性。
- 再在 RVS-Movie 上验证是否同样保持速度和 QA。

## 8. 结果文件

远程结果目录：

```text
results/rvs_ego_recency_20260603/r64_t0p3_recency0
results/rvs_ego_recency_20260603/r64_t0p3_recency1
results/rvs_ego_recency_20260603/r64_t0p3_recency2
results/rvs_ego_recency_20260603/r64_t0p3_recency4
results/rvs_ego_recency_20260603/r64_t0p3_recency8
```

关键分析文件：

```text
analysis/overlap.json
analysis/dense_vs_recency_compare.json
analysis/dense_vs_recency_judge_qwen25vl7b.json
analysis/category_summary/category_summary_all.json
```

# Semantic Coverage Floor 探针实验记录（2026-06-04）

## 1. 实验目的

上一轮大规模验证显示：

- semantic admission 相比 periodic baseline 能用更少视觉 token 达到相近或更好的 QA 表现；
- RVS-Movie 的 judge 结果暴露出剧情型长视频的长程覆盖问题；
- 仅靠 semantic drift + recency 可能漏掉慢变化但后续问题需要的片段。

因此本轮实现并验证 `Semantic + Coverage Floor + Recency`：

```text
keep frame if:
  1. semantic drift is large;
  2. or it is a periodic coverage anchor;
  3. or it is inside the recent K-frame window.
```

设计约束：

- coverage 默认关闭，旧实验完全不变；
- coverage keep 默认不更新 semantic anchor，避免破坏语义变化轨迹；
- online stage 仍然 query-independent。

## 2. 代码修改

新增参数：

- `semantic_coverage_interval`
- `semantic_coverage_updates_anchor`

修改文件：

- `model/vision_accelerator/semantic_stream.py`
- `model/vit_patch.py`
- `video_qa/base.py`
- `video_qa/run_eval.py`
- `video_qa/rekv_stream_vqa.py`
- `scripts/run_semantic_stream_sweep.py`

本地验证：

- `python -m py_compile ...` 通过；
- 使用 `.conda-envs/vit-sparse-gpu/python.exe` 跑 gate smoke test 通过；
- 默认 `coverage_interval=0` 时只保留 reference；
- `coverage_interval=2` 时额外保留第 2、4 帧，并正确记录 `coverage_kept_frames=2`。

## 3. RVS-Movie 探针

配置：

```text
semantic refresh=64
threshold=0.3
recency_keep=4
always_recent qrb=4
coverage_interval=32 / 48
repeat=1
```

对齐 dense 后处理口径：

| 方法 | kept / input | token reduction | speedup | token-F1 | W/T/L |
|---|---:|---:|---:|---:|---:|
| semantic only | 86 / 811 | 89.40% | 6.82x | 0.0589 | 4 / 18 / 2 |
| coverage 32 | 97 / 811 | 88.04% | 6.52x | 0.0579 | 5 / 16 / 3 |
| coverage 48 | 97 / 811 | 88.04% | 6.57x | 0.0619 | 5 / 17 / 2 |

Movie 上 `coverage=48` 的字符串指标略优于 semantic only：

- token-F1：0.0589 -> 0.0619；
- wins：4 -> 5；
- losses：保持 2；
- 但 speedup：6.82x -> 6.57x。

`coverage=32` 更密，但没有带来更好结果，说明 coverage 不是越密越好。

## 4. RVS-Movie Judge

对 `coverage=48` 跑 Qwen2.5-VL judge：

| 方法 | valid / total | dense acc | sparse acc | sparse-only | dense-only | better / same / worse |
|---|---:|---:|---:|---:|---:|---:|
| semantic only | 19 / 24 | 26.3% | 26.3% | 2 | 2 | 2 / 3 / 14 |
| coverage 48 | 18 / 24 | 22.2% | 33.3% | 2 | 0 | 2 / 4 / 12 |

解释：

1. coverage 48 在 Movie 上降低了 `dense-only correct`，说明它确实补回了一些 semantic-only 漏掉的信息。
2. sparse acc 从 26.3% 提到 33.3%，但 parse 样本数从 19 降到 18，因此只能作为积极信号，不能作为最终结论。
3. relative worse 从 14 降到 12，说明 coverage 对剧情型长视频有帮助。

## 5. RVS-Ego 反例

同样跑 `coverage=48`：

| 方法 | kept / input | token reduction | speedup | token-F1 | W/T/L |
|---|---:|---:|---:|---:|---:|
| semantic only | 157 / 2046 | 92.33% | 3.60x | 0.2894 | 7 / 17 / 0 |
| coverage 48 | 189 / 2046 | 90.76% | 3.32x | 0.2776 | 8 / 14 / 2 |

Ego 上 coverage 48 增加了 32 帧写入，但：

- speedup 下降；
- token-F1 下降；
- losses 从 0 增到 2。

这说明固定 coverage floor 不能作为无条件默认策略。它能缓解 Movie 的长程覆盖，但会干扰 Ego 的速度/质量平衡。

## 6. 方向性结论

本轮最重要的发现不是“coverage floor 直接成为最终方法”，而是：

> 长程覆盖确实是 semantic-only 的短板，但 coverage 必须是有条件的，而不是固定周期补帧。

更适合论文的方法抽象应从：

```text
semantic trigger + fixed coverage floor + recency
```

升级为：

```text
semantic trigger + uncertainty/diversity-aware coverage + recency
```

这样可以避免方法看起来像简单工程堆叠。三个模块分别对应明确的信息需求：

- semantic trigger：捕捉高变化事件；
- recency：保证当前状态；
- adaptive coverage：当流的语义状态过于单一或长时间没有新信息时，补充长程可追溯性。

## 7. 下一步建议

不要继续盲扫更密的 periodic coverage。下一步应设计更优雅的 adaptive coverage：

1. `staleness-aware coverage`：只有距离上次非 recency keep 超过阈值时才补帧。
2. `diversity-aware coverage`：只有当前帧与 sparse state 中已有锚点足够不同但又未达到 drift_keep 时才补帧。
3. `budget-aware coverage`：每个视频或时间窗口有 coverage token budget，优先补最有代表性的帧。

优先实现顺序：

1. 先做 `staleness-aware coverage`，因为它最简单、最容易解释；
2. 再做 `diversity-aware coverage`，作为更强版本；
3. 最后用 Movie judge 验证是否降低 relative worse，同时检查 Ego 不再增加 loss。

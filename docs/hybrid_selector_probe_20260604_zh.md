# Hybrid Semantic Selector 实验记录（2026-06-04）

## 1. 实验目的

前两轮结果形成了一个明确矛盾：

```text
vit_embedding selector（ViT 嵌入选择器）：语义信号更强，但选择开销太高。
raw_rgb selector（原始 RGB 选择器）：选择极快，但语义信号太粗。
```

本轮目标是验证 `two-stage selector（两阶段选择器）` 是否能缓解这个矛盾：

```text
先用 raw_rgb（原始 RGB）做低成本候选筛选，
再只对候选帧提取 vit_embedding（ViT 嵌入）做最终预算 Top-K 选择。
```

这对应论文中的方法动机：

```text
在 fixed visual budget（固定视觉预算）下，
用最小感知成本找到 semantic novelty（语义新颖性）最大的帧。
```

## 2. 新增方法

新增配置：

```text
semantic_selection_feature_source = hybrid
semantic_candidate_multiplier = 4
```

执行流程：

```text
输入视频帧
  -> Stage 1（第一阶段）：raw_rgb signature（原始 RGB 签名）预筛选 4K 个候选帧
  -> Stage 2（第二阶段）：只对候选帧计算 ViT embedding（ViT 嵌入）
  -> 在每个 budget window（预算窗口）内选择 Top-K 帧
  -> 只写入被选帧的视觉 token（视觉令牌）和 LLM KV cache（大模型键值缓存）
```

代码改动：

- `model/vision_accelerator/semantic_stream.py`
  - 新增候选帧选择接口；
  - 在只对候选计算二阶段特征时，仍按完整输入帧统计 `input_frames（输入帧）`、`skipped_frames（跳过帧）`。
- `model/vit_patch.py`
  - 新增 `hybrid（混合）` 编码路径；
  - 第一阶段不调用 ViT；
  - 第二阶段只对候选帧调用 ViT embedding（ViT 嵌入）。
- `video_qa/base.py`、`video_qa/run_eval.py`、`scripts/run_semantic_stream_sweep.py`
  - 新增命令行参数和结果字段。

## 3. 本地验证

本地 smoke test（冒烟测试）：

```text
输入 12 帧
budget_window_size = 6
budget_keep_per_window = 1
candidate_multiplier = 2
```

输出：

```text
candidates（候选帧）= [0, 3, 4, 8, 9]
kept_positions（最终保留位置）= [0, 2, 4]
input_frames（输入帧）= 12
kept_frames（保留帧）= 3
skipped_frames（跳过帧）= 9
budget_kept_frames（预算保留帧）= 2
prefilter_skips（预筛跳过帧）= 7
```

解释：

- 最终 3 帧来自 `1 个 reference frame（参考帧） + 2 个 budget_keep（预算保留帧）`；
- 统计仍覆盖完整 12 帧；
- 证明两阶段候选筛选没有破坏全局帧计数。

## 4. 远程实验设置

环境：

- 远程代码提交：`5cf5a2c`
- 远程 conda 环境：`base`
- GPU：CUDA 可用，8 张 GPU 可见
- 模型：`llava_ov_7b`
- 数据集：`RVS-Movie（电影剧情视频问答）`
- fps（帧率）：`1.0 fps（每秒 1 帧）`
- QA 数量：`24`
- 视频数量：`8`

核心配置：

```text
refresh_interval = 999999
skip_threshold = 0.0
recency_keep_frames = 4
semantic_selection_policy = budget_topk
semantic_selection_feature_source = hybrid
semantic_candidate_multiplier = 4
semantic_budget_window_size = 96
semantic_budget_keep_per_window = 1
query_retrieval_policy = always_recent
latest_retrieval_blocks = 4
```

## 5. 同视频 smoke 对照

先在单视频 3 QA 上对比三种选择源：

| 方法 | input / kept（输入/保留帧） | token reduction（令牌减少） | encode time（编码时间） | rule QA（规则 QA） |
|---|---:|---:|---:|---:|
| `raw_rgb（原始 RGB）` | 472 / 10 | 97.88% | 1.113s | 3 / 3 |
| `hybrid（混合）` | 472 / 10 | 97.88% | 1.485s | 3 / 3 |
| `vit_embedding（ViT 嵌入）` | 472 / 10 | 97.88% | 5.380s | 3 / 3 |

观察：

- `hybrid（混合）` 比 `vit_embedding（ViT 嵌入）` 快 `3.62x`；
- `hybrid（混合）` 比 `raw_rgb（原始 RGB）` 慢 `1.33x`；
- 这个单视频过于简单，三者 QA 都是 `3/3`，不能区分语义质量。

## 6. 完整 24 QA 结果

### 6.1 与 dense baseline（密集基线）对比

| 方法 | kept / input（保留/输入帧） | token reduction（令牌减少） | encode time（编码时间） | speedup vs dense（相对密集加速） | token-F1（词重叠 F1） | W/T/L vs dense（胜/平/负） |
|---|---:|---:|---:|---:|---:|---:|
| `dense（密集）` | 4067 / 4067 | 0.00% | 373.555s | 1.00x | 0.0551 | 0 / 24 / 0 |
| `hybrid（混合）` | 85 / 4067 | 97.91% | 9.597s | 38.92x | 0.0503 | 1 / 19 / 4 |

### 6.2 与 periodic sampling（周期采样）对比

| 方法 | kept / input（保留/输入帧） | token reduction（令牌减少） | encode time（编码时间） | speedup（相对周期采样） | token-F1（词重叠 F1） | W/T/L（胜/平/负） |
|---|---:|---:|---:|---:|---:|---:|
| `periodic（周期采样）` | 99 / 4067 | 97.57% | 27.579s | 1.00x | 0.0410 | - |
| `hybrid（混合）` | 85 / 4067 | 97.91% | 9.597s | 2.87x | 0.0503 | 2 / 20 / 2 |

### 6.3 与单信号选择器对比

| 方法 | kept / input（保留/输入帧） | encode time（编码时间） | speedup vs dense（相对密集加速） | token-F1（词重叠 F1） | W/T/L vs dense（胜/平/负） |
|---|---:|---:|---:|---:|---:|
| `raw_rgb（原始 RGB）` | 85 / 4067 | 4.656s | 80.24x | 0.0343 | 0 / 20 / 4 |
| `hybrid（混合）` | 85 / 4067 | 9.597s | 38.92x | 0.0503 | 1 / 19 / 4 |
| `vit_embedding（ViT 嵌入）` | 85 / 4067 | 29.881s | 12.50x | 0.0552 | 1 / 21 / 2 |

## 7. 关键发现

### 7.1 Hybrid 已经明显优于 periodic baseline

相对 `periodic sampling（周期采样）`：

```text
保留帧数：99 -> 85
编码时间：27.579s -> 9.597s
速度提升：2.87x
token-F1：0.0410 -> 0.0503
W/T/L：2 / 20 / 2
```

这说明在严格 `1.0 fps（每秒 1 帧）` 流式设置下，语义增量选择不再只是“和均匀采样差不多”，而是开始同时拿到：

- 更少视觉写入；
- 更低编码成本；
- 更高 token-F1 粗指标；
- 没有明显逐题崩坏。

### 7.2 Hybrid 解决了 ViT 选择器的主要开销

相对 `vit_embedding selector（ViT 嵌入选择器）`：

```text
编码时间：29.881s -> 9.597s
速度提升：3.11x
token-F1：0.0552 -> 0.0503
```

这说明两阶段设计可以大幅降低选择成本，同时保留大部分语义选择质量。

### 7.3 Hybrid 修复了 raw_rgb 的主要质量问题

相对 `raw_rgb selector（原始 RGB 选择器）`：

```text
编码时间：4.656s -> 9.597s
token-F1：0.0343 -> 0.0503
W/T/L vs dense：0 / 20 / 4 -> 1 / 19 / 4
```

这说明第二阶段 `ViT embedding（ViT 嵌入）` 的确补足了低级 RGB 信号的语义不足。

## 8. 阶段性结论

本轮得到当前为止最有论文价值的正向结果：

```text
hybrid two-stage semantic selector（混合两阶段语义选择器）
在 1fps RVS-Movie 上同时优于 periodic sampling（周期采样）的帧数、速度和 token-F1 粗指标。
```

这比之前单纯 `drift_keep（漂移保留）` 或纯 `raw_rgb（原始 RGB）` 更符合顶会方法设计：

- 它不是堆工程技巧，而是由 `selection quality-cost dilemma（选择质量-成本矛盾）` 推导出来；
- 它的形式足够简洁：低成本候选筛选 + 高质量语义复核；
- 它直接服务最终目标：把 `dense visual stream（密集视觉流）` 转换为 `sparse semantic stream（稀疏语义流）`。

## 9. 下一步

下一步建议：

1. 扫 `candidate_multiplier（候选倍数）= 2, 4, 8`，验证候选规模对速度和质量的影响；
2. 在 `RVS-Ego（第一视角视频问答）` 上复验，确认不是 Movie 特例；
3. 用 Qwen2.5-VL judge（模型裁判）评估 hybrid vs periodic，避免只靠 token-F1；
4. 继续下载或接入 `OVO-Bench（在线视频理解基准）` / `StreamingBench（流式视频理解基准）`，寻找更能体现语义增量选择优势的任务。


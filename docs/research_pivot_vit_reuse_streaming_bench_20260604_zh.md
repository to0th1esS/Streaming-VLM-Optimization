# 研究方向修正：ViT 帧间复用与真实流式 Benchmark（2026-06-04）

## 1. 当前问题

严格 `0.5 fps（每秒 0.5 帧）` 和 `1.0 fps（每秒 1 帧）` 实验显示，当前固定阈值 `semantic gate（语义门控）` 没有稳定胜过 `periodic sampling（周期/均匀采样）`。

这说明当前方法如果只被描述为：

```text
content-aware frame skipping（内容感知跳帧）
```

是不够的。审稿人会自然攻击：

```text
Why not just uniformly sample frames?
为什么不直接均匀采样？
```

这个攻击目前成立。

## 2. 但这不等于方向失败

当前实验暴露的是方法实现太弱，而不是研究目标错误。

真正目标不是“比均匀采样多跳几帧”，而是：

```text
turn dense visual stream into sparse semantic stream
把密集视觉流转化为稀疏语义流
```

这个目标包含两个同时发生的压缩：

1. `visual computation compression（视觉计算压缩）`
   - 减少 ViT 编码开销；
   - 不只是少送帧，而是利用帧间冗余做增量更新。

2. `visual context compression（视觉上下文压缩）`
   - 减少写入 LLM 上下文或 KV cache 的视觉 token；
   - 只保留对未来 QA 有语义价值的状态。

当前代码主要做了第二点的一部分，即 frame-level admission（帧级准入），还没有充分发挥第一点。

## 3. 为什么当前 ViT 加速没有超过均匀采样

### 3.1 目前主要是 frame-level skipping（帧级跳过）

当前主实验的核心逻辑是：

```text
keep frame or skip frame
保留整帧或跳过整帧
```

而 `periodic sampling（周期/均匀采样）` 也在做同样事情，只是不看内容。

所以在高 fps 场景中，periodic 非常强：

- 相邻帧冗余很高；
- 每隔几十秒保留一帧已经能覆盖粗粒度场景；
- 它天然控制 token budget（token 预算）。

### 3.2 当前 semantic threshold（语义阈值）没有预算约束

当前逻辑是：

```text
if drift > threshold: keep
如果语义漂移大于阈值，就保留
```

问题是：

- 不同 fps 下 drift distribution（语义漂移分布）不同；
- 不同数据集下动作和场景变化密度不同；
- 固定阈值不能保证保留帧数远低于 uniform baseline（均匀采样基线）。

因此它会出现：

```text
semantic keeps more frames than periodic
语义方法反而比周期采样保留更多帧
```

### 3.3 ViT internal reuse（ViT 内部复用）还没有成为主收益

我们前面实现过 ViT 稀疏更新，但当前 VLM 主实验里主要收益来自是否写入整帧，而不是：

```text
partial token update inside ViT
ViT 内部 token 级部分更新
```

这导致：

- periodic 只要少编码帧，就能获得巨大速度收益；
- 我们的 semantic gate 如果保留更多帧，速度就天然吃亏；
- ViT token 级复用没有在主实验中放大出来。

## 4. 为什么仍然必须保留 ViT 帧间复用

如果最终方法只做 frame selection（选帧），论文贡献会很弱，因为 uniform sampling 太强。

我们真正要做的是：

```text
two-level sparse update
两级稀疏更新
```

第一层：

```text
frame admission（帧级准入）
决定哪些帧值得写入语义状态
```

第二层：

```text
token/layer update（token/层级增量更新）
对未写入或低变化帧，也可以只更新少量 ViT token 或浅层特征
```

这样才能和 periodic 拉开差距：

| 方法 | 处理方式 | 局限 |
|---|---|---|
| periodic sampling（周期/均匀采样） | 只保留固定间隔帧 | 跳过帧完全不可见 |
| frame semantic skipping（帧级语义跳过） | 按内容保留整帧 | 仍然是整帧级决策 |
| our target（目标方法） | 帧级稀疏写入 + ViT 内部增量更新 | 能利用未写入帧的局部变化 |

这也是更适合顶会的方法点：

> Not only selecting fewer frames, but maintaining a low-cost evolving visual state.
> 不只是少选帧，而是维护一个低成本演化的视觉状态。

## 5. 帧数量为什么理论上应低于均匀采样

用户判断是合理的：如果我们做的是真正的 incremental semantic selection（增量语义选择），保留帧数应该低于 uniform sampling（均匀采样）。

但前提是保留规则不是固定阈值，而是：

```text
event-driven admission
事件驱动准入
```

也就是：

- 长时间静止或重复动作时，几乎不写入；
- 突发事件、对象变化、状态转移时，集中写入；
- 每个窗口有固定预算，超预算时只保留最有新颖性的帧。

当前固定阈值没有做到这一点。下一版应改成：

```text
budget-aware top-k semantic admission
预算感知的语义 Top-K 准入
```

即在每个 streaming window（流式窗口）内，只保留语义新颖性最高的 K 帧，K 明确小于 periodic 的帧数。

## 6. 当前数据集的问题

RVS 子集太小：

- 每个数据集只有 8 个视频；
- 每个视频 3 个问题；
- 任务较粗，很多问题均匀采样就能答；
- 不够体现 fine-grained temporal awareness（细粒度时间意识）。

这会天然削弱 semantic preservation（语义保留）的优势。

我们需要更适配流式场景的数据集。

## 7. 更合适的数据集

### 7.1 OVO-Bench（Online Video Benchmark，在线视频理解基准）

OVO-Bench 是 CVPR 2025 benchmark（基准），强调 online video understanding（在线视频理解）和 temporal awareness（时间意识）。

官方论文描述其包含：

- 644 个视频；
- 约 2800 个精细时间戳 QA；
- 三种在线场景：
  - `backward tracing（向后追溯）`
  - `real-time perception（实时感知）`
  - `forward active responding（前向主动响应）`

这比 RVS 更适合验证：

```text
what to remember, what to update, and when to answer
记住什么、更新什么、何时回答
```

### 7.2 StreamingBench（流式视频理解基准）

StreamingBench 更偏通用 streaming MLLM benchmark（流式多模态大模型基准）。

公开介绍显示它包含：

- 900 个视频；
- 4500 个 QA；
- 每个视频多个时间点问题；
- 三类能力：
  - `real-time visual understanding（实时视觉理解）`
  - `omni-source understanding（多源理解）`
  - `contextual understanding（上下文理解）`

它适合验证我们的方法是否能在更真实流式设置下保持 QA。

### 7.3 OVO-S-Bench（流式空间智能基准）

OVO-S-Bench 是更新的 streaming spatial intelligence benchmark（流式空间智能基准）。

它更适合后续扩展到 spatial memory（空间记忆）和 current-state QA（当前状态问答），但不是当前第一优先级。

## 8. 下一步实验设计

### Step 1：接入 OVO-Bench / StreamingBench 小子集

先不要全量下载 200GB 级别数据。先做：

```text
metadata first, videos subset later
先接元数据，再拉小视频子集
```

选择标准：

- 有 query timestamp（问题时间戳）；
- 有 evidence interval（证据时间段）更好；
- 视频长度大于 5 分钟；
- 问题类型包含 backward tracing（向后追溯）和 real-time perception（实时感知）。

### Step 2：做 budget-matched baseline（预算匹配基线）

必须公平比较：

```text
periodic keeps K frames
周期采样保留 K 帧

ours keeps <= K frames
我们保留不超过 K 帧
```

不能再让 semantic 自然保留更多帧。

### Step 3：恢复 ViT internal reuse（ViT 内部复用）

新方法应包含两条输出：

1. `write decision（写入决策）`
   - 哪些帧写入 LLM 上下文；

2. `update decision（更新决策）`
   - 对没有写入的帧，是否进行低成本 ViT 局部更新；
   - 例如只更新 motion-sensitive token（运动敏感 token）或浅层摘要。

### Step 4：主指标换成三元指标

不只看 speedup（加速比）：

| 指标 | 中文含义 |
|---|---|
| online compute | 在线视觉计算成本 |
| context write | 写入上下文的视觉 token 数 |
| QA accuracy | 问答准确性 |

最终目标是：

```text
better QA under lower compute and lower context write
在更低计算和更低写入下获得更好 QA
```

## 9. 当前方向性结论

现在不能再把方法写成：

```text
semantic frame skipping beats uniform sampling
语义跳帧超过均匀采样
```

而应该写成：

```text
query-independent sparse semantic state with incremental visual update
查询无关的稀疏语义状态与增量视觉更新
```

核心贡献应是：

1. `budget-aware semantic admission（预算感知语义准入）`
2. `incremental ViT state update（增量 ViT 状态更新）`
3. `streaming benchmark validation（流式基准验证）`

只有这三点合在一起，才有机会形成顶会级别的故事。

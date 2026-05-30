# Turbo-ViT-v1 实验记录

本文档用于记录 Turbo-ViT-v1 本地实验的每一版修改、结果和结论，便于后续回溯。

## v0 Dense Baseline 与层级冗余分析脚手架

目标：

- 搭建第一版本地实验框架。
- 跑通逐帧 Dense ViT 编码时延统计。
- 采集每层 hidden states，并计算相邻帧层级相似度。

实现：

- 新增自包含的 TinyViT encoder。
- 新增 synthetic redundant video stream 生成器。
- 新增 dense stream encoding，每帧记录 latency。
- 新增相邻帧 layer-wise cosine similarity 分析。
- 输出 JSON、CSV、SVG。

输出：

```text
results/turbovit_v1/v0_dense_baseline/dense_summary.json
results/turbovit_v1/v0_dense_baseline/dense_latency.csv
results/turbovit_v1/v0_dense_baseline/layer_redundancy.csv
results/turbovit_v1/v0_dense_baseline/layer_redundancy.svg
```

本地结果：

```text
mean latency/frame: 1.543 ms
total latency/video: 37.041 ms for 24 frames
layer-0 adjacent cosine: 0.999740
last-layer adjacent cosine: 0.999933
```

结论：

- synthetic 流视频具有很强的跨帧冗余。
- TinyViT 中层数越深，相邻帧特征相似度越高，符合我们对视觉编码冗余的初步预期。
- 这一版只是 dense baseline 和分析脚手架，不是加速结果。

## v1 功能正确版 Turbo-ViT

目标：

- 实现第一版 Turbo-ViT-v1 机制。
- 比较 dense 逐帧编码与 reference-frame 复用 + dynamic token 重算。
- 本地扫 `refresh_interval` 和 `dynamic_ratio`。

实现：

- 在 TinyViT block 中拆出 Q/K/V projection。
- reference frame 完整编码，并缓存每层 key 和 output。
- 非 reference frame：
  - 计算当前层 key；
  - 与 reference key 做 cosine similarity；
  - 选择低相似度 token 作为 dynamic token；
  - 只对 dynamic token 执行 attention + MLP；
  - 将结果 scatter 回 reference layer output。
- 新增 final output cosine 和 MSE 作为 fidelity 指标。

单点结果（`N=4`, `r=0.5`）：

```text
speedup: 0.697x
mean output cosine: 0.999954
mean output mse: 0.00009158
mean selector time/frame: 0.759 ms
mean sparse compute time/frame: 1.226 ms
```

ablation 早期结果：

```text
best speed: N=2, r=0.5
speedup: 1.449x
mean output cosine: 0.999985
mean output mse: 0.0000301

best fidelity: N=2, r=0.75
speedup: 1.135x
mean output cosine: 0.999993
mean output mse: 0.0000144
```

结论：

- v1 机制功能正确，能够保持较高 feature fidelity。
- CPU 小模型 latency 有噪声，但能看到部分设置下存在加速空间。
- `refresh_interval` 变大后 stale-reference drift 增加。
- selector 和 sparse compute 的额外开销很大，是 v1 收益受限的主要原因。

## v1 真实视频 Sanity Check

目标：

- 不只看 synthetic stream，加入一个真实视频小样例。
- 保持本地轻量验证，不依赖完整 benchmark。

实现：

- 新增真实视频加载器。
- 默认下载 Big Buck Bunny 小视频：
  `https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4`
- 视频保存到 `data/turbovit_v1/big_buck_bunny.mp4`，不进入 git。
- 使用 `imageio + imageio-ffmpeg` 解码，不依赖系统 ffmpeg。

单点结果（`N=4`, `r=0.5`）：

```text
speedup: 1.026x
mean output cosine: 0.922687
mean output mse: 0.154624
mean selector time/frame: 0.612 ms
mean sparse compute time/frame: 0.998 ms
```

真实视频 ablation：

```text
best speed: N=4, r=0.75
speedup: 1.347x
mean output cosine: 0.967529
mean output mse: 0.064941

best fidelity: N=2, r=0.75
speedup: 1.055x
mean output cosine: 0.977259
mean output mse: 0.045482
```

结论：

- 真实视频比 synthetic 更难，feature fidelity 明显下降。
- 低 dynamic ratio 在真实运动场景下风险较高。
- `r=0.75` 更稳。
- 这支持后续做 adaptive refresh、dual-anchor 或 segment-level decision。

## v1 Drift 与 Time Breakdown

目标：

- 显式量化 stale-reference drift。
- 显式拆解 selector、sparse compute、其他开销。

实现：

- `run_ablation.py` 新增输出：
  - `drift_by_distance.csv`
  - `time_breakdown.csv`
  - `best_speed_drift.svg`
  - `best_speed_time_breakdown.svg`
- drift 按 `distance_from_reference = frame_idx % refresh_interval` 聚合。
- time breakdown 拆成：
  - dense baseline latency；
  - selector time；
  - sparse compute time；
  - other/reference/scatter overhead。

synthetic rerun：

```text
best speed config: N=4, r=0.5
speedup: 0.951x
mean output cosine: 0.999954
selector: 0.673 ms/frame
sparse compute: 1.151 ms/frame
other/reference: 0.693 ms/frame
```

real-video rerun：

```text
best speed config: N=4, r=0.75
speedup: 0.921x
mean output cosine: 0.967529
selector: 0.691 ms/frame
sparse compute: 1.197 ms/frame
other/reference: 0.649 ms/frame
```

重要注意：

- 当前是 CPU tiny model microbench，latency 抖动明显，不能作为最终速度结论。
- 稳定结论是：
  - 距离 reference 越远，drift 越明显；
  - 真实视频 drift 明显大于 synthetic；
  - selector + sparse compute 开销足以抵消收益；
  - v1 需要降低判别开销，并改善 reference 管理。

下一步：

- 实现 v2 segment-level decision：先做低成本帧级 drift 判断，再决定 skip / sparse / dense，减少每层逐 token 判别频率。

## v2 分段级决策原型

目标：

- 降低 v1 中“每层、每帧、每 token 判别”的开销。
- 在进入逐层 sparse update 之前，先用一个便宜的帧级 drift 指标做路由。

实现：

- 新增 `encode_stream_turbovit_v2`。
- 对每个非 reference frame，先计算当前 patch embedding 与 rolling anchor 的 MSE。
- 决策策略：
  - `dense`：强制刷新，或 frame drift 高于 `dense_threshold`；
  - `skip`：frame drift 低于 `skip_threshold`，直接复用 anchor output；
  - `sparse`：中等 drift，执行 v1 风格逐层 sparse update。
- sparse frame 会更新 rolling anchor 状态，后续帧会相对最新近似视觉状态计算 drift。

默认真实视频配置：

```text
refresh_interval: 4
dynamic_ratio: 0.75
skip_threshold: 0.0005
dense_threshold: 0.006
```

修正状态一致性后的真实视频结果：

```text
speedup: 0.940x
mean output cosine: 0.990327
mean output mse: 0.019346
dense frames: 7
sparse frames: 10
skip frames: 7
selector: 0.420 ms/frame
sparse compute: 0.724 ms/frame
```

更保守 skip 配置：

```text
skip_threshold: 0.0001
dense_threshold: 0.006
speedup: 0.769x
mean output cosine: 0.995885
dense frames: 7
sparse frames: 15
skip frames: 2
```

结论：

- segment-level decision 可以减少进入逐层 selector 的帧数。
- skip 阈值形成了清晰的速度/保真度权衡。
- CPU latency 仍然不是最终性能结论，但这个版本已经暴露了重要研究旋钮：先做便宜的帧级路由，再决定是否进入昂贵的逐层 sparse update。
- 当前 patch embedding MSE 作为 routing signal 仍然太粗糙，可能跳过一些下游特征变化明显的帧。后续应考虑 low-layer feature drift、motion-aware drift 或 dual-anchor interpolation。

## 真实 ViT-B/16 本地小规模验证

目标：

- 从 TinyViT 原型推进到真实 ViT block 结构，验证当前 dense / Turbo-v1 实验框架是否能适配标准 ViT。
- 先在本地 CPU 小样本上验证功能链路、层级冗余曲线和逐层 sparse update 的开销构成。
- 明确本地验证环境的边界：本地用于机制认证和趋势分析，预训练大模型和大规模统计仍放到远程服务器。

实现：

- 新增 `TorchvisionViTWrapper`，接入 `torchvision.models.vit_b_16`。
- 为 torchvision 的 `EncoderBlock` 增加轻量 adapter，将：
  - `ln_1` 映射为 `norm1`；
  - `self_attention` 映射为 `attn`；
  - `ln_2` 映射为 `norm2`；
  - 补齐 `_project_qkv`、`_split_heads`、`_merge_heads`；
  - 提供 `forward_with_caches`，使现有 Turbo-v1 sparse encoder 可以直接复用。
- 新增本地脚本：
  - `scripts/run_real_vit_dense_local.ps1`
  - `scripts/run_real_vit_turbo_local.ps1`
- 新增实验脚本：
  - `experiments/turbovit_v1/scripts/run_real_vit_dense.py`
  - `experiments/turbovit_v1/scripts/run_real_vit_turbo.py`

### Step 1：真实 ViT-B/16 Dense baseline

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_real_vit_dense_local.ps1
```

配置：

```text
model: torchvision.vit_b_16
weights: none
device: cpu
video: Big Buck Bunny real video
num_frames: 6
frame_stride: 4
image_size: 224
```

结果：

```text
mean latency/frame: 53.634 ms
total latency: 321.805 ms
layer0 adjacent cosine: 0.826982
last-layer adjacent cosine: 0.860020
```

层级相似度：

```text
layer 0:  0.826982
layer 1:  0.834603
layer 2:  0.848927
layer 3:  0.851878
layer 4:  0.853393
layer 5:  0.857855
layer 6:  0.855885
layer 7:  0.855342
layer 8:  0.856583
layer 9:  0.858566
layer 10: 0.858311
layer 11: 0.860020
```

分析：

- 即使在未加载预训练权重的真实 ViT-B/16 架构上，相邻真实视频帧的 hidden states 仍表现出较高相似度。
- 相似度从浅层到深层整体略有上升，支持“跨帧复用在 ViT 编码器内部存在可利用冗余”这一基础判断。
- 但当前权重是随机初始化，不能解释语义稳定性，只能证明结构链路和冗余分析工具可运行。
- 后续论文实验必须使用预训练视觉 encoder，尤其是 CLIP / SigLIP / LLaVA-OneVision 对应的 vision tower，才能形成有效论文证据。

### Step 2：真实 ViT-B/16 Turbo-v1，dynamic_ratio = 0.5

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_real_vit_turbo_local.ps1 -NumFrames 6 -FrameStride 4 -RefreshInterval 4 -DynamicRatio 0.5
```

结果：

```text
refresh_interval: 4
dynamic_ratio: 0.5
dense mean latency/frame: 64.166 ms
turbo mean latency/frame: 50.286 ms
speedup: 1.276x
mean output cosine: 0.893802
mean output mse: 0.212398
reference frames: 2 / 6
selector: 10.119 ms/frame
sparse compute: 18.825 ms/frame
```

分析：

- 在真实 ViT-B/16 上，选择性更新确实能减少编码时间，本地 CPU 小样本得到约 `1.276x` 加速。
- 但 `r=0.5` 的输出保真度不足，mean cosine 下降到 `0.894`。
- 这说明真实 ViT 的逐层状态传播比 TinyViT 更敏感，粗暴固定比例 token 更新会产生明显 feature drift。
- 这支持后续不应只追求低 dynamic ratio，而要研究 adaptive token selection、layer-wise ratio 和 reference 管理。

### Step 3：真实 ViT-B/16 Turbo-v1，dynamic_ratio = 0.75

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_real_vit_turbo_local.ps1 -OutputDir results\turbovit_v1\real_vit_turbo_r075 -NumFrames 6 -FrameStride 4 -RefreshInterval 4 -DynamicRatio 0.75
```

结果：

```text
refresh_interval: 4
dynamic_ratio: 0.75
dense mean latency/frame: 64.145 ms
turbo mean latency/frame: 57.071 ms
speedup: 1.124x
mean output cosine: 0.949251
mean output mse: 0.101501
reference frames: 2 / 6
selector: 10.269 ms/frame
sparse compute: 24.975 ms/frame
```

分析：

- `r=0.75` 明显提升保真度，mean cosine 从 `0.894` 提升到 `0.949`。
- 加速比从 `1.276x` 降到 `1.124x`，形成了符合预期的速度/保真度权衡。
- selector 开销基本不随 `dynamic_ratio` 改变，主要由每层 QKV 计算、cosine similarity、top-k/gather 决定。
- sparse compute 随 dynamic token 数增加而上升，这是后续优化的关键瓶颈。

### Step 4：预训练 torchvision ViT-B/16 尝试

命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_real_vit_dense_local.ps1 -OutputDir results\turbovit_v1\real_vit_dense_imagenet -Weights imagenet -NumFrames 4 -FrameStride 4
```

结果：

```text
weights file downloaded:
C:\Users\Administrator\.cache\torch\hub\checkpoints\vit_b_16-c867db91.pth
size: 346328529 bytes

load failed:
DefaultCPUAllocator: not enough memory
```

分析：

- 本地已经成功下载 ImageNet 预训练权重，但加载阶段因为本地 CPU 内存不足失败。
- 这说明本地认证环境适合跑 TinyViT 和未加载权重的真实 ViT-B/16 小样本，但不适合承载完整预训练 ViT-B/16 的稳定实验。
- 预训练真实 ViT 实验应转移到远程服务器，并统一从 `/home/mllm/models` 取模型。
- 本地仍保留 `weights=imagenet` 入口，后续如果本地内存或机器配置提升，可以直接复用同一命令。

### 本轮结论

- 当前实现已经从 TinyViT 原型推进到真实 `torchvision.vit_b_16` 架构，dense、layer-wise redundancy、Turbo-v1 sparse update 均可运行。
- 真实 ViT 小样本显示：
  - 跨帧冗余存在；
  - 固定比例 token sparse update 可以带来初步加速；
  - 低 dynamic ratio 的 feature drift 明显；
  - selector 开销是固定成本，sparse compute 随更新 token 数增长。
- 论文叙事上，这一轮可作为“最小真实 ViT 机制认证”，但不能作为最终主结果，因为当前本地真实 ViT 权重为随机初始化。

下一步建议：

- 在远程服务器上跑预训练 ViT-B/16 / CLIP ViT / SigLIP vision tower 的同一组脚本。
- 将当前 `dynamic_ratio` 从全层固定值升级为 layer-wise policy：浅层更保守、深层更激进。
- 将 token selector 从单纯 top-ratio 改为 threshold + ratio cap，避免静态背景和运动主体被同等处理。
- 对 v2 继续推进 dual-anchor / rolling-anchor，让 stale reference drift 成为可控变量。

## GPU 实验记录：真实 ViT-B/16 on A100

背景：

- 本地机器有 RTX 5070，但最初认证环境安装的是 CPU 版 PyTorch，因此上一轮真实 ViT 默认跑在 CPU。
- 已单独创建本地 GPU 环境 `.conda-envs/vit-sparse-gpu`，安装 `torch 2.11.0+cu128` 和 `torchvision 0.26.0+cu128`。
- 本地 PyTorch 可以识别 RTX 5070，但创建 CUDA tensor 时出现 `CUDA error: out of memory`，即使 `nvidia-smi` 显示仍有空闲显存。该问题更像 Windows/WDDM + 当前 CUDA wheel + RTX 50 系列运行时兼容问题。
- 因此本轮正式 GPU 验证切到远程服务器 A100：

```text
torch: 2.5.1+cu124
torchvision: 0.20.1+cu124
cuda_available: True
gpu: NVIDIA A100-SXM4-80GB
```

### Step 1：真实 ViT-B/16 Turbo-v1 GPU sweep

新增脚本：

```text
experiments/turbovit_v1/scripts/run_synthetic_vit_turbo.py
scripts/run_synthetic_vit_turbo_local.ps1
```

远程命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_synthetic_vit_turbo \
  --output-dir results/turbovit_v1/synthetic_vit_turbo_gpu_warm \
  --weights none \
  --num-frames 24 \
  --refresh-interval 4 \
  --dynamic-ratios 0.25,0.5,0.75 \
  --device cuda
```

结果：

```text
dense latency/frame: 6.864 ms

r=0.25:
  turbo latency/frame: 7.574 ms
  speedup: 0.907x
  cosine: 0.999756

r=0.50:
  turbo latency/frame: 7.901 ms
  speedup: 0.869x
  cosine: 0.999869

r=0.75:
  turbo latency/frame: 7.993 ms
  speedup: 0.859x
  cosine: 0.999952
```

分析：

- 校正 CUDA warmup 后，v1 在 A100 上没有超过 dense baseline。
- 原因不是跨帧冗余不存在，而是当前功能版 sparse update 没有高效 kernel：
  - 每层都要算 QKV；
  - 每层都要 cosine + top-k；
  - gather/scatter 破坏连续计算；
  - token 级稀疏没有转化成 GPU 上足够高效的矩阵计算。
- 这直接解释了为什么“逐层逐 token 判别 + 选择性重算”的朴素 v1 很难成为最终系统。
- 论文叙事中可以把 v1 定位为机制验证版，而不是最终加速版。

### Step 2：真实 ViT-B/16 Turbo-v2 GPU routing sweep

新增脚本：

```text
experiments/turbovit_v1/scripts/run_synthetic_vit_v2.py
scripts/run_synthetic_vit_v2_local.ps1
```

远程 24 帧快速验证：

```text
skip=0.000100:
  speedup: 0.899x
  cosine: 0.999952
  dense/sparse/skip: 6/18/0

skip=0.000500:
  speedup: 1.309x
  cosine: 0.999837
  dense/sparse/skip: 6/6/12

skip=0.001000:
  speedup: 1.954x
  cosine: 0.999794
  dense/sparse/skip: 6/6/12
```

24 帧样本中两个阈值的决策分布相同但耗时差异较大，说明短序列 GPU timing 噪声仍明显。因此追加 96 帧验证。

远程 96 帧命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_synthetic_vit_v2 \
  --output-dir results/turbovit_v1/synthetic_vit_v2_gpu_96f \
  --weights none \
  --num-frames 96 \
  --refresh-interval 4 \
  --dynamic-ratio 0.75 \
  --skip-thresholds 0.0001,0.0005,0.001 \
  --dense-threshold 0.006 \
  --device cuda
```

96 帧结果：

```text
dense latency/frame: 6.354 ms

skip=0.000100:
  turbo latency/frame: 6.130 ms
  speedup: 1.036x
  cosine: 0.999970
  mse: 0.0000626
  dense/sparse/skip: 24/52/20

skip=0.000500:
  turbo latency/frame: 2.742 ms
  speedup: 2.317x
  cosine: 0.999905
  mse: 0.0001929
  dense/sparse/skip: 24/10/62

skip=0.001000:
  turbo latency/frame: 2.339 ms
  speedup: 2.717x
  cosine: 0.999880
  mse: 0.0002417
  dense/sparse/skip: 24/6/66
```

分析：

- v2 结果明显优于 v1，说明“先做便宜帧级路由，再决定是否进入逐层 sparse update”是正确方向。
- 当 `skip_threshold` 增大，更多帧直接复用 rolling anchor output，速度显著提升。
- 保真度下降很小，synthetic streaming 场景下 cosine 仍保持在 `0.99988+`。
- 这说明当前最有价值的研究点不是单纯降低 token ratio，而是减少进入逐层 selector 的次数。
- 对应论文贡献可以表述为：
  - v1 证明 ViT 内部存在跨帧冗余；
  - v1 同时暴露逐层 token 判别在 GPU 上的开销瓶颈；
  - v2 通过帧级 routing 将昂贵 token selector 变成条件触发，从而释放真实加速。

### Step 3：预训练 ViT 权重状态

尝试远程直接下载 torchvision ImageNet ViT-B/16 权重失败：

```text
expected hash: c867db91
actual hash: e3b0c442...
```

含义：

- 远程下载得到空文件，属于网络/下载链路问题，不是模型代码问题。
- 本机已有完整权重文件，但 scp 到远程两次超时，第二次中断时远程文件约 254MB，完整文件应约 330MB。

当前处理：

- 暂时不把 ImageNet 预训练结果写入主实验结论。
- 下一步应把模型文件统一放入 `/home/mllm/models`，并使用可恢复传输方式或服务器侧稳定下载工具。

本轮 GPU 结论：

- v1 真实 ViT GPU 结果支持“朴素逐层 token sparse 不够”的判断。
- v2 真实 ViT GPU 结果给出第一组正向加速证据：`2.3x - 2.7x`，但目前仍是 synthetic + random weights。
- 下一轮必须补齐：
  - 预训练 vision encoder；
  - 真实视频输入；
  - 更长视频序列；
  - 多 seed / 多片段统计；
  - v2 routing signal 从 patch MSE 升级为 low-layer feature drift 或 dual-anchor drift。

## 预训练 CLIP ViT-L/14 实验

目标：

- 从随机初始化的 torchvision ViT，推进到真实预训练 vision encoder。
- 使用远程已有模型 `/home/mllm/models/clip-vit-large-patch14-336`。
- 验证 v1/v2 在预训练 CLIP ViT-L/14 上是否仍成立。

模型配置：

```text
backbone: HuggingFace CLIPVisionModel
model path: /home/mllm/models/clip-vit-large-patch14-336
image_size: 336
patch_size: 14
hidden_size: 1024
num_layers: 24
num_heads: 16
device: A100 cuda
```

实现：

- 新增 `HFCLIPVisionWrapper`。
- 将 CLIP 的 `CLIPEncoderLayer` 适配到已有 sparse 接口：
  - `layer_norm1 -> norm1`
  - `self_attn.q_proj/k_proj/v_proj -> _project_qkv`
  - `self_attn.out_proj -> attn.out_proj`
  - `layer_norm2 -> norm2`
  - `mlp -> mlp`
- `run_synthetic_vit_turbo.py` 和 `run_synthetic_vit_v2.py` 新增 `--backbone clip` 和 `--model-path`。

### Step 1：CLIP ViT-L/14 Turbo-v1

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_synthetic_vit_turbo \
  --backbone clip \
  --model-path /home/mllm/models/clip-vit-large-patch14-336 \
  --output-dir results/turbovit_v1/clip_vit_v1_gpu_24f \
  --num-frames 24 \
  --refresh-interval 4 \
  --dynamic-ratios 0.25,0.5,0.75 \
  --device cuda
```

结果：

```text
dense latency/frame: 38.633 ms

r=0.25:
  turbo latency/frame: 26.418 ms
  speedup: 1.462x
  cosine: 0.977407
  mse: 0.058597

r=0.50:
  turbo latency/frame: 29.594 ms
  speedup: 1.305x
  cosine: 0.986267
  mse: 0.035599

r=0.75:
  turbo latency/frame: 32.480 ms
  speedup: 1.189x
  cosine: 0.994333
  mse: 0.014774
```

分析：

- 与 torchvision ViT-B/16 不同，CLIP ViT-L/14 足够大，v1 的 token sparse 已经能带来正向加速。
- 但速度/保真度权衡明显：
  - `r=0.25` 加速最高，但 cosine 只有 `0.977`；
  - `r=0.75` 保真度较好，但加速降到 `1.19x`。
- 这说明对于大视觉编码器，token-level sparse 有价值，但固定比例更新仍不是最优策略。

### Step 2：CLIP ViT-L/14 Turbo-v2 threshold sweep

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_synthetic_vit_v2 \
  --backbone clip \
  --model-path /home/mllm/models/clip-vit-large-patch14-336 \
  --output-dir results/turbovit_v1/clip_vit_v2_gpu_96f \
  --num-frames 96 \
  --refresh-interval 4 \
  --dynamic-ratio 0.75 \
  --skip-thresholds 0.0005,0.001,0.0011,0.0012 \
  --dense-threshold 0.006 \
  --device cuda
```

结果：

```text
dense latency/frame: 37.796 ms

skip=0.0005:
  turbo latency/frame: 17.016 ms
  speedup: 2.221x
  cosine: 0.994853
  dense/sparse/skip: 24/23/49

skip=0.0010:
  turbo latency/frame: 13.510 ms
  speedup: 2.798x
  cosine: 0.992097
  dense/sparse/skip: 24/12/60

skip=0.0011:
  turbo latency/frame: 12.551 ms
  speedup: 3.011x
  cosine: 0.991094
  dense/sparse/skip: 24/9/63

skip=0.0012:
  turbo latency/frame: 12.254 ms
  speedup: 3.084x
  cosine: 0.990815
  dense/sparse/skip: 24/8/64
```

分析：

- v2 在预训练 CLIP ViT-L/14 上显著优于 v1。
- 当 `skip_threshold=0.001` 时，已经可以达到 `2.80x`，同时 cosine 仍保持 `0.992`。
- 更激进的 `skip=0.0012` 可达到 `3.08x`，但 cosine 降到 `0.991`。
- 这组结果是目前最接近论文主线的证据：先做帧级 routing，再条件触发 token sparse，比逐帧逐层 token sparse 更有效。

### Step 3：CLIP ViT-L/14 refresh interval sweep

固定配置：

```text
dynamic_ratio: 0.75
skip_threshold: 0.001
dense_threshold: 0.006
num_frames: 96
```

结果：

```text
N=2:
  speedup: 1.882x
  cosine: 0.995848
  mse: 0.010719
  dense/sparse/skip: 48/3/45

N=4:
  speedup: 2.797x
  cosine: 0.992097
  mse: 0.020378
  dense/sparse/skip: 24/12/60

N=8:
  speedup: 3.432x
  cosine: 0.986977
  mse: 0.033537
  dense/sparse/skip: 12/19/65
```

分析：

- 刷新间隔越大，dense reference 帧越少，因此速度上升。
- 但 reference stale 问题也更明显，cosine 从 `0.996` 降到 `0.987`。
- 这正好支撑下一阶段的研究动机：单锚点 reference 不足，需要 dual-anchor 或 rolling-anchor correction。

### Step 4：CLIP ViT-L/14 dynamic ratio sweep under v2

固定配置：

```text
refresh_interval: 4
skip_threshold: 0.001
dense_threshold: 0.006
num_frames: 96
```

结果：

```text
r=0.25:
  speedup: 3.029x
  cosine: 0.989137
  mse: 0.027976
  dense/sparse/skip: 24/12/60

r=0.50:
  speedup: 2.907x
  cosine: 0.990361
  mse: 0.024837
  dense/sparse/skip: 24/12/60

r=0.75:
  speedup: 2.796x
  cosine: 0.992097
  mse: 0.020378
  dense/sparse/skip: 24/12/60
```

分析：

- 在 v2 下，dynamic ratio 只影响进入 sparse 分支的帧，不影响 skip/dense 分布。
- 降低 ratio 可以进一步提速，但保真度会下降。
- 当前更关键的变量仍是 routing 分布；一旦 skip 帧占比较高，dynamic ratio 的影响会弱于 refresh interval 和 skip threshold。

### 本轮 CLIP 结论

- 预训练 CLIP ViT-L/14 上，v1 已经能加速，但上限有限，且存在明显保真度损失。
- v2 routing 是当前最有潜力的方向：在 96 帧 synthetic streaming 上达到 `2.8x - 3.1x`，且 cosine 约 `0.991 - 0.992`。
- refresh interval sweep 清楚暴露 stale reference 问题，为 dual-anchor 提供直接实验动机。
- dynamic ratio sweep 说明 token 更新比例是二级旋钮，routing 策略是一级旋钮。

下一步：

- 接入真实视频解码，使用真实视频帧替代 synthetic frames。
- 在 CLIP ViT-L/14 上跑多视频、多 seed 统计。
- 实现 dual-anchor v3：
  - 短期 rolling anchor 处理局部连续性；
  - 长期 key anchor 防止语义漂移；
  - 对中间帧使用双锚点 drift 或插值判别。

## v3 Analysis：真实视频 failure mode 与 routing signal 分析

目标：

- 不急着实现最终 v3 加速算法，先做顶会论文所需的 failure mode 分析。
- 在真实视频上逐帧记录：
  - patch embedding drift；
  - low/mid/deep layer cosine；
  - dense/sparse/skip 决策；
  - final output cosine/MSE；
  - false skip；
  - signal 与 final error 的相关性。
- 用这些结果回答：为什么 synthetic 上 v2 很好，但真实视频上会失效；接下来应优化哪一部分。

新增脚本：

```text
experiments/turbovit_v1/scripts/run_v3_analysis.py
scripts/run_v3_analysis_local.ps1
```

输出文件：

```text
v3_analysis_summary.json
frame_analysis.csv
signal_correlations.csv
decision_summary.csv
false_skip.csv
policy_simulation.csv
timeline_output_cosine.svg
timeline_patch_drift.svg
scatter_patch_drift_vs_error.svg
scatter_best_signal_vs_error.svg
```

### Step 1：真实视频，frame_stride = 4

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_v3_analysis \
  --backbone clip \
  --model-path /home/mllm/models/clip-vit-large-patch14-336 \
  --video-source real \
  --video-path data/turbovit_v1/big_buck_bunny.mp4 \
  --output-dir results/turbovit_v1/clip_v3_analysis_real_48f \
  --num-frames 48 \
  --frame-stride 4 \
  --refresh-interval 4 \
  --dynamic-ratio 0.75 \
  --skip-threshold 0.001 \
  --dense-threshold 0.006 \
  --false-skip-cosine 0.99 \
  --device cuda
```

结果：

```text
dense latency/frame: 38.020 ms
turbo latency/frame: 38.802 ms
speedup: 0.980x
mean output cosine: 0.988983
false skip count: 0

decision distribution:
dense:  41 frames, cosine 1.0000
sparse: 7 frames,  cosine 0.9243
skip:   0 frames
```

分析：

- `frame_stride=4` 对真实视频来说跨度过大，patch drift 经常超过 `dense_threshold`，导致 routing 大量回退到 dense。
- 少数 sparse 帧的质量很差，mean cosine 只有 `0.924`。
- 这说明 synthetic 结果不能直接代表真实视频；真实视频中运动、光照、局部语义变化会显著放大 drift。
- 当前 v2 在该设置下不是有效加速方案。

signal 相关性：

```text
best signal: layer11_cos_to_long
pearson with final_error: -0.9946

patch_mse_to_rolling corr: -0.1033
patch_mse_to_long corr:    -0.0962
```

结论：

- patch embedding MSE 几乎不能预测最终误差。
- 中深层 feature cosine 与最终误差高度相关。
- v3 的 routing signal 不能继续只依赖 patch MSE。

### Step 2：真实视频逐帧输入，frame_stride = 1

命令：

```bash
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/conda run -n base \
  python -m experiments.turbovit_v1.scripts.run_v3_analysis \
  --backbone clip \
  --model-path /home/mllm/models/clip-vit-large-patch14-336 \
  --video-source real \
  --video-path data/turbovit_v1/big_buck_bunny.mp4 \
  --output-dir results/turbovit_v1/clip_v3_analysis_real_48f_stride1 \
  --num-frames 48 \
  --frame-stride 1 \
  --refresh-interval 4 \
  --dynamic-ratio 0.75 \
  --skip-threshold 0.001 \
  --dense-threshold 0.006 \
  --false-skip-cosine 0.99 \
  --device cuda
```

结果：

```text
dense latency/frame: 37.960 ms
turbo latency/frame: 28.471 ms
speedup: 1.333x
mean output cosine: 0.943008
false skip count: 9

decision distribution:
dense:  13 frames, cosine 1.0000
sparse: 24 frames, cosine 0.9633
skip:   11 frames, cosine 0.8314
```

分析：

- 逐帧输入确实带来了更多 reuse 机会，因此速度提升到 `1.33x`。
- 但质量明显不够，尤其 skip 分支严重误判。
- false skip 说明 patch drift 小不代表语义/深层特征稳定。
- 真实视频中的一些帧 patch MSE 低于阈值，但 final output cosine 只有 `0.73 - 0.82`。

signal 相关性：

```text
layer5_cos_to_long corr:  -0.7774
layer11_cos_to_long corr: -0.7663
final/output cosine to long corr: -0.7433

patch_mse_to_rolling corr: -0.1048
patch_mse_to_long corr:    -0.1210
```

结论：

- 中层特征比 patch MSE 更适合作为 skip confirmation。
- 单纯 patch-MSE routing 会把一些运动/语义变化明显的帧误判为可 skip。
- v3 需要 feature-aware routing，而不是只调 threshold。

### Step 3：feature gate policy simulation

新增输出：

```text
policy_simulation.csv
best_policy_simulation.json
```

模拟方式：

- 基于真实逐帧分析结果，离线模拟：
  - `gate_skip_to_dense`：如果 skip 帧的 layer cosine 低于阈值，则改为 dense；
  - `gate_reuse_to_dense`：如果 skip/sparse 帧的 layer cosine 低于阈值，则改为 dense。
- 这是上界分析，不代表当前实际 runtime，因为 layer feature 是 dense oracle 得到的。
- 目的不是报告最终性能，而是验证“加 feature gate 是否能消除 false skip”。

关键结果：

```text
gate_skip_to_dense, layer11_cos_to_long >= 0.999:
  speedup: 1.100x
  mean cosine: 0.9816
  false skip: 0

gate_reuse_to_dense, layer11_cos_to_long >= 0.999:
  speedup: 1.043x
  mean cosine: 0.99997
  false skip: 0
```

分析：

- feature gate 可以消除 false skip，但会牺牲大量速度。
- 如果只 gate skip，仍保留 sparse 分支，mean cosine 只有 `0.982`，说明 sparse 分支本身也需要改进。
- 如果 gate skip+sparse，质量几乎恢复 dense，但速度只剩 `1.04x`，说明直接用深层 feature gate 的成本/保守性太高。
- 这支持一个重要研究判断：v3 不能只是“加一个更深层阈值”，而要设计低成本、渐进式、双锚点的 routing。

### Step 4：真实视频 sparse 分支 dynamic ratio 分析

设置：

```text
frame_stride: 1
refresh_interval: 4
skip_threshold: 0.0001
dense_threshold: 0.006
num_frames: 48
```

结果：

```text
r=0.75:
  speedup: 1.131x
  mean cosine: 0.975341
  false skip: 0
  dense/sparse/skip: 13/33/2
  sparse cosine: 0.9642

r=0.90:
  speedup: 1.021x
  mean cosine: 0.989725
  false skip: 0
  dense/sparse/skip: 13/33/2
  sparse cosine: 0.9851

r=1.00:
  speedup: 1.016x
  mean cosine: 0.999988
  false skip: 0
  dense/sparse/skip: 13/33/2
  sparse cosine: 1.0000
```

分析：

- 提高 dynamic ratio 能恢复 sparse 分支质量。
- `r=0.9` 接近可用，但速度收益几乎消失。
- `r=1.0` 验证 sparse 实现正确，输出几乎等于 dense，但这已经不是真正稀疏加速。
- 说明真实视频上不能依赖固定比例 token update，必须做 adaptive token selection 或 layer-wise update。

### 本轮真实视频结论

1. Synthetic 上成立的 patch-MSE routing，在真实视频上明显不够可靠。
2. 真实视频中，patch MSE 与最终误差相关性很低。
3. 中层/深层 feature cosine 对最终误差预测更强。
4. skip 分支是速度来源，但也是主要风险来源。
5. sparse 分支在 `r=0.75` 下质量不够；提高到 `r=0.9` 质量接近可用但速度收益很小。
6. v3 的核心不应是单纯调阈值，而应是：
   - dual-anchor；
   - feature-aware routing；
   - adaptive ratio；
   - 对 skip 和 sparse 分支分别设置更严格的进入条件。

下一步优化方向：

- 实现 v3 dual-anchor analysis：
  - long anchor：最近 dense keyframe；
  - rolling anchor：最近 accepted output；
  - candidate skip 必须同时满足 rolling drift 和 long-anchor feature consistency。
- 实现 low-cost staged routing：
  - Stage 0：patch MSE 快速过滤明显 dense 帧；
  - Stage 1：只对候选 skip 帧计算前 K 层 feature gate；
  - Stage 2：不确定帧进入 sparse；
  - Stage 3：高风险帧 dense refresh。
- 实现 adaptive dynamic ratio：
  - drift 越大，ratio 越高；
  - 靠近 reference 的帧允许更低 ratio；
  - 高风险层使用更高 ratio。

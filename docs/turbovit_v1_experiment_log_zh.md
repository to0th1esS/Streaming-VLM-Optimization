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

## v3 Staged Routing 在线原型

目标：

- 将上一节的离线 `policy_simulation` 推进到真实 runtime 原型。
- 不再使用 dense oracle 的 layer feature，而是在在线推理中只对候选 skip 帧额外跑前 K 层 feature gate。
- 验证三个问题：
  1. feature gate 能否减少真实视频 false skip；
  2. gate 的额外耗时是否可接受；
  3. sparse 分支是否仍是质量/速度瓶颈。

新增实现：

```text
experiments/turbovit_v1/methods/turbovit_v3.py
experiments/turbovit_v1/scripts/run_turbovit_v3.py
scripts/run_turbovit_v3_local.ps1
```

在线 routing 逻辑：

```text
1. forced reference 或 patch_mse >= dense_threshold:
   -> dense

2. patch_mse <= skip_threshold:
   -> 先运行前 K 层 feature gate
   -> 如果 gate feature 同时接近 rolling anchor 和 long anchor:
      -> skip
   -> 否则:
      -> gate_dense

3. 中间 drift:
   -> sparse update
```

工程修正：

- 初版 gate 失败后会重新从头 dense，造成重复计算。
- 已修正为：gate 失败时从 gate layer 之后继续前向，复用已经算过的前 K 层。
- 新增 feature gate warmup，降低首次路径计时污染。

### Step 1：固定 ratio，扫 feature gate threshold

设置：

```text
backbone: CLIP ViT-L/14
video: Big Buck Bunny
num_frames: 48
frame_stride: 1
refresh_interval: 4
dynamic_ratio: 0.9
skip_threshold: 0.001
dense_threshold: 0.006
feature_gate_layer: 5
```

结果：

```text
feature_skip_threshold=0.98:
  speedup: 0.968x
  mean cosine: 0.989705
  false skip: 2

feature_skip_threshold=0.99:
  speedup: 0.953x
  mean cosine: 0.995353
  false skip: 1

feature_skip_threshold=0.999:
  speedup: 0.948x
  mean cosine: 0.995353
  false skip: 1
```

分析：

- 在线 feature gate 明显提升质量，但初版低于 dense。
- 主要原因是 gate 失败后重复计算前 K 层。

### Step 2：复用 gate prefix 后的结果

设置：

```text
dynamic_ratio: 0.9
skip_threshold: 0.001
dense_threshold: 0.006
```

结果：

```text
gate_layer=5, threshold=0.99:
  speedup: 1.020x
  mean cosine: 0.995353
  false skip: 1
  dense/gate_dense/skip/sparse: 13/13/3/19

gate_layer=2, threshold=0.9995:
  speedup: 1.027x
  mean cosine: 0.995353
  false skip: 1
  dense/gate_dense/skip/sparse: 13/13/3/19

gate_layer=0, threshold=0.9999:
  speedup: 1.006x
  mean cosine: 0.995801
  false skip: 0
  dense/gate_dense/skip/sparse: 13/15/2/18
```

分析：

- 复用 gate prefix 后，v3 从低于 dense 变为略高于 dense。
- 更浅的 gate layer 可以降低 `gate_dense` 成本。
- `gate_layer=0` 可以消除 false skip，但更保守，skip 更少，速度下降。
- 当前在线 v3 的主要瓶颈不是 gate 本身，而是：
  - skip 帧太少；
  - sparse 帧仍接近 dense 耗时；
  - 为了保证质量需要较高 dynamic ratio。

### Step 3：固定 gate，扫 ratio 和 skip threshold

设置：

```text
feature_gate_layer: 2
feature_skip_threshold: 0.9995
dense_threshold: 0.006
```

结果：

```text
r=0.75, skip=0.001:
  speedup: 1.087x
  mean cosine: 0.988549
  false skip: 1
  dense/gate_dense/skip/sparse: 13/13/3/19

r=0.75, skip=0.0005:
  speedup: 1.151x
  mean cosine: 0.974874
  false skip: 1
  dense/skip/sparse: 13/3/32

r=0.90, skip=0.0005:
  speedup: 1.040x
  mean cosine: 0.989307
  false skip: 1
  dense/skip/sparse: 13/3/32
```

分析：

- 降低 ratio 可以提升速度，但 sparse 质量下降明显。
- 降低 skip threshold 后，更多帧进入 sparse 而不是 gate_dense，速度上升但质量下降。
- `r=0.75, skip=0.001` 是当前较好的折中点，但 cosine 仍低于理想论文主结果。

### Step 4：adaptive dynamic ratio

实现：

```text
sparse_dynamic_ratio =
  dynamic_ratio_min + alpha * (dynamic_ratio_max - dynamic_ratio_min)

alpha = clamp((frame_drift - skip_threshold) / (dense_threshold - skip_threshold), 0, 1)
```

设置与结果：

```text
gate_layer=2, ratio 0.75 -> 0.95, skip=0.0005:
  speedup: 1.106x
  mean cosine: 0.978085
  false skip: 1

gate_layer=2, ratio 0.75 -> 1.00, skip=0.0005:
  speedup: 1.104x
  mean cosine: 0.978532
  false skip: 1

gate_layer=2, ratio 0.75 -> 0.95, skip=0.001:
  speedup: 1.065x
  mean cosine: 0.990499
  false skip: 1

gate_layer=0, ratio 0.75 -> 0.95, skip=0.001:
  speedup: 1.043x
  mean cosine: 0.990965
  false skip: 0
```

分析：

- adaptive ratio 方向是合理的，但当前线性映射提升有限。
- 原因是大多数 sparse 帧的 patch drift 数值不高，映射后的 ratio 仍接近下界。
- 这再次说明 patch drift 不是足够好的 ratio 控制信号。
- 要让 adaptive ratio 真正有效，应该用 feature drift 或 token-level drift 来调节 ratio。

### 本轮 v3 在线原型结论

1. 在线 feature gate 能显著减少 false skip。
2. gate prefix 复用是必要工程优化，否则 gate 会直接吃掉收益。
3. 真实视频上当前 v3 最好只是小幅超过 dense，约 `1.04x - 1.09x`，质量较稳。
4. 若追求更高速度，固定低 ratio sparse 会带来明显质量下降。
5. 当前最核心瓶颈已经明确：
   - skip 机会不足；
   - sparse 分支性价比不足；
   - patch drift 不能很好控制 skip 和 dynamic ratio。

下一步研究重点：

- 不再单纯调全局 threshold。
- 做 feature-aware adaptive ratio：
  - 用 low/mid-layer token drift 而不是 patch MSE 决定 ratio；
  - 对高风险层或高风险 token 自动提高 ratio；
  - 对背景/稳定 token 使用更低 ratio。
- 做 dual-anchor sparse：
  - sparse token 不只复用 long/rolling 的单一状态；
  - 根据 long/rolling 一致性选择复用来源或插值。
- 做真实视频多片段统计：
  - 当前 Big Buck Bunny 单片段有明显场景依赖；
  - 需要静态镜头、缓慢运动、快速运动、镜头切换、小目标运动五类子集。

## v4：Semantic-Stability Adaptive Reuse 原型

方法设计收敛：

前面的 v1/v2/v3 不应在最终论文中以“工程尝试堆叠”的形式呈现。根据实验现象，可以抽象成一个更干净的核心 insight：

```text
真实流视频中的可复用性不是由 patch-level change 决定，
而是由视觉编码器内部的 semantic state stability 决定。
```

因此 v4 不再把 patch drift、feature gate、ratio 分成多个割裂模块，而是引入一个统一的语义稳定性分数：

```text
S_t = min(
  cosine(P_t, P_rolling),
  cosine(P_t, P_long)
)
```

其中：

- `P_t`：当前帧在浅层 probe layer 的语义状态；
- `P_rolling`：最近一次 accepted state 的 rolling anchor；
- `P_long`：最近 dense keyframe 的 long anchor；
- `S_t` 同时用于决定：
  - skip；
  - sparse；
  - dense；
  - sparse 的 adaptive token ratio。

这比 v3 的“先 patch 再 gate”更适合作为最终方法叙事，因为它不是补丁式规则，而是一个统一准则：

```text
semantic stability controls reuse.
```

新增实现：

```text
experiments/turbovit_v1/methods/turbovit_v4.py
experiments/turbovit_v1/scripts/run_turbovit_v4.py
scripts/run_turbovit_v4_local.ps1
```

v4 routing：

```text
1. forced keyframe:
   dense, update rolling anchor and long anchor

2. non-keyframe:
   run shallow probe layer
   compute semantic stability S_t to rolling and long anchors

3. if patch drift high or S_t low:
   dense refresh

4. if patch drift low and S_t very high:
   skip

5. otherwise:
   sparse update with adaptive ratio
```

adaptive ratio：

```text
ratio_t = ratio_min + alpha * (ratio_max - ratio_min)
alpha = (skip_feature_threshold - S_t)
        / (skip_feature_threshold - dense_feature_threshold)
```

直觉：

- `S_t` 越高，帧越稳定，可以用更低 ratio；
- `S_t` 越低，帧越不稳定，ratio 自动升高；
- 如果低到不可复用，则直接 dense。

### Step 1：v4 三档语义稳定性配置

真实视频设置：

```text
backbone: CLIP ViT-L/14
video: Big Buck Bunny
num_frames: 48
frame_stride: 1
refresh_interval: 4
probe_layer: 2
```

结果：

```text
balanced:
  ratio: 0.75 -> 1.00
  skip_feature_threshold: 0.9995
  dense_feature_threshold: 0.98
  speedup: 1.063x
  mean cosine: 0.989540
  false skip: 1
  dense/skip/sparse: 14/3/31

conservative:
  ratio: 0.80 -> 1.00
  skip_feature_threshold: 0.9997
  dense_feature_threshold: 0.99
  speedup: 1.034x
  mean cosine: 0.995961
  false skip: 1
  dense/skip/sparse: 20/3/25

aggressive:
  ratio: 0.70 -> 0.95
  skip_feature_threshold: 0.9990
  dense_feature_threshold: 0.97
  speedup: 1.113x
  mean cosine: 0.978023
  false skip: 1
  dense/skip/sparse: 13/3/32
```

分析：

- v4 形成了清晰 Pareto：
  - aggressive 更快但质量下降；
  - conservative 更稳但速度下降；
  - balanced 居中。
- 但三个配置仍有 1 个 false skip，说明 skip 条件还需更严格。

### Step 2：strict skip

将 `skip_feature_threshold` 提高到 `0.9999`。

48 帧结果：

```text
strict skip:
  ratio: 0.75 -> 1.00
  skip_feature_threshold: 0.9999
  dense_feature_threshold: 0.98
  speedup: 1.045x
  mean cosine: 0.990123
  false skip: 0
  dense/skip/sparse: 14/2/32

strict conservative:
  ratio: 0.80 -> 1.00
  skip_feature_threshold: 0.9999
  dense_feature_threshold: 0.99
  speedup: 1.013x
  mean cosine: 0.996372
  false skip: 0
  dense/skip/sparse: 20/2/26
```

分析：

- strict skip 可以消除 false skip。
- 代价是 skip 帧更少，因此速度下降。
- 这说明真实视频里 skip 必须非常保守，不能作为唯一加速来源。

### Step 3：96 帧稳定性验证

配置：

```text
ratio: 0.75 -> 1.00
skip_feature_threshold: 0.9999
dense_feature_threshold: 0.98
num_frames: 96
```

结果：

```text
speedup: 1.019x
mean cosine: 0.994589
false skip: 0
dense/skip/sparse: 54/2/40

dense latency/frame: 37.874 ms
v4 latency/frame: 37.184 ms
sparse mean ratio: 0.856
sparse mean semantic stability: 0.9915
```

分析：

- 96 帧结果说明 strict v4 的质量稳定，false skip 维持为 0。
- 但加速只有 `1.02x`，说明当前真实视频片段中：
  - skip 机会非常少；
  - dense refresh 比例较高；
  - sparse 分支仍接近 dense 成本。
- 这不是方法设计失败，而是进一步定位出最终优化重点：

```text
语义稳定性用于帧级 routing 是必要的，
但还不够。
下一步要把语义稳定性推进到 token-level reuse。
```

### 当前方向性结论

截至 v4，我们可以把最终论文方法收敛为一个优雅方向：

```text
Semantic-Stability Guided Streaming ViT Reuse
```

它的核心不是“做了 patch MSE、feature gate、adaptive ratio 几个工程模块”，而是：

```text
利用双锚点估计当前视觉语义状态的稳定性，
再用稳定性统一决定帧级路由与 token 级复用强度。
```

现有实验支持三个 insight：

1. **Patch change is not semantic stability**
   - patch MSE 与 final error 相关性弱；
   - 真实视频 false skip 证明低 patch drift 不等于可复用。

2. **Dual-anchor semantic stability is safer**
   - rolling anchor 捕捉局部连续性；
   - long anchor 防止 rolling state 漂移；
   - `min(sim rolling, sim long)` 是一个自然稳定性下界。

3. **Frame-level routing alone is insufficient**
   - strict routing 可保质量但速度有限；
   - 真正要释放速度，必须把稳定性用于 token-level reuse。

下一步最重要：

- 实现 `token-level semantic stability`：
  - 在 probe layer 计算每个 token 到 rolling/long anchor 的稳定性；
  - 每个 token 得到自己的 reuse score；
  - 高稳定 token 复用；
  - 低稳定 token 重算；
  - 中间 token 可根据 long/rolling 更接近的一侧选择复用来源。
- 这会把 v4 从“帧级稳定性路由”推进为真正适合论文最终方法的：

```text
Semantic-Stability Guided Token Reuse
```

## 2026-06-01：v5 Token-Level Semantic Dual-Anchor Reuse 初步验证

### 目标

这一轮正式开始验证 token 级双锚点复用：

```text
不再只判断“整帧是否稳定”，
而是在 probe layer 后为每个 token 估计语义稳定性，
稳定 token 复用，动态 token 重算。
```

核心机制：

- rolling anchor 表示短期连续状态；
- long/reference anchor 表示关键帧全局状态；
- 每个 token 计算到 rolling / long 的相似性；
- 使用 `min(sim_rolling, sim_long)` 作为 token 语义稳定性下界；
- 稳定性最低的一部分 token 进入重算路径；
- 静态 token 根据 `sim_rolling >= sim_long` 选择 rolling 或 long cache 作为复用来源。

这一步的研究意义是：把 v4 的“帧级 semantic routing”推进为更接近最终论文方法的“token-level semantic reuse”。

### 实验设置

```text
model: CLIP ViT-L/14@336
weights: /home/mllm/models/clip-vit-large-patch14-336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
device: remote A100
probe_layer: 2
skip_feature_threshold: 0.9999
dense_feature_threshold: 0.98
```

### refresh interval 与动态 token 比例扫描

| 配置 | speedup | mean cosine | false skip | dense/skip/sparse | sparse ratio | rolling/long reuse |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| N=4, ratio 0.60->0.95 | 1.068x | 0.984326 | 0 | 48/2/46 | 0.751 | 0.904 / 0.096 |
| N=8, ratio 0.60->0.95 | 1.074x | 0.984488 | 0 | 45/2/49 | 0.772 | 0.926 / 0.074 |
| N=12, ratio 0.60->0.95 | 1.082x | 0.985228 | 0 | 43/2/51 | 0.766 | 0.923 / 0.077 |
| N=8, ratio 0.80->1.00 | 1.031x | 0.993598 | 0 | 45/2/49 | 0.899 | 0.926 / 0.074 |
| N=12, ratio 0.80->1.00 | 1.030x | 0.992976 | 0 | 43/2/51 | 0.895 | 0.923 / 0.077 |
| N=12, ratio 0.90->1.00 | 1.026x | 0.996097 | 0 | 43/2/51 | 0.947 | 0.923 / 0.077 |
| N=16, ratio 0.80->1.00 | 1.037x | 0.992580 | 0 | 44/2/50 | 0.901 | 0.924 / 0.076 |
| N=16, ratio 0.90->1.00 | 1.022x | 0.995540 | 0 | 44/2/50 | 0.951 | 0.924 / 0.076 |

### 关键拆解

以 N=16, ratio 0.90->1.00 为例：

```text
dense latency/frame: 37.752 ms
v5 latency/frame: 36.945 ms
speedup: 1.022x
mean cosine: 0.995540
false skip: 0

sparse frames: 50 / 96
sparse mean latency: 36.012 ms
probe: 1.087 ms
token selector: 0.174 ms
selector/scatter: 1.702 ms
sparse compute: 6.429 ms
rolling reuse: 92.4%
long reuse: 7.6%
```

和 v1/v3 的逐层判别相比，v5 的 token selector 成本已经明显下降，说明“先用轻量 probe 得到 token 稳定性，再统一指导后续复用”这个方向是成立的。

但当前端到端加速仍然有限，主要原因不是判别开销，而是 sparse 分支本身还接近 dense：

```text
真实 ViT-L 中，动态 token 比例一旦提高到 0.9 以上，
质量可以稳定到 cosine 0.995+，
但 sparse compute + scatter 仍然吃掉了主要收益。
```

### 关于加大关键帧刷新间隔

这轮验证了 `N=4/8/12/16`。

现象：

- 从 N=4 增大到 N=12，在低 ratio 配置下 speedup 从 1.068x 提升到 1.082x；
- 继续到 N=16 后，高保真配置仍保持 false skip 为 0；
- 但 N 增大带来的收益是温和的，不是数量级提升；
- 原因是当前真实视频里真正的 dense 决策并不只来自固定 key frame refresh，还来自 semantic stability 低时的主动回退。

因此，`refresh_interval` 可以作为释放速度的辅助旋钮，但不能单独构成最终方法的核心贡献。

### 关于 dual-anchor / rolling-anchor correction

当前结果说明可以正式进入这一方向，但需要明确它在论文方法中的角色：

```text
rolling anchor 是主复用来源，
long anchor 是漂移校正与稳定性下界约束。
```

在这段真实视频中，static token 约 92% 选择 rolling anchor，约 8% 选择 long anchor。这说明 long anchor 不是高频主路径，但它提供两个关键价值：

1. 防止 rolling state 一路漂移后被错误认为稳定；
2. 让 token stability 使用 `min(rolling, long)` 形成更保守的可复用下界。

这比“多加一个 anchor 做工程修补”的表述更适合作为论文 insight：

```text
短期连续性负责复用效率，
长期参考状态负责漂移约束，
二者共同定义 token-level semantic stability。
```

### 阶段性结论

当前 v5 支持以下方向判断：

1. 可以开始研究 dual-anchor / rolling-anchor correction，但它应作为“语义稳定性定义”的组成部分，而不是独立技巧。
2. 加大关键帧间隔能带来一定速度收益，但在真实视频里无法单独大幅提速。
3. 下一步真正值得优化的是 sparse execution path：
   - 减少 scatter/gather；
   - 合并 token selection 与 sparse compute；
   - 探索 block/token-group 级复用，避免单 token 索引导致 GPU kernel 低效；
   - 将 dense 回退从硬阈值改为更连续的 drift budget。
4. 最终论文方法应收敛为一个统一表述：

```text
Dual-Anchor Semantic Stability Guided Token Reuse
```

它不是多个工程尝试堆叠，而是一个清晰机制：

```text
用短期 rolling anchor 和长期 reference anchor 共同估计视觉 token 的语义稳定性，
再由稳定性统一决定 token 是否重算、从哪个 anchor 复用、以及何时刷新状态。
```

## 2026-06-01：v5 Anchor Ablation

### 目的

上一组结果说明 long anchor 被使用比例约 8%，但这还不足以证明 dual-anchor 必要。

因此增加 `anchor_mode` 消融：

```text
dual:         stability = min(sim_rolling, sim_long)，静态 token 选择更接近的 anchor
rolling_only: stability = sim_rolling，所有静态 token 复用 rolling anchor
long_only:    stability = sim_long，所有静态 token 复用 long/reference anchor
```

实验配置保持一致：

```text
model: CLIP ViT-L/14@336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
refresh_interval: 16
ratio: 0.80 -> 1.00
probe_layer: 2
device: remote A100
```

### 结果

| anchor mode | speedup | mean cosine | min cosine | p10 cosine | false skip | dense/skip/sparse | sparse ratio |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| dual | 1.033x | 0.992580 | 0.943300 | 0.976184 | 0 | 44/2/50 | 0.901 |
| rolling_only | 1.053x | 0.985255 | 0.890192 | 0.964159 | 0 | 28/2/66 | 0.872 |
| long_only | 1.035x | 0.991414 | 0.931265 | 0.974055 | 0 | 44/2/50 | 0.900 |

### 分析

rolling-only 的速度最高，原因很明确：

- rolling similarity 偏乐观；
- dense 回退帧从 dual 的 44 帧降到 28 帧；
- sparse 帧从 50 帧升到 66 帧；
- 平均动态 token 比例也更低。

但它的质量明显下降：

```text
mean cosine: 0.992580 -> 0.985255
min cosine:  0.943300 -> 0.890192
p10 cosine:  0.976184 -> 0.964159
```

这说明 rolling anchor 捕捉短期连续性很有效，但会高估可复用性；如果只看 rolling 状态，方法会更积极地复用，但会产生局部帧的大幅漂移。

long-only 比 rolling-only 更稳，但它没有利用 rolling anchor 的短期连续优势；其均值和低分位都略弱于 dual。

因此 dual-anchor 的价值不是“多一个缓存来源”，而是：

```text
rolling anchor 提供短期复用机会，
long anchor 提供长期漂移约束，
min(sim_rolling, sim_long) 把二者合成一个保守的 semantic stability 下界。
```

这组消融可以支撑论文中的方法设计：

1. 单 rolling anchor 会过度自信，速度更快但低质量帧风险增大；
2. 单 long anchor 稳定但缺少短期自适应；
3. dual-anchor 在速度与保真之间更均衡，并显著改善最差帧和低分位质量。

### 对下一步优化的启发

当前最值得继续推进的是：

- 保留 dual-anchor semantic stability 作为最终方法主线；
- 不再把“大幅加大 refresh interval”作为主要加速来源；
- 重点优化 sparse path 的 GPU 友好实现；
- 尝试 token-group / block-wise reuse，让复用不再依赖大量零散 gather/scatter；
- 引入 drift budget，让 rolling-only 的积极性在可控范围内释放，而不是完全依赖硬阈值回退。

## 2026-06-01：v6 Token-Group / Block-Wise Reuse 初步验证

### 目的

v5 的主要瓶颈已经不是 token selector，而是 sparse path 的零散 gather/scatter 与动态 token 计算不够 GPU 友好。

因此 v6 验证一个自然假设：

```text
如果不再选择离散 token，
而是按空间邻域选择动态 token group，
是否能让复用更稳定，并为后续 block-wise kernel 友好实现提供依据？
```

v6 保留 v5 的 dual-anchor semantic stability，但改动动态 token 选择方式：

- CLS token 始终重算；
- patch token 按 `group_size x group_size` 空间块分组；
- 每个块用 mean/min stability 得到 group score；
- 选择稳定性最低的若干组作为动态区域；
- 组内 token 全部重算，其余 token 继续按 rolling/long anchor 复用。

### 实验设置

```text
model: CLIP ViT-L/14@336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
refresh_interval: 16
probe_layer: 2
anchor_mode: dual
device: remote A100
```

### 高保真配置：ratio 0.80 -> 1.00

| method | group size | speedup | mean cosine | min cosine | p10 cosine | false skip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 token top-k | - | 1.033x | 0.992580 | 0.943300 | 0.976184 | 0 |
| v6 group top-k | 2 | 1.003x | 0.993238 | 0.942533 | 0.980694 | 0 |
| v6 group top-k | 3 | 1.019x | 0.994342 | 0.928517 | 0.984824 | 0 |

拆解观察：

```text
v6 group3 sparse latency: 36.229 ms
v5 sparse latency:        35.366 ms

v6 group3 sparse compute: 6.392 ms
v5 sparse compute:        6.507 ms

v6 group3 selector/scatter: 1.683 ms
v5 selector/scatter:        1.725 ms
```

v6 的 sparse compute 和 scatter 有轻微下降，但整体 sparse latency 没有明显领先。原因是当前实现仍然使用同一套 gather/scatter sparse path；空间块只改变了“选哪些 token”，没有真正改变 GPU kernel 的执行形态。

质量方面，v6 在高保真配置下提升了 mean/p10 cosine，说明空间块选择有一定语义合理性：动态区域往往局部连续，块状重算能减少孤立 token 错判。

### 激进配置：ratio 0.60 -> 0.95

| method | group size | speedup | mean cosine | min cosine | p10 cosine | false skip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v5 token top-k | - | 1.081x | 0.983931 | 0.911828 | 0.949083 | 0 |
| v6 group top-k | 3 | 1.058x | 0.983210 | 0.871338 | 0.947479 | 0 |
| v6 group top-k | 4 | 1.057x | 0.982639 | 0.890678 | 0.942818 | 0 |

激进配置下，v6 没有优于 v5。

这说明当前 block-wise selection 并不是“天然更稳”。当动态 token budget 变小后，块状选择会带来两个问题：

1. 一个 block 内并非所有 token 都同等动态，组内重算会浪费 budget；
2. 如果动态区域形状细碎，固定正方形 block 会漏掉部分真正变化 token。

### 阶段性结论

v6 给出一个重要负结果：

```text
只把 token top-k 换成 spatial group top-k，
不足以解决真实 ViT sparse path 的速度瓶颈。
```

它带来的启发更重要：

1. **选择粒度与执行粒度必须一起设计**
   - 现在 v6 只是 block-wise selection；
   - 但后端仍是 token gather/scatter；
   - 所以 GPU 友好性没有真正释放。

2. **块状复用适合高保真，但不适合盲目激进压缩**
   - 高保真配置下 p10 cosine 提升；
   - 激进配置下 min cosine 下降；
   - 固定空间块会在 token budget 低时变得粗糙。

3. **下一版不应继续堆 selection heuristic**
   - 继续换 group score 或 group size 的收益有限；
   - 更关键的是把 sparse path 变成真正的 segment/block execution。

### 下一步方法方向

v6 之后，论文方法主线应保持：

```text
Dual-Anchor Semantic Stability Guided Token Reuse
```

但工程实现要从“token sparse”转为“segment-aware reuse”：

```text
先用 dual-anchor stability 得到 token-level score，
再把相邻高动态 token 合并成少量 segment，
对 segment 做连续 gather / batched compute / scatter，
避免大量离散 token 索引。
```

这比简单 block-wise top-k 更自然：

- token-level score 保留精细判别；
- segment 合并提供 GPU 友好的连续执行；
- segment 长度可自适应，不被固定正方形 block 限制；
- 论文表述也更优雅：从 semantic stability 到 reusable visual segments。

## 2026-06-01：v7 Adaptive Segment-Aware Reuse

### 目的

v6 的负结果说明，固定空间块不是足够优雅的最终形态。它改变了选择粒度，但没有解决两个问题：

1. 固定正方形 block 会浪费动态 token budget；
2. 后端仍然是离散 token gather/scatter，没有真正变成连续执行。

v7 改为更自然的 adaptive segment：

```text
保留 token-level semantic stability 的精细判断，
再把相邻动态 token 合并成少量连续 segment。
```

这一步的论文动机更清晰：

```text
语义稳定性决定哪些视觉 token 需要更新；
视觉连续性把这些 token 组织成可执行片段；
最终目标是从 token reuse 走向 visual segment reuse。
```

### 方法

v7 仍然使用 v5 的 dual-anchor stability：

```text
stability_i = min(sim_i_to_rolling, sim_i_to_long)
```

之后：

- 先按 token stability 选择最低的一批候选动态 patch；
- 在每一行 patch grid 内，把相邻候选 token 合并为 segment；
- 允许填补长度不超过 `segment_max_gap` 的小间隙；
- 允许把过短 segment 扩展到 `min_segment_len`；
- CLS token 始终重算；
- 静态 token 仍按 rolling/long 更接近的一侧复用。

### 实验设置

```text
model: CLIP ViT-L/14@336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
refresh_interval: 16
probe_layer: 2
anchor_mode: dual
device: remote A100
```

### 高保真配置：ratio 0.80 -> 1.00

| method | segment config | speedup | mean cosine | min cosine | p10 cosine | false skip |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v5 token top-k | - | 1.033x | 0.992580 | 0.943300 | 0.976184 | 0 |
| v7 segment | gap=1, min_len=2 | 1.000x | 0.994970 | 0.950907 | 0.986075 | 0 |

v7 明显提升了保真度：

```text
mean cosine: 0.992580 -> 0.994970
min cosine:  0.943300 -> 0.950907
p10 cosine:  0.976184 -> 0.986075
```

但速度下降到接近 dense，说明当前实现仍然不是最终高效形态。

### 激进配置：ratio 0.60 -> 0.95

| method | segment config | speedup | mean cosine | min cosine | p10 cosine | false skip |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v5 token top-k | - | 1.081x | 0.983931 | 0.911828 | 0.949083 | 0 |
| v7 segment | gap=1, min_len=2 | 1.034x | 0.989186 | 0.920362 | 0.968506 | 0 |
| v7 segment | gap=1, min_len=1 | 1.031x | 0.987762 | 0.919364 | 0.964808 | 0 |
| v7 segment | gap=0, min_len=2 | 1.046x | 0.987391 | 0.913890 | 0.962776 | 0 |

激进配置下，v7 的结论更清楚：

- 速度不如 v5；
- 但质量显著高于 v5；
- 尤其 p10 cosine 从 0.949 提升到 0.969。

也就是说，segment 化不是当前工程实现里的速度收益来源，而是质量稳定性来源。

### Segment 统计

典型 sparse 帧统计：

```text
ratio 0.80->1.00, gap=1, min_len=2:
  mean segment count: 31.7
  mean segment length: 17.36
  expansion ratio: 1.027

ratio 0.60->0.95, gap=1, min_len=2:
  mean segment count: 40.38
  mean segment length: 12.25
  expansion ratio: 1.053

ratio 0.60->0.95, gap=0, min_len=2:
  mean segment count: 58.34
  mean segment length: 8.58
  expansion ratio: 1.027
```

这说明动态 token 在真实视频中确实可以被组织成连续片段，但片段数量仍然不少。

### 阶段性结论

v7 给出了一个比 v6 更有论文价值的 insight：

```text
Token-level semantic stability 应该保留；
但执行上不应该直接处理孤立 token。
更合理的中间表示是 semantic visual segments。
```

但目前 v7 仍然没有突破速度边界，因为它只是在选择后生成 segment，真正的 sparse compute 仍复用 v5 的 gather/scatter 实现。

因此，“前端 ViT-only 优化”已经基本触摸到当前原型边界：

1. 判别开销已经很低；
2. 加大 refresh interval 收益有限；
3. dual-anchor 提升稳定性，但不是速度主来源；
4. segment 能提升质量，但当前后端没有利用连续性；
5. 真正的速度突破需要联合执行设计，而不是继续换 selector。

下一步应该从两个方向合并推进：

```text
ViT side:  segment-aware sparse execution
LLM side:  REKV-style streaming cache management
```

也就是说，后续不应只问“ViT 编码快了多少”，而应问：

```text
当前帧哪些视觉 segment 真的需要进入 LLM cache？
哪些 segment 可以只更新 ViT 表征但不触发长期 KV 写入？
哪些稳定 segment 可以沿用旧视觉 token 与旧语言 KV？
```

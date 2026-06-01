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

### 轻量后端 cache policy simulation

为了不直接把完整 VLM benchmark 引入当前闭环，先做了一个轻量模拟：

```text
如果后端语言模型支持复用稳定视觉 token 的 KV，
那么每帧实际需要重新写入多少视觉 token？
```

设置：

```text
visual_tokens_per_frame: 576
输入: v5/v7 逐帧 decision 与 dynamic_ratio_observed
规则:
  dense  帧写入 576 tokens
  skip   帧写入 0 tokens
  sparse 帧写入 dynamic_ratio_observed * 576 tokens
```

结果：

| method | config | ViT speedup | mean cosine | estimated visual token reduction |
| --- | --- | ---: | ---: | ---: |
| v5 | N=16, ratio 0.80->1.00 | 1.033x | 0.992580 | 7.2% |
| v7 | N=16, ratio 0.80->1.00 | 1.000x | 0.994970 | 6.0% |
| v5 | N=16, ratio 0.60->0.95 | 1.081x | 0.983931 | 13.7% |
| v7 | N=16, ratio 0.60->0.95 | 1.034x | 0.989186 | 11.6% |

这说明：

1. 前端复用信号确实可以转化为后端视觉 token 写入减少；
2. v5 更激进，token reduction 更高，但质量更低；
3. v7 更稳，token reduction 略低，但保真明显更好；
4. 接入 REKV 的价值不是简单叠加加速，而是用同一个 semantic stability 信号同时控制视觉重算与语言 KV 写入。

因此，下一阶段应该开始做 ViT+REKV 联合验证，但顺序应保持：

```text
cache policy simulation -> 小规模真实 VLM QA -> 大规模 benchmark
```

## 2026-06-01：v8 Layer-Aware Static K/V Reuse

### 动机

前面的问题是：我们是否已经利用了 ViT 内部层间特性？

答案是：

```text
v1 利用了，但太重：每层都重新判断动态 token；
v5/v7 减少了判断次数：只在 probe layer 判断一次；
但 v5/v7 后续层仍然偏 dense：每层仍对完整 hidden states 做 Q/K/V projection。
```

因此 v8 验证一个更深入的层内复用假设：

```text
动态 token mask 不需要每层重新判断；
静态 token 的 K/V 也不应该每层重新投影。
```

### 方法

v8 在 reference / rolling / long anchor 中额外缓存每层：

```text
key
value
output
```

对于当前帧：

1. 前几层 probe；
2. 用 dual-anchor semantic stability 得到动态 segment；
3. 后续层只对动态 segment 计算 Q/K/V；
4. 静态 token 的 K/V 从 rolling/long anchor cache 复用；
5. 动态 query attend 到 mixed K/V；
6. 静态 output 继续复用 anchor output。

这一步对应的论文概念是：

```text
Layer-Aware Dual-Anchor K/V Reuse
```

### 实验设置

```text
model: CLIP ViT-L/14@336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
refresh_interval: 16
probe_layer: 2
anchor_mode: dual
segment config: gap=1, min_len=2
device: remote A100
```

### 结果

| method | config | speedup | mean cosine | sparse latency | sparse compute | KV projection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| v7 segment | 0.80->1.00 | 1.000x | 0.994970 | 37.798 ms | 6.427 ms | full QKV |
| v8 layer-KV | 0.80->1.00 | 0.995x | 0.994917 | 38.096 ms | 9.404 ms | 1.714 ms |
| v7 segment | 0.60->0.95 | 1.034x | 0.989186 | 35.262 ms | 6.536 ms | full QKV |
| v8 layer-KV | 0.60->0.95 | 1.026x | 0.989014 | 35.519 ms | 9.524 ms | 1.730 ms |

同时测试了一个 anchor cache mixing 优化：

```text
where mode:
  full torch.where over rolling/long cache

scatter mode:
  使用多数 anchor 作为 base，只 scatter 少数 anchor token
```

结果 scatter mode 更慢：

```text
0.80->1.00: speedup 0.891x
0.60->0.95: speedup 0.929x
```

说明小索引 `nonzero/scatter` 的 kernel overhead 比 full `where` 更差。

### 分析

v8 证明了一个很重要的点：

```text
从数学计算量上看，复用静态 K/V 是合理的；
但在当前 PyTorch 原型中，混合 K/V cache 的内存操作成本超过了省下来的 Q/K/V projection。
```

也就是说，层间 K/V 复用这个方向是方法上合理的，但不能用当前这种逐层 Python/PyTorch tensor 拼接实现来证明速度。

这进一步说明：

1. **减少动态 token 判断次数已经完成**
   - v5/v7/v8 都只在 probe layer 判断一次；
   - 不再是 v1 的逐层判别。

2. **真正的瓶颈已经转移**
   - 不是 selector；
   - 不是 semantic stability；
   - 而是 mixed K/V construction、scatter、非连续 token kernel。

3. **层间特性仍然值得保留为论文方法**
   - 因为 K/V 复用提供了清晰的理论计算节省；
   - 但需要 fused segment execution 或者和后端 REKV cache 共同设计。

### 方向结论

v8 不应作为最终工程速度版本，但它给最终方法提供了一个关键 insight：

```text
ViT 内部层间复用不应该表现为每层重新选择 token；
而应该表现为一次 semantic routing 后的跨层 K/V propagation。
```

下一步如果继续前端，应做：

```text
fused segment K/V reuse
```

如果推进系统，应做：

```text
ViT segment stability -> LLM/REKV cache write/retrieval policy
```

目前更推荐后者，因为当前 PyTorch 前端原型已经反复显示：选择策略越来越合理，但单靠非融合 sparse tensor 操作很难释放真实速度。

## 2026-06-01：Speed-First 目标转向实验

### 背景

前面所有版本主要在保证视觉特征保真，尤其是控制 final feature cosine 和 false skip。

但如果目标是超越 STC，并且不完全限定在它的评测赛道上，下一阶段必须把目标从：

```text
高特征保真下的小幅加速
```

切换为：

```text
速度优先，任务保真兜底。
```

原因是 STC 的高加速并不是在严格保持每帧视觉特征 cosine 的前提下实现的，而是通过 benchmark QA 精度来约束压缩误差。因此，我们需要先找到 speed-first 的速度上界和 Pareto 区间。

### 实验设置

```text
model: CLIP ViT-L/14@336
video: Big Buck Bunny real video
num_frames: 96
frame_stride: 1
device: remote A100
method: v5 dual-anchor token reuse
主要变化:
  - 放宽 dense fallback
  - 降低 sparse dynamic ratio
  - 增大 refresh_interval
```

### 结果

| config | speedup | latency reduction | mean cosine | false skip | visual token reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| N=16, ratio 0.25->0.75, relaxed | 1.364x | 26.7% | 0.877436 | 1 | 51.8% |
| N=32, ratio 0.25->0.75, relaxed | 1.363x | 26.6% | 0.882127 | 1 | - |
| N=16, ratio 0.40->0.85, relaxed | 1.288x | 22.4% | 0.905866 | 1 | 40.5% |
| N=16, ratio 0.55->0.90, mid | 1.168x | 14.4% | 0.955865 | 1 | 25.2% |
| N=16, ratio 0.70->0.95, mid | 1.119x | 10.6% | 0.973667 | 1 | 17.5% |
| N=16, ratio 0.55->0.90, no-skip | 1.132x | 11.7% | 0.956092 | 0 | 22.8% |

其中：

```text
latency reduction = 1 - 1 / speedup
visual token reduction 来自 cache policy simulation
```

### Breakdown

最激进配置：

```text
N=16, ratio 0.25->0.75:
  dense/skip/sparse: 8/4/84
  sparse adaptive ratio: 0.455
  sparse latency: 26.916 ms
  speedup: 1.364x
  cosine: 0.877
```

中间配置：

```text
N=16, ratio 0.55->0.90:
  dense/skip/sparse: 17/4/75
  sparse adaptive ratio: 0.731
  sparse latency: 31.697 ms
  speedup: 1.168x
  cosine: 0.956
```

关闭 skip 后：

```text
N=16, ratio 0.55->0.90, no-skip:
  dense/sparse: 17/79
  false skip: 0
  speedup: 1.132x
  cosine: 0.956
```

### 分析

这组实验说明：

1. 纯 ViT 前端在真实视频上可以达到约 `1.36x` speedup，但对应 cosine 只有约 `0.88`，不能直接作为高保真主结果。
2. 如果希望维持 cosine 约 `0.95+`，目前较现实的 speedup 是 `1.13x-1.17x`。
3. 速度主要来自两个因素：
   - dense fallback 从之前的 44 帧降到 8-17 帧；
   - sparse dynamic ratio 从约 0.90 降到 0.45-0.73。
4. 关闭 skip 可以消除 false skip，但速度从 1.168x 降到 1.132x。
5. speed-first 配置已经让理论视觉 token 写入减少到 `22.8%-51.8%`，这说明后端 REKV/LLM cache 联合设计的空间明显变大。

### 与 STC 的关系

STC 的 ReKV 主结果是：

```text
ViT encoding latency ↓24.5%
LLM prefilling latency ↓45.3%
```

当前 speed-first v5 的最激进点：

```text
ViT latency ↓26.7%
visual token writing ↓51.8%
```

这说明如果只看速度上界，我们已经能接近甚至超过 STC-Cacher 的 ViT latency reduction。但问题是特征保真不足。

因此，下一步不应该继续单纯追求 feature cosine，而应转向：

```text
任务保真约束下的速度最大化。
```

也就是：

```text
用小规模 QA sanity check 判断 cosine 0.90-0.96 的视觉特征变化是否真的影响答案。
```

如果 QA 仍稳定，那么 speed-first 配置就可能成为比 STC 更强的候选；如果 QA 不稳定，则需要引入后端 cache/pruner 来补偿。

### 下一步结论

后续主线应该是：

```text
Dual-Anchor Semantic Stability
  -> speed-first ViT reuse
  -> REKV-style visual cache/pruning
  -> task-level QA fidelity
```

这比单纯复刻 STC 更有潜力，因为我们的 dual-anchor 和 segment stability 可以同时服务于：

1. ViT 动态重算；
2. 视觉 token 写入；
3. LLM cache 检索；
4. QA 任务保真控制。

---

## 2026-06-01：真实 LLaVA-OneVision tiny QA 闭环验证

### 目标

这一轮不是验证最终 Turbo-ViT 方法，而是先确认：

1. 当前仓库中的 LLaVA-OneVision + ReKV streaming QA 链路可以在远程 A100 上跑通；
2. `vit_patch_hf` 的 ViT sparse 入口可以被真实 VLM 调用；
3. sparse 后的答案是否在极小 QA sanity 上立即崩坏；
4. 为后续 RVS / OVO / StreamingBench 小规模实验补齐端到端计时字段。

### 工程修复

| 文件 | 修改 |
| --- | --- |
| `model/patch.py` | 兼容新版 transformers 中 Qwen2 RoPE 从 attention 层迁移到 model 层的情况 |
| `model/patch.py` | 兼容新版 `Qwen2Attention` 不再暴露 `num_heads` / `num_key_value_heads` 属性，改从 config 回退读取 |
| `model/patch.py` | 兼容新版 Qwen2 decoder layer 不再返回 KV cache 的协议，将 ReKV cache 暂存在 attention 模块上再由外层读取 |
| `model/llava_onevision_rekv.py` | 将 LLaVA-OneVision 每帧视觉 token block size 从 60 修正为 196 |
| `video_qa/rekv_stream_vqa.py` | 增加 streaming QA 分段计时：video load、init prompt、incremental video encode、QA decode、累计耗时 |

其中 `block_size=196` 是关键修正。失败日志中 `global_k.shape=[1, 14, 392, 64]`，392 对应 2 帧乘以 196 token；原先的 60 会导致 ReKV block 检索断言失败。

### 实验设置

```text
model: LLaVA-OneVision-Qwen2-0.5B
vision tower: SigLIP / OneVision vision tower
backend: ReKV streaming QA
video: data/turbovit_v1/big_buck_bunny.mp4
tiny QA: scripts/prepare_tiny_streaming_qa.py 生成的 3 个 sanity 问题
remote GPU: A100
sparse patch: model/vit_patch.py 当前主代码入口版
cache_interval: 2
update_token_ratio: 0.25
retrieve_size: 4
retrieve_chunk_size: 1
debug: true
```

### 0.25 fps 结果

| mode | loaded frames | encoded frames until last QA | cumulative video encode | last elapsed | answer sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 15 | 4 | 0.283s | 2.433s | 3/3 语义正确 |
| sparse | 15 | 4 | 0.368s | 2.480s | 3/3 语义正确 |

结论：在极短输入下，sparse 入口功能稳定，但没有加速。4 个实际编码帧太少，selector、cache、scatter 等固定开销占主导。

### 1 fps 结果

| mode | loaded frames | encoded frames until last QA | cumulative video encode | last elapsed | answer sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| dense | 60 | 16 | 0.572s | 3.379s | 3/3 语义正确 |
| sparse | 60 | 16 | 1.126s | 4.402s | 3/3 语义基本正确 |

逐问题 video encode：

| mode | Q1 new frames | Q1 encode | Q2 new frames | Q2 encode | Q3 new frames | Q3 encode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 8 | 0.331s | 4 | 0.131s | 4 | 0.110s |
| sparse | 8 | 0.585s | 4 | 0.271s | 4 | 0.270s |

结论：当前主代码 `vit_patch_hf` 的稀疏入口在真实 LLaVA-OneVision 上是可运行的，但不是当前论文候选的高效实现。它逐层计算完整 K、选择 token、clone/reference、scatter 混合输出，真实 GPU 上这些开销超过了少算 Q/V/MLP 的收益。

### 与前面 Turbo-ViT v5/v7/v8 的关系

这组 QA 实验不能和 v5/v7/v8 的 CLIP ViT-L 原型加速数值直接等同，原因是：

1. 主代码 `model/vit_patch.py` 是最早的隔离入口版，只保留 ViT sparse update 功能和后处理入口；
2. 论文候选逻辑目前在 `experiments/turbovit_v1` 中，尤其是 v5 dual-anchor、v7 segment-aware routing、v8 K/V reuse；
3. QA 链路验证的是“真实 VLM 端到端可接入”和“任务答案是否立刻失真”，不是最终速度上界；
4. 该结果再次确认：最终方法不能停留在朴素 token sparse scatter，而要把 dual-anchor semantic stability 进一步转化为 block/segment 级执行和后端 cache 写入控制。

### 方法设计启示

```text
当前主代码 sparse patch:
  功能可用
  QA sanity 稳定
  速度不够

下一步论文方法:
  不应呈现为“逐层 token scatter 工程优化”
  应呈现为“语义稳定性驱动的统一流式视觉记忆机制”
```

也就是说，最终 paper 中应弱化中间工程尝试，保留为 insight：

1. 短时流式视频中存在大量稳定视觉 token，但逐层逐 token 判别会吃掉收益；
2. 单参考帧会 stale，dual-anchor 能提供 long-term semantic prior 与 rolling correction；
3. token 级选择信号本身还可以服务于三个位置：ViT recomputation、visual token writing、LLM cache retrieval；
4. 真正的速度提升应来自减少进入 sparse path 的次数、减少写入后端 cache 的 token 数，以及 block/segment 级 GPU 友好执行。

### 下一步

1. 将 `experiments/turbovit_v1` 中 v5/v7 的 dual-anchor/segment stability 接到真实 LLaVA-OneVision vision tower，而不是继续使用主代码入口版；
2. 保留当前 `vit_output_postprocess` 入口，把 visual token writing / pruning 作为后处理实验位点；
3. 用 tiny QA 先做任务保真 sanity，再准备 RVS 小子集真实视频；
4. 在 RVS 小子集上记录 dense QA answer、Turbo-ViT answer、video encode latency、QA decode latency、visual token 写入比例、ReKV retrieval block 命中变化。

---

## 2026-06-01：真实 OneVision/SigLIP 上的 v7/v9 方法迁移实验

### 实验目的

这组实验的目的不是继续在离线 CLIP 原型上调参，而是回答一个更接近论文方法设计的问题：

> 前面在 CLIP ViT-L 上得到的 dual-anchor / segment-aware reuse 机制，迁移到真实 LLaVA-OneVision 的 SigLIP vision tower 后，瓶颈和有效信号是否仍然成立？

具体要验证四件事：

1. **机制可迁移性**：v7 的 dual-anchor segment-aware routing 能否直接运行在真实 OneVision/SigLIP vision tower 上；
2. **速度瓶颈定位**：真实 SigLIP 上到底是 routing、token selector、sparse execution，还是 dense fallback 吃掉收益；
3. **方法主线取舍**：最终论文方法应继续走 token-level sparse recomputation，还是转向 low-cost anchor gate + correction；
4. **顶会可发表性原则**：最终方法不能呈现为一串工程补丁，而要抽象为一个统一、简洁、有 insight 的流式视觉记忆机制。

### 顶会方法设计原则

这轮实验开始后，我们把方法设计原则明确为：

```text
最终 paper 中只呈现最终优雅方法，
中间尝试只作为 insight 来源，不作为方法堆叠。
```

因此，每个候选机制都必须回答：

1. 它是否能被解释为一个统一信号或统一原则；
2. 它是否同时服务于 ViT recomputation、visual token writing、LLM cache retrieval 中至少两个位置；
3. 它是否减少系统复杂性，而不是增加一堆难以解释的 if-else；
4. 它是否有清晰的定量证据支撑，而不是只在单个 case 上偶然有效。

### 工程实现

本轮新增：

| 文件 | 作用 |
| --- | --- |
| `experiments/turbovit_v1/models/hf_siglip_vit.py` | 将真实 `SiglipVisionModel` 包装成实验接口，支持 `forward_with_layers` / `forward_with_caches` |
| `experiments/turbovit_v1/scripts/run_onevision_v7.py` | 在真实 LLaVA-OneVision vision tower 上运行 v7 dual-anchor segment-aware reuse |
| `experiments/turbovit_v1/methods/turbovit_v9.py` | 新增 v9 AnchorGate：只用低成本 patch embedding drift 做 skip/dense 路由 |
| `experiments/turbovit_v1/scripts/run_onevision_v9.py` | 在真实 OneVision/SigLIP 上运行 v9 AnchorGate |

同时修改了 v7 的 segment selector：

```text
原来默认 ViT token = CLS + square patch grid
现在兼容 SigLIP token = square patch grid，无 CLS token
```

这是必要修正，因为 OneVision/SigLIP 的视觉 token 序列没有 CLIP 风格 CLS token。

### 实验设置

```text
model: LLaVA-OneVision-Qwen2-0.5B
vision tower: SigLIPVisionModel
video: Big Buck Bunny
remote GPU: A100
dtype: float16
num_frames: 32
main clip: start_frame=120, frame_stride=1
baseline: dense SigLIP vision tower
```

### v7 迁移实验：dual-anchor segment-aware sparse

#### 实验 A：保守高保真配置

```text
start_frame: 0
frame_stride: 8
refresh_interval: 8
sparse_ratio: 0.55 -> 0.90
old CLIP thresholds
```

结果：

| speedup | latency reduction | mean cosine | min cosine | decision |
| ---: | ---: | ---: | ---: | --- |
| 0.789x | -26.7% | 0.9993 | 0.9985 | all dense |

分析：

旧 CLIP 阈值迁移到 SigLIP 后过于保守，32 帧全部进入 dense。这个结果证明了机制可运行，但不能证明 sparse 有效。

#### 实验 B：强制 sparse 压力测试

```text
start_frame: 0
frame_stride: 8
refresh_interval: 16
sparse_ratio: 0.25 -> 0.75
dense fallback disabled
```

结果：

| speedup | latency reduction | mean cosine | min cosine | dense/sparse |
| ---: | ---: | ---: | ---: | ---: |
| 0.518x | -93.1% | 0.5962 | 0.4290 | 2 / 30 |

breakdown：

| branch | frames | mean latency | probe | token selector | selector | sparse compute | observed ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 2 | 8.99ms | 0 | 0 | 0 | 0 | 1.0 |
| sparse | 30 | 17.52ms | 1.00ms | 2.20ms | 2.01ms | 7.42ms | 0.522 |

分析：

真实 OneVision/SigLIP 上，v7 sparse path 明显比 dense 慢。主要原因不是没有冗余，而是当前 PyTorch sparse execution 需要大量 gather/scatter、base clone、mixed output 构造；即使 observed dynamic ratio 只有 0.52，端到端仍然慢。

#### 实验 C：连续帧 + skip/sparse 混合

```text
start_frame: 0
frame_stride: 1
refresh_interval: 16
skip threshold relaxed
```

结果：

| speedup | latency reduction | mean cosine | min cosine | dense/skip/sparse |
| ---: | ---: | ---: | ---: | ---: |
| 0.606x | -65.0% | 0.6997 | 0.5767 | 2 / 7 / 23 |

breakdown：

| branch | frames | mean latency |
| --- | ---: | ---: |
| dense | 2 | 8.14ms |
| skip | 7 | 1.35ms |
| sparse | 23 | 18.99ms |

分析：

这个结果很关键：

1. skip 分支确实快，说明“复用”本身有速度空间；
2. sparse 分支仍然慢，说明 token-level sparse recomputation 不是当前真实 OneVision 上的主方向；
3. 只要大量帧进入 sparse，整体就会变慢；
4. 因此最终方法应尽量把更多帧路由为 skip 或 cheap correction，而不是进入昂贵 sparse path。

### v7 skip-only 上界

为了隔离 skip 的速度上界，我们关闭 sparse，非参考帧直接复用 rolling anchor。

#### N=16，中段 clip

```text
start_frame: 120
frame_stride: 1
refresh_interval: 16
non-reference frames: skip
```

结果：

| speedup | latency reduction | mean cosine | min cosine | dense/skip |
| ---: | ---: | ---: | ---: | ---: |
| 1.889x | 47.1% | 0.6648 | 0.5381 | 2 / 30 |

#### N=4，中段 clip

结果：

| speedup | latency reduction | mean cosine | min cosine | dense/skip |
| ---: | ---: | ---: | ---: | ---: |
| 1.473x | 32.1% | 0.8523 | 0.7354 | 8 / 24 |

分析：

更频繁的 rolling correction 能显著提升保真，但速度下降。这个 trade-off 直接支持后续论文 insight：

```text
Long skip gives speed but accumulates drift;
rolling correction controls drift but costs dense refresh;
the missing component is a cheap correction between skip and dense.
```

### v9 AnchorGate：低成本 skip/dense 路由

v7 skip 分支虽然不做 sparse compute，但仍然要运行 probe 和 token selector，因此 skip 也不够便宜。为此新增 v9 AnchorGate：

```text
Reference frame: dense encode and cache output
Non-reference frame:
  1. only compute patch embedding
  2. compare embedding drift to rolling anchor
  3. if drift <= threshold: skip and reuse anchor output
  4. else: dense correction and update rolling anchor
```

这个版本故意不做 token-level sparse correction，用来隔离 low-cost gate 的价值。

#### v9 trade-off，中段 clip，N=16

| threshold | speedup | latency reduction | mean cosine | min cosine | dense/skip |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.030 | 3.987x | 74.9% | 0.7572 | 0.5991 | 4 / 28 |
| 0.010 | 2.150x | 53.5% | 0.8935 | 0.7476 | 12 / 20 |
| 0.005 | 1.645x | 39.2% | 0.9333 | 0.8472 | 16 / 16 |

分支耗时：

| config | dense mean | skip mean | embed cost |
| --- | ---: | ---: | ---: |
| threshold 0.030 | 9.07ms | 1.23ms | ~0.30ms |
| threshold 0.010 | 8.36ms | 1.54ms | ~0.29ms |
| threshold 0.005 | 8.77ms | 1.87ms | ~0.31ms |

注意：`threshold=0.030` 的第一帧 skip 有一次额外 warm-up 异常耗时，后续 skip 实际多在 `0.36ms-0.42ms`。后续正式实验需要增加 warm-up 或报告 median/p90。

### 当前方向性结论

这轮真实 OneVision/SigLIP 实验给出了比之前更明确的方向：

1. **逐 token sparse recomputation 不适合作为最终主方法核心**  
   在真实 OneVision 上，v7 sparse 帧平均 `17ms-19ms`，dense 只有约 `8.8ms`。非融合 gather/scatter sparse path 已经触摸到工程边界。

2. **skip/reuse 才是速度空间的主要来源**  
   v9 AnchorGate 只用 embedding drift 做路由，已经能达到 `1.65x-3.99x`，明显超过 STC-Cacher 报告的 ViT 侧 `24.5%` latency reduction。

3. **但纯 skip 的特征保真不足**  
   即使中等阈值 `0.005`，mean cosine 约 `0.933`，min cosine 约 `0.847`。如果直接把它作为最终方法，视觉特征漂移风险太大。

4. **下一步应该研究 cheap correction，而不是继续优化 token sparse scatter**  
   最自然的最终方法形态应是：

```text
AnchorGate:
  cheap semantic drift estimation
  -> skip stable frames
  -> dense refresh for unstable frames
  -> lightweight correction for mid-drift frames

Unified usage:
  ViT recomputation gate
  visual token writing gate
  ReKV/LLM cache retrieval gate
```

### 对最终 paper 方法的启示

最终方法不应写成：

```text
我们尝试了 v1/v2/v3/v4/v5/v7/v9，然后堆出一个系统。
```

而应抽象为：

```text
Streaming video contains temporally stable visual semantics,
but full feature recomputation and naive sparse recomputation are both inefficient.
We propose an anchor-conditioned semantic stability mechanism that decides
when to reuse, when to refresh, and when to correct visual memory.
```

中文主线可以理解为：

```text
锚点条件语义稳定性
  -> 低成本判断当前帧是否值得重算
  -> 稳定帧直接复用
  -> 漂移帧触发 rolling correction
  -> 同一信号控制视觉 token 写入和后端 cache 检索
```

这比单纯复刻 STC 更强，因为 STC 主要是“压缩 token 数”，而我们的方向是“先判断流式语义状态，再统一控制前端计算和后端记忆”。

### 下一步实验

下一步不要继续在 v7 sparse path 上投入太多。建议做：

1. **v10 Cheap Correction**  
   在 v9 AnchorGate 的 skip 和 dense 之间加入轻量校正，例如：
   - projected visual token residual correction；
   - layernorm-space affine correction；
   - low-rank residual from anchor embedding drift；
   - pooled token correction before `multi_modal_projector`。

2. **任务级 QA 验证**  
   把 v9/v10 接到 `vit_output_postprocess` 或 `_get_video_features`，用 tiny QA 检查 cosine 下降是否真的破坏答案。

3. **visual token writing gate**  
   对 skip 帧不写入全部视觉 tokens，只写入 anchor reference 或少量 correction tokens，直接联动 ReKV cache。

4. **正式报告指标扩展**  
   后续每次记录：
   - mean / median / p90 latency；
   - dense/skip/correct 分布；
   - mean/min/p10 cosine；
   - QA answer consistency；
   - visual token writing reduction；
   - ReKV cache memory / retrieval latency。

---

## 2026-06-01：目标修正为 QA-first Sparse Semantic Stream

### 修正背景

前面大量实验把逐帧视觉特征的 cosine / MSE 放在了过高的位置。这有一个问题：

```text
很多速度优先方案虽然不能逐帧还原 dense ViT feature，
但可能在 QA 任务上完全可接受，
并且能显著减少视觉 token 写入和 ReKV/LLM cache 压力。
```

因此，从这一轮开始，实验目标正式修正为：

```text
不再以逐帧视觉特征完全还原 dense ViT 为目标；
而是以 QA 性能基本不下降为约束，
最大化流式视觉处理与上下文写入效率。
```

更贴近最初研究动机的表述是：

```text
将密集视觉流转化为稀疏语义流。
```

这里的“稀疏”不只包括 ViT 计算稀疏，也包括：

1. 视觉 token 写入稀疏；
2. ReKV / LLM cache 存储稀疏；
3. 后续 retrieval 搜索空间稀疏；
4. QA 前上下文构造稀疏。

### 如何重新看待之前实验

之前实验不能被否定，而应重新分层解释：

| 之前指标 | 新定位 |
| --- | --- |
| feature cosine / MSE | 诊断视觉漂移，不作为主否决指标 |
| v5/v7/v8 sparse path speed | 证明非融合 token sparse recomputation 的工程瓶颈 |
| v9 AnchorGate | 证明低成本语义路由能释放速度空间 |
| skip-only 漂移曲线 | 说明长期复用需要 correction 或任务级约束 |
| cache policy simulation | 说明视觉 token 写入减少能直接服务后端 ReKV |

因此，之前结论应从：

```text
某方案 feature cosine 不够，所以不可用。
```

修正为：

```text
某方案造成了可观视觉漂移；
下一步需要检查这种漂移是否真的影响 QA。
如果 QA 稳定，则该方案仍可能是高价值速度优先方案。
```

### 新的论文目标

更适合 AAAI / 顶会口味的方法主线应是：

```text
Anchor-conditioned Semantic Stream

Dense visual stream
  -> semantic stability estimation
  -> sparse semantic event stream
  -> selective visual recomputation
  -> selective context/cache writing
  -> task-level QA preservation
```

这比“只加速 ViT”更有研究价值，因为它把视觉编码、上下文写入和缓存管理统一到同一个语义稳定性信号下。

### 新增实现：semantic stream writing gate

本轮新增一个真实 VLM 入口：

```text
model/vision_accelerator/semantic_stream.py
```

核心逻辑：

```text
For each frame:
  1. compute semantic signature from selected vision features
  2. compare to rolling anchor
  3. keep frame if:
       - first/reference frame
       - periodic refresh
       - drift exceeds threshold
  4. skip frame otherwise
  5. only kept frames are written into LLM/ReKV context
```

这个版本暂时不是为了减少 ViT 计算，因为它在 `_get_video_features` 后做 postprocess；它的第一目标是验证：

```text
QA 能否容忍视觉上下文写入稀疏化。
```

### 真实 tiny QA 验证

实验设置：

```text
model: LLaVA-OneVision-Qwen2-0.5B
backend: ReKV streaming QA
video: Big Buck Bunny
sample_fps: 1.0
QA: 3 个 sanity 问题
semantic stream: enabled
vit sparse patch: enabled
```

#### Dense 参考

之前 dense 1fps tiny QA：

```text
encoded frames until last QA: 16
visual tokens written: 3136
answers: 3/3 语义正确
```

#### Semantic stream，N=4，threshold=0.01

结果：

| metric | value |
| --- | ---: |
| input frames | 16 |
| kept frames | 7 |
| skipped frames | 9 |
| input visual tokens | 3136 |
| written visual tokens | 1372 |
| token writing reduction | 56.25% |
| QA sanity | 3/3 语义正确 |

答案：

| question | semantic stream answer |
| --- | --- |
| character | small, cute, friendly bunny |
| animated or real | Animated |
| setting | lush green forest / tree stump / stream |

#### Semantic stream，N=8，threshold=0.03

结果：

| metric | value |
| --- | ---: |
| input frames | 16 |
| kept frames | 6 |
| skipped frames | 10 |
| input visual tokens | 3136 |
| written visual tokens | 1176 |
| token writing reduction | 62.5% |
| QA sanity | 3/3 语义正确 |

答案：

| question | semantic stream answer |
| --- | --- |
| character | small, cute, friendly bunny |
| animated or real | Animated |
| setting | lush green forest / treehouse / stream / meadow |

### 当前解释

这组结果很重要，因为它说明：

1. 即使视觉特征没有逐帧还原 dense，QA 仍可能稳定；
2. 视觉上下文写入可以先获得 `56%-62%` 的 token reduction；
3. 这部分收益会直接转化为更小的 ReKV/LLM cache、较少的后续 retrieval 搜索空间、较低的 prefill/cache 更新压力；
4. 这比单纯讨论 ViT latency 更符合“流式视频推理系统”的研究目标。

需要注意：

```text
当前 semantic stream gate 仍然先编码视觉帧，再决定是否写入。
所以这一轮主要验证 cache/context 稀疏化，而不是 ViT compute 稀疏化。
```

这不是缺陷，而是有意拆分变量：

1. 先证明 QA 允许稀疏语义写入；
2. 再把同一 gate 前移，用于跳过或校正 ViT 计算；
3. 最后把 semantic stream 与 ReKV cache 写入 / retrieval 统一起来。

### 修正后的评价指标

后续主指标改为：

| 层级 | 主指标 |
| --- | --- |
| QA 任务 | answer consistency / benchmark score |
| 视觉流 | kept frame ratio / skipped frame ratio |
| context 写入 | visual token writing reduction |
| cache | KV cache size / retrieved block count / retrieval latency |
| 速度 | end-to-end latency / ViT latency / prefill latency |
| 诊断 | feature cosine / MSE / drift curve |

也就是说，feature cosine 仍然记录，但只用于解释：

```text
为什么某个 QA case 失败；
或为什么某段视频需要更频繁 refresh/correction。
```

### 下一步实验

下一步应直接做 QA-first ablation：

1. 在 tiny QA 上扫：
   - `semantic_refresh_interval = 2, 4, 8, 16`
   - `semantic_skip_threshold = 0.005, 0.01, 0.03`
2. 记录：
   - QA answer 是否一致；
   - written visual tokens；
   - ReKV cache memory；
   - encode / QA latency；
3. 再接 RVS 小子集真实视频；
4. 如果 QA 稳定，再将 semantic gate 前移，减少 ViT 计算；
5. 同时设计 v10 cheap correction，只对 QA 风险较高的 mid-drift frame 写入少量 correction tokens。

这条路线更符合最终论文叙事：

```text
Dense frames are not all semantic events.
We convert streaming video into sparse semantic events,
and only semantic events deserve computation and memory.
```

---

## 2026-06-01：Semantic Stream 的 compute gate 与 writing gate 融合

### 实验目的

上一轮 semantic stream 只在 ViT 输出后做视觉 token 写入过滤，因此主要证明：

```text
QA 可以容忍上下文/cache 写入稀疏化。
```

但它还没有充分验证：

```text
同一个语义稳定性信号能否同时减少 ViT 计算和视觉 token 写入。
```

本轮目标就是把两者融合成最小闭环：

1. 在完整 ViT 之前先做低成本 semantic gate；
2. 稳定帧直接跳过完整 vision tower；
3. 被跳过的帧也不写入 ReKV/LLM context；
4. 只有 refresh / drift frame 才完整编码并写入视觉 token；
5. 仍然用 QA answer consistency 作为主约束。

### 新增实现

新增开关：

```text
--enable_semantic_compute_gate true
--enable_vit_layer_sparse false
```

关键区别：

| mode | ViT compute | token writing |
| --- | --- | --- |
| postprocess writing gate | 所有帧仍完整 ViT 编码 | 只写入 keep frames |
| pre-ViT compute gate | 只有 keep frames 完整 ViT 编码 | 只写入 keep frames |

同时将旧的逐层 sparse patch 与 semantic gate 解耦：

```text
--enable_vit_layer_sparse false
```

原因是旧 layer sparse path 在真实 OneVision 上已经证明不够 GPU 友好，会干扰我们验证 semantic stream 的真实收益。

### 实验设置

```text
model: LLaVA-OneVision-Qwen2-0.5B
backend: ReKV streaming QA
video: Big Buck Bunny
sample_fps: 1.0
QA: 3 个 sanity 问题
semantic_refresh_interval: 8
semantic_skip_threshold: 0.03
enable_semantic_compute_gate: true
enable_vit_layer_sparse: false
```

### 结果

| mode | input frames | kept frames | written tokens | token reduction | cumulative video encode | QA sanity |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense baseline | 16 | 16 | 3136 | 0% | 0.572s | 3/3 |
| writing gate only, N=8 | 16 | 6 | 1176 | 62.5% | 0.728s | 3/3 |
| compute + writing gate, N=8 | 16 | 5 | 980 | 68.75% | 0.467s | 3/3 |

逐问题结果：

| question | answer |
| --- | --- |
| character | small, cute bunny character |
| animated or real | Animated |
| setting | lush green forest / moss-covered tree / stream |

### 解释

这轮验证了“加速计算”和“选择性 token 保留”的协同关系：

```text
同一个 semantic gate 先判断当前帧是否是新的语义事件。

如果不是：
  - 不完整跑 ViT；
  - 不写入 visual tokens；
  - 不增加 ReKV/LLM cache；
  - 不扩大后续 retrieval 搜索空间。

如果是：
  - 完整编码；
  - 写入 visual tokens；
  - 更新 rolling semantic anchor；
  - 作为后续帧的语义参考。
```

因此，semantic stream 的收益不是单点收益，而是级联收益：

1. **计算侧**：跳过稳定帧的完整 vision tower；
2. **上下文侧**：减少进入 LLM 的 visual tokens；
3. **cache 侧**：减少 ReKV KV cache 写入；
4. **retrieval 侧**：减少后续需要检索的视觉 block；
5. **任务侧**：只要 QA answer 不下降，就不要求逐帧 feature 还原。

### 当前限制

1. tiny QA 只有 3 个问题，只能作为 sanity，不是论文主结果；
2. 当前 gate signature 来自 patch embedding，仍需验证在更多视频和问题上的鲁棒性；
3. cumulative video encode 下降幅度还不大，原因包括：
   - tiny 样本太短；
   - 每帧仍需要 video processor 和 embedding gate；
   - Python per-frame 调度开销明显；
4. 但 token writing reduction 已经非常明显，说明 cache/context 侧收益已经成立。

### 修正后的方法雏形

现在的方法雏形可以写成：

```text
Anchor-conditioned Semantic Stream:

For each incoming frame:
  1. estimate semantic drift using a cheap anchor-conditioned signature
  2. if stable:
       skip visual recomputation
       skip visual token writing
  3. if unstable or periodic refresh:
       perform full visual encoding
       write visual tokens into streaming memory
       update semantic anchor
```

后续 v10 要补的是：

```text
stable: skip
mid-drift: cheap correction / small semantic token write
high-drift: dense refresh
```

这能避免纯 skip 的语义漂移，同时避免回到昂贵的 token-level sparse scatter。

### 下一步

下一步实验应从单点验证转为 QA-first sweep：

1. 自动扫：
   - refresh interval: 2 / 4 / 8 / 16
   - skip threshold: 0.005 / 0.01 / 0.03
   - compute gate on/off
2. 输出主表：
   - QA answer consistency；
   - kept frame ratio；
   - visual token writing reduction；
   - cumulative video encode；
   - QA latency；
   - estimated KV cache reduction；
3. 然后迁移到 RVS 小子集。

---

## 2026-06-01：QA-first sweep 与大模型 smoke test

### 实验目的

本轮继续贯彻新的核心目标：

```text
最终目标是高效流式视频 VLM 推理，
不是单独追求 ViT feature 逐帧还原。
```

因此这一轮实验重点转为：

1. 用 tiny QA 做第一张 QA-first semantic stream 主表；
2. 观察不同 refresh interval / drift threshold 下 QA 是否稳定；
3. 统计 visual token writing reduction，作为 cache/context 压缩的核心指标；
4. 用更大的 `LLaVA-OneVision-Qwen2-7B` 做 smoke test，确认方法不是只在 0.5B 小模型上可跑；
5. 为 RVS / VStream 这类真正 streaming benchmark 预留直接入口。

### 新增脚本

```text
scripts/run_semantic_stream_sweep.py
```

功能：

1. 自动调用 `video_qa.rekv_stream_vqa`；
2. 扫 `semantic_refresh_interval` 与 `semantic_skip_threshold`；
3. 汇总：
   - QA sanity pass；
   - kept / skipped frame；
   - visual token writing reduction；
   - cumulative video encode；
   - QA latency；
   - answers。

输出：

```text
results/semantic_stream_sweep/.../summary.csv
results/semantic_stream_sweep/.../summary.json
```

### 0.5B tiny QA sweep

实验设置：

```text
model: LLaVA-OneVision-Qwen2-0.5B
sample_fps: 1.0
enable_semantic_compute_gate: true
enable_vit_layer_sparse: false
refresh_interval: 4 / 8 / 16
skip_threshold: 0.01 / 0.03
```

结果：

| refresh | threshold | QA | kept frames | token reduction | encode time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.01 | 3/3 | 7/16 | 56.2% | 0.583s |
| 4 | 0.03 | 3/3 | 6/16 | 62.5% | 0.528s |
| 8 | 0.01 | 3/3 | 6/16 | 62.5% | 0.558s |
| 8 | 0.03 | 3/3 | 5/16 | 68.8% | 0.516s |
| 16 | 0.01 | 3/3 | 5/16 | 68.8% | 0.570s |
| 16 | 0.03 | 3/3 | 5/16 | 68.8% | 0.689s |

结论：

1. 六个配置 QA sanity 全部通过；
2. visual token writing reduction 稳定在 `56%-69%`；
3. 在这个 tiny clip 上，`refresh=8, threshold=0.03` 是当前较好的点：

```text
QA: 3/3
kept frames: 5/16
token reduction: 68.8%
encode: 0.516s
```

4. `refresh=16` 没有继续减少写入，说明在当前 gate 下 drift keep 已经触发，单纯增大 refresh 不一定继续压缩。

### 7B smoke test

远程模型已确认存在：

```text
model_zoo/llava-onevision-qwen2-7b-ov-hf
```

实验设置：

```text
model: LLaVA-OneVision-Qwen2-7B
refresh_interval: 8
skip_threshold: 0.03
enable_semantic_compute_gate: true
enable_vit_layer_sparse: false
```

结果：

| model | QA | kept frames | token reduction | answers |
| --- | ---: | ---: | ---: | --- |
| 7B semantic stream | 3/3 | 5/16 | 68.8% | rabbit / animated / forest-river-tree |

这说明 semantic stream 的 QA 和 token/cache 稀疏化逻辑可以扩展到更大 OneVision 7B 模型。

需要谨慎解释的是 latency：

```text
7B semantic stream cumulative encode: 1.966s
7B dense cumulative encode: 0.737s
```

这个单点暂时不能作为速度负结论，因为当前 compute gate 还有一个明确工程问题：

```text
gate 阶段先计算 patch embedding；
保留帧进入 _get_video_features 时又重新计算 embedding。
```

也就是说，当前实现已经验证了语义路由与 token/cache 稀疏化，但还没有工程优化到真正复用 gate embedding。下一步应避免重复 embedding，并减少 Python per-frame 调度开销。

### RVS / streaming benchmark 入口

本轮将 semantic stream 参数接入：

```text
video_qa/run_eval.py
```

新增可传参数：

```text
--enable_vit_sparse
--enable_vit_layer_sparse
--vit_cache_interval
--vit_update_token_ratio
--enable_semantic_stream
--enable_semantic_compute_gate
--semantic_refresh_interval
--semantic_skip_threshold
```

这样后续 RVS 数据到位后，可以直接跑：

```bash
python video_qa/run_eval.py \
  --model llava_ov_7b \
  --dataset rvs_ego \
  --sample_fps 1.0 \
  --retrieve_size 4 \
  --enable_vit_sparse true \
  --enable_vit_layer_sparse false \
  --enable_semantic_stream true \
  --enable_semantic_compute_gate true \
  --semantic_refresh_interval 8 \
  --semantic_skip_threshold 0.03 \
  --debug true
```

当前远程服务器尚未发现可直接使用的 RVS/Ego4D/MovieNet 视频目录；已有的是 annotation subset。下一步需要补齐真实 streaming 视频资产，优先：

1. RVS-Ego 小子集；
2. RVS-Movie 小子集；
3. 如果官方视频准备耗时，先构造 5-10 个真实公开视频的 streaming QA sanity set。

### 当前方向判断

这轮结果支持一个更本质的论文目标：

```text
Streaming VLM acceleration should not be framed only as visual encoder acceleration.
It is a semantic stream sparsification problem:
only semantic events should consume visual computation and memory.
```

所以后续主线应同时报告：

1. QA score / answer consistency；
2. visual token writing reduction；
3. KV cache growth reduction；
4. retrieval latency / retrieved blocks；
5. end-to-end streaming latency；
6. ViT compute latency。

feature cosine / MSE 只作为诊断项保留。

### 下一步

1. 优化 compute gate，避免 kept frames 重复计算 embedding；
2. 增加 median / p90 latency 统计，避免单次 warm-up 干扰；
3. 在 7B 上补一组更稳定的 repeated run；
4. 准备 RVS 真实视频小子集；
5. 把 semantic stream 写入策略与 ReKV cache manager 的实际 block 数、memory 数值关联起来。

## 2026-06-01：Embedding Reuse 后的 QA-first 语义流实验

### 实验目的

这一轮实验不是继续追求 dense ViT feature 的逐帧还原，而是验证一个更贴近流式 VLM 的假设：

```text
如果当前帧没有带来新的语义事件，就不应该继续消耗完整 ViT 编码、视觉 token 写入和后端 KV cache 空间。
```

因此本轮指标优先级调整为：

1. QA sanity / answer consistency 不下降；
2. visual token 写入显著减少；
3. ViT encode 时间下降；
4. feature cosine / MSE 仅作为诊断指标，不再作为第一约束。

### 修改内容

上一版 semantic compute gate 的问题是：

```text
所有帧先经过 vision embedding 得到轻量语义签名；
被保留帧随后又调用 _get_video_features，从 pixel values 重新计算一次 embedding。
```

本轮新增 `_get_video_features_from_embeddings`，使 compute gate 的路径变为：

```text
pixel values
  -> vision embeddings
  -> frame-level semantic signature
  -> keep / skip decision
  -> kept embeddings directly enter vision encoder
  -> multimodal projector
  -> write only kept visual tokens into LLM / ReKV cache
```

这一步把语义门控统一成一个更干净的论文表述：**anchor-conditioned semantic router before visual encoding and context writing**。它不是额外的工程补丁，而是把“是否值得进入视觉语义流”这个决策前移。

### 实验设置

```text
dataset: tiny Big Buck Bunny streaming QA sanity set
sample_fps: 1.0
frames observed before final QA: 16
model: LLaVA-OneVision-Qwen2-0.5B / 7B
semantic_refresh_interval: 8
semantic_skip_threshold: 0.03
enable_semantic_compute_gate: true
enable_vit_layer_sparse: false
GPU: A100 80GB
```

### 结果

| model | method | QA | kept frames | token writing reduction | cumulative encode | speedup vs dense | encode reduction |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5B | Dense | 3/3 | 16/16 | 0.0% | 0.566s | 1.00x | 0.0% |
| 0.5B | Semantic compute + write gate | 3/3 | 5/16 | 68.8% | 0.492s | 1.15x | 13.1% |
| 7B | Dense | 3/3 | 16/16 | 0.0% | 0.771s | 1.00x | 0.0% |
| 7B | Semantic compute + write gate | 3/3 | 5/16 | 68.8% | 0.503s | 1.53x | 34.8% |

对应结果文件：

```text
results/semantic_stream_sweep/tiny_0p5b_compute_embed_reuse/summary.csv
results/semantic_stream_sweep/tiny_7b_compute_embed_reuse/summary.csv
results/tiny_streaming_qa/dense_0p5b_1fps_current/1_0.csv
results/tiny_streaming_qa/dense_7b_1fps_current/1_0.csv
```

### 现象分析

1. QA sanity 没有下降。3 个问题都保持语义正确，说明在这个简单流式片段里，跳过 11/16 帧没有破坏回答所需语义。
2. visual token 写入下降 68.8%。这比单纯 ViT 加速更重要，因为它直接减少后端上下文写入、KV cache 增长和后续检索/注意力压力。
3. 7B 的收益明显大于 0.5B。0.5B 上只有 1.15x，而 7B 上达到 1.53x，说明语义流稀疏化的价值会随着后端模型、cache 管理和真实流式长度上升而放大。
4. embedding reuse 解决了上一轮 7B compute gate 变慢的问题。上一轮 7B semantic compute gate 为 1.966s，本轮降到 0.503s，说明之前的负收益主要来自重复 embedding 和调度路径，而不是方法本身不可行。
5. tiny set 仍然只能作为 sanity evidence，不能作为最终论文主结果。下一步必须进入更长、更真实的 streaming QA 数据，统计 end-to-end latency、token/cache growth 和 QA accuracy。

### 对论文方法的启发

最终方法不应表述成“若干工程 trick 的组合”，而应收敛为一个统一原则：

```text
Dense visual streams should be converted into sparse semantic streams.
Only anchor-changing or query-relevant semantic events are encoded, written, and retained.
```

在这个框架下：

1. compute gate 负责减少进入 ViT encoder 的帧；
2. write gate 负责减少写入 LLM/ReKV cache 的视觉 token；
3. anchor refresh 负责控制长期漂移；
4. 后续的 ReKV / streaming cache 管理可以作为同一个 semantic stream 的后端延伸，而不是独立模块堆叠。

### 下一步实验

1. 增加 repeated run / median / p90，避免单次 GPU warm-up 和调度噪声影响结论。
2. 增加真实 streaming QA 小子集，优先 RVS-Ego / RVS-Movie；如果视频资产暂时不可用，则先建立公开视频 sanity set。
3. 记录实际 ReKV cache block 数、写入 token 数和检索耗时，把“语义流稀疏化”从 ViT 侧扩展到后端 cache 侧。
4. 在更长帧序列上扫描更激进配置，例如 `refresh=16/32`、更高 skip threshold，并用 QA 约束决定可接受边界。

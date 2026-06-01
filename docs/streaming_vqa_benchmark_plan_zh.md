# 流式 VQA Benchmark 接入计划

## 当前判断

可以开始引入流式 VQA benchmark，但不要一开始全量下载。

当前最适合接入的是 VStream-QA 的实时子集：

```text
RVS-Ego
RVS-Movie
```

原因：

- 现有仓库已经有 `video_qa/rekv_stream_vqa.py`；
- 现有 `video_qa/run_eval.py` 已经支持 `rvs_ego` 和 `rvs_movie`；
- RVS 是在线 streaming QA 设定，和我们现在的目标一致；
- 标注文件很小，可以先下载；
- 视频/帧文件很大，应只在远程服务器按需准备。

## 数据来源

VStream-QA HuggingFace:

```text
https://huggingface.co/datasets/IVGSZ/VStream-QA
```

已使用的实时标注文件：

```text
vstream-realtime/test_qa_ego4d.json
vstream-realtime/rvs_ego.json
vstream-realtime/test_qa_movienet.json
vstream-realtime/rvs_movie.json
```

数据说明：

- RVS-Ego: 99 videos, 1465 QA
- RVS-Movie: 1000 videos, 1905 QA
- Ego4D / MovieNet 原始视频或关键帧需要按官方数据源准备；
- HuggingFace 上也有 realtime frame archives，但 RVS-Ego 接近 30GB，不适合本地直接拉全量。

## 已实现脚本

```text
scripts/prepare_vstream_rvs_subset.py
```

功能：

- 下载 VStream-QA realtime 标注；
- 转换成当前 `rekv_stream_vqa.py` 可读取的格式；
- 按 video 聚合 conversations；
- 保留 `start_time/end_time`；
- 生成小规模 sanity subset；
- 输出缺失视频文件检查结果。

使用方式：

```bash
python scripts/prepare_vstream_rvs_subset.py \
  --dataset all \
  --max-videos 8 \
  --max-questions-per-video 3
```

生成文件：

```text
data/rvs/ego/ego4d_oe.json
data/rvs/ego/ego4d_oe.summary.json
data/rvs/movie/movienet_oe.json
data/rvs/movie/movienet_oe.summary.json
data/vstream_qa/raw/rvs_ego/test_qa_ego4d.json
data/vstream_qa/raw/rvs_ego/rvs_ego.json
data/vstream_qa/raw/rvs_movie/test_qa_movienet.json
data/vstream_qa/raw/rvs_movie/rvs_movie.json
```

当前本地小子集：

```text
RVS-Ego:   8 videos, 24 questions
RVS-Movie: 8 videos, 24 questions
```

视频文件当前缺失，这是正常状态。

## 为什么不直接下载完整 benchmark

不建议现在直接全量下载，原因：

1. RVS-Ego realtime frame archive 约 28GB+；
2. MovieNet/Ego4D 原始数据有独立下载授权与组织方式；
3. 我们现在需要的是速度/质量 sanity check，不需要一开始跑全量 leaderboard；
4. 本地和 git 同步应保持轻量，benchmark 视频应只放远程服务器数据目录，不进入仓库。

## 推荐下一步

### Step 1：本地/远程标注准备

先只准备 annotation subset，确认格式链路。

### Step 2：远程准备少量真实视频

优先准备 RVS-Ego 的 3-5 个视频 clip：

```text
/home/mllm/datasets/vstream/rvs_ego/videos/{video_id}.mp4
```

然后将 `data/rvs/ego/ego4d_oe.json` 中的 `video_path` 指向远程视频路径，或在远程建立软链接。

### Step 3：小规模 QA sanity check

先跑：

```bash
python -m video_qa.run_eval \
  --model llava_ov_0.5b \
  --dataset rvs_ego \
  --sample_fps 0.01 \
  --retrieve_size 16 \
  --debug true
```

目的不是追求最终分数，而是验证：

- dense baseline QA 能跑通；
- speed-first Turbo-ViT 改变视觉特征后是否明显破坏答案；
- REKV cache policy 能否记录端到端 latency 和 token/cache 写入变化。

### Step 4：再接 StreamingBench / OVO-Bench

StreamingBench 和 OVO-Bench 更适合作为后续大规模论文评测：

- StreamingBench: 更接近通用 streaming MLLM benchmark；
- OVO-Bench: STC 论文主表使用，适合最终对比；
- 当前阶段不作为第一入口，避免被数据准备和评测协议拖慢。

## 当前研究策略

我们不必完全沿 STC 的赛道比较。

更合理的路径是：

```text
先在 RVS 小子集验证 speed-first ViT reuse 的 QA 保真，
再把 semantic stability 接入 REKV-style cache policy，
最后扩展到 OVO / StreamingBench 大规模验证。
```

这能支持一个更优雅的论文主线：

```text
Dual-anchor semantic stability is a unified signal
for visual recomputation, visual token writing, and LLM cache retrieval.
```

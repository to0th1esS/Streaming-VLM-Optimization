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

远程服务器状态：

```text
远程直接访问 HuggingFace 失败：
  urllib.error.URLError: [Errno 99] Cannot assign requested address

已采用 fallback：
  本地下载标注 -> scp 到远程仓库 data/ 目录
```

因此后续远程数据准备不应假设 HuggingFace 可直接访问。小标注可以从本地同步；大视频数据建议通过服务器可用的数据源、已有数据盘或手动上传/挂载。

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

---

## 2026-06-01 状态更新：真实 VLM tiny QA 已跑通

在接入 RVS 真实视频前，已先用 `Big Buck Bunny` 构造了 3 个 tiny QA sanity 问题，并在远程 A100 上跑通：

```text
model: LLaVA-OneVision-Qwen2-0.5B
backend: ReKV streaming QA
video: data/turbovit_v1/big_buck_bunny.mp4
annotation: data/tiny_streaming_qa/big_buck_bunny_qa.json
```

已验证两条链路：

| mode | status | answer sanity | note |
| --- | --- | --- | --- |
| dense ViT | pass | 3/3 语义正确 | 真实 LLaVA-OneVision + ReKV 链路可用 |
| current `vit_patch_hf` sparse | pass | 3/3 语义基本正确 | sparse 入口可被真实 VLM 调用 |

但当前主代码 `vit_patch_hf` 仍是入口版，不是最终论文候选实现。计时显示在 tiny QA 上它没有加速：

| setting | dense video encode | sparse video encode | conclusion |
| --- | ---: | ---: | --- |
| 0.25 fps / 4 encoded frames | 0.283s | 0.368s | 输入太短，固定开销占主导 |
| 1 fps / 16 encoded frames | 0.572s | 1.126s | 朴素 token scatter sparse path 不够 GPU 友好 |

因此后续 RVS 小子集不应直接使用主代码入口版作为最终方法，而应接入 `experiments/turbovit_v1` 中更接近论文主线的 dual-anchor / segment-aware 逻辑。

下一步数据侧工作：

1. 保持 annotation subset 由本地下载后同步到远程；
2. 在远程 `/home/mllm/datasets` 或同级数据盘准备 3-5 个真实 RVS clip；
3. 优先跑 dense VLM QA，确认 benchmark 样本和 prompt 格式正确；
4. 再接入 dual-anchor Turbo-ViT + visual token writing policy，比较答案、latency 和 token/cache 写入比例。

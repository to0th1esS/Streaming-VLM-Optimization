# 数据集资产准备记录

本文档记录 streaming VQA / VLM 实验所需数据集资产的下载状态、目录约定和后续操作。

## 目录约定

模型固定放在：

```text
/home/mllm/models
```

数据集固定放在：

```text
/home/mllm/datasets
```

当前 VStream-QA / RVS 资产目录：

```text
/home/mllm/datasets/vstream_qa
```

仓库内只保存小标注、脚本和软链接入口，不保存大视频/大帧包。

## 当前优先级

第一优先级是 VStream-QA realtime subsets：

```text
RVS-Ego
RVS-Movie
```

原因：

1. 与当前 `video_qa/rekv_stream_vqa.py` 的流式逐问题设置最接近；
2. 已经有 `data/rvs/ego/ego4d_oe.json` 和 `data/rvs/movie/movienet_oe.json` 小子集；
3. 可以直接用于比较 dense vs semantic stream 的 QA、latency、token writing reduction；
4. 体量约 30GB+，比一次性拉 StreamingBench / OVO-Bench 更适合作为第一批真实 benchmark。

## 已确认的数据源

VStream-QA:

```text
https://huggingface.co/datasets/IVGSZ/VStream-QA
```

远程服务器无法稳定直连 HuggingFace，但可以访问：

```text
https://hf-mirror.com
```

已成功通过 `hf-mirror` 列出并下载 metadata。

## VStream-QA 下载状态

已完成：

```text
/home/mllm/datasets/vstream_qa/vstream-realtime/rvs_ego.json
/home/mllm/datasets/vstream_qa/vstream-realtime/test_qa_ego4d.json
/home/mllm/datasets/vstream_qa/vstream-realtime/rvs_movie.json
/home/mllm/datasets/vstream_qa/vstream-realtime/test_qa_movienet.json
```

已启动后台下载：

```text
vstream-realtime/movienet_frames_online.zip
vstream-realtime/ego4d_frames_online.partaa
vstream-realtime/ego4d_frames_online.partab
vstream-realtime/ego4d_frames_online.partac
```

后台下载日志：

```text
/home/mllm/datasets/vstream_qa/download_vstream_assets.log
```

后台 PID 文件：

```text
/home/mllm/datasets/vstream_qa/download_vstream_assets.pid
```

启动命令：

```bash
cd /home/yangjin/1#Streaming-VLM-Optimization
nohup env HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python scripts/download_vstream_assets.py \
  --target-root /home/mllm/datasets/vstream_qa \
  --datasets rvs_movie,rvs_ego \
  --extract \
  --manifest /home/mllm/datasets/vstream_qa/download_manifest.json \
  > /home/mllm/datasets/vstream_qa/download_vstream_assets.log 2>&1 &
```

检查命令：

```bash
ps -p $(cat /home/mllm/datasets/vstream_qa/download_vstream_assets.pid) -o pid,etime,pcpu,pmem,cmd
du -sh /home/mllm/datasets/vstream_qa
tail -40 /home/mllm/datasets/vstream_qa/download_vstream_assets.log
```

## 已发现的结构问题

VStream-RVS 资产不是 mp4，而是图片帧目录。

MovieNet zip 内部示例：

```text
movienet_frames/tt0067116_0008_0008/shot_0387_img_0.jpg
movienet_frames/tt0067116_0008_0008/shot_0387_img_1.jpg
...
```

而 annotation 中的 video id 可能是：

```text
tt0067116_0008_0012
```

因此一个 QA clip 可能对应多个 frame 子目录。已新增：

```text
scripts/link_vstream_assets.py
```

用于把多个 frame 子目录软链接到仓库期望的 `data/rvs/.../videos/{video_id}.mp4` 目录入口。

同时 `video_qa/rekv_stream_vqa.py` 已支持递归读取 frame directory。

## 下载完成后的链接与检查

下载和解包完成后，在远程执行：

```bash
cd /home/yangjin/1#Streaming-VLM-Optimization

/root/miniconda3/bin/python scripts/link_vstream_assets.py \
  --asset-root /home/mllm/datasets/vstream_qa \
  --repo-root . \
  --datasets rvs_ego,rvs_movie \
  --output-json results/streaming_asset_check/link_summary.json

/root/miniconda3/bin/python scripts/check_streaming_video_assets.py \
  --output-json results/streaming_asset_check/summary_after_link.json
```

如果 `available_videos` 变为非零，即可开始 RVS 小子集 QA 验证。

## 后续可下载的大 benchmark

StreamingBench:

```text
dataset repo: mjuicem/StreamingBench
scale: 900 videos / 4500 QA
```

OVO-Bench:

```text
dataset repo: JoeLeelyf/OVO-Bench
scale: 644 videos / 3100 queries
```

这两个数据集适合最终论文主评测，但当前不建议与 VStream 同时下载，原因是：

1. 体量约 200GB 级别；
2. 当前代码还没有完整 evaluator 接入；
3. 先用 VStream-RVS 跑通真实流式 benchmark 更高效。

建议顺序：

1. VStream-RVS 下载、链接、跑通；
2. 接入 RVS dense vs semantic 主表；
3. 再下载 StreamingBench；
4. 最后下载 OVO-Bench 做 STC 相关对比。

## 2026-06-02 状态更新

### 下载进度

后台下载进程仍在运行：

```text
PID: 13450
target: /home/mllm/datasets/vstream_qa
current size: about 14GB
```

已完成：

```text
movienet_frames_online.zip: 2.36GB
metadata files: done
```

仍在下载：

```text
ego4d_frames_online.partaa
ego4d_frames_online.partab
ego4d_frames_online.partac
```

### RVS-Movie 解包与链接

MovieNet frame archive 已解包到：

```text
/home/mllm/datasets/vstream_qa/frames/rvs_movie
```

并通过软链接接入仓库：

```text
data/rvs/movie/videos/*.mp4
```

资产检查结果：

| dataset | videos | questions | available videos |
| --- | ---: | ---: | ---: |
| RVS-Movie subset | 8 | 24 | 8/8 |
| RVS-Ego subset | 8 | 24 | 0/8 |

因此当前已经可以跑 RVS-Movie 小子集；RVS-Ego 等 Ego4D 分片下载和解包完成后再接入。

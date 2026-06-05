import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 支持从仓库根目录直接执行 `python scripts/...py`。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.vit_patch import _raw_rgb_signatures


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def final_rows_by_video(path):
    # 一个视频可能包含多条中间记录；与正式评测一致，只使用最后一条累计记录。
    result = {}
    for row in read_csv(path):
        result[row["video_id"]] = row
    return result


def changed_video_outcomes(periodic_path, novelty_path):
    periodic = final_rows_by_video(periodic_path)
    novelty = final_rows_by_video(novelty_path)
    outcomes = {}
    for video_id, baseline in periodic.items():
        method = novelty.get(video_id)
        if method is None:
            continue
        baseline_score = int(baseline["ovo_official_score"])
        method_score = int(method["ovo_official_score"])
        if baseline_score == method_score:
            continue
        outcomes[video_id] = {
            "outcome": "win" if method_score > baseline_score else "loss",
            "task": baseline["benchmark_task"],
            "answer": baseline["answer"],
            "periodic_prediction": baseline["pred_answer"],
            "novelty_prediction": method["pred_answer"],
        }
    return outcomes


def evenly_sample_video(video_path, sample_fps):
    # 解码依赖仅在真实视频分析时加载，便于本地环境测试纯指标逻辑。
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    source_fps = float(reader.get_avg_fps())
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = max(float(sample_fps), 1.0)
    step = max(1, round(source_fps / sample_fps))
    source_indices = list(range(0, len(reader), step))
    return reader, source_indices


def window_proposals(
    frames,
    window_size,
    grid_size,
    z_threshold,
    signature_mode="grid_sample_stable",
):
    signatures = _raw_rgb_signatures(
        torch.from_numpy(frames),
        grid_size=grid_size,
        mode=signature_mode,
    )
    deltas = torch.zeros(signatures.shape[0], dtype=torch.float32)
    if signatures.shape[0] > 1:
        similarities = F.cosine_similarity(signatures[1:], signatures[:-1], dim=-1)
        deltas[1:] = (1.0 - similarities).clamp_min(0)

    proposals = []
    for window_start in range(0, len(frames), window_size):
        window_delta = deltas[window_start : window_start + window_size]
        if window_delta.numel() == 0:
            continue
        local_novelty = int(torch.argmax(window_delta).item())
        novelty_index = window_start + local_novelty
        mean = float(window_delta.mean().item())
        std = float(window_delta.std(unbiased=False).item())
        peak = float(window_delta[local_novelty].item())
        z_score = (peak - mean) / (std + 1e-6)
        if z_score < z_threshold:
            continue
        proposals.append(
            {
                "window_start": window_start,
                "window_length": int(window_delta.numel()),
                "periodic_index": window_start,
                "novelty_index": novelty_index,
                "raw_peak": peak,
                "raw_mean": mean,
                "raw_std": std,
                "raw_z_score": z_score,
            }
        )
    return proposals


def image_statistics(frame):
    pixels = frame.astype(np.float32) / 255.0
    luminance = pixels.mean(axis=-1)
    return {
        "mean_luminance": float(luminance.mean()),
        "dark_pixel_ratio": float((luminance < 0.08).mean()),
        "channel_std": float(pixels.std(axis=-1).mean()),
    }


def feature_statistics(features):
    # 全局签名描述帧级语义；空间离散度描述图像块之间的结构丰富度。
    global_signature = F.normalize(features.mean(dim=1).float(), dim=-1)
    centered = features.float() - features.float().mean(dim=1, keepdim=True)
    spatial_dispersion = centered.norm(dim=-1).mean(dim=-1)
    return global_signature, spatial_dispersion


def paired_token_change_statistics(left_features, right_features):
    token_cosines = F.cosine_similarity(
        left_features.float(),
        right_features.float(),
        dim=-1,
    )
    sorted_cosines = torch.sort(token_cosines).values
    bottom_count = max(1, math.ceil(sorted_cosines.numel() * 0.1))
    # 低分位和变化比例用于区分局部 token 创新与全局场景切换。
    return {
        "token_cosine_mean": float(token_cosines.mean().item()),
        "token_cosine_min": float(token_cosines.min().item()),
        "token_cosine_bottom10_mean": float(
            sorted_cosines[:bottom_count].mean().item()
        ),
        "token_change_fraction_0p90": float(
            (token_cosines < 0.90).float().mean().item()
        ),
        "token_change_fraction_0p95": float(
            (token_cosines < 0.95).float().mean().item()
        ),
        "token_change_fraction_0p99": float(
            (token_cosines < 0.99).float().mean().item()
        ),
    }


@torch.inference_mode()
def extract_layer_features(model, processor, frames, layers):
    pixel_values = processor.video_processor(
        frames,
        return_tensors="pt",
    ).pixel_values_videos.to(model.device, model.dtype)
    _, frame_count, channels, height, width = pixel_values.shape
    flat_pixels = pixel_values.view(frame_count, channels, height, width)
    hidden_states = model.vision_tower.vision_model.embeddings(flat_pixels)

    requested = set(layers)
    outputs = {0: hidden_states}
    max_layer = max(requested)
    encoder_layers = model.vision_tower.vision_model.encoder.layers
    for layer_index in range(max_layer):
        hidden_states = encoder_layers[layer_index](
            hidden_states,
            attention_mask=None,
            output_attentions=False,
        )[0]
        depth = layer_index + 1
        if depth in requested:
            outputs[depth] = hidden_states
    return outputs


def cosine_value(signatures, left, right):
    return float(F.cosine_similarity(signatures[left : left + 1], signatures[right : right + 1]).item())


def analyze_video(
    model,
    processor,
    sample,
    outcome,
    sample_fps,
    window_size,
    grid_size,
    z_threshold,
    signature_mode,
    layers,
):
    reader, source_indices = evenly_sample_video(sample["video_path"], sample_fps)
    frames = reader.get_batch(source_indices).asnumpy()
    proposals = window_proposals(
        frames,
        window_size,
        grid_size,
        z_threshold,
        signature_mode=signature_mode,
    )
    rows = []

    for proposal in proposals:
        periodic_index = proposal["periodic_index"]
        novelty_index = proposal["novelty_index"]
        # 邻帧用于衡量事件是否持续；所有索引限制在当前已观测视频范围内。
        selected_indices = sorted(
            {
                periodic_index,
                max(0, novelty_index - 1),
                novelty_index,
                min(len(frames) - 1, novelty_index + 1),
            }
        )
        index_to_position = {
            frame_index: position
            for position, frame_index in enumerate(selected_indices)
        }
        selected_frames = frames[selected_indices]
        layer_outputs = extract_layer_features(
            model,
            processor,
            selected_frames,
            layers,
        )

        row = {
            "video_id": sample["video_id"],
            "task": outcome["task"],
            "outcome": outcome["outcome"],
            **proposal,
            "novelty_offset": novelty_index - periodic_index,
            **{
                f"periodic_{key}": value
                for key, value in image_statistics(frames[periodic_index]).items()
            },
            **{
                f"novelty_{key}": value
                for key, value in image_statistics(frames[novelty_index]).items()
            },
        }

        periodic_position = index_to_position[periodic_index]
        novelty_position = index_to_position[novelty_index]
        previous_position = index_to_position[max(0, novelty_index - 1)]
        next_position = index_to_position[min(len(frames) - 1, novelty_index + 1)]
        for depth in layers:
            depth_features = layer_outputs[depth]
            signatures, dispersion = feature_statistics(depth_features)
            previous_similarity = cosine_value(
                signatures,
                novelty_position,
                previous_position,
            )
            next_similarity = cosine_value(
                signatures,
                novelty_position,
                next_position,
            )
            token_change = paired_token_change_statistics(
                depth_features[periodic_position],
                depth_features[novelty_position],
            )
            row.update(
                {
                    f"layer{depth}_periodic_novelty_cosine": cosine_value(
                        signatures,
                        periodic_position,
                        novelty_position,
                    ),
                    f"layer{depth}_novelty_neighbor_cosine": (
                        previous_similarity + next_similarity
                    )
                    / 2.0,
                    f"layer{depth}_novelty_neighbor_min_cosine": min(
                        previous_similarity,
                        next_similarity,
                    ),
                    f"layer{depth}_periodic_spatial_dispersion": float(
                        dispersion[periodic_position].item()
                    ),
                    f"layer{depth}_novelty_spatial_dispersion": float(
                        dispersion[novelty_position].item()
                    ),
                    **{
                        f"layer{depth}_periodic_novelty_{key}": value
                        for key, value in token_change.items()
                    },
                }
            )
        rows.append(row)
    return rows


def summarize(rows, layers):
    summary = {
        "windows": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "by_outcome": {},
    }
    for outcome in ("win", "loss"):
        selected = [row for row in rows if row["outcome"] == outcome]
        if not selected:
            continue
        metrics = {
            "raw_z_score",
            "novelty_dark_pixel_ratio",
            "novelty_channel_std",
        }
        for depth in layers:
            metrics.update(
                {
                    f"layer{depth}_periodic_novelty_cosine",
                    f"layer{depth}_novelty_neighbor_cosine",
                    f"layer{depth}_novelty_neighbor_min_cosine",
                    f"layer{depth}_novelty_spatial_dispersion",
                }
            )
        summary["by_outcome"][outcome] = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in sorted(metrics)
        }
    return summary


def parse_layers(value):
    layers = sorted({int(item) for item in value.split(",") if item.strip()})
    if not layers or layers[0] < 0:
        raise argparse.ArgumentTypeError("layers must contain non-negative integers")
    return layers


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析浅层 ViT 特征能否区分有效事件候选与错误新颖性候选。"
    )
    parser.add_argument("--subset-json", required=True)
    parser.add_argument("--periodic-evaluated", required=True)
    parser.add_argument("--novelty-evaluated", required=True)
    parser.add_argument("--model", default="llava_ov_7b")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument(
        "--signature-mode",
        choices=("grid_sample", "grid_sample_stable"),
        default="grid_sample_stable",
    )
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("0,1,3,6"))
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # 大模型依赖延迟加载，避免 --help 和本地纯函数测试强制安装完整 VLM 环境。
    from video_qa.base import MODELS

    outcomes = changed_video_outcomes(
        args.periodic_evaluated,
        args.novelty_evaluated,
    )
    samples = {
        row["video_id"]: row
        for row in json.loads(Path(args.subset_json).read_text(encoding="utf-8"))
        if row["video_id"] in outcomes
    }

    model_config = MODELS[args.model]
    model, processor = model_config["load_func"](
        model_path=model_config["model_path"],
        n_local=15000,
        topk=64,
        chunk_size=1,
        enable_vit_sparse=False,
    )
    model.eval()

    rows = []
    for video_id in sorted(samples):
        rows.extend(
            analyze_video(
                model,
                processor,
                samples[video_id],
                outcomes[video_id],
                sample_fps=args.sample_fps,
                window_size=args.window_size,
                grid_size=args.grid_size,
                z_threshold=args.z_threshold,
                signature_mode=args.signature_mode,
                layers=args.layers,
            )
        )

    result = summarize(rows, args.layers)
    result.update(
        {
            "changed_videos": len(samples),
            "layers": args.layers,
            "z_threshold": args.z_threshold,
            "signature_mode": args.signature_mode,
            "periodic_evaluated": args.periodic_evaluated,
            "novelty_evaluated": args.novelty_evaluated,
        }
    )
    write_csv(args.output_csv, rows)
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path

import torch
from decord import VideoReader, cpu

from model.llava_onevision_rekv import load_model
from model.vision_accelerator import SemanticStreamGate
from model.vision_accelerator import StructuredGridTokenReducer
from model.vit_patch import _raw_rgb_signatures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile dense, temporal-sparse, and structured visual encoding."
    )
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output-token-budget", type=int, default=121)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--model-path",
        default="model_zoo/llava-onevision-qwen2-7b-ov-hf",
    )
    return parser.parse_args()


def load_video(video_path, sample_fps):
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    if not math.isfinite(fps) or fps <= 0:
        fps = max(sample_fps, 1.0)
    step = max(1, round(fps / sample_fps))
    indices = list(range(0, len(reader), step))
    return reader.get_batch(indices).asnumpy()


def select_semantic_frames(video):
    gate = SemanticStreamGate(
        refresh_interval=96,
        recency_keep_frames=4,
        selection_policy="budget_topk",
        budget_window_size=96,
        budget_keep_per_window=1,
    )
    gate.set_recency_window(0, len(video))
    signatures = _raw_rgb_signatures(
        torch.from_numpy(video),
        grid_size=4,
        mode="grid_sample_stable",
    )
    return gate.select_indices_from_signatures(
        signatures,
        token_count=196,
    ).cpu()


@torch.inference_mode()
def encode_frames(
    model,
    processor,
    video,
    frame_indices,
    chunk_size,
    structured_reducer=None,
):
    total_ms = 0.0
    output_tokens = 0
    for start_idx in range(0, len(frame_indices), chunk_size):
        indices = frame_indices[start_idx : start_idx + chunk_size]
        frames = video[indices.tolist()]
        pixel_values = processor.video_processor(
            frames,
            return_tensors="pt",
        ).pixel_values_videos.to(model.device, model.dtype)
        flat_pixels = pixel_values.reshape(
            -1,
            pixel_values.shape[-3],
            pixel_values.shape[-2],
            pixel_values.shape[-1],
        )

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        vision_output = model.vision_tower(
            flat_pixels,
            output_hidden_states=True,
        )
        selected = vision_output.hidden_states[
            model.config.vision_feature_layer
        ]
        if model.config.vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        if structured_reducer is None:
            projected = model.multi_modal_projector(selected)
            encoded = model.apply_pooling(projected)
        else:
            reduced = structured_reducer(
                selected,
                batch_size=1,
                frames=len(indices),
            )
            encoded = model.multi_modal_projector(reduced)
        end.record()
        end.synchronize()
        total_ms += float(start.elapsed_time(end))
        output_tokens += int(encoded.shape[0] * encoded.shape[1])
    return total_ms, output_tokens


@torch.inference_mode()
def main():
    args = parse_args()
    model, processor = load_model(
        model_path=args.model_path,
        n_local=15000,
        topk=64,
        chunk_size=1,
        enable_vit_sparse=True,
        vit_sparse_config={
            "enable_vit_layer_sparse": False,
            "enable_semantic_stream": False,
        },
    )
    annotations = json.loads(Path(args.anno_path).read_text(encoding="utf-8"))
    annotations = [
        item
        for item in annotations
        if not item["video_id"].startswith("warmup-")
    ]
    structured_reducer = StructuredGridTokenReducer(
        args.output_token_budget
    )

    # 预热视觉编码和两条输出路径，排除首次 kernel（内核）编译。
    warmup_video = load_video(
        annotations[0]["video_path"],
        args.sample_fps,
    )[:2]
    warmup_indices = torch.arange(len(warmup_video))
    encode_frames(
        model,
        processor,
        warmup_video,
        warmup_indices,
        args.chunk_size,
    )
    encode_frames(
        model,
        processor,
        warmup_video,
        warmup_indices,
        args.chunk_size,
        structured_reducer,
    )

    results = []
    totals = {
        "input_frames": 0,
        "semantic_frames": 0,
        "dense_visual_ms": 0.0,
        "temporal_sparse_visual_ms": 0.0,
        "structured_visual_ms": 0.0,
        "dense_output_tokens": 0,
        "temporal_sparse_output_tokens": 0,
        "structured_output_tokens": 0,
    }
    for video_idx, item in enumerate(annotations):
        video = load_video(item["video_path"], args.sample_fps)
        dense_indices = torch.arange(len(video))
        semantic_indices = select_semantic_frames(video)
        structured_reducer.reset()

        # 交替执行顺序，降低持续升频或降频带来的系统偏差。
        if video_idx % 2:
            structured_ms, structured_tokens = encode_frames(
                model,
                processor,
                video,
                semantic_indices,
                args.chunk_size,
                structured_reducer,
            )
            dense_ms, dense_tokens = encode_frames(
                model,
                processor,
                video,
                dense_indices,
                args.chunk_size,
            )
        else:
            dense_ms, dense_tokens = encode_frames(
                model,
                processor,
                video,
                dense_indices,
                args.chunk_size,
            )
            structured_ms, structured_tokens = encode_frames(
                model,
                processor,
                video,
                semantic_indices,
                args.chunk_size,
                structured_reducer,
            )
        sparse_ms, sparse_tokens = encode_frames(
            model,
            processor,
            video,
            semantic_indices,
            args.chunk_size,
        )

        row = {
            "video_id": item["video_id"],
            "input_frames": len(video),
            "semantic_frames": int(semantic_indices.numel()),
            "dense_visual_ms": dense_ms,
            "temporal_sparse_visual_ms": sparse_ms,
            "structured_visual_ms": structured_ms,
            "dense_output_tokens": dense_tokens,
            "temporal_sparse_output_tokens": sparse_tokens,
            "structured_output_tokens": structured_tokens,
        }
        results.append(row)
        for key in totals:
            totals[key] += row[key]

    totals["temporal_sparse_speedup"] = (
        totals["dense_visual_ms"]
        / totals["temporal_sparse_visual_ms"]
    )
    totals["structured_speedup"] = (
        totals["dense_visual_ms"]
        / totals["structured_visual_ms"]
    )
    totals["frame_reduction"] = (
        1.0 - totals["semantic_frames"] / totals["input_frames"]
    )
    totals["structured_token_reduction"] = (
        1.0
        - totals["structured_output_tokens"]
        / totals["dense_output_tokens"]
    )
    output = {
        "config": vars(args),
        "per_video": results,
        "totals": totals,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

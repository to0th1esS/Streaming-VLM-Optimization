import argparse
import json
import statistics

import torch
from decord import VideoReader, cpu

from model.llava_onevision_rekv import load_model
from model.vision_accelerator import StructuredGridTokenReducer


def cuda_time_ms(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile dense and structured post-ViT visual projection."
    )
    parser.add_argument(
        "--model-path",
        default="model_zoo/llava-onevision-qwen2-7b-ov-hf",
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-token-budget", type=int, default=121)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


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

    reader = VideoReader(args.video_path, ctx=cpu(0), num_threads=1)
    frame_indices = torch.linspace(
        0,
        len(reader) - 1,
        steps=args.frames,
    ).round().long().tolist()
    frames = reader.get_batch(frame_indices).asnumpy()
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
    embeddings = model.vision_tower.vision_model.embeddings(flat_pixels)
    encoder_outputs = model.vision_tower.vision_model.encoder(
        inputs_embeds=embeddings,
        output_hidden_states=True,
    )
    selected = encoder_outputs.hidden_states[model.config.vision_feature_layer]
    if model.config.vision_feature_select_strategy == "default":
        selected = selected[:, 1:]

    reducer = StructuredGridTokenReducer(args.output_token_budget)

    def dense_tail():
        projected = model.multi_modal_projector(selected)
        return model.apply_pooling(projected)

    def structured_tail():
        reduced = reducer(
            selected,
            batch_size=1,
            frames=args.frames,
        )
        return model.multi_modal_projector(reduced)

    round_results = []
    for round_idx in range(args.rounds):
        order = (
            ("dense", dense_tail),
            ("structured", structured_tail),
        )
        if round_idx % 2:
            order = tuple(reversed(order))
        result = {}
        for name, function in order:
            result[name] = summarize(
                cuda_time_ms(function, args.warmup, args.repeats)
            )
        round_results.append(result)

    dense_medians = [
        result["dense"]["median_ms"] for result in round_results
    ]
    structured_medians = [
        result["structured"]["median_ms"] for result in round_results
    ]
    dense_median = statistics.median(dense_medians)
    structured_median = statistics.median(structured_medians)
    output = {
        "video_path": args.video_path,
        "frames": args.frames,
        "input_tokens_per_frame": int(selected.shape[1]),
        "dense_output_tokens_per_frame": 196,
        "structured_output_tokens_per_frame": args.output_token_budget,
        "rounds": round_results,
        "dense_tail_median_ms": dense_median,
        "structured_tail_median_ms": structured_median,
        "tail_speedup": dense_median / structured_median,
        "tail_latency_reduction": 1.0 - structured_median / dense_median,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import statistics

import torch
from decord import VideoReader, cpu

from model.llava_onevision_rekv import load_model
from model.vision_accelerator import (
    StructuredGridTokenSampler,
    StructuredGridTokenReducer,
    StructuredResidualTokenReducer,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile structured visual bandwidth components in one process."
    )
    parser.add_argument(
        "--model-path",
        default="model_zoo/llava-onevision-qwen2-7b-ov-hf",
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[121, 144])
    parser.add_argument("--residual-output-budget", type=int, default=121)
    parser.add_argument("--residual-base-budget", type=int, default=100)
    parser.add_argument("--post-output-budget", type=int, default=121)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def cuda_samples(function, warmup, repeats):
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

    reducers = {
        budget: StructuredGridTokenReducer(budget)
        for budget in args.budgets
    }
    residual_reducer = StructuredResidualTokenReducer(
        output_token_budget=args.residual_output_budget,
        base_token_budget=args.residual_base_budget,
    )
    post_reducer = StructuredGridTokenReducer(args.post_output_budget)
    post_sampler = StructuredGridTokenSampler(args.post_output_budget)
    reduced_features = {
        budget: reducer(
            selected,
            batch_size=1,
            frames=args.frames,
        )
        for budget, reducer in reducers.items()
    }
    residual_features = residual_reducer(
        selected,
        batch_size=1,
        frames=args.frames,
    )
    standard_pooled_features = model.apply_pooling(
        model.multi_modal_projector(selected)
    )

    def dense_projector():
        return model.multi_modal_projector(selected)

    def dense_pool():
        return model.apply_pooling(dense_projector())

    operations = {
        "dense_projector": dense_projector,
        "dense_projector_and_pool": dense_pool,
        "residual_pool": lambda: residual_reducer(
            selected,
            batch_size=1,
            frames=args.frames,
        ),
        "residual_projector": lambda: model.multi_modal_projector(
            residual_features
        ),
        "residual_pool_and_projector": lambda: model.multi_modal_projector(
            residual_reducer(
                selected,
                batch_size=1,
                frames=args.frames,
            )
        ),
        "post_projector_pool": lambda: post_reducer(
            standard_pooled_features,
            batch_size=1,
            frames=args.frames,
        ),
        "post_projector_sample": lambda: post_sampler(
            standard_pooled_features,
            batch_size=1,
            frames=args.frames,
        ),
        "dense_tail_and_post_pool": lambda: post_reducer(
            model.apply_pooling(model.multi_modal_projector(selected)),
            batch_size=1,
            frames=args.frames,
        ),
        "dense_tail_and_post_sample": lambda: post_sampler(
            model.apply_pooling(model.multi_modal_projector(selected)),
            batch_size=1,
            frames=args.frames,
        ),
    }
    for budget, reducer in reducers.items():
        # 分离规则池化和 projector，定位 token 形状在哪个组件产生硬件台阶。
        operations[f"pool_{budget}"] = (
            lambda reducer=reducer: reducer(
                selected,
                batch_size=1,
                frames=args.frames,
            )
        )
        operations[f"projector_{budget}"] = (
            lambda budget=budget: model.multi_modal_projector(
                reduced_features[budget]
            )
        )
        operations[f"pool_and_projector_{budget}"] = (
            lambda reducer=reducer: model.multi_modal_projector(
                reducer(
                    selected,
                    batch_size=1,
                    frames=args.frames,
                )
            )
        )

    round_results = []
    operation_names = list(operations)
    for round_idx in range(args.rounds):
        # 每轮循环移位执行顺序，降低升频、降频和缓存热度造成的固定顺序偏差。
        shift = round_idx % len(operation_names)
        ordered_names = operation_names[shift:] + operation_names[:shift]
        result = {}
        for name in ordered_names:
            result[name] = summarize(
                cuda_samples(
                    operations[name],
                    args.warmup,
                    args.repeats,
                )
            )
        round_results.append(result)

    aggregate = {}
    for name in operation_names:
        medians = [
            result[name]["median_ms"]
            for result in round_results
        ]
        aggregate[name] = {
            "round_medians_ms": medians,
            "median_ms": statistics.median(medians),
            "mean_ms": statistics.mean(medians),
            "stdev_ms": (
                statistics.stdev(medians)
                if len(medians) > 1
                else 0.0
            ),
        }

    output = {
        "video_path": args.video_path,
        "frames": args.frames,
        "input_tokens_per_frame": int(selected.shape[1]),
        "budgets": args.budgets,
        "residual_output_budget": args.residual_output_budget,
        "residual_base_budget": args.residual_base_budget,
        "post_output_budget": args.post_output_budget,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "rounds": args.rounds,
        "aggregate": aggregate,
        "round_results": round_results,
        "metric_definition": (
            "CUDA event latency in one process on the same fixed ViT features; "
            "language-model context write is intentionally excluded."
        ),
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

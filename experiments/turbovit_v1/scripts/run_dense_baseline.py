import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.redundancy import adjacent_layer_cosine
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.models.tiny_vit import TinyViTConfig, TinyViTEncoder
from experiments.turbovit_v1.utils.io import write_csv, write_json, write_line_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v1 dense baseline and redundancy analysis.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v0_dense_baseline")
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--drift-per-frame", type=float, default=0.015)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    video_config = SyntheticVideoConfig(
        num_frames=args.num_frames,
        image_size=args.image_size,
        drift_per_frame=args.drift_per_frame,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    model_config = TinyViTConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
    )
    video = make_redundant_video(video_config)
    model = TinyViTEncoder(model_config).to(device)

    dense_results = encode_stream_dense(model, video)
    latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    redundancy_rows = adjacent_layer_cosine(dense_results)
    latencies = torch.tensor([row["latency_ms"] for row in latency_rows])

    summary = {
        "experiment": "turbovit_v1_dense_baseline",
        "device": str(device),
        "torch_version": torch.__version__,
        "num_frames": args.num_frames,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "latency_ms_mean": float(latencies.mean().item()),
        "latency_ms_std": float(latencies.std(unbiased=False).item()),
        "latency_ms_total": float(latencies.sum().item()),
        "redundancy_layer0_cosine": redundancy_rows[0]["adjacent_cosine_mean"],
        "redundancy_last_layer_cosine": redundancy_rows[-1]["adjacent_cosine_mean"],
    }

    write_json(output_dir / "dense_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", latency_rows)
    write_csv(output_dir / "layer_redundancy.csv", redundancy_rows)
    write_line_svg(
        output_dir / "layer_redundancy.svg",
        redundancy_rows,
        x_key="layer",
        y_key="adjacent_cosine_mean",
        title="Turbo-ViT-v1 Layer-wise Adjacent Frame Similarity",
    )

    print("Turbo-ViT-v1 dense baseline completed")
    print(f"summary: {output_dir / 'dense_summary.json'}")
    print(f"mean latency/frame: {summary['latency_ms_mean']:.3f} ms")
    print(f"last-layer adjacent cosine: {summary['redundancy_last_layer_cosine']:.4f}")


if __name__ == "__main__":
    main()

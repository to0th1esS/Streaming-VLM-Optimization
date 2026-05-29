import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.eval.redundancy import adjacent_layer_cosine
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json, write_line_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Run dense baseline on torchvision ViT-B/16.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/real_vit_dense")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    model = TorchvisionViTWrapper(weights=args.weights).to(device)
    video = load_video_frames(
        ensure_video(Path(args.video_path), args.video_url),
        num_frames=args.num_frames,
        image_size=model.image_size,
        stride=args.frame_stride,
    )

    dense_results = encode_stream_dense(model, video, warmup_frames=1)
    latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    redundancy_rows = adjacent_layer_cosine(dense_results)
    latencies = torch.tensor([row["latency_ms"] for row in latency_rows])
    summary = {
        "experiment": "real_torchvision_vit_dense",
        "model": "torchvision.vit_b_16",
        "weights": args.weights,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "image_size": model.image_size,
        "latency_ms_mean": float(latencies.mean().item()),
        "latency_ms_total": float(latencies.sum().item()),
        "redundancy_layer0_cosine": redundancy_rows[0]["adjacent_cosine_mean"],
        "redundancy_last_layer_cosine": redundancy_rows[-1]["adjacent_cosine_mean"],
    }
    write_json(output_dir / "real_vit_dense_summary.json", summary)
    write_csv(output_dir / "real_vit_dense_latency.csv", latency_rows)
    write_csv(output_dir / "real_vit_layer_redundancy.csv", redundancy_rows)
    write_line_svg(
        output_dir / "real_vit_layer_redundancy.svg",
        redundancy_rows,
        x_key="layer",
        y_key="adjacent_cosine_mean",
        title="Torchvision ViT-B/16 Layer-wise Adjacent Frame Similarity",
    )

    print("Torchvision ViT dense run completed")
    print(f"summary: {output_dir / 'real_vit_dense_summary.json'}")
    print(f"mean latency/frame: {summary['latency_ms_mean']:.3f} ms")
    print(f"last-layer adjacent cosine: {summary['redundancy_last_layer_cosine']:.6f}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v1 import encode_stream_turbovit_v1
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v1 on torchvision ViT-B/16.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/real_vit_turbo")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def mean(rows, key):
    values = torch.tensor([float(row[key]) for row in rows])
    return float(values.mean().item())


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
    turbo_results = encode_stream_turbovit_v1(
        model,
        video,
        refresh_interval=args.refresh_interval,
        dynamic_ratio=args.dynamic_ratio,
        warmup_frames=1,
    )

    dense_latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    turbo_latency_rows = [
        {
            "frame_idx": result.frame_idx,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "selector_ms": result.selector_ms,
            "sparse_compute_ms": result.sparse_compute_ms,
            "dynamic_ratio_observed": result.dynamic_ratio_observed,
        }
        for result in turbo_results
    ]
    fidelity_rows = compare_outputs(dense_results, turbo_results)

    dense_total = sum(row["latency_ms"] for row in dense_latency_rows)
    turbo_total = sum(row["latency_ms"] for row in turbo_latency_rows)
    summary = {
        "experiment": "real_torchvision_vit_turbo_v1",
        "model": "torchvision.vit_b_16",
        "weights": args.weights,
        "device": str(device),
        "torch_version": torch.__version__,
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "image_size": model.image_size,
        "refresh_interval": args.refresh_interval,
        "dynamic_ratio": args.dynamic_ratio,
        "dense_latency_ms_mean": mean(dense_latency_rows, "latency_ms"),
        "turbo_latency_ms_mean": mean(turbo_latency_rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "turbo_latency_ms_total": turbo_total,
        "speedup": dense_total / turbo_total if turbo_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "mean_selector_ms": mean(turbo_latency_rows, "selector_ms"),
        "mean_sparse_compute_ms": mean(turbo_latency_rows, "sparse_compute_ms"),
        "reference_frames": sum(row["is_reference"] for row in turbo_latency_rows),
    }

    write_json(output_dir / "real_vit_turbo_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)
    write_csv(output_dir / "turbo_latency.csv", turbo_latency_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)

    print("Torchvision ViT Turbo-v1 run completed")
    print(f"summary: {output_dir / 'real_vit_turbo_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"mean output mse: {summary['mean_output_mse']:.8f}")


if __name__ == "__main__":
    main()

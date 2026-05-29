import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v2 import encode_stream_turbovit_v2
from experiments.turbovit_v1.models.tiny_vit import TinyViTConfig, TinyViTEncoder
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v2 segment-level decision prototype.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v2_segment_decision")
    parser.add_argument("--video-source", default="real", choices=["synthetic", "real"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--drift-per-frame", type=float, default=0.015)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.75)
    parser.add_argument("--skip-threshold", type=float, default=0.0005)
    parser.add_argument("--dense-threshold", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=0)
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
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    if args.video_source == "real":
        video = load_video_frames(
            ensure_video(Path(args.video_path), args.video_url),
            num_frames=args.num_frames,
            image_size=args.image_size,
            stride=args.frame_stride,
        )
    else:
        video = make_redundant_video(
            SyntheticVideoConfig(
                num_frames=args.num_frames,
                image_size=args.image_size,
                drift_per_frame=args.drift_per_frame,
                noise_std=args.noise_std,
                seed=args.seed,
            )
        )

    model = TinyViTEncoder(
        TinyViTConfig(
            image_size=args.image_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        )
    ).to(device)

    dense_results = encode_stream_dense(model, video)
    v2_results = encode_stream_turbovit_v2(
        model,
        video,
        refresh_interval=args.refresh_interval,
        dynamic_ratio=args.dynamic_ratio,
        skip_threshold=args.skip_threshold,
        dense_threshold=args.dense_threshold,
    )
    dense_latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    v2_latency_rows = [
        {
            "frame_idx": result.frame_idx,
            "decision": result.decision,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "selector_ms": result.selector_ms,
            "sparse_compute_ms": result.sparse_compute_ms,
            "dynamic_ratio_observed": result.dynamic_ratio_observed,
            "frame_drift": result.frame_drift,
        }
        for result in v2_results
    ]
    fidelity_rows = compare_outputs(dense_results, v2_results)
    dense_total = sum(row["latency_ms"] for row in dense_latency_rows)
    v2_total = sum(row["latency_ms"] for row in v2_latency_rows)
    summary = {
        "experiment": "turbovit_v2_segment_decision",
        "device": str(device),
        "torch_version": torch.__version__,
        "video_source": args.video_source,
        "refresh_interval": args.refresh_interval,
        "dynamic_ratio": args.dynamic_ratio,
        "skip_threshold": args.skip_threshold,
        "dense_threshold": args.dense_threshold,
        "dense_latency_ms_total": dense_total,
        "v2_latency_ms_total": v2_total,
        "speedup": dense_total / v2_total if v2_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "dense_frames": sum(row["decision"] == "dense" for row in v2_latency_rows),
        "sparse_frames": sum(row["decision"] == "sparse" for row in v2_latency_rows),
        "skip_frames": sum(row["decision"] == "skip" for row in v2_latency_rows),
        "mean_selector_ms": mean(v2_latency_rows, "selector_ms"),
        "mean_sparse_compute_ms": mean(v2_latency_rows, "sparse_compute_ms"),
    }

    write_json(output_dir / "v2_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)
    write_csv(output_dir / "v2_latency.csv", v2_latency_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)

    print("Turbo-ViT-v2 segment decision completed")
    print(f"summary: {output_dir / 'v2_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(
        "decisions: "
        f"dense={summary['dense_frames']} sparse={summary['sparse_frames']} skip={summary['skip_frames']}"
    )


if __name__ == "__main__":
    main()

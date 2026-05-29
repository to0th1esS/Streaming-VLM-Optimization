import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v1 import encode_stream_turbovit_v1
from experiments.turbovit_v1.models.tiny_vit import TinyViTConfig, TinyViTEncoder
from experiments.turbovit_v1.utils.io import write_bar_svg, write_csv, write_json, write_line_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v1 local ablations.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v1_ablation")
    parser.add_argument("--video-source", default="synthetic", choices=["synthetic", "real"])
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
    parser.add_argument("--refresh-intervals", default="2,4,8")
    parser.add_argument("--dynamic-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def parse_int_list(value: str):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def row_mean(rows, key):
    values = torch.tensor([float(row[key]) for row in rows])
    return float(values.mean().item())


def aggregate_drift(fidelity_rows, refresh_interval, dynamic_ratio):
    rows = []
    max_distance = max(row["frame_idx"] % refresh_interval for row in fidelity_rows)
    for distance in range(max_distance + 1):
        bucket = [row for row in fidelity_rows if row["frame_idx"] % refresh_interval == distance]
        if not bucket:
            continue
        rows.append(
            {
                "refresh_interval": refresh_interval,
                "dynamic_ratio": dynamic_ratio,
                "distance_from_reference": distance,
                "mean_output_cosine": row_mean(bucket, "output_cosine"),
                "mean_output_mse": row_mean(bucket, "output_mse"),
                "num_frames": len(bucket),
            }
        )
    return rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    if args.video_source == "real":
        video_path = ensure_video(Path(args.video_path), args.video_url)
        video = load_video_frames(
            video_path,
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
    dense_total = sum(result.latency_ms for result in dense_results)
    dense_mean = dense_total / len(dense_results)
    rows = []
    drift_rows = []
    breakdown_rows = []
    for refresh_interval in parse_int_list(args.refresh_intervals):
        for dynamic_ratio in parse_float_list(args.dynamic_ratios):
            turbo_results = encode_stream_turbovit_v1(
                model,
                video,
                refresh_interval=refresh_interval,
                dynamic_ratio=dynamic_ratio,
            )
            fidelity_rows = compare_outputs(dense_results, turbo_results)
            turbo_total = sum(result.latency_ms for result in turbo_results)
            turbo_mean = turbo_total / len(turbo_results)
            selector_mean = sum(result.selector_ms for result in turbo_results) / len(turbo_results)
            sparse_mean = sum(result.sparse_compute_ms for result in turbo_results) / len(turbo_results)
            other_mean = turbo_mean - selector_mean - sparse_mean
            rows.append(
                {
                    "refresh_interval": refresh_interval,
                    "dynamic_ratio": dynamic_ratio,
                    "dense_total_ms": dense_total,
                    "turbo_total_ms": turbo_total,
                    "speedup": dense_total / turbo_total if turbo_total > 0 else 0.0,
                    "turbo_latency_ms_mean": turbo_mean,
                    "mean_output_cosine": row_mean(fidelity_rows, "output_cosine"),
                    "mean_output_mse": row_mean(fidelity_rows, "output_mse"),
                    "mean_selector_ms": selector_mean,
                    "mean_sparse_compute_ms": sparse_mean,
                    "mean_other_ms": other_mean,
                    "reference_frames": sum(int(result.is_reference) for result in turbo_results),
                }
            )
            drift_rows.extend(aggregate_drift(fidelity_rows, refresh_interval, dynamic_ratio))
            breakdown_rows.extend(
                [
                    {
                        "refresh_interval": refresh_interval,
                        "dynamic_ratio": dynamic_ratio,
                        "component": "dense_baseline",
                        "latency_ms": dense_mean,
                    },
                    {
                        "refresh_interval": refresh_interval,
                        "dynamic_ratio": dynamic_ratio,
                        "component": "selector",
                        "latency_ms": selector_mean,
                    },
                    {
                        "refresh_interval": refresh_interval,
                        "dynamic_ratio": dynamic_ratio,
                        "component": "sparse_compute",
                        "latency_ms": sparse_mean,
                    },
                    {
                        "refresh_interval": refresh_interval,
                        "dynamic_ratio": dynamic_ratio,
                        "component": "other_and_reference",
                        "latency_ms": other_mean,
                    },
                ]
            )

    best_fidelity = max(rows, key=lambda row: row["mean_output_cosine"])
    best_speed = max(rows, key=lambda row: row["speedup"])
    summary = {
        "experiment": "turbovit_v1_ablation",
        "device": str(device),
        "torch_version": torch.__version__,
        "video_source": args.video_source,
        "video_path": args.video_path if args.video_source == "real" else "",
        "dense_total_ms": dense_total,
        "best_speed": best_speed,
        "best_fidelity": best_fidelity,
    }
    write_csv(output_dir / "ablation.csv", rows)
    write_csv(output_dir / "drift_by_distance.csv", drift_rows)
    write_csv(output_dir / "time_breakdown.csv", breakdown_rows)
    write_json(output_dir / "ablation_summary.json", summary)

    best_speed_drift = [
        row for row in drift_rows
        if row["refresh_interval"] == best_speed["refresh_interval"]
        and row["dynamic_ratio"] == best_speed["dynamic_ratio"]
    ]
    write_line_svg(
        output_dir / "best_speed_drift.svg",
        best_speed_drift,
        x_key="distance_from_reference",
        y_key="mean_output_cosine",
        title="Best-Speed Config Drift From Reference",
    )
    best_speed_breakdown = [
        row for row in breakdown_rows
        if row["refresh_interval"] == best_speed["refresh_interval"]
        and row["dynamic_ratio"] == best_speed["dynamic_ratio"]
    ]
    write_bar_svg(
        output_dir / "best_speed_time_breakdown.svg",
        best_speed_breakdown,
        label_key="component",
        value_key="latency_ms",
        title="Best-Speed Config Time Breakdown",
    )

    print("Turbo-ViT-v1 ablation completed")
    print(f"summary: {output_dir / 'ablation_summary.json'}")
    print(
        "best speed: "
        f"N={best_speed['refresh_interval']} r={best_speed['dynamic_ratio']} "
        f"speedup={best_speed['speedup']:.3f}x cosine={best_speed['mean_output_cosine']:.6f}"
    )


if __name__ == "__main__":
    main()

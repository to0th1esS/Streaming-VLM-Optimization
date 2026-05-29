import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v2 import encode_stream_turbovit_v2
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic Turbo-ViT-v2 routing sweep on torchvision ViT-B/16.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/synthetic_vit_v2")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.75)
    parser.add_argument("--skip-thresholds", default="0.0001,0.0005,0.001")
    parser.add_argument("--dense-threshold", type=float, default=0.006)
    parser.add_argument("--drift-per-frame", type=float, default=0.015)
    parser.add_argument("--noise-std", type=float, default=0.01)
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

    model = TorchvisionViTWrapper(weights=args.weights).to(device)
    video = make_redundant_video(
        SyntheticVideoConfig(
            num_frames=args.num_frames,
            image_size=model.image_size,
            drift_per_frame=args.drift_per_frame,
            noise_std=args.noise_std,
            seed=args.seed,
        )
    )

    dense_results = encode_stream_dense(model, video, warmup_frames=2)
    dense_latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    dense_total = sum(row["latency_ms"] for row in dense_latency_rows)
    encode_stream_turbovit_v2(
        model,
        video[: min(video.shape[0], args.refresh_interval + 1)],
        refresh_interval=args.refresh_interval,
        dynamic_ratio=args.dynamic_ratio,
        skip_threshold=float(args.skip_thresholds.split(",")[0].strip()),
        dense_threshold=args.dense_threshold,
        warmup_frames=1,
    )

    summary_rows = []
    for threshold_text in args.skip_thresholds.split(","):
        skip_threshold = float(threshold_text.strip())
        threshold_dir = output_dir / f"skip{skip_threshold:g}".replace(".", "p")
        turbo_results = encode_stream_turbovit_v2(
            model,
            video,
            refresh_interval=args.refresh_interval,
            dynamic_ratio=args.dynamic_ratio,
            skip_threshold=skip_threshold,
            dense_threshold=args.dense_threshold,
            warmup_frames=2,
        )
        turbo_latency_rows = [
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
            for result in turbo_results
        ]
        fidelity_rows = compare_outputs(dense_results, turbo_results)
        turbo_total = sum(row["latency_ms"] for row in turbo_latency_rows)
        row = {
            "skip_threshold": skip_threshold,
            "dense_threshold": args.dense_threshold,
            "dense_latency_ms_mean": mean(dense_latency_rows, "latency_ms"),
            "turbo_latency_ms_mean": mean(turbo_latency_rows, "latency_ms"),
            "dense_latency_ms_total": dense_total,
            "turbo_latency_ms_total": turbo_total,
            "speedup": dense_total / turbo_total if turbo_total > 0 else 0.0,
            "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
            "mean_output_mse": mean(fidelity_rows, "output_mse"),
            "mean_selector_ms": mean(turbo_latency_rows, "selector_ms"),
            "mean_sparse_compute_ms": mean(turbo_latency_rows, "sparse_compute_ms"),
            "dense_frames": sum(item["decision"] == "dense" for item in turbo_latency_rows),
            "sparse_frames": sum(item["decision"] == "sparse" for item in turbo_latency_rows),
            "skip_frames": sum(item["decision"] == "skip" for item in turbo_latency_rows),
        }
        summary_rows.append(row)
        write_csv(threshold_dir / "turbo_latency.csv", turbo_latency_rows)
        write_csv(threshold_dir / "fidelity.csv", fidelity_rows)
        write_json(threshold_dir / "summary.json", row)

    summary = {
        "experiment": "synthetic_torchvision_vit_turbo_v2",
        "model": "torchvision.vit_b_16",
        "weights": args.weights,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "num_frames": args.num_frames,
        "refresh_interval": args.refresh_interval,
        "dynamic_ratio": args.dynamic_ratio,
        "dense_threshold": args.dense_threshold,
        "drift_per_frame": args.drift_per_frame,
        "noise_std": args.noise_std,
        "rows": summary_rows,
    }
    write_json(output_dir / "synthetic_vit_v2_summary.json", summary)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)

    print("Synthetic torchvision ViT Turbo-v2 sweep completed")
    print(f"summary: {output_dir / 'synthetic_vit_v2_summary.json'}")
    for row in summary_rows:
        print(
            f"skip={row['skip_threshold']:.6f} "
            f"speedup={row['speedup']:.3f}x "
            f"cos={row['mean_output_cosine']:.6f} "
            f"dense/sparse/skip={row['dense_frames']}/{row['sparse_frames']}/{row['skip_frames']}"
        )


if __name__ == "__main__":
    main()

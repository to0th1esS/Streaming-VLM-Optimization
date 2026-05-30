import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v1 import encode_stream_turbovit_v1
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic streaming Turbo-ViT-v1 on torchvision ViT-B/16.")
    parser.add_argument("--backbone", default="torchvision", choices=["torchvision", "clip"])
    parser.add_argument("--model-path", default="/home/mllm/models/clip-vit-large-patch14-336")
    parser.add_argument("--output-dir", default="results/turbovit_v1/synthetic_vit_turbo")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratios", default="0.25,0.5,0.75")
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

    if args.backbone == "clip":
        from experiments.turbovit_v1.models.hf_clip_vit import HFCLIPVisionWrapper

        model = HFCLIPVisionWrapper(args.model_path).to(device)
        model_name = "hf_clip_vision"
        weights_name = "local_pretrained"
    else:
        model = TorchvisionViTWrapper(weights=args.weights).to(device)
        model_name = "torchvision.vit_b_16"
        weights_name = args.weights
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
    warmup_ratio = float(args.dynamic_ratios.split(",")[0].strip())
    encode_stream_turbovit_v1(
        model,
        video[: min(video.shape[0], args.refresh_interval + 1)],
        refresh_interval=args.refresh_interval,
        dynamic_ratio=warmup_ratio,
        warmup_frames=1,
    )

    summary_rows = []
    for ratio_text in args.dynamic_ratios.split(","):
        dynamic_ratio = float(ratio_text.strip())
        ratio_dir = output_dir / f"r{dynamic_ratio:g}".replace(".", "p")
        turbo_results = encode_stream_turbovit_v1(
            model,
            video,
            refresh_interval=args.refresh_interval,
            dynamic_ratio=dynamic_ratio,
            warmup_frames=2,
        )
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
        turbo_total = sum(row["latency_ms"] for row in turbo_latency_rows)
        row = {
            "dynamic_ratio": dynamic_ratio,
            "dense_latency_ms_mean": mean(dense_latency_rows, "latency_ms"),
            "turbo_latency_ms_mean": mean(turbo_latency_rows, "latency_ms"),
            "dense_latency_ms_total": dense_total,
            "turbo_latency_ms_total": turbo_total,
            "speedup": dense_total / turbo_total if turbo_total > 0 else 0.0,
            "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
            "mean_output_mse": mean(fidelity_rows, "output_mse"),
            "mean_selector_ms": mean(turbo_latency_rows, "selector_ms"),
            "mean_sparse_compute_ms": mean(turbo_latency_rows, "sparse_compute_ms"),
            "reference_frames": sum(item["is_reference"] for item in turbo_latency_rows),
        }
        summary_rows.append(row)
        write_csv(ratio_dir / "turbo_latency.csv", turbo_latency_rows)
        write_csv(ratio_dir / "fidelity.csv", fidelity_rows)
        write_json(ratio_dir / "summary.json", row)

    summary = {
        "experiment": "synthetic_torchvision_vit_turbo_v1",
        "model": model_name,
        "backbone": args.backbone,
        "model_path": args.model_path if args.backbone == "clip" else "",
        "weights": weights_name,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "num_frames": args.num_frames,
        "refresh_interval": args.refresh_interval,
        "drift_per_frame": args.drift_per_frame,
        "noise_std": args.noise_std,
        "rows": summary_rows,
    }
    write_json(output_dir / "synthetic_vit_turbo_summary.json", summary)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)

    print("Synthetic ViT Turbo-v1 sweep completed")
    print(f"summary: {output_dir / 'synthetic_vit_turbo_summary.json'}")
    for row in summary_rows:
        print(
            f"r={row['dynamic_ratio']:.2f} "
            f"speedup={row['speedup']:.3f}x "
            f"cos={row['mean_output_cosine']:.6f} "
            f"turbo_ms={row['turbo_latency_ms_mean']:.3f}"
        )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v3 import encode_stream_turbovit_v3
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v3 staged routing.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v3_staged")
    parser.add_argument("--backbone", default="clip", choices=["torchvision", "clip"])
    parser.add_argument("--model-path", default="/home/mllm/models/clip-vit-large-patch14-336")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--video-source", default="real", choices=["real", "synthetic"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.9)
    parser.add_argument("--dynamic-ratio-max", type=float, default=0.0)
    parser.add_argument("--skip-threshold", type=float, default=0.001)
    parser.add_argument("--dense-threshold", type=float, default=0.006)
    parser.add_argument("--feature-gate-layer", type=int, default=5)
    parser.add_argument("--feature-skip-threshold", type=float, default=0.98)
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
    return float(values.mean().item()) if values.numel() else 0.0


def load_model(args, device: torch.device):
    if args.backbone == "clip":
        from experiments.turbovit_v1.models.hf_clip_vit import HFCLIPVisionWrapper

        model = HFCLIPVisionWrapper(args.model_path).to(device)
        return model, "hf_clip_vision", "local_pretrained"
    model = TorchvisionViTWrapper(weights=args.weights).to(device)
    return model, "torchvision.vit_b_16", args.weights


def load_stream(args, image_size: int) -> torch.Tensor:
    if args.video_source == "real":
        return load_video_frames(
            ensure_video(Path(args.video_path), args.video_url),
            num_frames=args.num_frames,
            image_size=image_size,
            stride=args.frame_stride,
        )
    return make_redundant_video(
        SyntheticVideoConfig(
            num_frames=args.num_frames,
            image_size=image_size,
            drift_per_frame=args.drift_per_frame,
            noise_std=args.noise_std,
            seed=args.seed,
        )
    )


def summarize_decisions(rows):
    decision_rows = []
    for decision in sorted(set(row["decision"] for row in rows)):
        subset = [row for row in rows if row["decision"] == decision]
        decision_rows.append(
            {
                "decision": decision,
                "frames": len(subset),
                "mean_latency_ms": mean(subset, "latency_ms"),
                "mean_selector_ms": mean(subset, "selector_ms"),
                "mean_sparse_compute_ms": mean(subset, "sparse_compute_ms"),
                "mean_feature_gate_ms": mean(subset, "feature_gate_ms"),
                "mean_feature_gate_cos_min": mean(subset, "feature_gate_cos_min"),
                "mean_frame_drift": mean(subset, "frame_drift"),
            }
        )
    return decision_rows


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)

    model, model_name, weights_name = load_model(args, device)
    video = load_stream(args, image_size=model.image_size)

    dense_results = encode_stream_dense(model, video, warmup_frames=2)
    v3_results = encode_stream_turbovit_v3(
        model,
        video,
        refresh_interval=args.refresh_interval,
        dynamic_ratio=args.dynamic_ratio,
        dynamic_ratio_max=args.dynamic_ratio_max if args.dynamic_ratio_max > 0 else None,
        skip_threshold=args.skip_threshold,
        dense_threshold=args.dense_threshold,
        feature_gate_layer=args.feature_gate_layer,
        feature_skip_threshold=args.feature_skip_threshold,
        warmup_frames=2,
    )

    dense_latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    v3_latency_rows = [
        {
            "frame_idx": result.frame_idx,
            "decision": result.decision,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "selector_ms": result.selector_ms,
            "sparse_compute_ms": result.sparse_compute_ms,
            "feature_gate_ms": result.feature_gate_ms,
            "feature_gate_cos_to_rolling": result.feature_gate_cos_to_rolling,
            "feature_gate_cos_to_long": result.feature_gate_cos_to_long,
            "feature_gate_cos_min": result.feature_gate_cos_min,
            "dynamic_ratio_observed": result.dynamic_ratio_observed,
            "frame_drift": result.frame_drift,
            "patch_mse_to_long": result.patch_mse_to_long,
            "rolling_long_mse": result.rolling_long_mse,
        }
        for result in v3_results
    ]
    fidelity_rows = compare_outputs(dense_results, v3_results)
    decision_rows = summarize_decisions(v3_latency_rows)

    dense_total = sum(row["latency_ms"] for row in dense_latency_rows)
    v3_total = sum(row["latency_ms"] for row in v3_latency_rows)
    false_skip_count = sum(
        1
        for row, fidelity in zip(v3_latency_rows, fidelity_rows)
        if row["decision"] == "skip" and fidelity["output_cosine"] < 0.99
    )
    summary = {
        "experiment": "turbovit_v3_staged_routing",
        "model": model_name,
        "weights": weights_name,
        "backbone": args.backbone,
        "model_path": args.model_path if args.backbone == "clip" else "",
        "video_source": args.video_source,
        "video_path": args.video_path if args.video_source == "real" else "",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "refresh_interval": args.refresh_interval,
        "dynamic_ratio": args.dynamic_ratio,
        "dynamic_ratio_max": args.dynamic_ratio_max,
        "skip_threshold": args.skip_threshold,
        "dense_threshold": args.dense_threshold,
        "feature_gate_layer": args.feature_gate_layer,
        "feature_skip_threshold": args.feature_skip_threshold,
        "dense_latency_ms_mean": mean(dense_latency_rows, "latency_ms"),
        "v3_latency_ms_mean": mean(v3_latency_rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "v3_latency_ms_total": v3_total,
        "speedup": dense_total / v3_total if v3_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "false_skip_count": false_skip_count,
        "decision_summary": decision_rows,
    }

    write_json(output_dir / "v3_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)
    write_csv(output_dir / "v3_latency.csv", v3_latency_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)
    write_csv(output_dir / "decision_summary.csv", decision_rows)

    print("Turbo-ViT-v3 staged routing completed")
    print(f"summary: {output_dir / 'v3_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"false skip count: {false_skip_count}")


if __name__ == "__main__":
    main()

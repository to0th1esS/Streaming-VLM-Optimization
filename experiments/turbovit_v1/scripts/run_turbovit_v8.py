import argparse
from pathlib import Path

import torch

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v8 import encode_stream_turbovit_v8
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Turbo-ViT-v8 layer-aware KV reuse.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v8_layer_kv")
    parser.add_argument("--backbone", default="clip", choices=["torchvision", "clip"])
    parser.add_argument("--model-path", default="/home/mllm/models/clip-vit-large-patch14-336")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--video-source", default="real", choices=["real", "synthetic"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--num-frames", type=int, default=96)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=16)
    parser.add_argument("--sparse-ratio-min", type=float, default=0.8)
    parser.add_argument("--sparse-ratio-max", type=float, default=1.0)
    parser.add_argument("--probe-layer", type=int, default=2)
    parser.add_argument("--skip-patch-threshold", type=float, default=0.001)
    parser.add_argument("--dense-patch-threshold", type=float, default=0.006)
    parser.add_argument("--skip-feature-threshold", type=float, default=0.9999)
    parser.add_argument("--dense-feature-threshold", type=float, default=0.98)
    parser.add_argument("--anchor-mode", default="dual", choices=["dual", "rolling_only", "long_only"])
    parser.add_argument("--segment-max-gap", type=int, default=1)
    parser.add_argument("--min-segment-len", type=int, default=2)
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
                "mean_probe_ms": mean(subset, "probe_ms"),
                "mean_token_selector_ms": mean(subset, "token_selector_ms"),
                "mean_selector_ms": mean(subset, "selector_ms"),
                "mean_sparse_compute_ms": mean(subset, "sparse_compute_ms"),
                "mean_kv_projection_ms": mean(subset, "kv_projection_ms"),
                "mean_semantic_stability": mean(subset, "semantic_stability"),
                "mean_adaptive_ratio": mean(subset, "adaptive_ratio"),
                "mean_segment_count": mean(subset, "segment_count"),
                "mean_segment_len": mean(subset, "mean_segment_len"),
                "mean_segment_expansion_ratio": mean(subset, "segment_expansion_ratio"),
                "mean_rolling_reuse_ratio": mean(subset, "rolling_reuse_ratio"),
                "mean_long_reuse_ratio": mean(subset, "long_reuse_ratio"),
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
    v8_results = encode_stream_turbovit_v8(
        model,
        video,
        refresh_interval=args.refresh_interval,
        sparse_ratio_min=args.sparse_ratio_min,
        sparse_ratio_max=args.sparse_ratio_max,
        probe_layer=args.probe_layer,
        skip_patch_threshold=args.skip_patch_threshold,
        dense_patch_threshold=args.dense_patch_threshold,
        skip_feature_threshold=args.skip_feature_threshold,
        dense_feature_threshold=args.dense_feature_threshold,
        anchor_mode=args.anchor_mode,
        segment_max_gap=args.segment_max_gap,
        min_segment_len=args.min_segment_len,
        warmup_frames=2,
    )

    dense_latency_rows = [
        {"frame_idx": result.frame_idx, "latency_ms": result.latency_ms}
        for result in dense_results
    ]
    v8_latency_rows = [
        {
            "frame_idx": result.frame_idx,
            "decision": result.decision,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "probe_ms": result.probe_ms,
            "token_selector_ms": result.token_selector_ms,
            "selector_ms": result.selector_ms,
            "sparse_compute_ms": result.sparse_compute_ms,
            "kv_projection_ms": result.kv_projection_ms,
            "dynamic_ratio_observed": result.dynamic_ratio_observed,
            "adaptive_ratio": result.adaptive_ratio,
            "frame_drift": result.frame_drift,
            "semantic_stability": result.semantic_stability,
            "segment_count": result.segment_count,
            "mean_segment_len": result.mean_segment_len,
            "segment_expansion_ratio": result.segment_expansion_ratio,
            "rolling_reuse_ratio": result.rolling_reuse_ratio,
            "long_reuse_ratio": result.long_reuse_ratio,
        }
        for result in v8_results
    ]
    fidelity_rows = compare_outputs(dense_results, v8_results)
    decision_rows = summarize_decisions(v8_latency_rows)
    dense_total = sum(row["latency_ms"] for row in dense_latency_rows)
    v8_total = sum(row["latency_ms"] for row in v8_latency_rows)
    false_skip_count = sum(
        1
        for row, fidelity in zip(v8_latency_rows, fidelity_rows)
        if row["decision"] == "skip" and fidelity["output_cosine"] < 0.99
    )

    summary = {
        "experiment": "turbovit_v8_layer_aware_kv_reuse",
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
        "sparse_ratio_min": args.sparse_ratio_min,
        "sparse_ratio_max": args.sparse_ratio_max,
        "probe_layer": args.probe_layer,
        "skip_patch_threshold": args.skip_patch_threshold,
        "dense_patch_threshold": args.dense_patch_threshold,
        "skip_feature_threshold": args.skip_feature_threshold,
        "dense_feature_threshold": args.dense_feature_threshold,
        "anchor_mode": args.anchor_mode,
        "segment_max_gap": args.segment_max_gap,
        "min_segment_len": args.min_segment_len,
        "dense_latency_ms_mean": mean(dense_latency_rows, "latency_ms"),
        "v8_latency_ms_mean": mean(v8_latency_rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "v8_latency_ms_total": v8_total,
        "speedup": dense_total / v8_total if v8_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "false_skip_count": false_skip_count,
        "decision_summary": decision_rows,
    }
    write_json(output_dir / "v8_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_latency_rows)
    write_csv(output_dir / "v8_latency.csv", v8_latency_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)
    write_csv(output_dir / "decision_summary.csv", decision_rows)

    print("Turbo-ViT-v8 layer-aware KV reuse completed")
    print(f"summary: {output_dir / 'v8_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"false skip count: {false_skip_count}")


if __name__ == "__main__":
    main()

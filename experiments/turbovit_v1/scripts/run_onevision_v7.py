import argparse
from pathlib import Path

import imageio.v3 as iio
import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v7 import encode_stream_turbovit_v7
from experiments.turbovit_v1.models.hf_siglip_vit import HFSigLIPVisionWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Turbo-ViT-v7 on the real LLaVA-OneVision SigLIP vision tower."
    )
    parser.add_argument("--model-path", default="model_zoo/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--output-dir", default="results/turbovit_v1/onevision_v7")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--sparse-ratio-min", type=float, default=0.55)
    parser.add_argument("--sparse-ratio-max", type=float, default=0.9)
    parser.add_argument("--probe-layer", type=int, default=2)
    parser.add_argument("--skip-patch-threshold", type=float, default=0.001)
    parser.add_argument("--dense-patch-threshold", type=float, default=0.006)
    parser.add_argument("--skip-feature-threshold", type=float, default=0.9999)
    parser.add_argument("--dense-feature-threshold", type=float, default=0.98)
    parser.add_argument("--anchor-mode", default="dual", choices=["dual", "rolling_only", "long_only"])
    parser.add_argument("--segment-max-gap", type=int, default=1)
    parser.add_argument("--min-segment-len", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_video_uint8(video_path: Path, num_frames: int, frame_stride: int):
    frames = []
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if frame_idx % frame_stride != 0:
            continue
        if frame.ndim == 2:
            frame = frame[..., None].repeat(3, axis=-1)
        frames.append(frame[..., :3])
        if len(frames) >= num_frames:
            break
    if len(frames) < num_frames:
        raise RuntimeError(f"Only decoded {len(frames)} frames from {video_path}; need {num_frames}.")
    return frames


def mean(rows, key):
    values = torch.tensor([float(row[key]) for row in rows])
    return float(values.mean().item()) if values.numel() else 0.0


def summarize_decisions(rows):
    summary = []
    for decision in sorted(set(row["decision"] for row in rows)):
        subset = [row for row in rows if row["decision"] == decision]
        summary.append(
            {
                "decision": decision,
                "frames": len(subset),
                "mean_latency_ms": mean(subset, "latency_ms"),
                "mean_probe_ms": mean(subset, "probe_ms"),
                "mean_token_selector_ms": mean(subset, "token_selector_ms"),
                "mean_selector_ms": mean(subset, "selector_ms"),
                "mean_sparse_compute_ms": mean(subset, "sparse_compute_ms"),
                "mean_dynamic_ratio_observed": mean(subset, "dynamic_ratio_observed"),
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
    return summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    processor = LlavaOnevisionProcessor.from_pretrained(args.model_path)
    full_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    vision_tower = full_model.vision_tower.to(device=device, dtype=dtype)
    model = HFSigLIPVisionWrapper(vision_tower).eval()

    raw_frames = load_video_uint8(Path(args.video_path), args.num_frames, args.frame_stride)
    pixel_values = processor.video_processor(raw_frames, return_tensors="pt").pixel_values_videos[0]

    dense_results = encode_stream_dense(model, pixel_values, warmup_frames=2)
    v7_results = encode_stream_turbovit_v7(
        model,
        pixel_values,
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

    dense_rows = [{"frame_idx": result.frame_idx, "latency_ms": result.latency_ms} for result in dense_results]
    v7_rows = [
        {
            "frame_idx": result.frame_idx,
            "decision": result.decision,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "probe_ms": result.probe_ms,
            "token_selector_ms": result.token_selector_ms,
            "selector_ms": result.selector_ms,
            "sparse_compute_ms": result.sparse_compute_ms,
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
        for result in v7_results
    ]
    fidelity_rows = compare_outputs(dense_results, v7_results)
    decision_rows = summarize_decisions(v7_rows)
    dense_total = sum(row["latency_ms"] for row in dense_rows)
    v7_total = sum(row["latency_ms"] for row in v7_rows)

    summary = {
        "experiment": "onevision_siglip_turbovit_v7_bridge",
        "experiment_purpose": (
            "Validate whether the dual-anchor segment-aware reuse policy developed on CLIP can run on the real "
            "LLaVA-OneVision SigLIP vision tower before connecting it to full streaming QA."
        ),
        "paper_principle": (
            "Use this result as a mechanism-transfer test, not as the final method claim. The final AAAI-style "
            "method should present semantic stability as a unified signal for recomputation, visual-token writing, "
            "and LLM cache control instead of a stack of ad-hoc engineering patches."
        ),
        "model_path": args.model_path,
        "vision_tower": "LLaVA-OneVision SigLIP",
        "video_path": args.video_path,
        "num_frames": args.num_frames,
        "frame_stride": args.frame_stride,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
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
        "dense_latency_ms_mean": mean(dense_rows, "latency_ms"),
        "v7_latency_ms_mean": mean(v7_rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "v7_latency_ms_total": v7_total,
        "speedup": dense_total / v7_total if v7_total > 0 else 0.0,
        "latency_reduction": 1.0 - (v7_total / dense_total) if dense_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "min_output_cosine": min(float(row["output_cosine"]) for row in fidelity_rows),
        "decision_summary": decision_rows,
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "onevision_v7_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_rows)
    write_csv(output_dir / "v7_latency.csv", v7_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)
    write_csv(output_dir / "decision_summary.csv", decision_rows)

    print("OneVision Turbo-ViT-v7 bridge completed")
    print(f"summary: {output_dir / 'onevision_v7_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"latency reduction: {summary['latency_reduction'] * 100:.1f}%")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"min output cosine: {summary['min_output_cosine']:.6f}")


if __name__ == "__main__":
    main()

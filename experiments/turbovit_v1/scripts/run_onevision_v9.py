import argparse
from pathlib import Path

import torch
from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

from experiments.turbovit_v1.eval.fidelity import compare_outputs
from experiments.turbovit_v1.methods.dense_vit import encode_stream_dense
from experiments.turbovit_v1.methods.turbovit_v9 import encode_stream_turbovit_v9
from experiments.turbovit_v1.models.hf_siglip_vit import HFSigLIPVisionWrapper
from experiments.turbovit_v1.scripts.run_onevision_v7 import (
    load_video_uint8,
    mean,
    resolve_device,
    resolve_dtype,
)
from experiments.turbovit_v1.utils.io import write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Turbo-ViT-v9 AnchorGate on the real LLaVA-OneVision SigLIP vision tower."
    )
    parser.add_argument("--model-path", default="model_zoo/llava-onevision-qwen2-0.5b-ov-hf")
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--output-dir", default="results/turbovit_v1/onevision_v9")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--refresh-interval", type=int, default=8)
    parser.add_argument("--skip-patch-threshold", type=float, default=0.01)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def summarize_decisions(rows):
    summary = []
    for decision in sorted(set(row["decision"] for row in rows)):
        subset = [row for row in rows if row["decision"] == decision]
        summary.append(
            {
                "decision": decision,
                "frames": len(subset),
                "mean_latency_ms": mean(subset, "latency_ms"),
                "mean_embed_ms": mean(subset, "embed_ms"),
                "mean_frame_drift": mean(subset, "frame_drift"),
                "mean_dynamic_ratio_observed": mean(subset, "dynamic_ratio_observed"),
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

    raw_frames = load_video_uint8(Path(args.video_path), args.num_frames, args.start_frame, args.frame_stride)
    pixel_values = processor.video_processor(raw_frames, return_tensors="pt").pixel_values_videos[0]

    dense_results = encode_stream_dense(model, pixel_values, warmup_frames=2)
    v9_results = encode_stream_turbovit_v9(
        model,
        pixel_values,
        refresh_interval=args.refresh_interval,
        skip_patch_threshold=args.skip_patch_threshold,
        warmup_frames=2,
    )

    dense_rows = [{"frame_idx": result.frame_idx, "latency_ms": result.latency_ms} for result in dense_results]
    v9_rows = [
        {
            "frame_idx": result.frame_idx,
            "decision": result.decision,
            "is_reference": int(result.is_reference),
            "latency_ms": result.latency_ms,
            "embed_ms": result.embed_ms,
            "frame_drift": result.frame_drift,
            "dynamic_ratio_observed": result.dynamic_ratio_observed,
        }
        for result in v9_results
    ]
    fidelity_rows = compare_outputs(dense_results, v9_results)
    decision_rows = summarize_decisions(v9_rows)
    dense_total = sum(row["latency_ms"] for row in dense_rows)
    v9_total = sum(row["latency_ms"] for row in v9_rows)

    summary = {
        "experiment": "onevision_siglip_turbovit_v9_anchor_gate",
        "experiment_purpose": (
            "Test whether a low-cost rolling-anchor gate can replace per-token sparse routing on the real "
            "LLaVA-OneVision SigLIP tower. This isolates the value of skip/dense routing before adding any "
            "token-level correction."
        ),
        "paper_principle": (
            "A top-conference method should be conceptually simple: first decide whether a frame is semantically "
            "stable enough to reuse, then spend extra computation only on frames that need correction. This run "
            "measures the pure gate before adding correction modules."
        ),
        "model_path": args.model_path,
        "vision_tower": "LLaVA-OneVision SigLIP",
        "video_path": args.video_path,
        "num_frames": args.num_frames,
        "start_frame": args.start_frame,
        "frame_stride": args.frame_stride,
        "device": str(device),
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "",
        "refresh_interval": args.refresh_interval,
        "skip_patch_threshold": args.skip_patch_threshold,
        "dense_latency_ms_mean": mean(dense_rows, "latency_ms"),
        "v9_latency_ms_mean": mean(v9_rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "v9_latency_ms_total": v9_total,
        "speedup": dense_total / v9_total if v9_total > 0 else 0.0,
        "latency_reduction": 1.0 - (v9_total / dense_total) if dense_total > 0 else 0.0,
        "mean_output_cosine": mean(fidelity_rows, "output_cosine"),
        "mean_output_mse": mean(fidelity_rows, "output_mse"),
        "min_output_cosine": min(float(row["output_cosine"]) for row in fidelity_rows),
        "decision_summary": decision_rows,
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "onevision_v9_summary.json", summary)
    write_csv(output_dir / "dense_latency.csv", dense_rows)
    write_csv(output_dir / "v9_latency.csv", v9_rows)
    write_csv(output_dir / "fidelity.csv", fidelity_rows)
    write_csv(output_dir / "decision_summary.csv", decision_rows)

    print("OneVision Turbo-ViT-v9 AnchorGate completed")
    print(f"summary: {output_dir / 'onevision_v9_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"latency reduction: {summary['latency_reduction'] * 100:.1f}%")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"min output cosine: {summary['min_output_cosine']:.6f}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import torch
import torch.nn.functional as F

from experiments.turbovit_v1.data.real_video import DEFAULT_VIDEO_URL, ensure_video, load_video_frames
from experiments.turbovit_v1.data.synthetic_stream import SyntheticVideoConfig, make_redundant_video
from experiments.turbovit_v1.methods.dense_vit import _synchronize_if_needed
from experiments.turbovit_v1.methods.turbovit_v2 import _sparse_from_reference
from experiments.turbovit_v1.models.torchvision_vit import TorchvisionViTWrapper
from experiments.turbovit_v1.utils.io import write_csv, write_json, write_line_svg


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Turbo-ViT routing signals and failure modes.")
    parser.add_argument("--output-dir", default="results/turbovit_v1/v3_analysis")
    parser.add_argument("--backbone", default="clip", choices=["torchvision", "clip"])
    parser.add_argument("--model-path", default="/home/mllm/models/clip-vit-large-patch14-336")
    parser.add_argument("--weights", default="none", choices=["none", "imagenet"])
    parser.add_argument("--video-source", default="real", choices=["real", "synthetic"])
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--video-url", default=DEFAULT_VIDEO_URL)
    parser.add_argument("--num-frames", type=int, default=48)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--dynamic-ratio", type=float, default=0.75)
    parser.add_argument("--skip-threshold", type=float, default=0.001)
    parser.add_argument("--dense-threshold", type=float, default=0.006)
    parser.add_argument("--false-skip-cosine", type=float, default=0.99)
    parser.add_argument("--signal-layers", default="0,2,5,11,23")
    parser.add_argument("--drift-per-frame", type=float, default=0.015)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


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


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item())


def tensor_mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left - right) ** 2).item())


def mean(rows: List[Dict], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) not in ("", None)]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if float(denom.item()) == 0.0:
        return 0.0
    return float(((x * y).sum() / denom).item())


def write_scatter_svg(path: Path, rows: List[Dict], x_key: str, y_key: str, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    filtered = [row for row in rows if row.get(x_key) not in ("", None) and row.get(y_key) not in ("", None)]
    if not filtered:
        path.write_text("", encoding="utf-8")
        return
    width, height = 720, 420
    pad = 60
    xs = [float(row[x_key]) for row in filtered]
    ys = [float(row[y_key]) for row in filtered]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_max += 1.0

    def px(x):
        return pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad)

    def py(y):
        return height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad)

    dots = "\n".join(
        f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="#2563eb" fill-opacity="0.75" />'
        for x, y in zip(xs, ys)
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>
  <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#111827"/>
  <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#111827"/>
  {dots}
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="13">{x_key}</text>
  <text x="18" y="{height / 2}" text-anchor="middle" transform="rotate(-90 18 {height / 2})" font-family="Arial" font-size="13">{y_key}</text>
  <text x="{pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_min:.4f}</text>
  <text x="{width - pad}" y="{height - pad + 22}" text-anchor="middle" font-family="Arial" font-size="11">{x_max:.4f}</text>
  <text x="{pad - 8}" y="{py(y_min):.2f}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.4f}</text>
  <text x="{pad - 8}" y="{py(y_max):.2f}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.4f}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


@torch.inference_mode()
def collect_dense_oracle(model, video: torch.Tensor, device: torch.device, signal_layers: List[int]) -> List[Dict]:
    model.eval()
    video = video.to(device)
    records = []
    valid_layers = [layer for layer in signal_layers if 0 <= layer < len(model.blocks)]

    for frame_idx in range(min(2, video.shape[0])):
        model.forward_with_layers(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        _synchronize_if_needed(device)
        start = perf_counter()
        output, layer_outputs = model.forward_with_layers(frame)
        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0
        embedding = model.embed(frame)
        selected_layers = {layer: layer_outputs[layer].detach().cpu() for layer in valid_layers}
        records.append(
            {
                "frame_idx": frame_idx,
                "dense_latency_ms": latency_ms,
                "output": output.detach().cpu(),
                "embedding": embedding.detach().cpu(),
                "layers": selected_layers,
            }
        )
    return records


@torch.inference_mode()
def analyze_v2_routing(model, video: torch.Tensor, dense_records: List[Dict], args, device: torch.device) -> List[Dict]:
    model.eval()
    video = video.to(device)
    for frame_idx in range(min(2, video.shape[0])):
        model.forward_with_caches(video[frame_idx : frame_idx + 1])
    _synchronize_if_needed(device)

    rows = []
    ref_caches = []
    ref_embed = None
    ref_output = None
    rolling_idx = None
    long_idx = None
    long_embed = None

    for frame_idx in range(video.shape[0]):
        frame = video[frame_idx : frame_idx + 1]
        forced_reference = (frame_idx % args.refresh_interval == 0) or not ref_caches

        _synchronize_if_needed(device)
        start = perf_counter()
        current_embed = model.embed(frame)
        patch_mse_to_rolling = float(torch.mean((current_embed - ref_embed) ** 2).item()) if ref_embed is not None else 0.0
        patch_mse_to_long = float(torch.mean((current_embed - long_embed) ** 2).item()) if long_embed is not None else 0.0
        rolling_long_mse = float(torch.mean((ref_embed - long_embed) ** 2).item()) if ref_embed is not None and long_embed is not None else 0.0
        selector_ms = 0.0
        sparse_compute_ms = 0.0
        dynamic_ratio_observed = 1.0

        if forced_reference or patch_mse_to_rolling >= args.dense_threshold:
            output, ref_caches = model.forward_with_caches(frame)
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            long_embed = current_embed.detach()
            rolling_idx = frame_idx
            long_idx = frame_idx
            decision = "dense"
            is_reference = True
        elif patch_mse_to_rolling <= args.skip_threshold and ref_output is not None:
            output = ref_output
            decision = "skip"
            is_reference = False
            dynamic_ratio_observed = 0.0
        else:
            output, ref_caches, selector_ms, sparse_compute_ms, dynamic_ratio_observed = _sparse_from_reference(
                model,
                frame,
                ref_caches,
                args.dynamic_ratio,
            )
            ref_embed = current_embed.detach()
            ref_output = output.detach()
            rolling_idx = frame_idx
            decision = "sparse"
            is_reference = False

        _synchronize_if_needed(device)
        latency_ms = (perf_counter() - start) * 1000.0

        dense_record = dense_records[frame_idx]
        dense_output = dense_record["output"]
        output_cpu = output.detach().cpu()
        output_cosine = tensor_cosine(dense_output, output_cpu)
        output_mse = tensor_mse(dense_output, output_cpu)
        final_error = 1.0 - output_cosine
        row = {
            "frame_idx": frame_idx,
            "decision": decision,
            "is_reference": int(is_reference),
            "latency_ms": latency_ms,
            "dense_latency_ms": dense_record["dense_latency_ms"],
            "selector_ms": selector_ms,
            "sparse_compute_ms": sparse_compute_ms,
            "dynamic_ratio_observed": dynamic_ratio_observed,
            "patch_mse_to_rolling": patch_mse_to_rolling,
            "patch_mse_to_long": patch_mse_to_long,
            "rolling_long_mse": rolling_long_mse,
            "distance_from_rolling": frame_idx - rolling_idx if rolling_idx is not None else 0,
            "distance_from_long": frame_idx - long_idx if long_idx is not None else 0,
            "output_cosine": output_cosine,
            "output_mse": output_mse,
            "final_error": final_error,
            "false_skip": int(decision == "skip" and output_cosine < args.false_skip_cosine),
        }

        if frame_idx > 0:
            prev_record = dense_records[frame_idx - 1]
            row["patch_mse_to_prev_oracle"] = tensor_mse(dense_record["embedding"], prev_record["embedding"])
            row["output_cosine_to_prev_oracle"] = tensor_cosine(dense_output, prev_record["output"])
        else:
            row["patch_mse_to_prev_oracle"] = 0.0
            row["output_cosine_to_prev_oracle"] = 1.0

        if long_idx is not None:
            long_record = dense_records[long_idx]
            row["oracle_output_cosine_to_long"] = tensor_cosine(dense_output, long_record["output"])
            row["oracle_patch_mse_to_long"] = tensor_mse(dense_record["embedding"], long_record["embedding"])
            for layer_idx, layer in dense_record["layers"].items():
                long_layer = long_record["layers"][layer_idx]
                row[f"layer{layer_idx}_cos_to_long"] = tensor_cosine(layer, long_layer)
                row[f"layer{layer_idx}_mse_to_long"] = tensor_mse(layer, long_layer)
        else:
            row["oracle_output_cosine_to_long"] = 1.0
            row["oracle_patch_mse_to_long"] = 0.0
            for layer_idx in dense_record["layers"]:
                row[f"layer{layer_idx}_cos_to_long"] = 1.0
                row[f"layer{layer_idx}_mse_to_long"] = 0.0

        rows.append(row)
    return rows


def summarize(rows: List[Dict], args, model_name: str, weights_name: str, device: torch.device, image_size: int) -> Dict:
    dense_total = sum(float(row["dense_latency_ms"]) for row in rows)
    turbo_total = sum(float(row["latency_ms"]) for row in rows)
    decisions = sorted(set(row["decision"] for row in rows))
    decision_rows = []
    for decision in decisions:
        subset = [row for row in rows if row["decision"] == decision]
        decision_rows.append(
            {
                "decision": decision,
                "frames": len(subset),
                "mean_latency_ms": mean(subset, "latency_ms"),
                "mean_output_cosine": mean(subset, "output_cosine"),
                "mean_output_mse": mean(subset, "output_mse"),
                "mean_patch_mse_to_rolling": mean(subset, "patch_mse_to_rolling"),
            }
        )
    false_skip_rows = [row for row in rows if row["false_skip"]]
    return {
        "experiment": "turbovit_v3_routing_analysis",
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
        "image_size": image_size,
        "refresh_interval": args.refresh_interval,
        "dynamic_ratio": args.dynamic_ratio,
        "skip_threshold": args.skip_threshold,
        "dense_threshold": args.dense_threshold,
        "false_skip_cosine": args.false_skip_cosine,
        "dense_latency_ms_mean": mean(rows, "dense_latency_ms"),
        "turbo_latency_ms_mean": mean(rows, "latency_ms"),
        "dense_latency_ms_total": dense_total,
        "turbo_latency_ms_total": turbo_total,
        "speedup": dense_total / turbo_total if turbo_total > 0 else 0.0,
        "mean_output_cosine": mean(rows, "output_cosine"),
        "mean_output_mse": mean(rows, "output_mse"),
        "false_skip_count": len(false_skip_rows),
        "false_skip_rate_among_all": len(false_skip_rows) / len(rows) if rows else 0.0,
        "decision_summary": decision_rows,
    }


def build_correlation_rows(rows: List[Dict]) -> List[Dict]:
    candidate_keys = [
        "patch_mse_to_rolling",
        "patch_mse_to_long",
        "rolling_long_mse",
        "distance_from_rolling",
        "distance_from_long",
        "patch_mse_to_prev_oracle",
        "output_cosine_to_prev_oracle",
        "oracle_output_cosine_to_long",
        "oracle_patch_mse_to_long",
    ]
    for key in rows[0].keys() if rows else []:
        if key.startswith("layer") and (key.endswith("_cos_to_long") or key.endswith("_mse_to_long")):
            candidate_keys.append(key)
    y = [float(row["final_error"]) for row in rows]
    corr_rows = []
    for key in candidate_keys:
        values = [row.get(key) for row in rows]
        if any(value in ("", None) for value in values):
            continue
        xs = [float(value) for value in values]
        corr = pearson(xs, y)
        corr_rows.append(
            {
                "signal": key,
                "pearson_with_final_error": corr,
                "abs_pearson": abs(corr),
            }
        )
    return sorted(corr_rows, key=lambda item: item["abs_pearson"], reverse=True)


def simulate_feature_gate_policies(rows: List[Dict], false_skip_cosine: float) -> List[Dict]:
    signals = [
        key
        for key in rows[0].keys()
        if key.startswith("layer") and key.endswith("_cos_to_long")
    ] if rows else []
    thresholds = [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
    policy_rows = []
    dense_total = sum(float(row["dense_latency_ms"]) for row in rows)
    for signal in signals:
        for threshold in thresholds:
            for mode in ["gate_skip_to_dense", "gate_reuse_to_dense"]:
                latency_total = 0.0
                cosines = []
                mses = []
                kept_dense = 0
                kept_sparse = 0
                kept_skip = 0
                remedied_dense = 0
                false_skip = 0
                for row in rows:
                    decision = row["decision"]
                    should_gate = float(row[signal]) < threshold
                    gate_skip = mode == "gate_skip_to_dense" and decision == "skip" and should_gate
                    gate_reuse = mode == "gate_reuse_to_dense" and decision in ("skip", "sparse") and should_gate
                    if gate_skip or gate_reuse:
                        latency_total += float(row["dense_latency_ms"])
                        cosines.append(1.0)
                        mses.append(0.0)
                        remedied_dense += 1
                        kept_dense += 1
                    else:
                        latency_total += float(row["latency_ms"])
                        cosine = float(row["output_cosine"])
                        cosines.append(cosine)
                        mses.append(float(row["output_mse"]))
                        if decision == "dense":
                            kept_dense += 1
                        elif decision == "sparse":
                            kept_sparse += 1
                        elif decision == "skip":
                            kept_skip += 1
                            if cosine < false_skip_cosine:
                                false_skip += 1
                policy_rows.append(
                    {
                        "signal": signal,
                        "threshold": threshold,
                        "mode": mode,
                        "speedup": dense_total / latency_total if latency_total > 0 else 0.0,
                        "mean_output_cosine": sum(cosines) / len(cosines) if cosines else 0.0,
                        "mean_output_mse": sum(mses) / len(mses) if mses else 0.0,
                        "false_skip_count": false_skip,
                        "kept_dense_or_remedied": kept_dense,
                        "kept_sparse": kept_sparse,
                        "kept_skip": kept_skip,
                        "remedied_to_dense": remedied_dense,
                    }
                )
    return sorted(
        policy_rows,
        key=lambda item: (item["false_skip_count"], -item["mean_output_cosine"], -item["speedup"]),
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    signal_layers = [int(item.strip()) for item in args.signal_layers.split(",") if item.strip()]
    output_dir = Path(args.output_dir)

    model, model_name, weights_name = load_model(args, device)
    video = load_stream(args, image_size=model.image_size)
    dense_records = collect_dense_oracle(model, video, device, signal_layers)
    frame_rows = analyze_v2_routing(model, video, dense_records, args, device)
    summary = summarize(frame_rows, args, model_name, weights_name, device, model.image_size)
    corr_rows = build_correlation_rows(frame_rows)
    policy_rows = simulate_feature_gate_policies(frame_rows, args.false_skip_cosine)
    decision_rows = summary["decision_summary"]
    false_skip_rows = [row for row in frame_rows if row["false_skip"]]

    write_json(output_dir / "v3_analysis_summary.json", summary)
    write_csv(output_dir / "frame_analysis.csv", frame_rows)
    write_csv(output_dir / "signal_correlations.csv", corr_rows)
    write_csv(output_dir / "policy_simulation.csv", policy_rows)
    write_csv(output_dir / "decision_summary.csv", decision_rows)
    write_csv(output_dir / "false_skip.csv", false_skip_rows)
    write_line_svg(output_dir / "timeline_output_cosine.svg", frame_rows, "frame_idx", "output_cosine", "Output cosine over stream")
    write_line_svg(output_dir / "timeline_patch_drift.svg", frame_rows, "frame_idx", "patch_mse_to_rolling", "Patch drift to rolling anchor")
    write_scatter_svg(output_dir / "scatter_patch_drift_vs_error.svg", frame_rows, "patch_mse_to_rolling", "final_error", "Patch drift vs final error")
    if corr_rows:
        best_signal = corr_rows[0]["signal"]
        write_scatter_svg(output_dir / "scatter_best_signal_vs_error.svg", frame_rows, best_signal, "final_error", f"{best_signal} vs final error")
    if policy_rows:
        write_json(output_dir / "best_policy_simulation.json", policy_rows[0])

    print("Turbo-ViT v3 routing analysis completed")
    print(f"summary: {output_dir / 'v3_analysis_summary.json'}")
    print(f"speedup: {summary['speedup']:.3f}x")
    print(f"mean output cosine: {summary['mean_output_cosine']:.6f}")
    print(f"false skip count: {summary['false_skip_count']}")
    if corr_rows:
        print(f"best signal: {corr_rows[0]['signal']} corr={corr_rows[0]['pearson_with_final_error']:.4f}")
    if policy_rows:
        print(
            "best simulated policy: "
            f"{policy_rows[0]['mode']} {policy_rows[0]['signal']} >= {policy_rows[0]['threshold']} "
            f"speedup={policy_rows[0]['speedup']:.3f}x "
            f"cos={policy_rows[0]['mean_output_cosine']:.6f} "
            f"false_skip={policy_rows[0]['false_skip_count']}"
        )


if __name__ == "__main__":
    main()

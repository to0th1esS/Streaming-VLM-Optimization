import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def summarize(root, methods):
    metrics_by_method = {
        method: load_json(root / method / "metrics.json") for method in methods
    }
    dense_time = float(metrics_by_method["dense"]["total_encode_video_sec"])
    rows = []
    for method in methods:
        metrics = metrics_by_method[method]
        encode_time = float(metrics["total_encode_video_sec"])
        row = {
            "method": method,
            "samples": int(metrics["samples"]),
            "official_accuracy": float(metrics["official_three_group_average"]),
            "strict_accuracy": float(metrics["strict_three_group_average"]),
            "encode_video_sec": encode_time,
            "speedup_vs_dense": dense_time / encode_time if encode_time else 0.0,
            "semantic_input_frames": int(metrics["semantic_input_frames"]),
            "semantic_kept_frames": int(metrics["semantic_kept_frames"]),
            "semantic_candidate_frames": int(metrics.get("semantic_candidate_frames", 0)),
            "semantic_preprocessed_frames": int(metrics.get("semantic_preprocessed_frames", 0)),
            "semantic_token_reduction": float(metrics["semantic_token_reduction"]),
        }
        for stage, seconds in metrics.get("semantic_timing_sec", {}).items():
            row[f"{stage}_sec"] = float(seconds)
        for key, value in metrics.get("vit_layer_sparse", {}).items():
            row[f"vit_{key}"] = value
        for key, value in metrics.get("vit_output_reduction", {}).items():
            row[f"vit_output_{key}"] = value
        for group in ("backward", "realtime", "forward"):
            group_metrics = metrics.get("per_group", {}).get(group, {})
            row[f"{group}_official_accuracy"] = float(
                group_metrics.get("official_macro_accuracy", 0.0)
            )
            row[f"{group}_strict_accuracy"] = float(
                group_metrics.get("strict_macro_accuracy", 0.0)
            )
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine dense, periodic, and hybrid OVO-Bench metrics."
    )
    parser.add_argument("--root", default="results/ovo_bench/validation")
    parser.add_argument("--methods", default="dense,periodic,hybrid_cm2")
    parser.add_argument(
        "--output-csv",
        default="results/ovo_bench/validation/summary.csv",
    )
    parser.add_argument(
        "--output-json",
        default="results/ovo_bench/validation/summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if "dense" not in methods:
        raise ValueError("methods must include dense")
    rows = summarize(Path(args.root), methods)
    write_csv(args.output_csv, rows)
    write_json(args.output_json, rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

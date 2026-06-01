import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def split_floats(value):
    return [float(item) for item in value.split(",") if item]


def split_ints(value):
    return [int(item) for item in value.split(",") if item]


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def tiny_answer_pass(row):
    question = row["question"].lower()
    pred = row["pred_answer"].lower()
    if "character" in question:
        return any(word in pred for word in ["rabbit", "bunny"])
    if "animated" in question or "real-world" in question:
        return "animat" in pred
    if "setting" in question:
        return any(word in pred for word in ["forest", "nature", "outdoor", "green", "stream", "tree", "meadow"])
    return bool(pred.strip())


def summarize_run(rows, config):
    final = rows[-1]
    input_tokens = int(float(final.get("semantic_input_tokens", 0)))
    written_tokens = int(float(final.get("semantic_written_tokens", 0)))
    input_frames = int(float(final.get("semantic_input_frames", 0)))
    kept_frames = int(float(final.get("semantic_kept_frames", 0)))
    skipped_frames = int(float(final.get("semantic_skipped_frames", 0)))
    qa_passes = [tiny_answer_pass(row) for row in rows]
    token_reduction = 1.0 - (written_tokens / input_tokens) if input_tokens else 0.0
    frame_reduction = 1.0 - (kept_frames / input_frames) if input_frames else 0.0
    return {
        **config,
        "qa_pass": int(all(qa_passes)),
        "qa_pass_count": sum(int(item) for item in qa_passes),
        "qa_total": len(qa_passes),
        "input_frames": input_frames,
        "kept_frames": kept_frames,
        "skipped_frames": skipped_frames,
        "kept_frame_ratio": kept_frames / input_frames if input_frames else 0.0,
        "frame_reduction": frame_reduction,
        "input_tokens": input_tokens,
        "written_tokens": written_tokens,
        "token_reduction": token_reduction,
        "cumulative_encode_video_sec": float(final.get("cumulative_encode_video_sec", 0.0)),
        "elapsed_video_sec": float(final.get("elapsed_video_sec", 0.0)),
        "qa_sec_sum": sum(float(row.get("qa_sec", 0.0)) for row in rows),
        "pred_answers": " | ".join(row["pred_answer"] for row in rows),
    }


def run_one(args, output_dir: Path, refresh_interval: int, threshold: float, compute_gate: bool):
    tag = (
        f"{args.model}_fps{args.sample_fps:g}_r{refresh_interval}_t{threshold:g}_"
        f"compute{int(compute_gate)}_layer{int(args.enable_vit_layer_sparse)}"
    ).replace(".", "p")
    save_dir = output_dir / "runs" / tag
    cmd = [
        sys.executable,
        "-m",
        "video_qa.rekv_stream_vqa",
        "--model",
        args.model,
        "--anno_path",
        args.anno_path,
        "--save_dir",
        str(save_dir),
        "--sample_fps",
        str(args.sample_fps),
        "--n_local",
        str(args.n_local),
        "--retrieve_size",
        str(args.retrieve_size),
        "--retrieve_chunk_size",
        str(args.retrieve_chunk_size),
        "--enable_vit_sparse",
        "true",
        "--enable_vit_layer_sparse",
        str(args.enable_vit_layer_sparse).lower(),
        "--vit_cache_interval",
        str(args.vit_cache_interval),
        "--vit_update_token_ratio",
        str(args.vit_update_token_ratio),
        "--enable_semantic_stream",
        "true",
        "--enable_semantic_compute_gate",
        str(compute_gate).lower(),
        "--semantic_refresh_interval",
        str(refresh_interval),
        "--semantic_skip_threshold",
        str(threshold),
        "--debug",
        str(args.debug).lower(),
    ]
    subprocess.run(cmd, check=True)
    rows = read_csv(save_dir / "1_0.csv")
    return summarize_run(
        rows,
        {
            "model": args.model,
            "sample_fps": args.sample_fps,
            "refresh_interval": refresh_interval,
            "skip_threshold": threshold,
            "compute_gate": int(compute_gate),
            "enable_vit_layer_sparse": int(args.enable_vit_layer_sparse),
            "save_dir": str(save_dir),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run QA-first semantic stream sweeps on a tiny streaming QA set.")
    parser.add_argument("--model", default="llava_ov_0.5b")
    parser.add_argument("--anno-path", default="data/tiny_streaming_qa/big_buck_bunny_qa.json")
    parser.add_argument("--output-dir", default="results/semantic_stream_sweep/tiny")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--n-local", type=int, default=15000)
    parser.add_argument("--retrieve-size", type=int, default=4)
    parser.add_argument("--retrieve-chunk-size", type=int, default=1)
    parser.add_argument("--vit-cache-interval", type=int, default=2)
    parser.add_argument("--vit-update-token-ratio", type=float, default=0.25)
    parser.add_argument("--enable-vit-layer-sparse", type=str2bool, default=False)
    parser.add_argument("--refresh-intervals", default="2,4,8,16")
    parser.add_argument("--thresholds", default="0.005,0.01,0.03")
    parser.add_argument("--compute-gates", default="true,false")
    parser.add_argument("--debug", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    refresh_intervals = split_ints(args.refresh_intervals)
    thresholds = split_floats(args.thresholds)
    compute_gates = [str2bool(item) for item in args.compute_gates.split(",") if item]
    rows = []
    for compute_gate in compute_gates:
        for refresh_interval in refresh_intervals:
            for threshold in thresholds:
                row = run_one(args, output_dir, refresh_interval, threshold, compute_gate)
                rows.append(row)
                write_csv(output_dir / "summary.csv", rows)
                write_json(output_dir / "summary.json", rows)
                print(
                    f"done compute={compute_gate} refresh={refresh_interval} threshold={threshold}: "
                    f"qa={row['qa_pass_count']}/{row['qa_total']} "
                    f"tokens=-{row['token_reduction'] * 100:.1f}% "
                    f"encode={row['cumulative_encode_video_sec']:.3f}s"
                )

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    print(f"summary: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

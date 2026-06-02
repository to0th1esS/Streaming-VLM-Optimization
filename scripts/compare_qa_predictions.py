import argparse
import csv
import json
from pathlib import Path

from evaluate_open_qa_overlap import token_f1


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


def row_key(row):
    return (
        row.get("video_id", ""),
        row.get("question", ""),
        row.get("answer", ""),
    )


def final_by_video(rows):
    by_video = {}
    for row in rows:
        by_video[row.get("video_id", "")] = row
    return by_video


def summarize_runtime(rows):
    by_video = final_by_video(rows)
    total_encode_sec = sum(float(row.get("cumulative_encode_video_sec", 0) or 0) for row in by_video.values())
    input_frames = sum(int(float(row.get("semantic_input_frames", 0) or 0)) for row in by_video.values())
    kept_frames = sum(int(float(row.get("semantic_kept_frames", 0) or 0)) for row in by_video.values())
    input_tokens = sum(float(row.get("semantic_input_tokens", 0) or 0) for row in by_video.values())
    written_tokens = sum(float(row.get("semantic_written_tokens", 0) or 0) for row in by_video.values())
    return {
        "videos": len(by_video),
        "total_encode_video_sec": total_encode_sec,
        "semantic_input_frames": input_frames,
        "semantic_kept_frames": kept_frames,
        "semantic_token_reduction": 1.0 - written_tokens / input_tokens if input_tokens else 0.0,
    }


def mean(values):
    return sum(values) / len(values) if values else 0.0


def compare_rows(baseline_rows, method_rows, epsilon):
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    method_by_key = {row_key(row): row for row in method_rows}
    common_keys = [key for key in baseline_by_key if key in method_by_key]
    compared = []
    for key in common_keys:
        baseline = baseline_by_key[key]
        method = method_by_key[key]
        baseline_f1 = token_f1(baseline.get("pred_answer", ""), baseline.get("answer", ""))
        method_f1 = token_f1(method.get("pred_answer", ""), method.get("answer", ""))
        delta = method_f1 - baseline_f1
        if delta > epsilon:
            outcome = "win"
        elif delta < -epsilon:
            outcome = "loss"
        else:
            outcome = "tie"
        compared.append(
            {
                "video_id": key[0],
                "question": key[1],
                "answer": key[2],
                "baseline_pred": baseline.get("pred_answer", ""),
                "method_pred": method.get("pred_answer", ""),
                "baseline_token_f1": baseline_f1,
                "method_token_f1": method_f1,
                "delta_token_f1": delta,
                "outcome": outcome,
            }
        )
    return compared


def summarize_comparison(compared, baseline_rows, method_rows, epsilon):
    baseline_runtime = summarize_runtime(baseline_rows)
    method_runtime = summarize_runtime(method_rows)
    baseline_scores = [float(row["baseline_token_f1"]) for row in compared]
    method_scores = [float(row["method_token_f1"]) for row in compared]
    outcomes = {"win": 0, "tie": 0, "loss": 0}
    for row in compared:
        outcomes[row["outcome"]] += 1
    speedup = (
        baseline_runtime["total_encode_video_sec"] / method_runtime["total_encode_video_sec"]
        if method_runtime["total_encode_video_sec"]
        else 0.0
    )
    return {
        "samples": len(compared),
        "epsilon": epsilon,
        "baseline_mean_token_f1": mean(baseline_scores),
        "method_mean_token_f1": mean(method_scores),
        "delta_mean_token_f1": mean(method_scores) - mean(baseline_scores),
        "baseline_f1_ge_0_3": mean([score >= 0.3 for score in baseline_scores]),
        "method_f1_ge_0_3": mean([score >= 0.3 for score in method_scores]),
        "baseline_f1_ge_0_5": mean([score >= 0.5 for score in baseline_scores]),
        "method_f1_ge_0_5": mean([score >= 0.5 for score in method_scores]),
        "baseline_f1_ge_0_7": mean([score >= 0.7 for score in baseline_scores]),
        "method_f1_ge_0_7": mean([score >= 0.7 for score in method_scores]),
        "wins": outcomes["win"],
        "ties": outcomes["tie"],
        "losses": outcomes["loss"],
        "win_rate": outcomes["win"] / len(compared) if compared else 0.0,
        "tie_rate": outcomes["tie"] / len(compared) if compared else 0.0,
        "loss_rate": outcomes["loss"] / len(compared) if compared else 0.0,
        "baseline_runtime": baseline_runtime,
        "method_runtime": method_runtime,
        "speedup": speedup,
        "latency_reduction": 1.0 - 1.0 / speedup if speedup else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare dense and semantic QA predictions question by question.")
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--method-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    baseline_rows = read_csv(Path(args.baseline_path))
    method_rows = read_csv(Path(args.method_path))
    compared = compare_rows(baseline_rows, method_rows, args.epsilon)
    summary = summarize_comparison(compared, baseline_rows, method_rows, args.epsilon)
    summary["baseline_path"] = args.baseline_path
    summary["method_path"] = args.method_path
    write_csv(Path(args.output_csv), compared)
    write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

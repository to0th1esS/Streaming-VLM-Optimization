import argparse
import csv
import json
import statistics
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


def percentile(values, percent):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percent
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


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


def _load_json_list(value):
    if value is None:
        return []
    if not str(value).strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _matches_term_group(pred, group):
    if isinstance(group, str):
        return group.lower() in pred
    if isinstance(group, list):
        return any(str(term).lower() in pred for term in group)
    return False


def rule_based_answer_pass(row):
    pred = row["pred_answer"].lower()
    eval_all = _load_json_list(row.get("eval_all"))
    eval_any = _load_json_list(row.get("eval_any"))
    eval_not = _load_json_list(row.get("eval_not"))
    if eval_all or eval_any or eval_not:
        all_ok = all(_matches_term_group(pred, group) for group in eval_all)
        any_ok = True if not eval_any else any(_matches_term_group(pred, group) for group in eval_any)
        not_ok = not any(_matches_term_group(pred, group) for group in eval_not)
        return all_ok and any_ok and not_ok
    return tiny_answer_pass(row)


def summarize_run(rows, config):
    final = rows[-1]
    input_tokens = int(float(final.get("semantic_input_tokens", 0)))
    written_tokens = int(float(final.get("semantic_written_tokens", 0)))
    input_frames = int(float(final.get("semantic_input_frames", 0)))
    kept_frames = int(float(final.get("semantic_kept_frames", 0)))
    skipped_frames = int(float(final.get("semantic_skipped_frames", 0)))
    recency_kept_frames = int(float(final.get("semantic_recency_kept_frames", 0)))
    coverage_kept_frames = int(float(final.get("semantic_coverage_kept_frames", 0)))
    budget_kept_frames = int(float(final.get("semantic_budget_kept_frames", 0)))
    qa_passes = [rule_based_answer_pass(row) for row in rows]
    latest_recent_queries = sum(row.get("query_route", "") == "latest_recent" for row in rows)
    always_recent_queries = sum(row.get("query_route", "") == "always_recent" for row in rows)
    recent_routed_queries = latest_recent_queries + always_recent_queries
    token_reduction = 1.0 - (written_tokens / input_tokens) if input_tokens else 0.0
    frame_reduction = 1.0 - (kept_frames / input_frames) if input_frames else 0.0
    return {
        **config,
        "qa_pass": int(all(qa_passes)),
        "qa_pass_count": sum(int(item) for item in qa_passes),
        "qa_total": len(qa_passes),
        "latest_recent_queries": latest_recent_queries,
        "always_recent_queries": always_recent_queries,
        "recent_routed_queries": recent_routed_queries,
        "input_frames": input_frames,
        "kept_frames": kept_frames,
        "skipped_frames": skipped_frames,
        "recency_kept_frames": recency_kept_frames,
        "coverage_kept_frames": coverage_kept_frames,
        "budget_kept_frames": budget_kept_frames,
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


def aggregate_rows(rows):
    grouped = {}
    for row in rows:
        key = (
            row["model"],
            row["sample_fps"],
            row["refresh_interval"],
            row["skip_threshold"],
            row["compute_gate"],
            row["enable_vit_layer_sparse"],
            row["semantic_recency_keep_frames"],
            row["semantic_recency_updates_anchor"],
            row["semantic_coverage_interval"],
            row["semantic_coverage_updates_anchor"],
            row["semantic_selection_policy"],
            row["semantic_budget_window_size"],
            row["semantic_budget_keep_per_window"],
            row["enable_query_aware_retrieval"],
            row["query_retrieval_policy"],
            row["latest_retrieval_blocks"],
        )
        grouped.setdefault(key, []).append(row)

    aggregates = []
    for key, group in grouped.items():
        encode_values = [float(row["cumulative_encode_video_sec"]) for row in group]
        elapsed_values = [float(row["elapsed_video_sec"]) for row in group]
        qa_values = [int(row["qa_pass"]) for row in group]
        first = group[0]
        aggregates.append(
            {
                "model": key[0],
                "sample_fps": key[1],
                "refresh_interval": key[2],
                "skip_threshold": key[3],
                "compute_gate": key[4],
                "enable_vit_layer_sparse": key[5],
                "semantic_recency_keep_frames": key[6],
                "semantic_recency_updates_anchor": key[7],
                "semantic_coverage_interval": key[8],
                "semantic_coverage_updates_anchor": key[9],
                "semantic_selection_policy": key[10],
                "semantic_budget_window_size": key[11],
                "semantic_budget_keep_per_window": key[12],
                "enable_query_aware_retrieval": key[13],
                "query_retrieval_policy": key[14],
                "latest_retrieval_blocks": key[15],
                "repeats": len(group),
                "qa_pass_rate": sum(qa_values) / len(qa_values),
                "qa_pass_count_mean": statistics.mean(int(row["qa_pass_count"]) for row in group),
                "qa_total": first["qa_total"],
                "latest_recent_queries_mean": statistics.mean(float(row["latest_recent_queries"]) for row in group),
                "always_recent_queries_mean": statistics.mean(float(row["always_recent_queries"]) for row in group),
                "recent_routed_queries_mean": statistics.mean(float(row["recent_routed_queries"]) for row in group),
                "input_frames": first["input_frames"],
                "kept_frames_mean": statistics.mean(float(row["kept_frames"]) for row in group),
                "recency_kept_frames_mean": statistics.mean(float(row["recency_kept_frames"]) for row in group),
                "coverage_kept_frames_mean": statistics.mean(float(row["coverage_kept_frames"]) for row in group),
                "budget_kept_frames_mean": statistics.mean(float(row["budget_kept_frames"]) for row in group),
                "token_reduction_mean": statistics.mean(float(row["token_reduction"]) for row in group),
                "encode_mean_sec": statistics.mean(encode_values),
                "encode_median_sec": statistics.median(encode_values),
                "encode_p90_sec": percentile(encode_values, 0.9),
                "elapsed_mean_sec": statistics.mean(elapsed_values),
                "elapsed_median_sec": statistics.median(elapsed_values),
            }
        )
    return aggregates


def run_one(args, output_dir: Path, refresh_interval: int, threshold: float, compute_gate: bool, repeat_idx: int):
    tag = (
        f"{args.model}_fps{args.sample_fps:g}_r{refresh_interval}_t{threshold:g}_"
        f"compute{int(compute_gate)}_layer{int(args.enable_vit_layer_sparse)}_"
        f"recent{args.semantic_recency_keep_frames}_"
        f"anchor{int(args.semantic_recency_updates_anchor)}_"
        f"cov{args.semantic_coverage_interval}_"
        f"covanchor{int(args.semantic_coverage_updates_anchor)}_"
        f"sel{args.semantic_selection_policy}_"
        f"bw{args.semantic_budget_window_size}_"
        f"bk{args.semantic_budget_keep_per_window}_"
        f"qa{int(args.enable_query_aware_retrieval)}_"
        f"policy{args.query_retrieval_policy}_"
        f"qrb{args.latest_retrieval_blocks}_rep{repeat_idx}"
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
        "--semantic_recency_keep_frames",
        str(args.semantic_recency_keep_frames),
        "--semantic_recency_updates_anchor",
        str(args.semantic_recency_updates_anchor).lower(),
        "--semantic_coverage_interval",
        str(args.semantic_coverage_interval),
        "--semantic_coverage_updates_anchor",
        str(args.semantic_coverage_updates_anchor).lower(),
        "--semantic_selection_policy",
        args.semantic_selection_policy,
        "--semantic_budget_window_size",
        str(args.semantic_budget_window_size),
        "--semantic_budget_keep_per_window",
        str(args.semantic_budget_keep_per_window),
        "--enable_query_aware_retrieval",
        str(args.enable_query_aware_retrieval).lower(),
        "--query_retrieval_policy",
        args.query_retrieval_policy,
        "--latest_retrieval_blocks",
        str(args.latest_retrieval_blocks),
        "--latest_query_terms",
        args.latest_query_terms,
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
            "semantic_recency_keep_frames": args.semantic_recency_keep_frames,
            "semantic_recency_updates_anchor": int(args.semantic_recency_updates_anchor),
            "semantic_coverage_interval": args.semantic_coverage_interval,
            "semantic_coverage_updates_anchor": int(args.semantic_coverage_updates_anchor),
            "semantic_selection_policy": args.semantic_selection_policy,
            "semantic_budget_window_size": args.semantic_budget_window_size,
            "semantic_budget_keep_per_window": args.semantic_budget_keep_per_window,
            "enable_query_aware_retrieval": int(args.enable_query_aware_retrieval),
            "query_retrieval_policy": args.query_retrieval_policy,
            "latest_retrieval_blocks": args.latest_retrieval_blocks,
            "repeat_idx": repeat_idx,
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
    parser.add_argument("--semantic-recency-keep-frames", type=int, default=0)
    parser.add_argument("--semantic-recency-updates-anchor", type=str2bool, default=False)
    parser.add_argument("--semantic-coverage-interval", type=int, default=0)
    parser.add_argument("--semantic-coverage-updates-anchor", type=str2bool, default=False)
    parser.add_argument(
        "--semantic-selection-policy",
        choices=["threshold", "budget_topk"],
        default="threshold",
    )
    parser.add_argument("--semantic-budget-window-size", type=int, default=0)
    parser.add_argument("--semantic-budget-keep-per-window", type=int, default=1)
    parser.add_argument("--enable-query-aware-retrieval", type=str2bool, default=False)
    parser.add_argument(
        "--query-retrieval-policy",
        choices=["internal", "latest_recent", "always_recent"],
        default="latest_recent",
    )
    parser.add_argument("--latest-retrieval-blocks", type=int, default=0)
    parser.add_argument(
        "--latest-query-terms",
        default="latest,current,currently,now,setting,where,last frame,latest clip,latest video frame",
    )
    parser.add_argument("--repeats", type=int, default=1)
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
                for repeat_idx in range(args.repeats):
                    row = run_one(args, output_dir, refresh_interval, threshold, compute_gate, repeat_idx)
                    rows.append(row)
                    aggregates = aggregate_rows(rows)
                    write_csv(output_dir / "summary.csv", rows)
                    write_json(output_dir / "summary.json", rows)
                    write_csv(output_dir / "aggregate_summary.csv", aggregates)
                    write_json(output_dir / "aggregate_summary.json", aggregates)
                    print(
                        f"done compute={compute_gate} refresh={refresh_interval} threshold={threshold} "
                        f"repeat={repeat_idx}: "
                        f"qa={row['qa_pass_count']}/{row['qa_total']} "
                        f"token_reduction={row['token_reduction'] * 100:.1f}% "
                        f"coverage={row['coverage_kept_frames']} "
                        f"budget={row['budget_kept_frames']} "
                        f"encode={row['cumulative_encode_video_sec']:.3f}s"
                    )

    write_csv(output_dir / "summary.csv", rows)
    write_json(output_dir / "summary.json", rows)
    aggregates = aggregate_rows(rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregates)
    write_json(output_dir / "aggregate_summary.json", aggregates)
    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"aggregate summary: {output_dir / 'aggregate_summary.csv'}")


if __name__ == "__main__":
    main()

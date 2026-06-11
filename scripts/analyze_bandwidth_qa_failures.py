import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


KEY_FIELDS = ("video_id", "benchmark_task", "official_id", "query_index")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    return tuple(str(row.get(field, "")) for field in KEY_FIELDS)


def as_int(row, key):
    return int(float(row.get(key, 0) or 0))


def as_float(row, key):
    return float(row.get(key, 0) or 0)


def flip_name(baseline_score, candidate_score):
    if baseline_score == candidate_score:
        return "stable_correct" if baseline_score else "stable_wrong"
    return "positive_flip" if candidate_score else "negative_flip"


def compare_rows(baseline_rows, candidate_rows):
    baseline_by_key = {row_key(row): row for row in baseline_rows}
    candidate_by_key = {row_key(row): row for row in candidate_rows}
    if baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(baseline_by_key.keys() - candidate_by_key.keys())
        missing_baseline = sorted(candidate_by_key.keys() - baseline_by_key.keys())
        raise ValueError(
            "Baseline and candidate samples differ: "
            f"missing_candidate={missing_candidate[:3]}, "
            f"missing_baseline={missing_baseline[:3]}"
        )

    comparisons = []
    for key in sorted(baseline_by_key):
        baseline = baseline_by_key[key]
        candidate = candidate_by_key[key]
        baseline_score = as_int(baseline, "ovo_official_score")
        candidate_score = as_int(candidate, "ovo_official_score")
        kept_frames = as_int(candidate, "semantic_kept_frames")
        recency_frames = as_int(candidate, "semantic_recency_kept_frames")
        comparisons.append(
            {
                "video_id": candidate.get("video_id", ""),
                "benchmark_group": candidate.get("benchmark_group", ""),
                "benchmark_task": candidate.get("benchmark_task", ""),
                "official_id": candidate.get("official_id", ""),
                "query_index": candidate.get("query_index", ""),
                "flip": flip_name(baseline_score, candidate_score),
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "baseline_answer": baseline.get("pred_answer", ""),
                "candidate_answer": candidate.get("pred_answer", ""),
                "ground_truth": candidate.get("answer", ""),
                "question": candidate.get("question", ""),
                "loaded_frames": as_int(candidate, "loaded_frames"),
                "kept_frames": kept_frames,
                "recency_kept_frames": recency_frames,
                # 该比例用于判断固定近期锚点是否挤压了历史事件帧预算。
                "recency_share": (
                    recency_frames / kept_frames if kept_frames else 0.0
                ),
                "written_tokens": as_int(candidate, "semantic_written_tokens"),
                "online_video_processing_sec": as_float(
                    candidate, "cumulative_encode_video_sec"
                ),
            }
        )
    return comparisons


def summarize_group(rows, group_field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_field]].append(row)

    summaries = {}
    for group_name, group_rows in sorted(grouped.items()):
        summaries[group_name] = {
            "samples": len(group_rows),
            "baseline_micro_accuracy": mean(
                row["baseline_score"] for row in group_rows
            ),
            "candidate_micro_accuracy": mean(
                row["candidate_score"] for row in group_rows
            ),
            "positive_flips": sum(
                row["flip"] == "positive_flip" for row in group_rows
            ),
            "negative_flips": sum(
                row["flip"] == "negative_flip" for row in group_rows
            ),
        }
    return summaries


def summarize_flip_cohorts(rows):
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row["flip"]].append(row)

    summaries = {}
    for cohort_name, cohort_rows in sorted(cohorts.items()):
        summaries[cohort_name] = {
            "samples": len(cohort_rows),
            "mean_loaded_frames": mean(
                row["loaded_frames"] for row in cohort_rows
            ),
            "mean_kept_frames": mean(row["kept_frames"] for row in cohort_rows),
            "mean_recency_share": mean(
                row["recency_share"] for row in cohort_rows
            ),
            "mean_written_tokens": mean(
                row["written_tokens"] for row in cohort_rows
            ),
        }
    return summaries


def build_report(baseline_rows, candidate_rows):
    comparisons = compare_rows(baseline_rows, candidate_rows)
    by_group = summarize_group(comparisons, "benchmark_group")
    group_deltas = {
        group_name: (
            values["candidate_micro_accuracy"]
            - values["baseline_micro_accuracy"]
        )
        for group_name, values in by_group.items()
    }
    return comparisons, {
        "samples": len(comparisons),
        # 逐样本微平均仅用于定位翻转；论文主质量仍使用 OVO 三组宏平均。
        "baseline_micro_accuracy": mean(
            row["baseline_score"] for row in comparisons
        ),
        "candidate_micro_accuracy": mean(
            row["candidate_score"] for row in comparisons
        ),
        "baseline_three_group_macro": mean(
            values["baseline_micro_accuracy"]
            for values in by_group.values()
        ),
        "candidate_three_group_macro": mean(
            values["candidate_micro_accuracy"]
            for values in by_group.values()
        ),
        "worst_group_delta": min(group_deltas.values(), default=0.0),
        "group_deltas": group_deltas,
        "by_group": by_group,
        "by_task": summarize_group(comparisons, "benchmark_task"),
        "by_flip": summarize_flip_cohorts(comparisons),
        "metric_definitions": {
            "positive_flip": "基线回答错误、候选方法回答正确。",
            "negative_flip": "基线回答正确、候选方法回答错误。",
            "micro_accuracy": (
                "所有查询逐样本等权平均，仅用于失效定位；"
                "不替代 OVO-Bench 三任务组宏平均主指标。"
            ),
            "worst_group_delta": (
                "候选方法相对基线在三个任务组中最小的准确率变化；"
                "用于防止总体平均掩盖单类任务退化。"
            ),
            "recency_share": "候选方法近期锚点帧数除以全部保留帧数。",
            "online_video_processing_sec": (
                "帧到达后的 encode_video 累计同步墙钟时间；不含离线文件读取。"
            ),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="定位空间语义带宽压缩造成的 QA 正负翻转。"
    )
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    comparisons, report = build_report(
        read_csv(args.baseline_csv),
        read_csv(args.candidate_csv),
    )
    write_csv(args.output_csv, comparisons)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TASK_GROUPS = {
    "EPM": "backward",
    "ASI": "backward",
    "HLD": "backward",
    "OCR": "realtime",
    "ACR": "realtime",
    "ATR": "realtime",
    "STU": "realtime",
    "FPD": "realtime",
    "OJR": "realtime",
    "REC": "forward",
    "SSR": "forward",
    "CRR": "forward",
}


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


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def extract_choice(response):
    match = re.search(r"(?<![A-Z])([A-H])(?![A-Z])", response.upper())
    return match.group(1) if match else ""


def official_compatible_score(task, response, ground_truth):
    if not response:
        return 0
    if task in {"REC"}:
        digits = "".join(re.findall(r"\d+", response))
        return int(digits == str(ground_truth))
    if task in {"SSR", "CRR"}:
        if (response == "N" and ground_truth == "No") or (
            response == "Y" and ground_truth == "Yes"
        ):
            return 1
        return int(str(ground_truth) in response)
    return int(str(ground_truth) in response)


def strict_score(task, response, ground_truth):
    if task == "REC":
        numbers = re.findall(r"\d+", response)
        return int(len(numbers) == 1 and numbers[0] == str(ground_truth))
    if task in {"SSR", "CRR"}:
        normalized = response.strip().lower().rstrip(".")
        return int(normalized == str(ground_truth).lower())
    return int(extract_choice(response) == str(ground_truth))


def evaluate_rows(rows):
    evaluated = []
    for row in rows:
        task = row.get("benchmark_task", "")
        if task not in TASK_GROUPS:
            raise ValueError(f"Missing or unknown benchmark_task: {task}")
        response = row.get("pred_answer", "")
        ground_truth = row.get("answer", "")
        evaluated.append(
            {
                **row,
                "ovo_official_score": official_compatible_score(
                    task, response, ground_truth
                ),
                "ovo_strict_score": strict_score(task, response, ground_truth),
            }
        )
    return evaluated


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize(evaluated):
    task_official = defaultdict(list)
    task_strict = defaultdict(list)
    for row in evaluated:
        task = row["benchmark_task"]
        task_official[task].append(int(row["ovo_official_score"]))
        task_strict[task].append(int(row["ovo_strict_score"]))

    per_task = {}
    group_task_scores = defaultdict(list)
    group_task_strict_scores = defaultdict(list)
    for task in sorted(task_official):
        group = TASK_GROUPS[task]
        official_accuracy = mean(task_official[task])
        strict_accuracy = mean(task_strict[task])
        per_task[task] = {
            "group": group,
            "samples": len(task_official[task]),
            "official_accuracy": official_accuracy,
            "strict_accuracy": strict_accuracy,
        }
        group_task_scores[group].append(official_accuracy)
        group_task_strict_scores[group].append(strict_accuracy)

    per_group = {}
    for group in ("backward", "realtime", "forward"):
        if group not in group_task_scores:
            continue
        per_group[group] = {
            "tasks": len(group_task_scores[group]),
            "official_macro_accuracy": mean(group_task_scores[group]),
            "strict_macro_accuracy": mean(group_task_strict_scores[group]),
        }

    final_by_video = {}
    for row in evaluated:
        final_by_video[row.get("video_id", "")] = row
    input_tokens = sum(
        float(row.get("semantic_input_tokens", 0) or 0)
        for row in final_by_video.values()
    )
    written_tokens = sum(
        float(row.get("semantic_written_tokens", 0) or 0)
        for row in final_by_video.values()
    )
    total_patch_tokens = sum(
        int(float(row.get("vit_total_patch_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    updated_patch_tokens = sum(
        int(float(row.get("vit_updated_patch_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    output_input_tokens = sum(
        int(float(row.get("vit_output_input_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    output_tokens = sum(
        int(float(row.get("vit_output_tokens", 0) or 0))
        for row in final_by_video.values()
    )
    kv_cache_memory_bytes = [
        int(float(row.get("kv_cache_memory_bytes", 0) or 0))
        for row in final_by_video.values()
    ]

    return {
        "samples": len(evaluated),
        "per_task": per_task,
        "per_group": per_group,
        "official_three_group_average": mean(
            [value["official_macro_accuracy"] for value in per_group.values()]
        ),
        "strict_three_group_average": mean(
            [value["strict_macro_accuracy"] for value in per_group.values()]
        ),
        "total_encode_video_sec": sum(
            float(row.get("cumulative_encode_video_sec", 0) or 0)
            for row in final_by_video.values()
        ),
        "semantic_input_frames": sum(
            int(float(row.get("semantic_input_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_kept_frames": sum(
            int(float(row.get("semantic_kept_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_candidate_frames": sum(
            int(float(row.get("semantic_candidate_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_preprocessed_frames": sum(
            int(float(row.get("semantic_preprocessed_frames", 0) or 0))
            for row in final_by_video.values()
        ),
        "semantic_token_reduction": (
            1.0 - written_tokens / input_tokens if input_tokens else 0.0
        ),
        "semantic_timing_sec": {
            key: sum(
                float(row.get(f"semantic_{key}_sec", 0) or 0)
                for row in final_by_video.values()
            )
            for key in (
                "proposal",
                "preprocess",
                "embedding",
                "verification",
                "vit_encoder",
                "context_write",
            )
        },
        "vit_layer_sparse": {
            "dense_frames": sum(
                int(float(row.get("vit_dense_frames", 0) or 0))
                for row in final_by_video.values()
            ),
            "sparse_frames": sum(
                int(float(row.get("vit_sparse_frames", 0) or 0))
                for row in final_by_video.values()
            ),
            "dense_sec": sum(
                float(row.get("vit_dense_sec", 0) or 0)
                for row in final_by_video.values()
            ),
            "sparse_sec": sum(
                float(row.get("vit_sparse_sec", 0) or 0)
                for row in final_by_video.values()
            ),
            "total_patch_tokens": total_patch_tokens,
            "updated_patch_tokens": updated_patch_tokens,
            "planned_update_ratio": (
                updated_patch_tokens / total_patch_tokens
                if total_patch_tokens
                else 0.0
            ),
        },
        "vit_output_reduction": {
            "input_tokens": output_input_tokens,
            "output_tokens": output_tokens,
            "reduction_ratio": (
                1.0 - output_tokens / output_input_tokens
                if output_input_tokens
                else 0.0
            ),
        },
        "kv_cache_memory": {
            # 均值用于跨方法比较，峰值用于检查长视频的缓存压力。
            "mean_bytes": (
                mean(kv_cache_memory_bytes)
                if kv_cache_memory_bytes
                else 0.0
            ),
            "max_bytes": max(kv_cache_memory_bytes, default=0),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ReKV CSV predictions with the OVO-Bench offline protocol."
    )
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    evaluated = evaluate_rows(read_csv(args.pred_path))
    summary = summarize(evaluated)
    summary["pred_path"] = args.pred_path
    write_csv(args.output_csv, evaluated)
    write_json(args.output_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median


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
DEFAULT_TASKS = ",".join(TASK_GROUPS)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def load_excluded_source_ids(subset_json):
    if not subset_json:
        return set()
    rows = load_json(subset_json)
    return {int(row["official_id"]) for row in rows}


def format_multiple_choice_prompt(question, options):
    formatted_options = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n"
        "Respond only with the letter of the best option."
    )


def format_forward_prompt(item, query):
    task = item["task"]
    if task == "REC":
        return (
            "Count how many times people in the video complete the following action: "
            f"{item['activity']}. Count one complete motion as one occurrence. "
            "Respond with a single number only."
        )
    if task == "SSR":
        return (
            "Determine whether the person is currently performing this tutorial step: "
            f"{query['step']}. Respond only with Yes or No."
        )
    if task == "CRR":
        return (
            "Based on the video observed so far, especially the latest frames, determine "
            "whether there is enough visual information to answer this question: "
            f"{item['question']} Respond only with Yes or No."
        )
    raise ValueError(f"Unsupported forward task: {task}")


def normalize_video_path(chunked_dir, file_name):
    normalized_dir = str(chunked_dir).rstrip("/\\")
    return f"{normalized_dir}/{file_name}".replace("\\", "/")


def make_row(item, chunked_dir, query_index=None):
    task = item["task"]
    group = TASK_GROUPS[task]
    official_id = int(item["id"])

    if group in {"backward", "realtime"}:
        question = format_multiple_choice_prompt(item["question"], item["options"])
        answer = chr(65 + int(item["gt"]))
        end_time = float(item["realtime"])
        video_name = f"{official_id}.mp4"
        row_video_id = f"ovo-{official_id}"
        answer_type = "multiple_choice"
    else:
        query = item["test_info"][query_index]
        question = format_forward_prompt(item, query)
        end_time = float(query["realtime"])
        video_name = f"{official_id}_{query_index}.mp4"
        row_video_id = f"ovo-{official_id}-{query_index}"
        if task == "REC":
            answer = str(query["count"])
            answer_type = "exact_number"
        else:
            answer = "Yes" if int(query["type"]) == 1 else "No"
            answer_type = "yes_no"

    return {
        "video_id": row_video_id,
        "video_path": normalize_video_path(chunked_dir, video_name),
        "benchmark": "ovo_bench",
        "benchmark_group": group,
        "benchmark_task": task,
        "official_id": official_id,
        "query_index": -1 if query_index is None else int(query_index),
        "original_video": item.get("video", ""),
        "conversations": [
            {
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "start_time": 0.0,
                "end_time": end_time,
            }
        ],
    }


def item_end_time(item):
    if TASK_GROUPS[item["task"]] != "forward":
        return float(item["realtime"])
    return max(
        (float(query["realtime"]) for query in item.get("test_info", [])),
        default=0.0,
    )


def evenly_spaced_indices(length, limit):
    if limit <= 0 or limit >= length:
        return list(range(length))
    if limit == 1:
        return [length // 2]
    return [
        round(index * (length - 1) / (limit - 1))
        for index in range(limit)
    ]


def select_source_items(items, limit, policy, fold_count=1, fold_index=0):
    if fold_count < 1:
        raise ValueError("fold_count must be >= 1")
    if fold_index < 0 or fold_index >= fold_count:
        raise ValueError("fold_index must satisfy 0 <= fold_index < fold_count")
    if policy == "head":
        ranked = list(items)
    elif policy == "duration_stratified":
        ranked = sorted(items, key=lambda item: (item_end_time(item), int(item["id"])))
    else:
        raise ValueError(f"Unsupported source selection policy: {policy}")

    folded = ranked[fold_index::fold_count]
    if limit <= 0 or limit >= len(folded):
        return folded
    if policy == "head":
        return folded[:limit]
    return [folded[index] for index in evenly_spaced_indices(len(folded), limit)]


def select_query_indices(queries, limit, policy):
    if limit <= 0 or limit >= len(queries):
        return list(range(len(queries)))
    if policy == "head":
        return list(range(limit))
    if policy == "time_stratified":
        ranked = sorted(
            range(len(queries)),
            key=lambda index: (float(queries[index]["realtime"]), index),
        )
        return [ranked[index] for index in evenly_spaced_indices(len(ranked), limit)]
    raise ValueError(f"Unsupported query selection policy: {policy}")


def convert_annotations(
    annotations,
    chunked_dir,
    tasks,
    max_source_items_per_task=0,
    max_queries_per_source=0,
    source_selection="head",
    query_selection="head",
    source_fold_count=1,
    source_fold_index=0,
):
    selected_tasks = set(tasks)
    source_counts = Counter()
    rows = []
    items_by_task = {
        task: [
            item
            for item in annotations
            if item.get("task") == task
        ]
        for task in tasks
    }
    selected_ids = {
        int(item["id"])
        for task, items in items_by_task.items()
        for item in select_source_items(
            items,
            max_source_items_per_task,
            source_selection,
            fold_count=source_fold_count,
            fold_index=source_fold_index,
        )
    }

    for item in annotations:
        task = item.get("task")
        if task not in selected_tasks:
            continue
        if task not in TASK_GROUPS:
            raise ValueError(f"Unknown OVO-Bench task: {task}")
        if int(item["id"]) not in selected_ids:
            continue

        source_counts[task] += 1
        if TASK_GROUPS[task] != "forward":
            rows.append(make_row(item, chunked_dir))
            continue

        queries = item.get("test_info", [])
        query_indices = select_query_indices(
            queries,
            max_queries_per_source,
            query_selection,
        )
        for query_index in query_indices:
            rows.append(make_row(item, chunked_dir, query_index=query_index))

    return rows, source_counts


def summarize(rows, source_counts):
    task_queries = Counter(row["benchmark_task"] for row in rows)
    group_queries = Counter(row["benchmark_group"] for row in rows)
    times_by_task = {}
    for row in rows:
        task = row["benchmark_task"]
        times_by_task.setdefault(task, []).append(
            float(row["conversations"][0]["end_time"])
        )
    missing_paths = [
        row["video_path"] for row in rows if not Path(row["video_path"]).exists()
    ]
    return {
        "source_items": sum(source_counts.values()),
        "queries": len(rows),
        "source_items_by_task": dict(sorted(source_counts.items())),
        "queries_by_task": dict(sorted(task_queries.items())),
        "queries_by_group": dict(sorted(group_queries.items())),
        "query_time_by_task": {
            task: {
                "min": min(times),
                "median": median(times),
                "max": max(times),
            }
            for task, times in sorted(times_by_task.items())
        },
        "available_video_files": len(rows) - len(missing_paths),
        "missing_video_files": len(missing_paths),
        "missing_video_examples": missing_paths[:20],
    }


def parse_tasks(value):
    tasks = [task.strip().upper() for task in value.split(",") if task.strip()]
    unknown = sorted(set(tasks) - set(TASK_GROUPS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown tasks: {','.join(unknown)}")
    return tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert official OVO-Bench annotations to the ReKV streaming VQA format."
    )
    parser.add_argument("--source-json", required=True)
    parser.add_argument(
        "--chunked-dir",
        default="/home/mllm/datasets/ovo_bench/chunked_videos",
    )
    parser.add_argument(
        "--output-json",
        default="data/ovo_bench/ovo_rekv_subset.json",
    )
    parser.add_argument("--tasks", type=parse_tasks, default=parse_tasks(DEFAULT_TASKS))
    parser.add_argument("--max-source-items-per-task", type=int, default=2)
    parser.add_argument("--max-queries-per-source", type=int, default=2)
    parser.add_argument(
        "--source-selection",
        choices=("head", "duration_stratified"),
        default="head",
    )
    parser.add_argument(
        "--query-selection",
        choices=("head", "time_stratified"),
        default="head",
    )
    parser.add_argument("--source-fold-count", type=int, default=1)
    parser.add_argument("--source-fold-index", type=int, default=0)
    parser.add_argument(
        "--exclude-subset-json",
        default="",
        help="Exclude every official source id already present in another converted subset.",
    )
    parser.add_argument(
        "--require-videos",
        action="store_true",
        help="Fail if any converted chunked video is absent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = load_json(args.source_json)
    excluded_source_ids = load_excluded_source_ids(args.exclude_subset_json)
    if excluded_source_ids:
        annotations = [
            item
            for item in annotations
            if int(item["id"]) not in excluded_source_ids
        ]
    rows, source_counts = convert_annotations(
        annotations,
        chunked_dir=args.chunked_dir,
        tasks=args.tasks,
        max_source_items_per_task=args.max_source_items_per_task,
        max_queries_per_source=args.max_queries_per_source,
        source_selection=args.source_selection,
        query_selection=args.query_selection,
        source_fold_count=args.source_fold_count,
        source_fold_index=args.source_fold_index,
    )
    summary = summarize(rows, source_counts)
    summary.update(
        {
            "source_json": str(args.source_json).replace("\\", "/"),
            "output_json": str(args.output_json).replace("\\", "/"),
            "chunked_dir": str(args.chunked_dir).replace("\\", "/"),
            "source_selection": args.source_selection,
            "query_selection": args.query_selection,
            "source_fold_count": args.source_fold_count,
            "source_fold_index": args.source_fold_index,
            "exclude_subset_json": str(args.exclude_subset_json).replace("\\", "/"),
            "excluded_source_items": len(excluded_source_ids),
        }
    )
    if args.require_videos and summary["missing_video_files"]:
        raise FileNotFoundError(
            f"{summary['missing_video_files']} required chunked videos are missing"
        )

    write_json(args.output_json, rows)
    write_json(Path(args.output_json).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

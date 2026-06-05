import argparse
import json
from collections import Counter
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
DEFAULT_TASKS = ",".join(TASK_GROUPS)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


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


def convert_annotations(
    annotations,
    chunked_dir,
    tasks,
    max_source_items_per_task=0,
    max_queries_per_source=0,
):
    selected_tasks = set(tasks)
    source_counts = Counter()
    rows = []

    for item in annotations:
        task = item.get("task")
        if task not in selected_tasks:
            continue
        if task not in TASK_GROUPS:
            raise ValueError(f"Unknown OVO-Bench task: {task}")
        if (
            max_source_items_per_task > 0
            and source_counts[task] >= max_source_items_per_task
        ):
            continue

        source_counts[task] += 1
        if TASK_GROUPS[task] != "forward":
            rows.append(make_row(item, chunked_dir))
            continue

        queries = item.get("test_info", [])
        if max_queries_per_source > 0:
            queries = queries[:max_queries_per_source]
        for query_index in range(len(queries)):
            rows.append(make_row(item, chunked_dir, query_index=query_index))

    return rows, source_counts


def summarize(rows, source_counts):
    task_queries = Counter(row["benchmark_task"] for row in rows)
    group_queries = Counter(row["benchmark_group"] for row in rows)
    missing_paths = [
        row["video_path"] for row in rows if not Path(row["video_path"]).exists()
    ]
    return {
        "source_items": sum(source_counts.values()),
        "queries": len(rows),
        "source_items_by_task": dict(sorted(source_counts.items())),
        "queries_by_task": dict(sorted(task_queries.items())),
        "queries_by_group": dict(sorted(group_queries.items())),
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
        "--require-videos",
        action="store_true",
        help="Fail if any converted chunked video is absent.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = load_json(args.source_json)
    rows, source_counts = convert_annotations(
        annotations,
        chunked_dir=args.chunked_dir,
        tasks=args.tasks,
        max_source_items_per_task=args.max_source_items_per_task,
        max_queries_per_source=args.max_queries_per_source,
    )
    summary = summarize(rows, source_counts)
    summary.update(
        {
            "source_json": str(args.source_json).replace("\\", "/"),
            "output_json": str(args.output_json).replace("\\", "/"),
            "chunked_dir": str(args.chunked_dir).replace("\\", "/"),
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

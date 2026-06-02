import argparse
import csv
import json
from pathlib import Path


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


def infer_category(question: str):
    question = question.lower()
    if question.startswith(("is ", "does ", "did ", "are ", "was ", "were ", "has ", "have ")):
        return "yes_no"
    if "latest" in question or "setting" in question or "where" in question:
        return "scene_latest"
    if any(term in question for term in ["activity", "task", "action", "step", "perform"]):
        return "action"
    if any(term in question for term in ["object", "device", "fixture", "tool"]):
        return "object"
    return "other"


def summarize(rows, comparison):
    grouped = {}
    for row in rows:
        grouped.setdefault(infer_category(row.get("question", "")), []).append(row)

    summary = []
    for category, group in sorted(grouped.items()):
        count = len(group)
        dense_correct = sum(row.get("dense_correct") == "1" for row in group)
        sparse_correct = sum(row.get("sparse_correct") == "1" for row in group)
        wins = sum(row.get("dense_correct") == "0" and row.get("sparse_correct") == "1" for row in group)
        losses = sum(row.get("dense_correct") == "1" and row.get("sparse_correct") == "0" for row in group)
        summary.append(
            {
                "comparison": comparison,
                "category": category,
                "samples": count,
                "dense_correct": dense_correct,
                "sparse_correct": sparse_correct,
                "dense_accuracy": dense_correct / count if count else 0.0,
                "sparse_accuracy": sparse_correct / count if count else 0.0,
                "wins": wins,
                "ties": count - wins - losses,
                "losses": losses,
            }
        )
    return summary


def collect_cases(rows, case_type):
    cases = []
    for row in rows:
        dense = row.get("dense_correct") == "1"
        sparse = row.get("sparse_correct") == "1"
        if case_type == "sparse_only" and not (sparse and not dense):
            continue
        if case_type == "dense_only" and not (dense and not sparse):
            continue
        cases.append(
            {
                "case_type": case_type,
                "category": infer_category(row.get("question", "")),
                "video_id": row.get("video_id", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "baseline_pred": row.get("baseline_pred", ""),
                "method_pred": row.get("method_pred", ""),
                "judge_reason": row.get("judge_reason", ""),
            }
        )
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize LLM judge QA results by heuristic question category.")
    parser.add_argument("--inputs", required=True, help="Comma-separated judged CSV files.")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_summary = []
    all_cases = []
    for item in [value for value in args.inputs.split(",") if value]:
        path = Path(item)
        rows = read_csv(path)
        comparison = path.stem
        summary = summarize(rows, comparison)
        cases = collect_cases(rows, "sparse_only") + collect_cases(rows, "dense_only")
        write_csv(output_dir / f"{comparison}_category_summary.csv", summary)
        write_json(output_dir / f"{comparison}_category_summary.json", summary)
        write_csv(output_dir / f"{comparison}_correctness_cases.csv", cases)
        write_json(output_dir / f"{comparison}_correctness_cases.json", cases)
        all_summary.extend(summary)
        all_cases.extend(cases)

    write_csv(output_dir / "category_summary_all.csv", all_summary)
    write_json(output_dir / "category_summary_all.json", all_summary)
    write_csv(output_dir / "correctness_cases_all.csv", all_cases)
    write_json(output_dir / "correctness_cases_all.json", all_cases)
    print(json.dumps({"summaries": all_summary, "cases": all_cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

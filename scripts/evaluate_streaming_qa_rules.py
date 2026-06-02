import argparse
import csv
import json
from pathlib import Path


def load_json_list(value):
    if value is None:
        return []
    value = str(value).strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def matches_term_group(prediction, group):
    if isinstance(group, str):
        return group.lower() in prediction
    if isinstance(group, list):
        return any(str(term).lower() in prediction for term in group)
    return False


def evaluate_row(row):
    prediction = row["pred_answer"].lower()
    eval_all = load_json_list(row.get("eval_all"))
    eval_any = load_json_list(row.get("eval_any"))
    eval_not = load_json_list(row.get("eval_not"))

    all_ok = all(matches_term_group(prediction, group) for group in eval_all)
    any_ok = True if not eval_any else any(matches_term_group(prediction, group) for group in eval_any)
    not_ok = not any(matches_term_group(prediction, group) for group in eval_not)
    passed = all_ok and any_ok and not_ok
    return {
        **row,
        "rule_pass": int(passed),
        "rule_all_ok": int(all_ok),
        "rule_any_ok": int(any_ok),
        "rule_not_ok": int(not_ok),
    }


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate streaming QA CSV files with per-sample rule metadata.")
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [evaluate_row(row) for row in read_csv(Path(args.pred_path))]
    total = len(rows)
    passed = sum(int(row["rule_pass"]) for row in rows)
    summary = {
        "pred_path": args.pred_path,
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0.0,
        "failures": [
            {
                "video_id": row.get("video_id"),
                "question": row.get("question"),
                "answer": row.get("answer"),
                "pred_answer": row.get("pred_answer"),
                "all_ok": row["rule_all_ok"],
                "any_ok": row["rule_any_ok"],
                "not_ok": row["rule_not_ok"],
            }
            for row in rows
            if not int(row["rule_pass"])
        ],
    }
    write_csv(Path(args.output_csv), rows)
    write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

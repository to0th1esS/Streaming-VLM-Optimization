import argparse
import csv
import json
import re
from pathlib import Path


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def tokenize(text):
    return [token for token in re.findall(r"[a-z0-9]+", str(text).lower()) if token not in STOPWORDS]


def token_f1(prediction, answer):
    pred_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    if not pred_tokens or not answer_tokens:
        return 0.0
    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in answer_tokens:
        count = pred_counts.get(token, 0)
        if count:
            overlap += 1
            pred_counts[token] = count - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


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
    parser = argparse.ArgumentParser(description="Evaluate open-ended QA predictions with token-overlap F1.")
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = read_csv(Path(args.pred_path))
    evaluated = []
    for row in rows:
        score = token_f1(row.get("pred_answer", ""), row.get("answer", ""))
        evaluated.append({**row, "token_f1": score})

    scores = [float(row["token_f1"]) for row in evaluated]
    final = rows[-1] if rows else {}
    semantic_input_tokens = float(final.get("semantic_input_tokens", 0) or 0)
    semantic_written_tokens = float(final.get("semantic_written_tokens", 0) or 0)
    summary = {
        "pred_path": args.pred_path,
        "samples": len(evaluated),
        "mean_token_f1": sum(scores) / len(scores) if scores else 0.0,
        "min_token_f1": min(scores) if scores else 0.0,
        "max_token_f1": max(scores) if scores else 0.0,
        "cumulative_encode_video_sec": float(final.get("cumulative_encode_video_sec", 0) or 0),
        "semantic_input_frames": int(float(final.get("semantic_input_frames", 0) or 0)),
        "semantic_kept_frames": int(float(final.get("semantic_kept_frames", 0) or 0)),
        "semantic_token_reduction": (
            1.0 - semantic_written_tokens / semantic_input_tokens if semantic_input_tokens else 0.0
        ),
    }
    write_csv(Path(args.output_csv), evaluated)
    write_json(Path(args.output_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

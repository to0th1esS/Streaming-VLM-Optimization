import argparse
import json
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def check_annotation(path):
    rows = load_json(path)
    expected = []
    for item in rows:
        video_path = Path(item["video_path"])
        expected.append(
            {
                "video_id": item.get("video_id", ""),
                "video_path": str(video_path).replace("\\", "/"),
                "exists": video_path.exists(),
                "questions": len(item.get("conversations", [])),
            }
        )

    available = [item for item in expected if item["exists"]]
    missing = [item for item in expected if not item["exists"]]
    return {
        "annotation_path": str(path).replace("\\", "/"),
        "videos": len(expected),
        "questions": sum(item["questions"] for item in expected),
        "available_videos": len(available),
        "missing_videos": len(missing),
        "missing_examples": missing[:20],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether streaming QA annotation video assets exist locally.")
    parser.add_argument(
        "--anno-paths",
        nargs="+",
        default=[
            "data/rvs/ego/ego4d_oe.json",
            "data/rvs/movie/movienet_oe.json",
            "data/streaming_qa_hard/bbb_semantic_events_qa.json",
        ],
    )
    parser.add_argument("--output-json", default="results/streaming_asset_check/summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = []
    for path_value in args.anno_paths:
        path = Path(path_value)
        if not path.exists():
            summaries.append(
                {
                    "annotation_path": str(path).replace("\\", "/"),
                    "exists": False,
                    "videos": 0,
                    "questions": 0,
                    "available_videos": 0,
                    "missing_videos": 0,
                    "missing_examples": [],
                }
            )
            continue
        summary = check_annotation(path)
        summary["exists"] = True
        summaries.append(summary)

    write_json(Path(args.output_json), summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_big_buck_bunny_annotation(video_path: str):
    return [
        {
            "video_id": "big_buck_bunny_tiny",
            "video_path": video_path,
            "conversations": [
                {
                    "question": "What kind of character appears in the video?",
                    "answer": "An animated rabbit appears in the video.",
                    "answer_type": "sanity_scene",
                    "start_time": 0,
                    "end_time": 8,
                    "gt_duration": 8,
                },
                {
                    "question": "Is this video animated or real-world footage?",
                    "answer": "It is animated.",
                    "answer_type": "sanity_style",
                    "start_time": 0,
                    "end_time": 12,
                    "gt_duration": 12,
                },
                {
                    "question": "What is the general setting shown in the video?",
                    "answer": "An outdoor animated nature scene.",
                    "answer_type": "sanity_scene",
                    "start_time": 0,
                    "end_time": 16,
                    "gt_duration": 16,
                },
            ],
        }
    ]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a tiny streaming QA sanity annotation.")
    parser.add_argument("--video-path", default="data/turbovit_v1/big_buck_bunny.mp4")
    parser.add_argument("--output", default="data/tiny_streaming_qa/big_buck_bunny_qa.json")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    write_json(output, build_big_buck_bunny_annotation(args.video_path))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

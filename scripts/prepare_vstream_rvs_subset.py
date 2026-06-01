import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


BASE_URL = "https://huggingface.co/datasets/IVGSZ/VStream-QA/resolve/main/vstream-realtime"
FILES = {
    "rvs_ego": {
        "qa": "test_qa_ego4d.json",
        "clips": "rvs_ego.json",
        "output": "data/rvs/ego/ego4d_oe.json",
        "video_dir": "data/rvs/ego/videos",
    },
    "rvs_movie": {
        "qa": "test_qa_movienet.json",
        "clips": "rvs_movie.json",
        "output": "data/rvs/movie/movienet_oe.json",
        "video_dir": "data/rvs/movie/videos",
    },
}


def download(url: str, path: Path, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    print(f"download: {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_clip_index(clips: Iterable[Dict]) -> Dict[str, Dict]:
    return {str(item["video_id"]): item for item in clips}


def normalize_time(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        match = re.search(r"shot_(\d+)", value)
        if match:
            return int(match.group(1))
        try:
            return float(value)
        except ValueError:
            return 0
    return 0


def convert_to_rekv_stream_format(
    qa_items: Iterable[Dict],
    clip_index: Dict[str, Dict],
    video_dir: Path,
    max_videos: int,
    max_questions_per_video: int,
) -> List[Dict]:
    grouped = defaultdict(list)
    for item in qa_items:
        grouped[str(item["video_id"])].append(item)

    rows = []
    for video_id in sorted(grouped.keys()):
        if max_videos > 0 and len(rows) >= max_videos:
            break
        clip_meta = clip_index.get(video_id, {})
        conversations = []
        for qa in sorted(grouped[video_id], key=lambda row: (normalize_time(row.get("end_time", 0)), str(row.get("id", "")))):
            if max_questions_per_video > 0 and len(conversations) >= max_questions_per_video:
                break
            conversations.append(
                {
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "answer_type": qa.get("answer_type", ""),
                    "start_time": normalize_time(qa.get("start_time", 0)),
                    "end_time": normalize_time(qa.get("end_time", qa.get("duration", 0))),
                    "gt_duration": qa.get("gt_duration", 0),
                }
            )
        if not conversations:
            continue
        rows.append(
            {
                "video_id": video_id,
                "video_path": str(video_dir / f"{video_id}.mp4").replace("\\", "/"),
                "original_video": clip_meta.get("original_video", clip_meta.get("movie_id", "")),
                "clip_start_time": clip_meta.get("start_time", clip_meta.get("start_shot", 0)),
                "clip_end_time": clip_meta.get("end_time", clip_meta.get("end_shot", 0)),
                "conversations": conversations,
            }
        )
    return rows


def summarize(rows: List[Dict], video_dir: Path) -> Dict:
    expected_paths = [Path(item["video_path"]) for item in rows]
    missing = [str(path).replace("\\", "/") for path in expected_paths if not path.exists()]
    return {
        "videos": len(rows),
        "questions": sum(len(item["conversations"]) for item in rows),
        "expected_video_dir": str(video_dir).replace("\\", "/"),
        "available_video_files": len(expected_paths) - len(missing),
        "missing_video_files": len(missing),
        "missing_video_examples": missing[:10],
    }


def prepare_dataset(name: str, raw_dir: Path, overwrite: bool, max_videos: int, max_questions_per_video: int) -> Dict:
    config = FILES[name]
    qa_path = raw_dir / name / config["qa"]
    clips_path = raw_dir / name / config["clips"]
    download(f"{BASE_URL}/{config['qa']}", qa_path, overwrite=overwrite)
    download(f"{BASE_URL}/{config['clips']}", clips_path, overwrite=overwrite)

    output_path = Path(config["output"])
    video_dir = Path(config["video_dir"])
    rows = convert_to_rekv_stream_format(
        load_json(qa_path),
        build_clip_index(load_json(clips_path)),
        video_dir,
        max_videos=max_videos,
        max_questions_per_video=max_questions_per_video,
    )
    write_json(output_path, rows)
    summary = summarize(rows, video_dir)
    summary.update(
        {
            "dataset": name,
            "annotation_path": str(output_path).replace("\\", "/"),
            "raw_qa_path": str(qa_path).replace("\\", "/"),
            "raw_clip_path": str(clips_path).replace("\\", "/"),
        }
    )
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare small VStream-QA RVS subsets for ReKV streaming VQA.")
    parser.add_argument("--dataset", choices=["rvs_ego", "rvs_movie", "all"], default="rvs_ego")
    parser.add_argument("--raw-dir", default="data/vstream_qa/raw")
    parser.add_argument("--max-videos", type=int, default=8)
    parser.add_argument("--max-questions-per-video", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = ["rvs_ego", "rvs_movie"] if args.dataset == "all" else [args.dataset]
    summaries = []
    for dataset in datasets:
        summaries.append(
            prepare_dataset(
                dataset,
                raw_dir=Path(args.raw_dir),
                overwrite=args.overwrite,
                max_videos=args.max_videos,
                max_questions_per_video=args.max_questions_per_video,
            )
        )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

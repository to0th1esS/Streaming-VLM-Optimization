import argparse
import json
import os
import re
from pathlib import Path


DATASETS = {
    "rvs_ego": {
        "annotation": "data/rvs/ego/ego4d_oe.json",
        "repo_video_dir": "data/rvs/ego/videos",
        "frames_root": "frames/rvs_ego",
    },
    "rvs_movie": {
        "annotation": "data/rvs/movie/movienet_oe.json",
        "repo_video_dir": "data/rvs/movie/videos",
        "frames_root": "frames/rvs_movie",
    },
}


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_frame_dirs(frames_root, video_id):
    candidates = [
        frames_root / video_id,
        frames_root / f"{video_id}.mp4",
        frames_root / "frames" / video_id,
        frames_root / "videos" / video_id,
        frames_root / "movienet_frames" / video_id,
        frames_root / "ego4d_frames" / video_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return [candidate]

    movie_match = re.match(r"(.+)_([0-9]{4})_([0-9]{4})$", video_id)
    if movie_match:
        movie_id, start_value, end_value = movie_match.groups()
        start_idx = int(start_value)
        end_idx = int(end_value)
        segment_dirs = []
        for path in frames_root.rglob(f"{movie_id}_*"):
            if not path.is_dir():
                continue
            segment_match = re.match(rf"{re.escape(movie_id)}_([0-9]{{4}})_([0-9]{{4}})$", path.name)
            if not segment_match:
                continue
            segment_start = int(segment_match.group(1))
            segment_end = int(segment_match.group(2))
            if start_idx <= segment_start <= end_idx or start_idx <= segment_end <= end_idx:
                segment_dirs.append(path)
        if segment_dirs:
            return sorted(segment_dirs)

    matches = [path for path in frames_root.rglob(video_id) if path.is_dir()]
    return matches


def link_dataset(name, asset_root, repo_root):
    config = DATASETS[name]
    annotation_path = repo_root / config["annotation"]
    repo_video_dir = repo_root / config["repo_video_dir"]
    frames_root = asset_root / config["frames_root"]
    repo_video_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(annotation_path)
    linked = []
    missing = []
    for row in rows:
        video_id = row["video_id"]
        sources = find_frame_dirs(frames_root, video_id)
        target = repo_video_dir / f"{video_id}.mp4"
        if not sources:
            missing.append({"video_id": video_id, "target": str(target), "reason": "source_frame_dir_not_found"})
            continue
        if target.exists() or target.is_symlink():
            if len(sources) == 1 and target.is_symlink() and Path(os.readlink(target)).resolve() == sources[0].resolve():
                linked.append({"video_id": video_id, "sources": [str(sources[0])], "target": str(target), "status": "already_linked"})
                continue
            if target.is_dir():
                linked.append({"video_id": video_id, "sources": [str(source) for source in sources], "target": str(target), "status": "target_dir_exists"})
                continue
            target.unlink()
        if len(sources) == 1:
            os.symlink(sources[0], target, target_is_directory=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
            for source in sources:
                link_path = target / source.name
                if not link_path.exists():
                    os.symlink(source, link_path, target_is_directory=True)
        linked.append({"video_id": video_id, "sources": [str(source) for source in sources], "target": str(target), "status": "linked"})

    return {
        "dataset": name,
        "annotation": str(annotation_path),
        "frames_root": str(frames_root),
        "linked": len(linked),
        "missing": len(missing),
        "linked_examples": linked[:10],
        "missing_examples": missing[:10],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Link extracted VStream frame directories into repo annotation paths.")
    parser.add_argument("--asset-root", default="/home/mllm/datasets/vstream_qa")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--datasets", default="rvs_ego,rvs_movie")
    parser.add_argument("--output-json", default="results/streaming_asset_check/link_summary.json")
    return parser.parse_args()


def main():
    args = parse_args()
    asset_root = Path(args.asset_root)
    repo_root = Path(args.repo_root)
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summaries = [link_dataset(name, asset_root, repo_root) for name in selected]
    output_path = repo_root / args.output_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

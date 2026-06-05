import argparse
import json
from pathlib import Path


EXPECTED_ARCHIVES = {
    "source_video_parts": [
        f"src_videos.tar.parta{letter}" for letter in "abcde"
    ],
    "chunked_video_parts": [
        f"chunked_videos.tar.parta{letter}" for letter in "abcdefghijklmno"
    ],
}


def inspect_file(path):
    path = Path(path)
    return {
        "path": str(path).replace("\\", "/"),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }


def inspect_archives(root):
    result = {}
    for group, file_names in EXPECTED_ARCHIVES.items():
        files = [inspect_file(root / file_name) for file_name in file_names]
        result[group] = {
            "expected_parts": len(files),
            "available_parts": sum(int(item["exists"]) for item in files),
            "available_bytes": sum(item["bytes"] for item in files),
            "missing_parts": [
                Path(item["path"]).name for item in files if not item["exists"]
            ],
            "files": files,
        }
    return result


def inspect_subset(subset_json):
    if not subset_json:
        return None
    path = Path(subset_json)
    if not path.is_file():
        return {
            "path": str(path).replace("\\", "/"),
            "exists": False,
        }

    rows = json.loads(path.read_text(encoding="utf-8"))
    video_paths = [Path(row["video_path"]) for row in rows]
    missing = [str(path).replace("\\", "/") for path in video_paths if not path.is_file()]
    return {
        "path": str(path).replace("\\", "/"),
        "exists": True,
        "queries": len(rows),
        "available_videos": len(video_paths) - len(missing),
        "missing_videos": len(missing),
        "missing_video_examples": missing[:20],
    }


def inspect_annotation(root):
    candidates = [
        root / "ovo_bench_new.json",
        root / "data" / "ovo_bench_new.json",
    ]
    files = [inspect_file(path) for path in candidates]
    return {
        "available": any(item["exists"] for item in files),
        "candidates": files,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check OVO-Bench archive parts and optional converted subset videos."
    )
    parser.add_argument(
        "--root",
        default="/home/mllm/datasets/ovo_bench",
    )
    parser.add_argument("--subset-json", default="")
    parser.add_argument(
        "--output-json",
        default="results/ovo_bench/assets_summary.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    summary = {
        "root": str(root).replace("\\", "/"),
        "root_exists": root.is_dir(),
        "annotation": inspect_annotation(root),
        "archives": inspect_archives(root),
        "extracted_chunked_dir": {
            "path": str(root / "chunked_videos").replace("\\", "/"),
            "exists": (root / "chunked_videos").is_dir(),
        },
        "subset": inspect_subset(args.subset_json),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

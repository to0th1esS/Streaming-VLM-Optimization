import argparse
import json
from pathlib import Path


TARGETS = {
    "vstream_rvs": {
        "description": "VStream-RVS（当前已接入的流式视频 QA 子集）",
        "local_paths": [
            "data/rvs/ego/ego4d_oe.json",
            "data/rvs/movie/movienet_oe.json",
            "data/vstream_qa/raw/rvs_ego/test_qa_ego4d.json",
            "data/vstream_qa/raw/rvs_movie/test_qa_movienet.json",
        ],
    },
    "streamingbench": {
        "description": "StreamingBench（流式视频理解基准）",
        "repo": "mjuicem/StreamingBench",
        "local_paths": [
            "streamingbench",
            "StreamingBench",
            "data/streamingbench",
            "data/StreamingBench",
        ],
    },
    "ovo_bench": {
        "description": "OVO-Bench（在线视频理解基准）",
        "repo": "JoeLeelyf/OVO-Bench",
        "local_paths": [
            "ovo_bench",
            "OVO-Bench",
            "data/ovo_bench",
            "data/OVO-Bench",
        ],
    },
    "ovbench": {
        "description": "OVBench（在线视频理解基准，若后续区分 OVO-Bench 需单独确认协议）",
        "local_paths": [
            "data/ovbench",
            "data/OVBench",
        ],
    },
}


def inspect_path(path: Path):
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    if not exists:
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "type": "file",
            "bytes": path.stat().st_size,
        }
    files = [item for item in path.rglob("*") if item.is_file()]
    total_bytes = sum(item.stat().st_size for item in files)
    return {
        "path": str(path),
        "exists": True,
        "type": "directory" if is_dir else "other",
        "files": len(files),
        "bytes": total_bytes,
    }


def main():
    parser = argparse.ArgumentParser(description="Check local entry points for streaming video benchmarks.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--extra-roots", default="/home/mllm/datasets", help="Comma-separated extra roots to inspect.")
    parser.add_argument("--output-json", default="results/streaming_benchmark_targets/summary.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    extra_roots = [Path(item).resolve() for item in args.extra_roots.split(",") if item.strip()]
    roots = [repo_root] + extra_roots

    summary = {}
    for name, config in TARGETS.items():
        checks = []
        for root in roots:
            for rel in config.get("local_paths", []):
                checks.append(inspect_path(root / rel))
        summary[name] = {
            "description": config["description"],
            "repo": config.get("repo", ""),
            "checks": checks,
            "available": any(item.get("exists") for item in checks),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

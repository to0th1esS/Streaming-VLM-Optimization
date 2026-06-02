import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "IVGSZ/VStream-QA"
REPO_TYPE = "dataset"
REALTIME_FILES = {
    "metadata": [
        "vstream-realtime/rvs_ego.json",
        "vstream-realtime/test_qa_ego4d.json",
        "vstream-realtime/rvs_movie.json",
        "vstream-realtime/test_qa_movienet.json",
    ],
    "rvs_ego": [
        "vstream-realtime/ego4d_frames_online.partaa",
        "vstream-realtime/ego4d_frames_online.partab",
        "vstream-realtime/ego4d_frames_online.partac",
    ],
    "rvs_movie": [
        "vstream-realtime/movienet_frames_online.zip",
    ],
}


def run(cmd, cwd=None):
    print("exec:", " ".join(str(item) for item in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def download_file(filename, target_root, endpoint):
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=filename,
        local_dir=str(target_root),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return Path(local_path)


def extract_movie_zip(target_root):
    archive = target_root / "vstream-realtime" / "movienet_frames_online.zip"
    extract_dir = target_root / "frames" / "rvs_movie"
    if not archive.exists():
        return {"archive": str(archive), "status": "missing"}
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extract_complete"
    if marker.exists():
        return {"archive": str(archive), "extract_dir": str(extract_dir), "status": "already_extracted"}
    run(["unzip", "-q", "-o", str(archive), "-d", str(extract_dir)])
    marker.write_text("ok\n", encoding="utf-8")
    return {"archive": str(archive), "extract_dir": str(extract_dir), "status": "extracted"}


def extract_ego_parts(target_root):
    archive_dir = target_root / "vstream-realtime"
    parts = [archive_dir / Path(name).name for name in REALTIME_FILES["rvs_ego"]]
    missing = [str(path) for path in parts if not path.exists()]
    if missing:
        return {"parts": [str(path) for path in parts], "status": "missing_parts", "missing": missing}

    combined = archive_dir / "ego4d_frames_online.zip"
    extract_dir = target_root / "frames" / "rvs_ego"
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extract_complete"
    if marker.exists():
        return {"archive": str(combined), "extract_dir": str(extract_dir), "status": "already_extracted"}

    if not combined.exists():
        with combined.open("wb") as output:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, output)
    run(["unzip", "-q", "-o", str(combined), "-d", str(extract_dir)])
    marker.write_text("ok\n", encoding="utf-8")
    return {"archive": str(combined), "extract_dir": str(extract_dir), "status": "extracted"}


def parse_args():
    parser = argparse.ArgumentParser(description="Download VStream-QA realtime assets to a remote data directory.")
    parser.add_argument("--target-root", default="/home/mllm/datasets/vstream_qa")
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument(
        "--datasets",
        default="metadata,rvs_movie,rvs_ego",
        help="Comma-separated subset from metadata,rvs_movie,rvs_ego.",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    target_root = Path(args.target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    requested = [item.strip() for item in args.datasets.split(",") if item.strip()]

    downloaded = []
    for dataset in requested:
        for filename in REALTIME_FILES[dataset]:
            path = download_file(filename, target_root, args.endpoint)
            downloaded.append({"dataset": dataset, "filename": filename, "path": str(path), "bytes": path.stat().st_size})

    extraction = []
    if args.extract:
        if "rvs_movie" in requested:
            extraction.append(extract_movie_zip(target_root))
        if "rvs_ego" in requested:
            extraction.append(extract_ego_parts(target_root))

    manifest_path = Path(args.manifest) if args.manifest else target_root / "download_manifest.json"
    manifest = {
        "repo_id": REPO_ID,
        "endpoint": args.endpoint,
        "target_root": str(target_root),
        "requested": requested,
        "downloaded": downloaded,
        "extraction": extraction,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import subprocess
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_ffmpeg(ffmpeg, source_path, output_path, end_time):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    copy_command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(end_time),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(
        copy_command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size:
        return "stream_copy"

    output_path.unlink(missing_ok=True)
    encode_command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(end_time),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-an",
        str(output_path),
    ]
    subprocess.run(encode_command, check=True)
    return "reencode"


def prepare_chunks(subset, annotations, source_dir, ffmpeg, overwrite=False):
    by_id = {int(item["id"]): item for item in annotations}
    results = []
    for row in subset:
        official_id = int(row["official_id"])
        source_path = source_dir / by_id[official_id]["video"]
        output_path = Path(row["video_path"])
        end_time = float(row["conversations"][0]["end_time"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source video: {source_path}")

        if output_path.is_file() and output_path.stat().st_size and not overwrite:
            mode = "existing"
        else:
            mode = run_ffmpeg(ffmpeg, source_path, output_path, end_time)
        results.append(
            {
                "video_id": row["video_id"],
                "official_id": official_id,
                "source_path": str(source_path),
                "output_path": str(output_path),
                "end_time": end_time,
                "mode": mode,
                "output_bytes": output_path.stat().st_size,
            }
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create only the OVO-Bench query clips required by a converted subset."
    )
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--subset-json", required=True)
    parser.add_argument(
        "--source-dir",
        default="/home/mllm/datasets/ovo_bench/src_videos",
    )
    parser.add_argument(
        "--ffmpeg",
        default="/root/miniconda3/envs/ragged_test/bin/ffmpeg",
    )
    parser.add_argument(
        "--output-json",
        default="results/ovo_bench/chunk_source_summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results = prepare_chunks(
        subset=load_json(args.subset_json),
        annotations=load_json(args.source_json),
        source_dir=Path(args.source_dir),
        ffmpeg=args.ffmpeg,
        overwrite=args.overwrite,
    )
    summary = {
        "clips": len(results),
        "stream_copy": sum(item["mode"] == "stream_copy" for item in results),
        "reencoded": sum(item["mode"] == "reencode" for item in results),
        "existing": sum(item["mode"] == "existing" for item in results),
        "results": results,
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

#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OVO_ROOT="${OVO_ROOT:-/home/mllm/datasets/ovo_bench}"
SUBSET_JSON="${SUBSET_JSON:-data/ovo_bench/ovo_rekv_subset.json}"
OUTPUT_JSON="${OUTPUT_JSON:-results/ovo_bench/extraction_assets.json}"
FILE_LIST="${FILE_LIST:-${OVO_ROOT}/subset_video_files.txt}"

parts=()
for suffix in a b c d e f g h i j k l m n o; do
  part="${OVO_ROOT}/chunked_videos.tar.parta${suffix}"
  if [[ ! -s "$part" ]]; then
    echo "Missing required archive part: $part" >&2
    exit 1
  fi
  parts+=("$part")
done

"$PYTHON_BIN" - "$SUBSET_JSON" "$FILE_LIST" <<'PY'
import json
import sys
from pathlib import Path

subset_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
rows = json.loads(subset_path.read_text(encoding="utf-8"))
names = sorted({Path(row["video_path"]).name for row in rows})
output_path.write_text("\n".join(names) + "\n", encoding="utf-8", newline="\n")
print(f"Prepared {len(names)} selective extraction targets: {output_path}")
PY

mkdir -p "$OVO_ROOT"
cat "${parts[@]}" | tar \
  --extract \
  --file - \
  --directory "$OVO_ROOT" \
  --wildcards \
  --no-anchored \
  --files-from "$FILE_LIST"

"$PYTHON_BIN" scripts/check_ovo_bench_assets.py \
  --root "$OVO_ROOT" \
  --subset-json "$SUBSET_JSON" \
  --output-json "$OUTPUT_JSON"

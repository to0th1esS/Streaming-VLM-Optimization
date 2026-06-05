#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
OVO_ROOT="${OVO_ROOT:-/home/mllm/datasets/ovo_bench}"
SOURCE_JSON="${SOURCE_JSON:-${OVO_ROOT}/ovo_bench_new.json}"
SUBSET_JSON="${SUBSET_JSON:-data/ovo_bench/ovo_rekv_subset.json}"
FILE_LIST="${FILE_LIST:-${OVO_ROOT}/subset_source_video_files.txt}"

parts=()
for suffix in a b c d e; do
  part="${OVO_ROOT}/src_videos.tar.parta${suffix}"
  if [[ ! -s "$part" ]]; then
    echo "Missing required source archive part: $part" >&2
    exit 1
  fi
  parts+=("$part")
done

"$PYTHON_BIN" - "$SOURCE_JSON" "$SUBSET_JSON" "$FILE_LIST" <<'PY'
import json
import sys
from pathlib import Path

source_json = Path(sys.argv[1])
subset_json = Path(sys.argv[2])
output_path = Path(sys.argv[3])

annotations = json.loads(source_json.read_text(encoding="utf-8"))
subset = json.loads(subset_json.read_text(encoding="utf-8"))
by_id = {int(item["id"]): item for item in annotations}
paths = sorted(
    {
        f"./src_videos/{by_id[int(row['official_id'])]['video']}"
        for row in subset
    }
)
output_path.write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
print(f"Prepared {len(paths)} source extraction targets: {output_path}")
PY

cat "${parts[@]}" | tar \
  --extract \
  --file - \
  --directory "$OVO_ROOT" \
  --files-from "$FILE_LIST"

echo "Extracted source videos:"
while IFS= read -r relative_path; do
  test -s "${OVO_ROOT}/${relative_path#./}"
  ls -lh "${OVO_ROOT}/${relative_path#./}"
done < "$FILE_LIST"

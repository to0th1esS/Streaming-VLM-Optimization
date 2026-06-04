#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-llava_ov_7b}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-64}"
REPEATS="${REPEATS:-1}"
OUT_ROOT="${OUT_ROOT:-results/fps_dataset_sensitivity_20260604}"
FPS_LIST="${FPS_LIST:-0.5,1.0}"
DATASETS="${DATASETS:-rvs_ego,rvs_movie}"
RUN_JUDGE="${RUN_JUDGE:-false}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/mllm/models/Qwen2.5-VL-7B-Instruct}"

# Time-normalized defaults inherited from the 0.2 fps pilot:
# semantic refresh 64 frames at 0.2 fps ~= 320 seconds.
# periodic interval 13 frames at 0.2 fps ~= 65 seconds.
SEMANTIC_REFRESH_SECONDS="${SEMANTIC_REFRESH_SECONDS:-320}"
PERIODIC_INTERVAL_SECONDS="${PERIODIC_INTERVAL_SECONDS:-65}"
SEMANTIC_THRESHOLD="${SEMANTIC_THRESHOLD:-0.3}"
RECENCY_KEEP_FRAMES="${RECENCY_KEEP_FRAMES:-4}"
LATEST_RETRIEVAL_BLOCKS="${LATEST_RETRIEVAL_BLOCKS:-4}"

dataset_anno() {
  case "$1" in
    rvs_ego) echo "data/rvs/ego/ego4d_oe.json" ;;
    rvs_movie) echo "data/rvs/movie/movienet_oe.json" ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

fps_tag() {
  echo "$1" | sed 's/\./p/g'
}

round_interval() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
fps = float(sys.argv[1])
seconds = float(sys.argv[2])
print(max(1, int(round(fps * seconds))))
PY
}

run_sweep() {
  local dataset="$1"
  local fps="$2"
  local method="$3"
  local output_dir="$4"
  local refresh="$5"
  local threshold="$6"
  local recency="$7"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/run_semantic_stream_sweep.py \
    --model "$MODEL" \
    --anno-path "$(dataset_anno "$dataset")" \
    --output-dir "$output_dir" \
    --sample-fps "$fps" \
    --retrieve-size "$RETRIEVE_SIZE" \
    --refresh-intervals "$refresh" \
    --thresholds "$threshold" \
    --compute-gates true \
    --repeats "$REPEATS" \
    --debug false \
    --semantic-recency-keep-frames "$recency" \
    --enable-query-aware-retrieval true \
    --query-retrieval-policy always_recent \
    --latest-retrieval-blocks "$LATEST_RETRIEVAL_BLOCKS"

  echo "finished $dataset fps=$fps method=$method refresh=$refresh threshold=$threshold recency=$recency"
}

postprocess_method() {
  local method_dir="$1"
  local dense_dir="$2"

  mkdir -p "$method_dir/analysis"
  for ((rep = 0; rep < REPEATS; rep++)); do
    local method_csv
    method_csv="$(find "$method_dir/runs" -path "*_rep${rep}/1_0.csv" | head -1)"
    local dense_csv
    dense_csv="$(find "$dense_dir/runs" -path "*_rep${rep}/1_0.csv" | head -1)"
    if [[ -z "$method_csv" || -z "$dense_csv" ]]; then
      echo "Missing CSV for rep=$rep method=$method_dir dense=$dense_dir" >&2
      exit 1
    fi
    mkdir -p "$method_dir/analysis/rep${rep}"
    "$PYTHON_BIN" scripts/evaluate_open_qa_overlap.py \
      --pred-path "$method_csv" \
      --output-csv "$method_dir/analysis/rep${rep}/overlap.csv" \
      --output-json "$method_dir/analysis/rep${rep}/overlap.json" >/dev/null
    "$PYTHON_BIN" scripts/compare_qa_predictions.py \
      --baseline-path "$dense_csv" \
      --method-path "$method_csv" \
      --output-csv "$method_dir/analysis/rep${rep}/dense_vs_method_compare.csv" \
      --output-json "$method_dir/analysis/rep${rep}/dense_vs_method_compare.json" >/dev/null
  done
}

maybe_judge_rep0() {
  local method_dir="$1"
  if [[ "$RUN_JUDGE" != "true" ]]; then
    return
  fi
  local compare_csv="$method_dir/analysis/rep0/dense_vs_method_compare.csv"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/judge_qa_pairwise.py \
    --input-csv "$compare_csv" \
    --output-csv "$method_dir/analysis/rep0/dense_vs_method_compare_judge_qwen25vl7b.csv" \
    --output-json "$method_dir/analysis/rep0/dense_vs_method_compare_judge_qwen25vl7b.json" \
    --judge-model "$JUDGE_MODEL" \
    --model-family qwen2_5_vl \
    --device cuda \
    --max-new-tokens 64
}

summarize_all() {
  OUT_ROOT="$OUT_ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
rows = []
for method_dir in sorted(out_root.glob("*/*/*")):
    if not method_dir.is_dir():
        continue
    parts = method_dir.relative_to(out_root).parts
    if len(parts) != 3:
        continue
    dataset, fps_tag, method = parts
    summary_path = method_dir / "summary.json"
    summary_rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for rep_dir in sorted((method_dir / "analysis").glob("rep*")):
        overlap_path = rep_dir / "overlap.json"
        compare_path = rep_dir / "dense_vs_method_compare.json"
        if not overlap_path.exists() or not compare_path.exists():
            continue
        overlap = json.loads(overlap_path.read_text())
        compare = json.loads(compare_path.read_text())
        run_summary = summary_rows[int(rep_dir.name.replace("rep", ""))] if summary_rows else {}
        rows.append({
            "dataset": dataset,
            "fps": fps_tag.replace("fps", "").replace("p", "."),
            "method": method,
            "repeat": rep_dir.name.replace("rep", ""),
            "refresh_interval": run_summary.get("refresh_interval", ""),
            "skip_threshold": run_summary.get("skip_threshold", ""),
            "recency_keep_frames": run_summary.get("semantic_recency_keep_frames", ""),
            "kept_frames": overlap["semantic_kept_frames"],
            "input_frames": overlap["semantic_input_frames"],
            "token_reduction": overlap["semantic_token_reduction"],
            "encode_sec": overlap["total_encode_video_sec"],
            "speedup_vs_dense": compare["speedup"],
            "mean_token_f1": overlap["mean_token_f1"],
            "wins": compare["wins"],
            "ties": compare["ties"],
            "losses": compare["losses"],
        })

summary_csv = out_root / "summary_all.csv"
summary_json = out_root / "summary_all.json"
if rows:
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

main() {
  mkdir -p "$OUT_ROOT"
  IFS=',' read -ra fps_values <<< "$FPS_LIST"
  IFS=',' read -ra dataset_values <<< "$DATASETS"

  for dataset in "${dataset_values[@]}"; do
    for fps in "${fps_values[@]}"; do
      local tag
      tag="fps$(fps_tag "$fps")"
      local dense_dir="$OUT_ROOT/$dataset/$tag/dense"
      local semantic_dir="$OUT_ROOT/$dataset/$tag/semantic_time_norm"
      local periodic_dir="$OUT_ROOT/$dataset/$tag/periodic_time_norm"
      local semantic_refresh
      semantic_refresh="$(round_interval "$fps" "$SEMANTIC_REFRESH_SECONDS")"
      local periodic_interval
      periodic_interval="$(round_interval "$fps" "$PERIODIC_INTERVAL_SECONDS")"

      run_sweep "$dataset" "$fps" dense "$dense_dir" 1 0 0
      run_sweep "$dataset" "$fps" semantic "$semantic_dir" "$semantic_refresh" "$SEMANTIC_THRESHOLD" "$RECENCY_KEEP_FRAMES"
      run_sweep "$dataset" "$fps" periodic "$periodic_dir" "$periodic_interval" 999 "$RECENCY_KEEP_FRAMES"

      postprocess_method "$dense_dir" "$dense_dir"
      postprocess_method "$semantic_dir" "$dense_dir"
      postprocess_method "$periodic_dir" "$dense_dir"
      maybe_judge_rep0 "$semantic_dir"
      maybe_judge_rep0 "$periodic_dir"
      summarize_all
    done
  done
}

main "$@"

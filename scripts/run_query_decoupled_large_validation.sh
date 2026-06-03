#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-llava_ov_7b}"
SAMPLE_FPS="${SAMPLE_FPS:-0.2}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-64}"
REPEATS="${REPEATS:-3}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/mllm/models/Qwen2.5-VL-7B-Instruct}"
RUN_JUDGE="${RUN_JUDGE:-false}"
OUT_ROOT="${OUT_ROOT:-results/large_validation_query_decoupled_20260603}"

DENSE_EGO="results/rvs_ego_repeats_20260602/dense_r1_t0"
DENSE_MOVIE="results/rvs_movie_repeats_20260602/dense_r1_t0"

run_sweep() {
  local dataset_name="$1"
  local anno_path="$2"
  local output_dir="$3"
  local refresh="$4"
  local threshold="$5"
  local policy="$6"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/run_semantic_stream_sweep.py \
    --model "$MODEL" \
    --anno-path "$anno_path" \
    --output-dir "$output_dir" \
    --sample-fps "$SAMPLE_FPS" \
    --retrieve-size "$RETRIEVE_SIZE" \
    --refresh-intervals "$refresh" \
    --thresholds "$threshold" \
    --compute-gates true \
    --repeats "$REPEATS" \
    --debug false \
    --semantic-recency-keep-frames 4 \
    --enable-query-aware-retrieval true \
    --query-retrieval-policy "$policy" \
    --latest-retrieval-blocks 4
}

postprocess_run() {
  local method_dir="$1"
  local dense_dir="$2"
  local compare_name="$3"

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
      --output-csv "$method_dir/analysis/rep${rep}/${compare_name}.csv" \
      --output-json "$method_dir/analysis/rep${rep}/${compare_name}.json" >/dev/null
  done
}

summarize_all() {
  "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

out_root = Path(os.environ.get("OUT_ROOT", "results/large_validation_query_decoupled_20260603"))
rows = []
for method_dir in sorted(out_root.glob("*/*")):
    if not method_dir.is_dir():
        continue
    for rep_dir in sorted((method_dir / "analysis").glob("rep*")):
        overlap_path = rep_dir / "overlap.json"
        compare_paths = list(rep_dir.glob("*compare*.json"))
        if not overlap_path.exists() or not compare_paths:
            continue
        overlap = json.loads(overlap_path.read_text())
        compare = json.loads(compare_paths[0].read_text())
        rows.append({
            "dataset": method_dir.parent.name,
            "method": method_dir.name,
            "repeat": rep_dir.name.replace("rep", ""),
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
summary_csv.parent.mkdir(parents=True, exist_ok=True)
if rows:
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
}

maybe_judge_rep0() {
  local method_dir="$1"
  local compare_name="$2"
  if [[ "$RUN_JUDGE" != "true" ]]; then
    return
  fi
  local compare_csv="$method_dir/analysis/rep0/${compare_name}.csv"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/judge_qa_pairwise.py \
    --input-csv "$compare_csv" \
    --output-csv "$method_dir/analysis/rep0/${compare_name}_judge_qwen25vl7b.csv" \
    --output-json "$method_dir/analysis/rep0/${compare_name}_judge_qwen25vl7b.json" \
    --judge-model "$JUDGE_MODEL" \
    --model-family qwen2_5_vl \
    --device cuda \
    --max-new-tokens 64
}

main() {
  mkdir -p "$OUT_ROOT"

  local ego_semantic="$OUT_ROOT/rvs_ego/semantic_r64_recency4_always_recent_qrb4"
  local ego_periodic="$OUT_ROOT/rvs_ego/periodic_r13_recency4_always_recent_qrb4"
  local movie_semantic="$OUT_ROOT/rvs_movie/semantic_r64_recency4_always_recent_qrb4"
  local movie_periodic="$OUT_ROOT/rvs_movie/periodic_r13_recency4_always_recent_qrb4"

  run_sweep rvs_ego data/rvs/ego/ego4d_oe.json "$ego_semantic" 64 0.3 always_recent
  run_sweep rvs_ego data/rvs/ego/ego4d_oe.json "$ego_periodic" 13 999 always_recent
  run_sweep rvs_movie data/rvs/movie/movienet_oe.json "$movie_semantic" 64 0.3 always_recent
  run_sweep rvs_movie data/rvs/movie/movienet_oe.json "$movie_periodic" 13 999 always_recent

  postprocess_run "$ego_semantic" "$DENSE_EGO" dense_vs_method_compare
  postprocess_run "$ego_periodic" "$DENSE_EGO" dense_vs_method_compare
  postprocess_run "$movie_semantic" "$DENSE_MOVIE" dense_vs_method_compare
  postprocess_run "$movie_periodic" "$DENSE_MOVIE" dense_vs_method_compare

  maybe_judge_rep0 "$ego_semantic" dense_vs_method_compare
  maybe_judge_rep0 "$ego_periodic" dense_vs_method_compare
  maybe_judge_rep0 "$movie_semantic" dense_vs_method_compare
  maybe_judge_rep0 "$movie_periodic" dense_vs_method_compare

  summarize_all
}

main "$@"

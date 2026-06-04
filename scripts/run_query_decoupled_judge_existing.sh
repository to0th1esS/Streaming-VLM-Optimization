#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-/home/mllm/models/Qwen2.5-VL-7B-Instruct}"
OUT_ROOT="${OUT_ROOT:-results/large_validation_query_decoupled_20260603}"
REPEAT_IDX="${REPEAT_IDX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
FORCE="${FORCE:-false}"

run_judge_for_method() {
  local method_dir="$1"
  local rep_dir="$method_dir/analysis/rep${REPEAT_IDX}"
  local compare_csv="$rep_dir/dense_vs_method_compare.csv"
  local judge_csv="$rep_dir/dense_vs_method_compare_judge_qwen25vl7b.csv"
  local judge_json="$rep_dir/dense_vs_method_compare_judge_qwen25vl7b.json"

  if [[ ! -f "$compare_csv" ]]; then
    echo "missing compare csv: $compare_csv" >&2
    return 1
  fi

  if [[ "$FORCE" != "true" && -f "$judge_json" && -f "$rep_dir/category_summary/category_summary_all.json" ]]; then
    echo "skip existing judge: $rep_dir"
    return
  fi

  echo "judge: $compare_csv"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" scripts/judge_qa_pairwise.py \
    --input-csv "$compare_csv" \
    --output-csv "$judge_csv" \
    --output-json "$judge_json" \
    --judge-model "$JUDGE_MODEL" \
    --model-family qwen2_5_vl \
    --device cuda \
    --max-new-tokens "$MAX_NEW_TOKENS"

  "$PYTHON_BIN" scripts/summarize_judge_categories.py \
    --inputs "$judge_csv" \
    --output-dir "$rep_dir/category_summary"
}

main() {
  local methods=(
    "$OUT_ROOT/rvs_ego/semantic_r64_recency4_always_recent_qrb4"
    "$OUT_ROOT/rvs_ego/periodic_r13_recency4_always_recent_qrb4"
    "$OUT_ROOT/rvs_movie/semantic_r64_recency4_always_recent_qrb4"
    "$OUT_ROOT/rvs_movie/periodic_r13_recency4_always_recent_qrb4"
  )

  for method_dir in "${methods[@]}"; do
    run_judge_for_method "$method_dir"
  done
}

main "$@"

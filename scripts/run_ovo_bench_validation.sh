#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-llava_ov_7b}"
SAMPLE_FPS="${SAMPLE_FPS:-1.0}"
N_LOCAL="${N_LOCAL:-15000}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-64}"
OVO_ROOT="${OVO_ROOT:-/home/mllm/datasets/ovo_bench}"
SOURCE_JSON="${SOURCE_JSON:-${OVO_ROOT}/ovo_bench_new.json}"
CHUNKED_DIR="${CHUNKED_DIR:-${OVO_ROOT}/chunked_videos}"
SUBSET_JSON="${SUBSET_JSON:-data/ovo_bench/ovo_rekv_subset.json}"
OUT_ROOT="${OUT_ROOT:-results/ovo_bench/validation}"
TASKS="${TASKS:-EPM,ASI,HLD,OCR,ACR,ATR,STU,FPD,OJR,REC,SSR,CRR}"
MAX_SOURCE_ITEMS_PER_TASK="${MAX_SOURCE_ITEMS_PER_TASK:-2}"
MAX_QUERIES_PER_SOURCE="${MAX_QUERIES_PER_SOURCE:-2}"
SOURCE_SELECTION="${SOURCE_SELECTION:-head}"
QUERY_SELECTION="${QUERY_SELECTION:-head}"
BUDGET_WINDOW="${BUDGET_WINDOW:-96}"
RECENCY_KEEP="${RECENCY_KEEP:-4}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-2}"
RAW_SIGNATURE_MODE="${RAW_SIGNATURE_MODE:-avg_pool}"
RAW_GRID_SIZE="${RAW_GRID_SIZE:-4}"
RAW_PROPOSAL_POLICY="${RAW_PROPOSAL_POLICY:-novelty_topk}"
SALIENCY_Z_THRESHOLD="${SALIENCY_Z_THRESHOLD:-4.0}"
PROFILE_BREAKDOWN="${PROFILE_BREAKDOWN:-false}"

prepare_subset() {
  "$PYTHON_BIN" scripts/prepare_ovo_bench_subset.py \
    --source-json "$SOURCE_JSON" \
    --chunked-dir "$CHUNKED_DIR" \
    --output-json "$SUBSET_JSON" \
    --tasks "$TASKS" \
    --max-source-items-per-task "$MAX_SOURCE_ITEMS_PER_TASK" \
    --max-queries-per-source "$MAX_QUERIES_PER_SOURCE" \
    --source-selection "$SOURCE_SELECTION" \
    --query-selection "$QUERY_SELECTION" \
    --require-videos
}

run_method() {
  local method="$1"
  shift
  local save_dir="$OUT_ROOT/$method"
  mkdir -p "$save_dir"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m video_qa.rekv_stream_vqa \
    --model "$MODEL" \
    --sample_fps "$SAMPLE_FPS" \
    --n_local "$N_LOCAL" \
    --retrieve_size "$RETRIEVE_SIZE" \
    --retrieve_chunk_size 1 \
    --save_dir "$save_dir" \
    --anno_path "$SUBSET_JSON" \
    --num_chunks 1 \
    --chunk_idx 0 \
    --enable_query_aware_retrieval false \
    --debug false \
    "$@"

  "$PYTHON_BIN" scripts/evaluate_ovo_bench.py \
    --pred-path "$save_dir/1_0.csv" \
    --output-csv "$save_dir/evaluated.csv" \
    --output-json "$save_dir/metrics.json"
}

main() {
  prepare_subset

  run_method dense \
    --enable_vit_sparse false \
    --enable_vit_layer_sparse false \
    --enable_semantic_stream false \
    --enable_semantic_compute_gate false

  run_method periodic \
    --enable_vit_sparse true \
    --enable_vit_layer_sparse false \
    --enable_semantic_stream true \
    --enable_semantic_compute_gate true \
    --semantic_refresh_interval "$BUDGET_WINDOW" \
    --semantic_skip_threshold 0 \
    --semantic_recency_keep_frames "$RECENCY_KEEP" \
    --semantic_selection_policy periodic \
    --semantic_selection_feature_source raw_rgb \
    --semantic_raw_signature_mode "$RAW_SIGNATURE_MODE" \
    --semantic_raw_grid_size "$RAW_GRID_SIZE" \
    --semantic_raw_proposal_policy "$RAW_PROPOSAL_POLICY" \
    --semantic_saliency_z_threshold "$SALIENCY_Z_THRESHOLD" \
    --semantic_profile_breakdown "$PROFILE_BREAKDOWN"

  run_method "hybrid_cm${CANDIDATE_MULTIPLIER}" \
    --enable_vit_sparse true \
    --enable_vit_layer_sparse false \
    --enable_semantic_stream true \
    --enable_semantic_compute_gate true \
    --semantic_refresh_interval 1000000 \
    --semantic_skip_threshold 0 \
    --semantic_recency_keep_frames "$RECENCY_KEEP" \
    --semantic_selection_policy budget_topk \
    --semantic_selection_feature_source hybrid \
    --semantic_candidate_multiplier "$CANDIDATE_MULTIPLIER" \
    --semantic_raw_signature_mode "$RAW_SIGNATURE_MODE" \
    --semantic_raw_grid_size "$RAW_GRID_SIZE" \
    --semantic_raw_proposal_policy "$RAW_PROPOSAL_POLICY" \
    --semantic_saliency_z_threshold "$SALIENCY_Z_THRESHOLD" \
    --semantic_profile_breakdown "$PROFILE_BREAKDOWN" \
    --semantic_budget_window_size "$BUDGET_WINDOW" \
    --semantic_budget_keep_per_window 1

  "$PYTHON_BIN" scripts/summarize_ovo_bench_validation.py \
    --root "$OUT_ROOT" \
    --output-csv "$OUT_ROOT/summary.csv" \
    --output-json "$OUT_ROOT/summary.json"
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/mllm/datasets}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-base}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

download_streamingbench_metadata() {
  mkdir -p "${DATA_ROOT}/streamingbench"
  HF_ENDPOINT="${HF_ENDPOINT}" "${CONDA_BIN}" run -n "${CONDA_ENV}" huggingface-cli download \
    mjuicem/StreamingBench \
    --repo-type dataset \
    --local-dir "${DATA_ROOT}/streamingbench" \
    --include 'StreamingBench/*.csv' 'README.md'
}

download_streamingbench_boundary_media() {
  mkdir -p "${DATA_ROOT}/streamingbench"
  HF_ENDPOINT="${HF_ENDPOINT}" "${CONDA_BIN}" run -n "${CONDA_ENV}" huggingface-cli download \
    mjuicem/StreamingBench \
    --repo-type dataset \
    --local-dir "${DATA_ROOT}/streamingbench" \
    --include 'Real-Time Visual Understanding_1-50.zip' \
              'Sequential Question Answering_1-25.zip'
}

download_ovo_bench() {
  mkdir -p "${DATA_ROOT}/ovo_bench"
  HF_ENDPOINT="${HF_ENDPOINT}" "${CONDA_BIN}" run -n "${CONDA_ENV}" huggingface-cli download \
    JoeLeelyf/OVO-Bench \
    --repo-type dataset \
    --local-dir "${DATA_ROOT}/ovo_bench"
}

case "${1:-metadata}" in
  metadata)
    download_streamingbench_metadata
    ;;
  streamingbench-media)
    download_streamingbench_boundary_media
    ;;
  ovo)
    download_ovo_bench
    ;;
  all)
    download_streamingbench_metadata
    download_streamingbench_boundary_media
    download_ovo_bench
    ;;
  *)
    echo "Usage: $0 {metadata|streamingbench-media|ovo|all}" >&2
    exit 2
    ;;
esac


#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Streaming-VLM-Optimization}"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV="${CONDA_ENV:-base}"
RESULT_PATH="${RESULT_PATH:-results/remote_vit_sparse_certification.json}"

cd "$REPO_DIR"
git pull --ff-only

"$CONDA_BIN" run -n "$CONDA_ENV" python scripts/verify_vit_sparse_patch.py \
  --experiment-name remote_vit_sparse_certification \
  --result-path "$RESULT_PATH"

cat "$RESULT_PATH"

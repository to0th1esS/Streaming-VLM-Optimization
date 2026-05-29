#!/usr/bin/env bash
set -euo pipefail

# Run this script on the remote GPU server after setting REPO_DIR and CONDA_ENV.
# It intentionally assumes model_zoo/, data/, and results/ are local to the server
# and are not synchronized through git.

REPO_DIR="${REPO_DIR:-$HOME/Streaming-VLM-Optimization}"
CONDA_ENV="${CONDA_ENV:-rekv}"
MODEL="${MODEL:-llava_ov_0.5b}"
DATASET="${DATASET:-qaego4d}"
NUM_CHUNKS="${NUM_CHUNKS:-1}"
SAMPLE_FPS="${SAMPLE_FPS:-0.5}"
N_LOCAL="${N_LOCAL:-15000}"
RETRIEVE_SIZE="${RETRIEVE_SIZE:-64}"

cd "$REPO_DIR"
git pull --ff-only

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python -m video_qa.run_eval \
  --num_chunks "$NUM_CHUNKS" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --sample_fps "$SAMPLE_FPS" \
  --n_local "$N_LOCAL" \
  --retrieve_size "$RETRIEVE_SIZE"

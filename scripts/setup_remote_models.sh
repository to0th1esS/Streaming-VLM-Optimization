#!/usr/bin/env bash
set -euo pipefail

# Prepare model directories on the remote server. Existing mirrors are linked
# into /home/models to avoid duplicating multi-GB checkpoints.

MODEL_ROOT="${MODEL_ROOT:-/home/models}"
MIRROR_ROOT="${MIRROR_ROOT:-/home/Streaming-VLM-Optimization/model_zoo}"
HF_HOME="${HF_HOME:-$MODEL_ROOT/.cache/huggingface}"
MODELS="${MODELS:-llava-onevision-qwen2-0.5b-ov-hf llava-onevision-qwen2-7b-ov-hf LanguageBind-Video-LLaVA-7B-hf LongVA-7B}"

mkdir -p "$MODEL_ROOT" "$HF_HOME"

repo_for_model() {
  case "$1" in
    llava-onevision-qwen2-0.5b-ov-hf) echo "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" ;;
    llava-onevision-qwen2-7b-ov-hf) echo "llava-hf/llava-onevision-qwen2-7b-ov-hf" ;;
    llava-onevision-qwen2-72b-ov-hf) echo "llava-hf/llava-onevision-qwen2-72b-ov-hf" ;;
    LanguageBind-Video-LLaVA-7B-hf) echo "LanguageBind/Video-LLaVA-7B-hf" ;;
    LongVA-7B) echo "lmms-lab/LongVA-7B" ;;
    *) echo "$1" ;;
  esac
}

for model_name in $MODELS; do
  target="$MODEL_ROOT/$model_name"
  mirror="$MIRROR_ROOT/$model_name"

  if [ -e "$target" ]; then
    echo "[model] exists: $target"
    continue
  fi

  if [ -d "$mirror" ]; then
    ln -s "$mirror" "$target"
    echo "[model] linked: $target -> $mirror"
    continue
  fi

  repo_id="$(repo_for_model "$model_name")"
  echo "[model] downloading $repo_id to $target"
  python -m pip install -q "huggingface_hub[cli]"
  HF_HOME="$HF_HOME" huggingface-cli download "$repo_id" \
    --local-dir "$target" \
    --local-dir-use-symlinks False
done

echo "[model] prepared under $MODEL_ROOT"

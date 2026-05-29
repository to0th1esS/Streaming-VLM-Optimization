#!/usr/bin/env bash
set -euo pipefail

# Prepare model directories on the remote server. Checkpoints live under
# MODEL_ROOT, while the repository's model_zoo/ contains symlinks so existing
# model_path values in video_qa/base.py keep working.

MODEL_ROOT="${MODEL_ROOT:-/home/mllm/models}"
MIRROR_ROOT="${MIRROR_ROOT:-/home/Streaming-VLM-Optimization/model_zoo}"
REPO_MODEL_ZOO="${REPO_MODEL_ZOO:-model_zoo}"
HF_HOME="${HF_HOME:-$MODEL_ROOT/.cache/huggingface}"
MODELS="${MODELS:-llava-onevision-qwen2-0.5b-ov-hf llava-onevision-qwen2-7b-ov-hf Video-LLaVA-7B-hf LongVA-7B}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

mkdir -p "$MODEL_ROOT" "$HF_HOME" "$REPO_MODEL_ZOO"

repo_for_model() {
  case "$1" in
    llava-onevision-qwen2-0.5b-ov-hf) echo "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" ;;
    llava-onevision-qwen2-7b-ov-hf) echo "llava-hf/llava-onevision-qwen2-7b-ov-hf" ;;
    llava-onevision-qwen2-72b-ov-hf) echo "llava-hf/llava-onevision-qwen2-72b-ov-hf" ;;
    Video-LLaVA-7B-hf) echo "LanguageBind/Video-LLaVA-7B-hf" ;;
    LongVA-7B) echo "lmms-lab/LongVA-7B" ;;
    *) echo "$1" ;;
  esac
}

mirror_candidates() {
  case "$1" in
    Video-LLaVA-7B-hf)
      echo "$MIRROR_ROOT/Video-LLaVA-7B-hf $MIRROR_ROOT/LanguageBind-Video-LLaVA-7B-hf"
      ;;
    *)
      echo "$MIRROR_ROOT/$1"
      ;;
  esac
}

find_existing_mirror() {
  for candidate in $(mirror_candidates "$1"); do
    if [ -d "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

for model_name in $MODELS; do
  target="$MODEL_ROOT/$model_name"
  repo_link="$REPO_MODEL_ZOO/$model_name"

  if [ "$FORCE_DOWNLOAD" = "1" ] && { [ -e "$target" ] || [ -L "$target" ]; }; then
    rm -rf "$target"
  fi

  if [ -e "$target" ]; then
    echo "[model] exists: $target"
  else
    if [ "$FORCE_DOWNLOAD" = "1" ]; then
      mirror=""
    else
      mirror="$(find_existing_mirror "$model_name" || true)"
    fi
    if [ -n "$mirror" ]; then
    ln -s "$mirror" "$target"
    echo "[model] linked: $target -> $mirror"
    else
      repo_id="$(repo_for_model "$model_name")"
      echo "[model] downloading $repo_id to $target"
      python -m pip install -q "huggingface_hub[cli]"
      HF_HOME="$HF_HOME" huggingface-cli download "$repo_id" \
        --local-dir "$target" \
        --local-dir-use-symlinks False
    fi
  fi

  if [ -e "$repo_link" ] || [ -L "$repo_link" ]; then
    rm -rf "$repo_link"
  fi
  ln -s "$target" "$repo_link"
  echo "[model_zoo] linked: $repo_link -> $target"
done

echo "[model] prepared under $MODEL_ROOT"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${KEY_EMBEDDING_VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export DOCUMENT_GRAPH_BASE_MODEL="${DOCUMENT_GRAPH_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
export KEY_EMBEDDING_BASE_MODEL="${KEY_EMBEDDING_BASE_MODEL:-$DOCUMENT_GRAPH_BASE_MODEL}"
export KEY_EMBEDDING_MODEL_NAME="${KEY_EMBEDDING_MODEL_NAME:-key-embedding-graph}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

mkdir -p "$HF_HOME"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.2.2+cu118"

python -m pip install \
    "transformers==4.41.2" \
    "accelerate==0.30.1" \
    "peft==0.11.1" \
    "bitsandbytes==0.43.1" \
    "trl==0.9.6" \
    "datasets==2.19.2" \
    "sentence-transformers==3.0.1" \
    "sentencepiece==0.2.0" \
    "protobuf==4.25.3" \
    "safetensors==0.4.3" \
    "numpy==1.26.4" \
    "scikit-learn==1.4.2" \
    "orjson==3.9.12" \
    "rich==15.0.0"

python - <<'PY'
import os
import torch

print("venv ready")
print("base model:", os.environ["KEY_EMBEDDING_BASE_MODEL"])
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo "activate: source $VENV_DIR/bin/activate"

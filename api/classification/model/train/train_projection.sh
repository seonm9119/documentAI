#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MODEL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${TRAIN_PROJECTION_VENV_DIR:-$SCRIPT_DIR/train_projection}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROJECTION_CACHE_DIR="$("$PYTHON_BIN" - <<'PY'
from api.classification.config import PROJECTION_CACHE_FOLDER

print(PROJECTION_CACHE_FOLDER)
PY
)"

mkdir -p "$PROJECTION_CACHE_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1+cu121"

python -m pip install \
    "numpy==1.26.4" \
    "transformers==4.41.2" \
    "sentence-transformers==3.3.1" \
    "safetensors==0.4.3" \
    "huggingface-hub==0.36.0"

python - <<'PY'
import torch
from sentence_transformers import SentenceTransformer
from api.classification.config import (
    PROJECTION_CACHE_FOLDER,
    PROJECTION_DEVICE,
    PROJECTION_ENCODER_MODEL,
)

print("key embedding projection venv ready")
print("python cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("cuda is required for key embedding projection training")

print("cuda device:", torch.cuda.get_device_name(0))
print("encoder model:", PROJECTION_ENCODER_MODEL)
print("device:", PROJECTION_DEVICE)
print("cache dir:", PROJECTION_CACHE_FOLDER)
print("sentence-transformers import:", SentenceTransformer.__name__)
PY

cat <<EOF

activate:
  source "$VENV_DIR/bin/activate"

train:
  cd "$PROJECT_DIR"
  python -m api.classification.model.train.train_projection

EOF

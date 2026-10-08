#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ -d .env ]]; then
  echo "ERROR: .env must be the DeepSeek key file, not a virtual-environment directory." >&2
  exit 1
fi
if [[ ! -f .env ]]; then
  echo "WARNING: root .env key file is missing; create it before using online parsing." >&2
fi

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ./multitq_src
python -m pip install -r multitq_src/requirements-dev.txt

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("wheel CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside .venv")
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
PY

echo "Virtual environment ready. Activate it with: source .venv/bin/activate"

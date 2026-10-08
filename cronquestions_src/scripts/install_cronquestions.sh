#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# For a fresh AutoDL CUDA 13.0 image. If torch 2.12.1 is already installed,
# pip reuses it without downloading another copy.
python -m pip install \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r cronquestions_src/requirements.txt
python -m pip install -e ./cronquestions_src --no-deps

python - <<'PY'
import torch
import decgc
print("torch:", torch.__version__)
print("CUDA wheel:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print("DECGC package:", decgc.__file__)
PY
PYTHONPATH="$PROJECT_ROOT/cronquestions_src${PYTHONPATH:+:$PYTHONPATH}" python -m decgc.cli --help >/dev/null
echo "CronQuestions environment is ready."

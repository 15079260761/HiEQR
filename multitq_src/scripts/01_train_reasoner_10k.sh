#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/four_loss_common.sh"

run_stage environment_check "${DECGC[@]}" doctor --config "$CONFIG"

python -B - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from decgc.config import load_config
from decgc.cli import _best_model, _run_dir

config, _ = load_config(Path(sys.argv[1]))
for stage in ("retriever", "reranker"):
    fallback = getattr(config.paths, f"pretrained_{stage}")
    best = _best_model(config, stage, fallback)
    assert best.resolve() != fallback.resolve(), (
        f"Trained {stage} checkpoint is missing; refusing to substitute the base model: {fallback}"
    )
    weights = [best / "model.safetensors", best / "pytorch_model.bin"]
    assert (best / "config.json").exists() and any(path.exists() for path in weights), (
        f"Existing trained {stage} best_model is missing: {best}"
    )
    print(f"Reusing trained {stage}: {best}")

reasoner = _run_dir(config, "reasoner")
print("Four-loss reasoner run:", reasoner)
PY

# Retriever/reranker hashes exclude the reasoner loss configuration. Their
# existing best_model directories are reused. The changed objective and data
# semantics create a distinct reasoner hash and therefore a new checkpoint.
run_stage train_reasoner_10k "${DECGC[@]}" train-reasoner \
  --config "$CONFIG" \
  --retriever-checkpoint best \
  --reranker-checkpoint best \
  --limit 10000 \
  --resume auto

python -B - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from decgc.config import load_config
from decgc.cli import _run_dir

config, _ = load_config(Path(sys.argv[1]))
run = _run_dir(config, "reasoner")
summary = json.loads((run / "training_summary.json").read_text(encoding="utf-8"))
assert summary["training_complete"] is True
assert summary["active_losses"] == ["answer", "prefix", "stop", "counterfactual"]
assert summary["removed_losses"] == ["ranking"]
assert summary["loss_weights"] == {
    "answer": 1.0,
    "ranking": 0.0,
    "prefix": 0.5,
    "stop": 0.3,
    "counterfactual": 0.3,
}
print("Four-loss reasoner verified:", run)
PY

#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/cronquestions_4loss_common.sh"

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
    weights = (best / "model.safetensors", best / "pytorch_model.bin")
    assert (best / "config.json").exists() and any(path.exists() for path in weights), (
        f"Completed CronQuestions five-loss {stage} checkpoint is missing: {best}"
    )
    print(f"Reusing completed CronQuestions {stage}: {best}")

for split in ("train", "valid"):
    manifest = config.paths.cache_root / "candidate_chains" / split / "manifest.json"
    assert manifest.exists(), f"Reusable {split} chain cache is missing: {manifest}"
    print(f"Reusing {split} chain cache: {manifest}")

reasoner = _run_dir(config, "reasoner")
print("New four-loss reasoner run:", reasoner)
PY

# No retriever/reranker preparation or standalone training occurs here. The
# reasoner starts from their completed same-dataset best checkpoints.
run_stage train_reasoner_10k "${DECGC[@]}" train-reasoner \
  --config "$CONFIG" \
  --retriever-checkpoint best \
  --reranker-checkpoint best \
  --limit 10000 \
  --resume auto

python -B - "$CONFIG" <<'PY'
import json, sys
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
    "answer": 1.0, "ranking": 0.0, "prefix": 0.5,
    "stop": 0.3, "counterfactual": 0.3,
}
print("Verified four-loss training summary:", run)
PY

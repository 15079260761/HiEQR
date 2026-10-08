#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${DECGC_GPU_LOCK:-/tmp/decgc-four-loss-main.lock}"
exec 9>"$LOCK_FILE"
echo "Waiting for the single-GPU experiment lock: $LOCK_FILE"
flock 9
echo "GPU experiment lock acquired by CronQuestions."
bash "$SCRIPT_DIR/01_train_reasoner_10k.sh"
bash "$SCRIPT_DIR/03_full_test.sh"

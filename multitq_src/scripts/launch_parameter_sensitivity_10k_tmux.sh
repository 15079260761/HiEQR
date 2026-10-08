#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DECGC_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SESSION="${PARAM_TMUX_SESSION:-multitq-param10k}"
RUN_TOKEN="${PARAM_RUN_TOKEN:-$(date '+%Y%m%d_%H%M%S')}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 2
fi

tmux new-session -d -s "$SESSION" -n prepare
tmux send-keys -t "$SESSION:prepare" \
  "cd '$PROJECT_ROOT' && PARAM_RUN_TOKEN='$RUN_TOKEN' CUDA_VISIBLE_DEVICES=0 bash multitq_3loss_src/scripts/prepare_parameter_sensitivity_10k.sh" C-m

tmux new-window -t "$SESSION" -n gpu0-prefix
tmux send-keys -t "$SESSION:gpu0-prefix" \
  "cd '$PROJECT_ROOT' && PARAM_RUN_TOKEN='$RUN_TOKEN' bash multitq_3loss_src/scripts/run_parameter_sensitivity_10k_worker.sh 0 prefix" C-m

tmux new-window -t "$SESSION" -n gpu1-stop
tmux send-keys -t "$SESSION:gpu1-stop" \
  "cd '$PROJECT_ROOT' && PARAM_RUN_TOKEN='$RUN_TOKEN' bash multitq_3loss_src/scripts/run_parameter_sensitivity_10k_worker.sh 1 stop" C-m

tmux new-window -t "$SESSION" -n gpu2-cf
tmux send-keys -t "$SESSION:gpu2-cf" \
  "cd '$PROJECT_ROOT' && PARAM_RUN_TOKEN='$RUN_TOKEN' bash multitq_3loss_src/scripts/run_parameter_sensitivity_10k_worker.sh 2 counterfactual" C-m

tmux new-window -t "$SESSION" -n collect
tmux send-keys -t "$SESSION:collect" \
  "cd '$PROJECT_ROOT' && PARAM_RUN_TOKEN='$RUN_TOKEN' bash multitq_3loss_src/scripts/collect_parameter_sensitivity_10k_when_done.sh" C-m

tmux select-window -t "$SESSION:prepare"
echo "Created tmux session: $SESSION"
echo "Run token: $RUN_TOKEN"
echo "Attach with: tmux attach -t $SESSION"
echo "Windows: prepare, gpu0-prefix, gpu1-stop, gpu2-cf, collect"

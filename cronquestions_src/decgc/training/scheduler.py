"""Linear warmup/decay scheduler."""

from transformers import get_linear_schedule_with_warmup


def build_scheduler(optimizer, total_steps: int):
    return get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), max(1, total_steps))


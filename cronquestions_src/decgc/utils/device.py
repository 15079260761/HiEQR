"""Runtime device and mixed-precision selection."""

import torch


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype(mode: str = "auto") -> torch.dtype | None:
    if mode == "fp16" or (mode == "auto" and torch.cuda.is_available()):
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    return None


def grad_scaler(enabled: bool):
    """Create a CUDA scaler using the current API with an old-Torch fallback."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

"""Atomic checkpoints with pointer manifests and compatibility checks."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from ..utils.hashing import file_hash
from ..utils.io import write_json
from .reproducibility import capture_rng, restore_rng


class CheckpointManager:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _git_commit(self) -> str | None:
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def save(self, name: str, model, optimizer, scheduler, scaler, metadata: dict[str, Any], best: bool = False) -> Path:
        target = self.root / f"{name}.pt"
        temporary = target.with_suffix(".pt.tmp")
        payload = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict() if scaler else None, "rng": capture_rng(),
            "git_commit": self._git_commit(), **metadata,
        }
        torch.save(payload, temporary)
        validation = torch.load(temporary, map_location="cpu", weights_only=False)
        if "model" not in validation or validation.get("config_hash") != metadata.get("config_hash"):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("Checkpoint integrity validation failed")
        os.replace(temporary, target)
        pointer = {"path": target.name, "sha256": file_hash(target)}
        write_json(self.root / "latest.json", pointer)
        if best:
            write_json(self.root / "best.json", pointer)
        return target

    def resolve(self, resume: str) -> Path | None:
        if resume in {"none", "None", ""}:
            return None
        if resume in {"auto", "latest"}:
            pointer = self.root / "latest.json"
        elif resume == "best":
            pointer = self.root / "best.json"
        else:
            return Path(resume)
        return self.root / json.loads(pointer.read_text(encoding="utf-8"))["path"] if pointer.exists() else None

    def load(self, path: Path, model, optimizer=None, scheduler=None, scaler=None,
             expected_config_hash: str | None = None) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if expected_config_hash and payload.get("config_hash") != expected_config_hash:
            raise RuntimeError("Checkpoint configuration is incompatible")
        model.load_state_dict(payload["model"])
        if optimizer and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        if scaler and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        restore_rng(payload["rng"])
        return payload


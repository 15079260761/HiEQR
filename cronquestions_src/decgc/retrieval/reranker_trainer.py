"""Actual-gradient cross-encoder fine-tuning with update verification."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..training.reproducibility import capture_rng
from ..training.record_split import split_records_by_query
from ..training.checkpoint_cleanup import cleanup_run
from ..utils.device import grad_scaler
from ..utils.hashing import file_hash, state_dict_hash
from ..utils.io import read_json, write_json


def train_reranker(model, records: list[dict], output: Path, settings: dict, device: torch.device,
                   resume: str = "auto") -> dict:
    if not records:
        raise RuntimeError("No reranker training records were constructed")
    output.mkdir(parents=True, exist_ok=True)
    model.to(device).train()
    if settings.get("gradient_checkpointing"):
        model.model.gradient_checkpointing_enable()
    before = state_dict_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
    use_amp = device.type == "cuda" and settings.get("mixed_precision", "auto") in {"auto", "fp16"}
    scaler = grad_scaler(use_amp)
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    training_records, validation_records = split_records_by_query(
        records,
        float(settings.get("validation_fraction", 0.05)),
    )
    start_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    pointer = output / "checkpoints" / "latest.json"
    if resume not in {"none", "None", ""} and pointer.exists():
        import json
        checkpoint_path = pointer.parent / json.loads(pointer.read_text(encoding="utf-8"))["path"]
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        global_step = int(state["global_step"])
        best_validation_loss = float(state.get("best_validation_loss", float("inf")))
        before = state.get("before_hash", before)
        if start_epoch >= int(settings["epochs"]) and (output / "training_summary.json").exists():
            return read_json(output / "training_summary.json")
    loader = DataLoader(
        training_records,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=lambda rows: rows,
    )
    validation_loader = DataLoader(
        validation_records,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
        collate_fn=lambda rows: rows,
    )
    maximum_gradient = 0.0
    progress_every = max(1, int(settings.get("progress_every_batches", 100)))
    for epoch in range(start_epoch, int(settings["epochs"])):
        epoch_started = time.monotonic()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, rows in enumerate(loader):
            labels = torch.tensor([r["label"] for r in rows], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model([r["query"] for r in rows], [r["text"] for r in rows], labels) / accumulation
            epoch_losses.append(float(loss.detach().float()) * accumulation)
            scaler.scale(loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                maximum_gradient = max(maximum_gradient, float(gradient))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            completed_batches = batch_index + 1
            if completed_batches == len(loader) or completed_batches % progress_every == 0:
                elapsed = max(time.monotonic() - epoch_started, 1.0e-6)
                rate = completed_batches / elapsed
                eta = (len(loader) - completed_batches) / rate if rate else 0.0
                print(
                    f"Reranker epoch {epoch + 1}/{int(settings['epochs'])}: "
                    f"{completed_batches}/{len(loader)} batches, "
                    f"mean loss {sum(epoch_losses) / len(epoch_losses):.6f}, "
                    f"{rate:.2f} batches/s, ETA {eta:.0f}s",
                    flush=True,
                )
        model.eval()
        validation_losses = []
        with torch.inference_mode():
            for rows in validation_loader:
                labels = torch.tensor(
                    [row["label"] for row in rows],
                    device=device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    validation_loss = model(
                        [row["query"] for row in rows],
                        [row["text"] for row in rows],
                        labels,
                    )
                validation_losses.append(
                    float(validation_loss.detach().float().cpu())
                )
        model.train()
        mean_validation_loss = sum(validation_losses) / len(validation_losses)
        improved = mean_validation_loss < best_validation_loss
        if improved:
            best_validation_loss = mean_validation_loss
        print(
            f"Reranker epoch {epoch + 1} validation loss "
            f"{mean_validation_loss:.6f}, best={'yes' if improved else 'no'}",
            flush=True,
        )
        checkpoint = output / "checkpoints" / f"epoch-{epoch + 1:03d}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": None,
                    "scaler": scaler.state_dict(), "epoch": epoch + 1, "global_step": global_step,
                    "before_hash": before, "best_validation_loss": best_validation_loss,
                    "rng": capture_rng()}, temporary)
        if "model" not in torch.load(temporary, map_location="cpu", weights_only=False):
            raise RuntimeError("Reranker checkpoint integrity failure")
        os.replace(temporary, checkpoint)
        pointer = {"path": checkpoint.name, "sha256": file_hash(checkpoint)}
        write_json(checkpoint.parent / "latest.json", pointer)
        if improved:
            write_json(checkpoint.parent / "best.json", pointer)
        cleanup_run(checkpoint.parent, keep_last=1, execute=True)
    best_pointer = read_json(output / "checkpoints" / "best.json")
    best_checkpoint = output / "checkpoints" / best_pointer["path"]
    best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    after = state_dict_hash(model)
    if maximum_gradient == 0 or before == after:
        raise RuntimeError("Reranker did not update")
    best = output / "best_model"
    model.model.save_pretrained(best)
    model.tokenizer.save_pretrained(best)
    summary = {
        "before_hash": before,
        "after_hash": after,
        "maximum_gradient_norm": maximum_gradient,
        "global_step": global_step,
        "training_records": len(training_records),
        "validation_records": len(validation_records),
        "best_validation_loss": best_validation_loss,
    }
    write_json(output / "training_summary.json", summary)
    return summary

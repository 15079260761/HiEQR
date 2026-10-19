"""Small-batch reasoner training over cached prefix features."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as functional

from ..evaluation.answer_normalizer import normalize_answer
from ..utils.hashing import object_hash, state_dict_hash
from ..utils.io import read_json, write_json
from ..utils.device import grad_scaler
from .checkpoint_manager import CheckpointManager
from .checkpoint_cleanup import cleanup_run
from .early_stopping import EarlyStopping


_REASONER_LOSS_DEFAULTS = {
    "prefix": 0.5,
    "stop": 0.3,
    "counterfactual": 0.3,
}


def reasoner_loss_weights(config) -> dict[str, float]:
    """Resolve the three terms in Equation 7 of the HiEQR paper."""
    settings = config.model_extra["loss"]
    weights = {
        name: float(settings.get(name, default))
        for name, default in _REASONER_LOSS_DEFAULTS.items()
    }
    if not any(weight != 0.0 for weight in weights.values()):
        raise ValueError("At least one reasoner loss coefficient must be non-zero")
    return weights


def loss_ablation_metadata(loss_weights: dict[str, float]) -> dict[str, object]:
    """Record which of the three Equation 7 objectives are active."""
    active = [name for name, weight in loss_weights.items() if weight != 0.0]
    removed = [name for name, weight in loss_weights.items() if weight == 0.0]
    return {
        "active_losses": active,
        "removed_losses": removed,
        "semantics": "direct_loss_term_ablation",
        "ranking_score_heads_retained": ["reliability", "stop", "answer", "interaction"],
        "counterfactual_forward_enabled": loss_weights["counterfactual"] != 0.0,
    }


def _forward_prefix_sequences(model, batch: dict, tensors: dict, device: torch.device):
    sequences = batch.get("fact_sequences")
    if model.retriever is None or not sequences:
        output = model.prefix_step(
            tensors["previous"],
            tensors["fact"],
            tensors["hop"],
            tensors["temporal"],
            tensors["type_score"],
        )
        return output, tensors["previous"]

    batch_size = len(sequences)
    hidden_size = model.chain_encoder.gru.hidden_size
    previous = torch.zeros(batch_size, hidden_size, device=device)
    final_outputs: list[dict | None] = [None] * batch_size
    pre_final_states: list[torch.Tensor | None] = [None] * batch_size
    maximum_hops = max(len(sequence) for sequence in sequences)
    for hop_index in range(maximum_hops):
        active = [index for index, sequence in enumerate(sequences) if hop_index < len(sequence)]
        active_index = torch.tensor(active, dtype=torch.long, device=device)
        questions = [batch["questions"][index] for index in active]
        fact_texts = [sequences[index][hop_index] for index in active]
        encoded = model.encode_live(questions, fact_texts)
        # CUDA autocast makes live BGE/GRU features FP16 while the recurrent
        # state is initially FP32. Keep the whole state buffer in the active
        # compute dtype before indexed reads/writes; torch.index_copy requires
        # exact dtype equality and must remain differentiable for earlier hops.
        if previous.dtype != encoded.dtype:
            previous = previous.to(encoded.dtype)
        before = previous.index_select(0, active_index)
        output = model.prefix_step(
            before,
            encoded,
            torch.full((len(active),), hop_index + 1, dtype=torch.long, device=device),
            tensors["temporal"].index_select(0, active_index),
            tensors["type_score"].index_select(0, active_index),
        )
        next_state = output["chain"]
        if previous.dtype != next_state.dtype:
            previous = previous.to(next_state.dtype)
            before = before.to(next_state.dtype)
        previous = previous.index_copy(0, active_index, next_state)
        for local_index, batch_index in enumerate(active):
            if hop_index + 1 == len(sequences[batch_index]):
                final_outputs[batch_index] = {
                    key: value[local_index]
                    for key, value in output.items()
                    if isinstance(value, torch.Tensor)
                }
                pre_final_states[batch_index] = before[local_index]

    if any(output is None for output in final_outputs) or any(state is None for state in pre_final_states):
        raise RuntimeError("Incomplete chain-prefix sequence batch")
    keys = final_outputs[0].keys()
    stacked = {
        key: torch.stack([output[key] for output in final_outputs])
        for key in keys
    }
    return stacked, torch.stack(pre_final_states)


def validation_ranking_metrics(
    model,
    batches: list[dict],
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float | int]:
    """Measure within-question weak-label Hits@1 for checkpoint selection."""
    if not batches:
        return {"hits1": 0.0, "loss": 0.0, "groups": 0}
    was_training = model.training
    model.eval()
    correct = 0
    eligible = 0
    losses = []
    with torch.inference_mode():
        for batch in batches:
            tensors = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                output, _ = _forward_prefix_sequences(model, batch, tensors, device)
                logits = output["reliability_logits"] + output["stop_logits"]
                if model.retriever is not None and batch.get("answer_texts"):
                    answer_logits = model.score_answers(
                        output["chain"],
                        batch["answer_texts"],
                    )
                    interaction_logits = model.score_interactions(
                        output["chain"],
                        batch["answer_texts"],
                        batch["group_ids"],
                    )
                    logits = logits + answer_logits + interaction_logits
                labels = tensors["prefix_label"].bool()
                for group_id in dict.fromkeys(batch["group_ids"]):
                    indices = [
                        index
                        for index, candidate_group in enumerate(batch["group_ids"])
                        if candidate_group == group_id
                    ]
                    answer_groups: dict[str, list[int]] = {}
                    for index in indices:
                        answer_groups.setdefault(
                            normalize_answer(batch["answer_texts"][index]),
                            [],
                        ).append(index)
                    answer_logits = []
                    answer_labels = []
                    for answer_indices in answer_groups.values():
                        answer_index = torch.tensor(
                            answer_indices,
                            dtype=torch.long,
                            device=device,
                        )
                        answer_logits.append(
                            logits.index_select(0, answer_index).max()
                        )
                        answer_labels.append(
                            labels.index_select(0, answer_index).any()
                        )
                    group_logits = torch.stack(answer_logits)
                    group_labels = torch.stack(answer_labels)
                    if not (
                        group_labels.any()
                        and (~group_labels).any()
                    ):
                        continue
                    eligible += 1
                    correct += int(group_labels[group_logits.argmax()].item())
                    losses.append(
                        torch.logsumexp(group_logits, dim=0)
                        - torch.logsumexp(group_logits[group_labels], dim=0)
                    )
    model.train(was_training)
    return {
        "hits1": correct / eligible if eligible else 0.0,
        "loss": (
            float(torch.stack(losses).mean().float().cpu()) if losses else 0.0
        ),
        "groups": eligible,
    }


def reasoner_runtime_preflight(
    model,
    batches: list[dict],
    device: torch.device,
    use_amp: bool,
    check_counterfactual: bool = True,
) -> dict[str, str | int]:
    """Exercise the complete live AMP path before optimizer state is changed."""
    if not batches:
        raise RuntimeError("Reasoner runtime preflight has no batches")
    batch = next(
        (
            candidate
            for candidate in batches
            if isinstance(candidate.get("cf_mask"), torch.Tensor)
            and candidate["cf_mask"].bool().any()
        ),
        batches[0],
    )
    tensors = {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }
    was_training = model.training
    model.eval()
    checked = []
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        output, pre_final = _forward_prefix_sequences(
            model,
            batch,
            tensors,
            device,
        )
        checked.extend(
            output[key]
            for key in ("chain", "reliability_logits", "stop_logits")
        )
        if model.retriever is not None and batch.get("answer_texts"):
            checked.append(
                model.score_answers(
                    output["chain"],
                    batch["answer_texts"],
                )
            )
            checked.append(
                model.score_interactions(
                    output["chain"],
                    batch["answer_texts"],
                    batch.get("group_ids", batch["questions"]),
                )
            )
        if (
            check_counterfactual
            and
            model.retriever is not None
            and batch.get("cf_fact_texts")
            and tensors.get("cf_owner_indices") is not None
            and tensors["cf_owner_indices"].numel() > 0
        ):
            owners = tensors["cf_owner_indices"]
            counterfactual = model.encode_live(
                batch["cf_questions"],
                batch["cf_fact_texts"],
            )
            counterfactual_output = model.prefix_step(
                pre_final.index_select(0, owners),
                counterfactual,
                tensors["hop"].index_select(0, owners),
                tensors["temporal"].index_select(0, owners) * 0,
                tensors["type_score"].index_select(0, owners) * 0,
            )
            checked.extend(
                counterfactual_output[key]
                for key in ("chain", "reliability_logits", "stop_logits")
            )
    model.train(was_training)
    if any(not torch.isfinite(value).all() for value in checked):
        raise RuntimeError("Reasoner runtime preflight produced non-finite tensors")
    return {
        "batch_items": len(batch["questions"]),
        "state_dtype": str(output["chain"].dtype),
        "amp": str(use_amp).lower(),
    }


def train_reasoner(
    model,
    batches: list[dict],
    optimizer,
    config,
    run_dir,
    resume: str = "auto",
    validation_batches: list[dict] | None = None,
    checkpoint_config_hash: str | None = None,
) -> dict:
    if not batches:
        raise RuntimeError("No cached chain batches available")
    manager = CheckpointManager(run_dir / "checkpoints")
    settings = config.model_extra["reasoner_training"]
    loss_weights = reasoner_loss_weights(config)
    ablation_metadata = loss_ablation_metadata(loss_weights)
    active_losses = ablation_metadata["active_losses"]
    effective_config_hash = checkpoint_config_hash or config.hash
    use_amp = torch.cuda.is_available() and settings.get("mixed_precision", "auto") in {"auto", "fp16"}
    scaler = grad_scaler(use_amp)
    start_epoch, global_step = 0, 0
    state = {}
    checkpoint = manager.resolve(resume)
    if checkpoint:
        state = manager.load(
            checkpoint,
            model,
            optimizer,
            scaler=scaler,
            expected_config_hash=effective_config_hash,
        )
        start_epoch, global_step = int(state["epoch"]), int(state["global_step"])
        summary_path = run_dir / "training_summary.json"
        if start_epoch >= int(settings["epochs"]) and summary_path.exists():
            return read_json(summary_path)
    early = EarlyStopping(
        int(settings["early_stopping_patience"]),
        best=float(state.get("best_metric", float("-inf"))),
        bad_epochs=int(state.get("patience_counter", 0)),
    )
    device = next(model.parameters()).device
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    progress_every = max(1, int(settings.get("progress_every_batches", 250)))
    preflight = reasoner_runtime_preflight(
        model,
        batches,
        device,
        use_amp,
        check_counterfactual=loss_weights["counterfactual"] != 0.0,
    )
    print(
        "Reasoner runtime preflight passed: "
        f"{preflight['batch_items']} items, "
        f"state={preflight['state_dtype']}, amp={preflight['amp']}",
        flush=True,
    )
    print(
        "Active reasoner losses: " + ", ".join(active_losses),
        flush=True,
    )
    print(
        "Direct-loss ablation; retained ranking score heads: "
        + ", ".join(ablation_metadata["ranking_score_heads_retained"]),
        flush=True,
    )
    final_epoch = start_epoch
    best_validation_hits1 = float(state.get("best_validation_hits1", 0.0))
    latest_validation = {"hits1": 0.0, "loss": 0.0, "groups": 0}
    for epoch in range(start_epoch, int(settings["epochs"])):
        model.train()
        losses = []
        epoch_started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(batches):
            tensors = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output, pre_final = _forward_prefix_sequences(model, batch, tensors, device)
                zero = output["reliability_logits"].sum() * 0.0
                prefix_loss = zero
                if loss_weights["prefix"] != 0.0:
                    prefix_loss = functional.binary_cross_entropy_with_logits(
                        output["reliability_logits"], tensors["prefix_label"].float()
                    )

                # Equation 7 contains exactly three direct objectives.  The
                # answer-support and interaction heads remain part of the
                # complete-chain inference score in Equation 10, but they do
                # not introduce additional training losses here.
                loss = loss_weights["prefix"] * prefix_loss
                if loss_weights["stop"] != 0.0:
                    stop_loss = functional.binary_cross_entropy_with_logits(
                        output["stop_logits"], tensors["stop_label"].float()
                    )
                    loss = loss + loss_weights["stop"] * stop_loss
                if (
                    loss_weights["counterfactual"] != 0.0
                    and
                    model.retriever is not None
                    and batch.get("cf_fact_texts")
                    and tensors.get("cf_owner_indices") is not None
                    and tensors["cf_owner_indices"].numel() > 0
                ):
                    owners = tensors["cf_owner_indices"]
                    counterfactual = model.encode_live(
                        batch["cf_questions"], batch["cf_fact_texts"]
                    )
                    cf_output = model.prefix_step(
                        pre_final.index_select(0, owners),
                        counterfactual,
                        tensors["hop"].index_select(0, owners),
                        tensors["temporal"].index_select(0, owners) * 0,
                        tensors["type_score"].index_select(0, owners) * 0,
                    )
                    margin = float(config.model_extra["loss"].get("counterfactual_margin", 0.2))
                    counterfactual_losses = functional.relu(
                        margin
                        - output["reliability_logits"].index_select(0, owners)
                        + cf_output["reliability_logits"]
                    )
                    counterfactual_loss = counterfactual_losses.mean()
                    loss = loss + loss_weights["counterfactual"] * counterfactual_loss
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(batches):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(loss.detach().float().item())
            global_step += 1
            completed_batches = batch_index + 1
            if (
                completed_batches == len(batches)
                or completed_batches % progress_every == 0
            ):
                elapsed = max(time.monotonic() - epoch_started, 1.0e-6)
                rate = completed_batches / elapsed
                eta = (len(batches) - completed_batches) / rate if rate else 0.0
                print(
                    f"Reasoner epoch {epoch + 1}/{int(settings['epochs'])}: "
                    f"{completed_batches}/{len(batches)} batches, "
                    f"mean loss {sum(losses) / len(losses):.6f}, "
                    f"{rate:.2f} batches/s, ETA {eta:.0f}s",
                    flush=True,
                )
        training_loss = sum(losses) / len(losses)
        latest_validation = validation_ranking_metrics(
            model,
            validation_batches or [],
            device,
            use_amp,
        )
        if latest_validation["groups"]:
            metric = float(latest_validation["hits1"]) - 1.0e-6 * float(
                latest_validation["loss"]
            )
        else:
            metric = -training_loss
        improved, stop = early.update(metric)
        print(
            f"Reasoner epoch {epoch + 1} validation: "
            f"weak Hits@1={float(latest_validation['hits1']):.6f}, "
            f"ranking loss={float(latest_validation['loss']):.6f}, "
            f"eligible questions={int(latest_validation['groups'])}, "
            f"best={'yes' if improved else 'no'}",
            flush=True,
        )
        if improved and latest_validation["groups"]:
            best_validation_hits1 = float(latest_validation["hits1"])
        manager.save(f"epoch-{epoch + 1:03d}", model, optimizer, None, scaler, {
            "epoch": epoch + 1, "global_step": global_step, "best_metric": early.best,
            "patience_counter": early.bad_epochs, "config_hash": effective_config_hash,
            "training_loss": training_loss,
            "validation": latest_validation,
            "best_validation_hits1": best_validation_hits1,
            "dataset_manifest_hash": object_hash([
                (item["questions"][0], float(item["prefix_label"][0])) for item in batches
            ]),
            "retriever_hash": state_dict_hash(model.retriever) if model.retriever is not None else "not-attached",
            "reranker_hash": state_dict_hash(model.reranker) if model.reranker is not None else "not-attached",
            "model_hash": state_dict_hash(model),
            "resolved_config": config.model_dump(mode="json"),
        }, best=improved)
        cleanup_run(manager.root, keep_last=1, execute=True)
        final_epoch = epoch + 1
        if stop:
            break
    result = {
        "epoch": final_epoch,
        "global_step": global_step,
        "best_metric": early.best,
        "best_validation_hits1": best_validation_hits1,
        "latest_validation": latest_validation,
        "training_complete": True,
        "active_losses": active_losses,
        "removed_losses": ablation_metadata["removed_losses"],
        "loss_ablation": ablation_metadata,
        "loss_weights": loss_weights,
    }
    write_json(run_dir / "training_summary.json", result)
    return result

"""Discriminative learning-rate optimizer construction."""

import torch


def build_optimizer(reasoner, retriever=None, reranker=None, settings=None):
    settings = settings or {}
    external = set()
    if retriever is not None:
        external.update(id(parameter) for parameter in retriever.parameters())
    if reranker is not None:
        external.update(id(parameter) for parameter in reranker.parameters())
    reasoner_parameters = [parameter for parameter in reasoner.parameters() if id(parameter) not in external]
    groups = [{"params": reasoner_parameters, "lr": float(settings.get("reasoner_learning_rate", 2e-4))}]
    if retriever is not None:
        groups.append({"params": [p for p in retriever.parameters() if p.requires_grad],
                       "lr": float(settings.get("retriever_learning_rate", 5e-6))})
    if reranker is not None:
        groups.append({"params": reranker.parameters(), "lr": float(settings.get("reranker_learning_rate", 5e-6))})
    return torch.optim.AdamW(groups)

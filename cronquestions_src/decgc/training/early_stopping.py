"""Patience-based early stopping state."""

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    best: float = float("-inf")
    bad_epochs: int = 0

    def update(self, metric: float) -> tuple[bool, bool]:
        improved = metric > self.best
        if improved:
            self.best, self.bad_epochs = metric, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


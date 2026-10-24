from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score


@dataclass(frozen=True)
class FoldMetric:
    fold: int
    score: float
    n_samples: int


def score_predictions(task_type: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    targets = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    if targets.shape != predictions.shape:
        raise ValueError(f"Target/prediction shape mismatch: {targets.shape} != {predictions.shape}")
    if task_type == "regression":
        return float(mean_absolute_error(targets, predictions))
    if task_type == "classification":
        return float(roc_auc_score(targets, predictions))
    raise ValueError(f"Unknown task type: {task_type}")


def summarize_folds(folds: list[FoldMetric]) -> dict[str, float | int]:
    if not folds:
        raise ValueError("At least one fold metric is required")
    values = np.asarray([fold.score for fold in folds], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n_folds": len(folds),
        "n_samples": sum(fold.n_samples for fold in folds),
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hpsafe_sota.stage1_tpot.official import (
    HIDDEN_DIM,
    STACKING_COUNT,
    XGBOOST_TREE_COUNT,
    load_official_pipeline_template,
)


@dataclass(frozen=True)
class ReplayOutput:
    prediction: np.ndarray
    uncertainty: np.ndarray
    hidden: np.ndarray
    component_predictions: np.ndarray


def _seed_pipeline(value: Any, *, seed: int, threads: int) -> tuple[Any, dict[str, Any]]:
    from sklearn.base import clone

    pipeline = clone(value)
    parameters = pipeline.get_params(deep=True)
    updates: dict[str, Any] = {}
    for name in parameters:
        if name.endswith("random_state"):
            updates[name] = int(seed)
        elif name.endswith("n_jobs"):
            updates[name] = int(threads)
    pipeline.set_params(**updates)
    return pipeline, updates


def fresh_official_pipeline(project_root: Any, *, seed: int, threads: int) -> tuple[Any, dict[str, Any]]:
    template = load_official_pipeline_template(project_root)
    return _seed_pipeline(template, seed=seed, threads=threads)


def _intermediate(value: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    current = np.asarray(features, dtype=np.float64)
    stacking: list[np.ndarray] = []
    for name, step in value.steps[:-1]:
        before = current
        current = np.asarray(step.transform(current))
        if name.startswith("stackingestimator-"):
            estimator = step.estimator
            prediction = np.asarray(estimator.predict(before), dtype=np.float64).reshape(-1)
            stacking.append(prediction)
    if len(stacking) != STACKING_COUNT:
        raise ValueError(f"Expected {STACKING_COUNT} TPOT stacking responses, got {len(stacking)}")
    return current, np.stack(stacking, axis=1), value.steps[-1][1]


def _leaf_matrix(final_estimator: Any, final_features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    booster = final_estimator.get_booster()
    leaves = booster.predict(xgb.DMatrix(final_features), pred_leaf=True)
    leaves = np.asarray(leaves, dtype=np.float64)
    if leaves.ndim == 1:
        leaves = leaves[:, None]
    if leaves.shape[1] != XGBOOST_TREE_COUNT:
        raise ValueError(
            f"Official TPOT-Mat output XGBoost tree count {leaves.shape[1]} != {XGBOOST_TREE_COUNT}"
        )
    return leaves


def fit_pipeline(
    pipeline: Any, x_train: np.ndarray, y_train: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    pipeline.fit(x, y)
    final_features, stacking, final = _intermediate(pipeline, x)
    leaves = _leaf_matrix(final, final_features)
    target_scale = float(max(np.std(y), 1.0e-12))
    leaf_scale = np.maximum(np.max(np.abs(leaves), axis=0), 1.0)
    recipe = {
        "target_mean": float(np.mean(y)),
        "target_scale": target_scale,
        "leaf_scale": leaf_scale.tolist(),
        "stacking_count": int(stacking.shape[1]),
        "tree_count": int(leaves.shape[1]),
        "hidden_dim": HIDDEN_DIM,
        "hidden_source": "three_stacking_predictions_plus_xgboost_leaf_path",
    }
    return pipeline, recipe


def replay_pipeline(pipeline: Any, recipe: dict[str, Any], features: np.ndarray) -> ReplayOutput:
    x = np.asarray(features, dtype=np.float64)
    final_features, stacking, final = _intermediate(pipeline, x)
    final_prediction = np.asarray(final.predict(final_features), dtype=np.float64).reshape(-1)
    pipeline_prediction = np.asarray(pipeline.predict(x), dtype=np.float64).reshape(-1)
    if not np.allclose(final_prediction, pipeline_prediction, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("TPOT-Mat intermediate replay differs from Pipeline.predict")
    leaves = _leaf_matrix(final, final_features)
    target_mean = float(recipe["target_mean"])
    target_scale = float(recipe["target_scale"])
    leaf_scale = np.asarray(recipe["leaf_scale"], dtype=np.float64)
    if leaf_scale.shape != (XGBOOST_TREE_COUNT,):
        raise ValueError("TPOT-Mat hidden leaf scale drifted")
    stacking_standard = (stacking - target_mean) / target_scale
    hidden = np.concatenate([stacking_standard, leaves / leaf_scale], axis=1).astype(np.float32)
    if hidden.shape != (len(x), HIDDEN_DIM) or not np.all(np.isfinite(hidden)):
        raise ValueError("TPOT-Mat hidden representation is invalid")
    components = np.concatenate([stacking, pipeline_prediction[:, None]], axis=1)
    uncertainty = np.std(components, axis=1).astype(np.float64)
    return ReplayOutput(
        prediction=pipeline_prediction,
        uncertainty=uncertainty,
        hidden=hidden,
        component_predictions=components,
    )

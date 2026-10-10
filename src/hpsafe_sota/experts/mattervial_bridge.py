from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hpsafe_sota.experts.events import ProgressHeartbeat
from hpsafe_sota.experts.external_cache import ExternalFeatureCache
from hpsafe_sota.experts.interface import ExpertBridge
from hpsafe_sota.experts.target_firewall import TargetAccessFirewall


def _supported(function: Any, values: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_KERAS_SCOPE_SAFE_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]*$")
_MODEL_TARGET_NAME_POLICY = "preserve_safe_else_task_and_target_hash"


def _resolve_ensemble_member_count(extra: dict[str, Any], *, smoke_test: bool) -> int:
    """Return the declared ensemble size, with a two-member smoke-test cap."""
    configured = int(extra.get("n_models", 20))
    if configured < 1:
        raise ValueError("n_models must be positive")
    return min(configured, 2) if smoke_test else configured


def _preprocessing_signature_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid preprocessing receipt: {path}")
    scientific = {
        key: value
        for key, value in payload.items()
        if key not in {"xgb_checkpoint_resumed", "cross_nmi_checkpoint_resumed"}
    }
    return _payload_sha256(scientific)


def _modnet_internal_target_name(task_name: str, scientific_target_name: str) -> str:
    """Return a deterministic Keras-safe target alias for MODNet internals.

    Matbench scientific labels such as ``log10(G_VRH)`` are metadata, not
    valid TensorFlow root-scope names.  The bridge must therefore keep them at
    the artifact boundary and use a stable alias inside MODNet/Keras.  A hash
    makes aliases collision-resistant without leaking punctuation back into a
    layer name.
    """

    task_name = str(task_name).strip()
    scientific_target_name = str(scientific_target_name).strip()
    if not task_name or not scientific_target_name:
        raise ValueError("MatterVial task and scientific target names must be non-empty")
    # Preserve already-safe names so completed jobs and checkpoints from the
    # same graph remain byte-compatible with the pre-fix bridge.
    if _KERAS_SCOPE_SAFE_TARGET.fullmatch(scientific_target_name) is not None:
        return scientific_target_name
    safe_task = re.sub(r"[^A-Za-z0-9]+", "_", task_name).strip("_")[:48] or "task"
    digest = hashlib.sha256(
        f"{task_name}\0{scientific_target_name}".encode("utf-8")
    ).hexdigest()[:12]
    internal = f"hpsafe_{safe_task}_{digest}"
    if _KERAS_SCOPE_SAFE_TARGET.fullmatch(internal) is None:
        raise RuntimeError(f"Generated invalid Keras target alias: {internal!r}")
    return internal


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            mode="w",
            encoding="utf-8",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _numeric_cross_nmi(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize MODNet's object-typed cross-NMI matrix.

    MODNet 0.4.5 creates the matrix from an empty DataFrame and fills it cell
    by cell.  Some pandas releases retain ``object`` columns, for which the
    relevance-redundancy ``idxmax`` reduction fails.  Converting at the bridge
    boundary preserves the computed values and leaves the pinned vendor source
    untouched.
    """

    if frame.ndim != 2 or frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Cross-NMI must be square, observed {frame.shape}")
    index = [str(value) for value in frame.index]
    columns = [str(value) for value in frame.columns]
    if index != columns or len(set(columns)) != len(columns):
        raise ValueError("Cross-NMI rows and columns must be identical and unique")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Cross-NMI contains NaN or infinity after numeric conversion")
    return pd.DataFrame(values, index=index, columns=columns, dtype=np.float64)


def _load_preselection_checkpoint(
    path: Path,
    *,
    signature: str,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, int]]] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("signature") != signature or payload.get("status") != "complete":
            return None
        selected = [str(value) for value in payload["selected_columns"]]
        if not selected or len(selected) != len(set(selected)):
            return None
        if any(value not in frame.columns for value in selected):
            return None
        selection_trace = [
            {key: int(entry[key]) for key in ("iteration", "before", "dropped", "after")}
            for entry in payload["selection_trace"]
        ]
        if selection_trace and selection_trace[-1]["after"] != len(selected):
            return None
        return frame.loc[:, selected], selection_trace
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_preselection_checkpoint(
    path: Path,
    *,
    signature: str,
    selected_columns: list[str],
    selection_trace: list[dict[str, int]],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "complete",
            "algorithm": "recursive_xgboost_importance_task_aware",
            "signature": signature,
            "selected_columns": selected_columns,
            "selected_feature_count": len(selected_columns),
            "selection_trace": selection_trace,
            "target_values_stored": False,
        },
    )


def _load_cross_nmi_checkpoint(
    root: Path,
    *,
    signature: str,
    allowed_columns: list[str],
) -> pd.DataFrame | None:
    receipt_path = root / "cross_nmi.json"
    matrix_path = root / "cross_nmi.npz"
    if not receipt_path.is_file() or not matrix_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("signature") != signature
            or receipt.get("status") != "complete"
            or receipt.get("matrix_sha256") != _sha256(matrix_path)
        ):
            return None
        with np.load(matrix_path, allow_pickle=False) as loaded:
            columns = loaded["columns"].astype(str).tolist()
            values = np.asarray(loaded["values"], dtype=np.float64)
        if not columns or len(columns) != len(set(columns)):
            return None
        if any(value not in allowed_columns for value in columns):
            return None
        return _numeric_cross_nmi(
            pd.DataFrame(values, index=columns, columns=columns, dtype=np.float64)
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cross_nmi_checkpoint(
    root: Path,
    *,
    signature: str,
    cross_nmi: pd.DataFrame,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    matrix_path = root / "cross_nmi.npz"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, suffix=".npz.tmp", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                columns=np.asarray(cross_nmi.columns.astype(str), dtype=np.str_),
                values=cross_nmi.to_numpy(dtype=np.float64),
            )
        os.replace(temporary, matrix_path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    _atomic_json(
        root / "cross_nmi.json",
        {
            "schema_version": 1,
            "status": "complete",
            "algorithm": "modnet_get_cross_nmi_numeric",
            "signature": signature,
            "matrix": matrix_path.name,
            "matrix_sha256": _sha256(matrix_path),
            "feature_count": int(cross_nmi.shape[0]),
            "dtype": "float64",
            "target_values_stored": False,
        },
    )


def _prediction_values(result: Any, target: str, *, classification: bool) -> np.ndarray:
    if hasattr(result, "columns"):
        columns = [str(column) for column in result.columns]
        if classification:
            positive = f"{target}_prob_1"
            matching = [column for column in columns if column.startswith(target)]
            if positive in columns:
                raw = result[positive].to_numpy()
            elif len(matching) >= 2:
                raw = result[matching[-1]].to_numpy()
            elif matching:
                raw = result[matching[0]].to_numpy()
            else:
                raw = result.to_numpy()
        else:
            matched = [column for column in columns if column == target]
            matched = matched or [column for column in columns if column.startswith(target)]
            if not matched:
                raise ValueError(f"Regression output is missing target {target!r}: {columns}")
            raw = result[matched[0]].to_numpy()
    else:
        raw = np.asarray(result)
    array = np.asarray(raw)
    if array.dtype == object:
        rows = [np.asarray(value, dtype=np.float64).reshape(-1) for value in array.reshape(-1)]
        matrix = np.stack(rows)
        return (matrix[:, 1] if classification and matrix.shape[1] >= 2 else matrix[:, 0]).reshape(-1)
    array = array.astype(np.float64)
    if classification and array.ndim == 2 and array.shape[1] >= 2:
        return array[:, 1].reshape(-1)
    return array.reshape(-1)


def _uncertainty_values(
    result: Any,
    expected_rows: int,
    *,
    target: str,
    classification: bool,
) -> np.ndarray:
    if classification and hasattr(result, "columns"):
        positive = f"{target}_prob_1"
        columns = [str(column) for column in result.columns]
        matching = [column for column in columns if column.startswith(target)]
        raw = (
            result[positive].to_numpy()
            if positive in columns
            else result[matching[-1]].to_numpy()
            if matching
            else result.to_numpy()
        )
    else:
        raw = result.to_numpy() if hasattr(result, "to_numpy") else np.asarray(result)
    array = np.asarray(raw)
    if array.dtype == object:
        rows = []
        for row in array.reshape(expected_rows, -1):
            cells = [np.asarray(value, dtype=np.float64).reshape(-1) for value in row]
            rows.append(float(np.nanmean(np.concatenate(cells))))
        return np.asarray(rows, dtype=np.float64)
    values = array.astype(np.float64).reshape(expected_rows, -1)
    return np.nanmean(values, axis=1)


def _write_live_predictions(
    path: Path,
    *,
    sample_ids: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    member_count: int,
    total_members: int,
    prediction_role: str,
    outer_fold: int,
    inner_fold: int | None,
) -> None:
    """Atomically publish target-free partial-ensemble predictions for monitoring."""
    ids = np.asarray(sample_ids).astype(str)
    predictions = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(y_std, dtype=np.float64).reshape(-1)
    if not (len(ids) == len(predictions) == len(uncertainty)):
        raise ValueError("Live prediction arrays have inconsistent row counts")
    if not np.isfinite(predictions).all() or not np.isfinite(uncertainty).all():
        raise ValueError("Live predictions must be finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(
                handle,
                sample_ids=ids,
                y_pred=predictions,
                y_std=uncertainty,
                member_count=np.asarray([member_count], dtype=np.int64),
                total_members=np.asarray([total_members], dtype=np.int64),
                prediction_role=np.asarray([prediction_role]),
                outer_fold=np.asarray([outer_fold], dtype=np.int64),
                inner_fold=np.asarray([-1 if inner_fold is None else inner_fold], dtype=np.int64),
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _seal_final_live_predictions(
    path: Path,
    *,
    sample_ids: np.ndarray,
    y_pred: np.ndarray,
    y_uncertainty: np.ndarray | None,
    member_count: int,
    prediction_role: str,
    outer_fold: int,
    inner_fold: int | None,
) -> dict[str, Any]:
    """Write the authoritative MODNet ensemble prediction to the live output.

    MODNet's ensemble-level prediction path applies its own post-processing, so the
    returned ensemble prediction is written directly. Existing live uncertainty is
    retained when the ensemble call does not return a separate uncertainty array.
    """

    ids = np.asarray(sample_ids).astype(str)
    current_prediction = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(ids) != len(current_prediction) or not np.isfinite(current_prediction).all():
        raise ValueError("Live prediction inputs are invalid")

    existing_sha256: str | None = None
    existing_max_abs_delta: float | None = None
    fallback_uncertainty: np.ndarray | None = None
    if path.is_file():
        existing_sha256 = _sha256(path)
        with np.load(path, allow_pickle=False) as loaded:
            if "y_true" in loaded.files:
                raise ValueError(f"Live snapshot contains forbidden targets: {path}")
            existing_ids = loaded["sample_ids"].astype(str)
            existing_predictions = np.asarray(loaded["y_pred"], dtype=np.float64).reshape(-1)
            fallback_uncertainty = np.asarray(loaded["y_std"], dtype=np.float64).reshape(-1)
            existing_members = int(loaded["member_count"][0])
            existing_total = int(loaded["total_members"][0])
            existing_role = str(loaded["prediction_role"][0])
            existing_outer = int(loaded["outer_fold"][0])
            existing_inner = int(loaded["inner_fold"][0])
        expected_inner = -1 if inner_fold is None else int(inner_fold)
        if (
            not np.array_equal(existing_ids, ids)
            or existing_predictions.shape != current_prediction.shape
            or fallback_uncertainty.shape != current_prediction.shape
            or existing_members != member_count
            or existing_total != member_count
            or existing_role != prediction_role
            or existing_outer != int(outer_fold)
            or existing_inner != expected_inner
        ):
            raise ValueError(f"Existing live snapshot identity is invalid: {path}")
        existing_max_abs_delta = float(np.max(np.abs(existing_predictions - current_prediction)))

    if y_uncertainty is None:
        if fallback_uncertainty is None:
            raise ValueError("Ensemble uncertainty and live uncertainty are both unavailable")
        uncertainty = fallback_uncertainty
        uncertainty_source = "partial_member_standard_deviation"
    else:
        uncertainty = np.asarray(y_uncertainty, dtype=np.float64).reshape(-1)
        uncertainty_source = "ensemble_uncertainty"
    if (
        uncertainty.shape != current_prediction.shape
        or not np.isfinite(uncertainty).all()
        or (uncertainty < 0).any()
    ):
        raise ValueError("Live uncertainty is invalid")

    _write_live_predictions(
        path,
        sample_ids=ids,
        y_pred=current_prediction,
        y_std=uncertainty,
        member_count=member_count,
        total_members=member_count,
        prediction_role=prediction_role,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    with np.load(path, allow_pickle=False) as loaded:
        if "y_true" in loaded.files:
            raise ValueError(f"Sealed live snapshot contains forbidden targets: {path}")
        sealed = np.asarray(loaded["y_pred"], dtype=np.float64).reshape(-1)
    sealed_max_abs_delta = float(np.max(np.abs(sealed - current_prediction)))
    if sealed_max_abs_delta != 0.0:
        raise RuntimeError(f"Live prediction seal is not byte-value exact: {path}")
    return {
        "schema_version": 1,
        "status": "sealed",
        "path": str(path),
        "target_values_read": False,
        "target_values_stored": False,
        "member_count": member_count,
        "prediction_role": prediction_role,
        "existing_sha256": existing_sha256,
        "existing_max_abs_delta_vs_ensemble": existing_max_abs_delta,
        "uncertainty_source": uncertainty_source,
        "sealed_sha256": _sha256(path),
        "sealed_max_abs_delta_vs_ensemble": sealed_max_abs_delta,
    }


def _fit_apply_latent_imputer(
    fit_latent: np.ndarray,
    prediction_latent: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Impute target-free latent values using fitting-partition statistics only.

    MatterVial support features can contain isolated NaN/Inf values.  Computing
    medians from the current fit partition avoids both label access and held-out
    feature-distribution leakage.  An entirely non-finite latent dimension is
    treated as a corrupt recipe and fails closed rather than being silently
    replaced by a constant.
    """
    fit = np.asarray(fit_latent, dtype=np.float64)
    prediction = np.asarray(prediction_latent, dtype=np.float64)
    if fit.ndim != 2 or prediction.ndim != 2 or fit.shape[1] != prediction.shape[1]:
        raise ValueError(
            "Fit and prediction latent arrays must be 2D with identical feature widths"
        )
    fit_finite = np.isfinite(fit)
    prediction_finite = np.isfinite(prediction)
    all_missing = np.flatnonzero(~fit_finite.any(axis=0))
    if len(all_missing):
        raise ValueError(
            "MVL latent has dimensions with no finite fit-partition values: "
            f"{all_missing[:10].tolist()}"
        )
    statistics = np.empty(fit.shape[1], dtype=np.float64)
    for column in range(fit.shape[1]):
        statistics[column] = np.median(fit[fit_finite[:, column], column])
    repaired = prediction.copy()
    bad_rows, bad_columns = np.nonzero(~prediction_finite)
    if len(bad_rows):
        repaired[bad_rows, bad_columns] = statistics[bad_columns]
    if not np.isfinite(repaired).all():
        raise RuntimeError("Fit-partition latent median imputation did not produce finite values")
    fit_bad_rows = np.flatnonzero(~fit_finite.all(axis=1))
    prediction_bad_rows = np.flatnonzero(~prediction_finite.all(axis=1))
    return repaired, {
        "policy": "fit_partition_finite_median_per_dimension",
        "target_values_accessed": False,
        "heldout_features_used_to_fit_statistics": False,
        "fit_nonfinite_value_count": int((~fit_finite).sum()),
        "fit_nonfinite_row_count": int(len(fit_bad_rows)),
        "prediction_nonfinite_value_count": int((~prediction_finite).sum()),
        "prediction_nonfinite_row_count": int(len(prediction_bad_rows)),
        "prediction_nonfinite_row_positions": prediction_bad_rows[:100].tolist(),
        "statistics_sha256": hashlib.sha256(
            statistics.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "all_nonfinite_fit_dimension_count": 0,
    }


def _recursive_xgb_preselection(
    frame: pd.DataFrame,
    targets: np.ndarray,
    *,
    target_threshold: int,
    drop_fraction: float,
    n_jobs: int,
    random_state: int,
    task_type: str,
    heartbeat: ProgressHeartbeat,
) -> tuple[pd.DataFrame, list[dict[str, int]]]:
    import xgboost as xgb

    if target_threshold < 1:
        raise ValueError("xgb_preselect_target must be positive")
    if not 0.0 < drop_fraction < 1.0:
        raise ValueError("xgb_preselect_fraction must be in (0, 1)")
    current = frame.copy()
    selection_trace: list[dict[str, int]] = []
    iteration = 0
    while current.shape[1] > target_threshold:
        iteration += 1
        heartbeat.update(
            "xgb_preselection",
            detail=f"iteration={iteration} features={current.shape[1]}",
        )
        numeric = (
            current.apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if task_type == "classification":
            model = xgb.XGBClassifier(
                n_jobs=n_jobs,
                random_state=random_state,
                objective="binary:logistic",
                eval_metric="logloss",
            )
            fit_targets = np.asarray(targets, dtype=np.int64)
        elif task_type == "regression":
            model = xgb.XGBRegressor(
                n_jobs=n_jobs,
                random_state=random_state,
                objective="reg:squarederror",
            )
            fit_targets = np.asarray(targets, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported MatterVial task type: {task_type}")
        model.fit(numeric.to_numpy(), fit_targets)
        current_count = current.shape[1]
        drop_count = max(1, int(current_count * drop_fraction))
        if current_count - drop_count < target_threshold:
            drop_count = current_count - target_threshold
        order = np.argsort(np.asarray(model.feature_importances_), kind="stable")
        to_drop = current.columns[order[:drop_count]]
        current = current.drop(columns=to_drop)
        selection_trace.append(
            {
                "iteration": iteration,
                "before": current_count,
                "dropped": int(drop_count),
                "after": int(current.shape[1]),
            }
        )
    return current, selection_trace


@dataclass
class PreparedMatterVialData:
    train_data: Any
    prediction_data: Any
    latent_prediction: np.ndarray
    scientific_target_name: str
    model_target_name: str
    receipt: Path
    heartbeat: ProgressHeartbeat


@dataclass
class FittedMatterVialModel:
    model: Any
    checkpoint: Path


class MatterVialMODNetBridge(ExpertBridge):
    """Task-frozen published MatterVial recipe with split-scoped fitting.

    Predictor blocks are supplied by an immutable per-task recipe.  The Stage2
    semantic latent is assembled only from pretrained, target-free cache blocks
    and deterministically right-padded to 288 dimensions when necessary.
    """

    def describe(self) -> dict[str, Any]:
        return {
            "bridge": "mattervial_modnet_multitask",
            "backend": "published_mattervial_features_plus_modnet_ensemble",
            "feature_selection_scope": "current_fit_partition_only",
            "prediction_target_visibility": "none_until_artifact_evaluation",
            "predictor_blocks": "task_frozen_request",
            "latent": "canonical_target_free_task_recipe_right_padded_288d",
            "pretraining_overlap_status": "requires_separate_structure_overlap_audit",
        }

    def prepare(self) -> PreparedMatterVialData:
        from modnet.preprocessing import MODData, get_cross_nmi

        definition = self.manifest.metadata["definition"]
        task_type = str(definition["task_type"])
        configured_task_type = str(self.request.extra.get("task_type", task_type))
        if configured_task_type != task_type:
            raise ValueError(
                f"MatterVial recipe task type {configured_task_type!r} != manifest {task_type!r}"
            )
        heartbeat = ProgressHeartbeat(
            self.request.output_path / "progress.json",
            interval_seconds=float(self.request.extra.get("heartbeat_seconds", 30)),
        )
        self._heartbeat = heartbeat
        heartbeat.__enter__()
        try:
            heartbeat.update("cache_validation", detail="opening immutable feature cache")
            cache_value = self.request.extra.get("feature_cache_dir")
            if not cache_value:
                raise ValueError("MatterVial bridge requires feature_cache_dir")
            cache = ExternalFeatureCache.open(
                Path(str(cache_value)),
                manifest=self.manifest,
                verify_hashes=bool(self.request.extra.get("verify_cache_hashes", False)),
            )
            predictor_blocks = tuple(str(value) for value in self.request.extra.get("predictor_blocks", []))
            latent_blocks = tuple(str(value) for value in self.request.extra.get("latent_blocks", []))
            if not predictor_blocks or not latent_blocks:
                raise ValueError("MatterVial recipe requires non-empty predictor_blocks and latent_blocks")
            cached_blocks = set(cache.receipt.get("blocks", {}))
            missing_blocks = sorted(set(predictor_blocks + latent_blocks).difference(cached_blocks))
            if missing_blocks:
                raise ValueError(f"MatterVial cache is missing frozen blocks: {missing_blocks}")
            train_positions = np.asarray(self.split.train_positions, dtype=np.int64)
            prediction_positions = np.asarray(self.split.prediction_positions, dtype=np.int64)
            firewall = TargetAccessFirewall(
                self.manifest,
                allowed_fit_positions=train_positions,
                prediction_positions=prediction_positions,
            )
            targets = firewall.targets_for_fit(train_positions)
            if not np.isfinite(np.asarray(targets, dtype=np.float64)).all():
                raise ValueError("Training targets contain NaN or infinity")

            heartbeat.update("load_predictor_blocks", detail="memory-mapped cache rows")
            train_frame = cache.frame(predictor_blocks, train_positions)
            prediction_frame = cache.frame(predictor_blocks, prediction_positions)
            if list(train_frame.columns) != list(prediction_frame.columns):
                raise RuntimeError("Training and prediction feature schemas differ")
            train_nonfinite = int((~np.isfinite(train_frame.to_numpy(dtype=np.float64))).sum())
            prediction_nonfinite = int(
                (~np.isfinite(prediction_frame.to_numpy(dtype=np.float64))).sum()
            )
            # MODNet learns its scaler/imputer on the fitting partition.  Mapping
            # infinity to NaN is target-independent and lets that train-only
            # imputer handle both train and held-out rows consistently.
            train_frame = train_frame.replace([np.inf, -np.inf], np.nan)
            prediction_frame = prediction_frame.replace([np.inf, -np.inf], np.nan)

            n_jobs = int(self.request.extra.get("feature_selection_n_jobs", 16))
            smoke_test = bool(self.request.extra.get("smoke_test", False))
            requested_n_feat = int(self.request.extra.get("n_feat", 512))
            effective_n_feat = min(requested_n_feat, 16) if smoke_test else requested_n_feat
            xgb_target = int(self.request.extra.get("xgb_preselect_target", 800))
            xgb_drop_fraction = float(self.request.extra.get("xgb_preselect_fraction", 0.1))
            if smoke_test:
                xgb_target = min(
                    xgb_target,
                    int(self.request.extra.get("smoke_xgb_preselect_target", 64)),
                )
                xgb_drop_fraction = max(
                    xgb_drop_fraction,
                    float(self.request.extra.get("smoke_xgb_preselect_fraction", 0.35)),
                )
            xgb_target = max(effective_n_feat, xgb_target)
            xgb_random_state = int(self.request.extra.get("xgb_random_state", 1))
            checkpoint_root = self.request.output_path / "preprocessing_checkpoints"
            fit_positions_sha256 = hashlib.sha256(
                train_positions.tobytes(order="C")
            ).hexdigest()
            input_columns = train_frame.columns.astype(str).tolist()
            input_columns_sha256 = hashlib.sha256(
                "\n".join(input_columns).encode("utf-8")
            ).hexdigest()
            preselection_signature = _payload_sha256(
                {
                    "schema_version": 1,
                    "algorithm": "recursive_xgboost_importance_task_aware",
                    "task_type": task_type,
                    "manifest_digest": self.manifest.digest,
                    "cache_key": cache.cache_key,
                    "fit_positions_sha256": fit_positions_sha256,
                    "input_columns_sha256": input_columns_sha256,
                    "target_threshold": xgb_target,
                    "drop_fraction": xgb_drop_fraction,
                    "n_jobs": n_jobs,
                    "random_state": xgb_random_state,
                }
            )
            preselection_path = checkpoint_root / "xgb_preselection.json"
            resumed_preselection = _load_preselection_checkpoint(
                preselection_path,
                signature=preselection_signature,
                frame=train_frame,
            )
            if resumed_preselection is None:
                selected_frame, xgb_selection_trace = _recursive_xgb_preselection(
                    train_frame,
                    np.asarray(targets),
                    target_threshold=xgb_target,
                    drop_fraction=xgb_drop_fraction,
                    n_jobs=n_jobs,
                    random_state=xgb_random_state,
                    task_type=task_type,
                    heartbeat=heartbeat,
                )
                _write_preselection_checkpoint(
                    preselection_path,
                    signature=preselection_signature,
                    selected_columns=selected_frame.columns.astype(str).tolist(),
                    selection_trace=xgb_selection_trace,
                )
                xgb_checkpoint_resumed = False
            else:
                selected_frame, xgb_selection_trace = resumed_preselection
                xgb_checkpoint_resumed = True
                heartbeat.update(
                    "xgb_preselection_resume",
                    detail=f"features={selected_frame.shape[1]}",
                )
            selected_columns = selected_frame.columns.astype(str).tolist()
            prediction_frame = prediction_frame.loc[:, selected_columns]
            scientific_target_name = str(definition["target"])
            model_target_name = _modnet_internal_target_name(
                self.request.task_name,
                scientific_target_name,
            )
            num_classes = {
                model_target_name: 2 if task_type == "classification" else 0
            }
            train_ids = self.manifest.sample_ids[train_positions].astype(str).tolist()
            prediction_ids = self.manifest.sample_ids[prediction_positions].astype(str).tolist()
            train_data = MODData(
                df_featurized=selected_frame,
                targets=np.asarray(targets).reshape(-1, 1),
                target_names=[model_target_name],
                structure_ids=train_ids,
                num_classes=num_classes,
            )
            heartbeat.update(
                "modnet_nmi_feature_selection",
                detail=f"rows={len(train_ids)} features={len(selected_columns)}",
            )
            nmi_n_samples = int(self.request.extra.get("feature_selection_n_samples", 6000))
            nmi_drop_threshold = float(
                self.request.extra.get("feature_selection_drop_threshold", 0.2)
            )
            if nmi_n_samples < 1:
                raise ValueError("feature_selection_n_samples must be positive")
            cross_nmi_signature = _payload_sha256(
                {
                    "schema_version": 1,
                    "algorithm": "modnet_get_cross_nmi_numeric",
                    "manifest_digest": self.manifest.digest,
                    "cache_key": cache.cache_key,
                    "fit_positions_sha256": fit_positions_sha256,
                    "selected_columns_sha256": hashlib.sha256(
                        "\n".join(selected_columns).encode("utf-8")
                    ).hexdigest(),
                    "n_samples": nmi_n_samples,
                    "drop_threshold": nmi_drop_threshold,
                    "random_state": int(self.request.seed),
                }
            )
            cross_nmi = _load_cross_nmi_checkpoint(
                checkpoint_root,
                signature=cross_nmi_signature,
                allowed_columns=selected_columns,
            )
            if cross_nmi is None:
                if len(selected_frame) > nmi_n_samples:
                    nmi_frame = selected_frame.sample(n=nmi_n_samples, random_state=12)
                else:
                    nmi_frame = selected_frame.copy()
                raw_cross_nmi, _ = get_cross_nmi(
                    nmi_frame,
                    return_entropy=True,
                    drop_thr=nmi_drop_threshold,
                    n_jobs=n_jobs,
                    random_state=int(self.request.seed),
                )
                cross_nmi = _numeric_cross_nmi(raw_cross_nmi)
                _write_cross_nmi_checkpoint(
                    checkpoint_root,
                    signature=cross_nmi_signature,
                    cross_nmi=cross_nmi,
                )
                cross_nmi_checkpoint_resumed = False
            else:
                cross_nmi_checkpoint_resumed = True
                heartbeat.update(
                    "modnet_cross_nmi_resume",
                    detail=f"features={cross_nmi.shape[0]} dtype=float64",
                )
            ranking_feature_count = min(effective_n_feat, int(cross_nmi.shape[0]))
            if ranking_feature_count < 1:
                raise RuntimeError("MODNet cross-NMI filtering removed every descriptor")
            feature_selection_values = {
                "n": ranking_feature_count,
                "cross_nmi": cross_nmi,
                "n_jobs": n_jobs,
                "use_precomputed_cross_nmi": False,
                "random_state": int(self.request.seed),
            }
            train_data.feature_selection(
                **_supported(train_data.feature_selection, feature_selection_values)
            )
            optimal_features = [str(value) for value in train_data.optimal_features]
            if not optimal_features:
                raise RuntimeError("MODNet NMI feature selection returned no descriptors")
            prediction_data = MODData(
                df_featurized=prediction_frame,
                structure_ids=prediction_ids,
            )
            prediction_data.optimal_features = optimal_features

            fit_latent_parts = [cache.block(name, train_positions) for name in latent_blocks]
            prediction_latent_parts = [
                cache.block(name, prediction_positions) for name in latent_blocks
            ]
            fit_latent = np.concatenate(fit_latent_parts, axis=1)
            latent = np.concatenate(prediction_latent_parts, axis=1)
            latent, latent_imputation = _fit_apply_latent_imputer(fit_latent, latent)
            expected_latent_dim = int(self.request.extra.get("latent_dim", 288))
            unpadded_latent_dim = int(latent.shape[1])
            if unpadded_latent_dim > expected_latent_dim:
                raise ValueError(
                    f"Target-free MatterVial latent width {unpadded_latent_dim} exceeds "
                    f"the frozen {expected_latent_dim}-dimensional Stage2 contract"
                )
            if unpadded_latent_dim < expected_latent_dim:
                latent = np.pad(
                    latent,
                    ((0, 0), (0, expected_latent_dim - unpadded_latent_dim)),
                    mode="constant",
                    constant_values=0.0,
                )
            if latent.shape != (len(prediction_positions), expected_latent_dim):
                raise ValueError(f"Canonical MatterVial latent has unexpected shape {latent.shape}")
            available = np.isfinite(latent).all(axis=1)
            if not available.all():
                raise RuntimeError("Latent contract contains non-finite values after imputation")

            firewall_path = firewall.write_receipt(
                self.request.output_path / "target_access_receipt.json"
            )
            receipt = self.request.output_path / "preprocessing_manifest.json"
            receipt_payload = {
                "schema_version": 1,
                "policy": "published_recipe_frozen_before_outer_test",
                "cache_dir": str(cache.root),
                "cache_key": cache.cache_key,
                "cache_receipt_sha256": _sha256(cache.root / "receipt.json"),
                "cache_hashes_verified_in_this_job": bool(
                    self.request.extra.get("verify_cache_hashes", False)
                ),
                "predictor_blocks": list(predictor_blocks),
                "latent_blocks": list(latent_blocks),
                "latent_dim": expected_latent_dim,
                "latent_unpadded_dim": unpadded_latent_dim,
                "latent_right_padding_dim": expected_latent_dim - unpadded_latent_dim,
                "latent_semantics": "pretrained_target_free_blocks_then_deterministic_right_padding",
                "task_type": task_type,
                "scientific_target_name": scientific_target_name,
                "model_internal_target_name": model_target_name,
                "model_internal_target_name_policy": _MODEL_TARGET_NAME_POLICY,
                "feature_selection_scope": "fit_partition_only",
                "xgb_selection_trace": xgb_selection_trace,
                "xgb_selected_feature_count": len(selected_columns),
                "xgb_target_feature_count": xgb_target,
                "xgb_drop_fraction": xgb_drop_fraction,
                "xgb_checkpoint_resumed": xgb_checkpoint_resumed,
                "xgb_checkpoint": str(preselection_path.relative_to(self.request.output_path)),
                "cross_nmi_dtype": str(cross_nmi.to_numpy().dtype),
                "cross_nmi_feature_count": int(cross_nmi.shape[0]),
                "cross_nmi_checkpoint_resumed": cross_nmi_checkpoint_resumed,
                "cross_nmi_checkpoint": str(
                    (checkpoint_root / "cross_nmi.json").relative_to(self.request.output_path)
                ),
                "feature_ranking_count": ranking_feature_count,
                "model_requested_feature_count": effective_n_feat,
                "modnet_optimal_feature_count": len(optimal_features),
                "modnet_optimal_features": optimal_features,
                "nonfinite_to_nan_policy": "outer_train_modnet_imputer",
                "train_nonfinite_input_count": train_nonfinite,
                "prediction_nonfinite_input_count": prediction_nonfinite,
                "fit_positions": train_positions.tolist(),
                "prediction_positions": prediction_positions.tolist(),
                "target_firewall_receipt": firewall_path.name,
                "target_firewall_receipt_sha256": _sha256(firewall_path),
                "preprocessing_fit_scope": "outer_train",
            }
            # Record latent preprocessing in the receipt.
            if (
                latent_imputation["fit_nonfinite_value_count"]
                or latent_imputation["prediction_nonfinite_value_count"]
            ):
                receipt_payload["latent_nonfinite_policy"] = latent_imputation
            _atomic_json(receipt, receipt_payload)
            heartbeat.update("prepared", detail=f"optimal_features={len(optimal_features)}")
            return PreparedMatterVialData(
                train_data=train_data,
                prediction_data=prediction_data,
                latent_prediction=np.asarray(latent, dtype=np.float32),
                scientific_target_name=scientific_target_name,
                model_target_name=model_target_name,
                receipt=receipt,
                heartbeat=heartbeat,
            )
        except Exception as exc:
            heartbeat.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def fit(self, prepared: PreparedMatterVialData) -> FittedMatterVialModel:
        from modnet.models import EnsembleMODNetModel, MODNetModel
        from sklearn.utils import resample

        random.seed(self.request.seed)
        np.random.seed(self.request.seed)
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.request.seed)
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        smoke_test = bool(self.request.extra.get("smoke_test", False))
        n_models = _resolve_ensemble_member_count(
            self.request.extra, smoke_test=smoke_test
        )
        epochs = int(self.request.extra.get("epochs", 1000))
        n_feat = int(self.request.extra.get("n_feat", 512))
        if smoke_test:
            epochs = min(epochs, 2)
            n_feat = min(n_feat, 16)
        n_feat = min(n_feat, len(prepared.train_data.optimal_features))
        definition = self.manifest.metadata["definition"]
        scientific_target = prepared.scientific_target_name
        target = prepared.model_target_name
        task_type = str(definition["task_type"])
        classification = task_type == "classification"
        if task_type not in {"regression", "classification"}:
            raise ValueError(f"Unsupported MatterVial task type: {task_type}")
        fit_started = time.monotonic()
        prepared.heartbeat.update(
            "modnet_ensemble_fit",
            detail=f"models={n_models} epochs={epochs} n_feat={n_feat}",
            completed=0,
            total=n_models * epochs,
            fit_elapsed_seconds=0.0,
        )

        heartbeat = prepared.heartbeat

        class EpochProgress(tf.keras.callbacks.Callback):
            def __init__(self, model_number: int) -> None:
                super().__init__()
                self.model_number = model_number
                # At most ~1% of a member between visible updates.  This keeps
                # the dashboard responsive without writing on every epoch.
                self.update_every = max(1, epochs // 100)

            def on_epoch_end(self, epoch: int, logs: Any = None) -> None:
                if (epoch + 1) % self.update_every != 0 and epoch + 1 != epochs:
                    return
                completed = min(
                    n_models * epochs,
                    (self.model_number - 1) * epochs + epoch + 1,
                )
                loss = None if not isinstance(logs, dict) else logs.get("loss")
                heartbeat.update(
                    "modnet_ensemble_fit",
                    detail=f"model={self.model_number}/{n_models} epoch={epoch + 1}/{epochs}",
                    completed=completed,
                    total=n_models * epochs,
                    ensemble_member=self.model_number,
                    ensemble_total_members=n_models,
                    member_epoch=epoch + 1,
                    member_total_epochs=epochs,
                    train_loss=(
                        None
                        if loss is None or not np.isfinite(float(loss))
                        else float(loss)
                    ),
                    fit_elapsed_seconds=round(time.monotonic() - fit_started, 3),
                )

        model = EnsembleMODNetModel(
            n_models=n_models,
            targets=[[[target]]],
            weights={target: 1.0},
            num_classes={target: 2 if classification else 0},
            num_neurons=self.request.extra.get("num_neurons", [[128], [32], [8], [8]]),
            n_feat=n_feat,
            act=str(self.request.extra.get("act", "relu")),
            random_state=int(self.request.seed),
        )
        configured_loss = self.request.extra.get("loss", None if classification else "mae")
        fit_values = {
            "lr": float(self.request.extra.get("lr", 0.005)),
            "epochs": epochs,
            "batch_size": int(self.request.extra.get("batch_size", 64)),
            "loss": configured_loss,
            "verbose": int(self.request.extra.get("verbose", 0)),
            "val_fraction": 0,
        }
        preprocessing_semantic_sha256 = _preprocessing_signature_sha256(prepared.receipt)
        recipe_signature_common = {
            "scientific_target_name": scientific_target,
            "model_internal_target_name": target,
            "model_internal_target_name_policy": _MODEL_TARGET_NAME_POLICY,
            "seed": int(self.request.seed),
            "smoke_test": smoke_test,
            "n_models": n_models,
            "n_feat": n_feat,
            "num_neurons": self.request.extra.get("num_neurons", [[128], [32], [8], [8]]),
            "act": str(self.request.extra.get("act", "relu")),
            "fit_values": fit_values,
        }
        recipe_signature_payload = {
            "schema_version": 2,
            "preprocessing_manifest_semantic_sha256": preprocessing_semantic_sha256,
            "preprocessing_signature_policy": (
                "scientific_recipe_fields"
            ),
            **recipe_signature_common,
        }
        recipe_signature = hashlib.sha256(
            json.dumps(
                recipe_signature_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        member_root = self.request.output_path / "ensemble_members"
        member_root.mkdir(parents=True, exist_ok=True)
        allow_resume = bool(self.request.extra.get("resume_members", True))
        resumed_members = 0
        resume_rejections: list[dict[str, Any]] = []
        member_receipts: list[dict[str, Any]] = []
        n_train = len(prepared.train_data.df_targets)
        n_prediction = len(self.split.prediction_positions)
        live_sum = np.zeros(n_prediction, dtype=np.float64)
        live_sum_squares = np.zeros(n_prediction, dtype=np.float64)
        live_prediction_path = self.request.output_path / "live_predictions.npz"
        prediction_ids = self.manifest.sample_ids[self.split.prediction_positions]

        def publish_live_prediction(member_index: int) -> None:
            nonlocal live_sum, live_sum_squares
            predict_values = {
                "return_prob": classification,
                "remap_out_of_bounds": not classification,
            }
            result = model.models[member_index].predict(
                prepared.prediction_data,
                **_supported(model.models[member_index].predict, predict_values),
            )
            if isinstance(result, tuple):
                result = result[0]
            values = _prediction_values(result, target, classification=classification)
            if values.shape != (n_prediction,) or not np.isfinite(values).all():
                raise ValueError("Partial ensemble member produced invalid predictions")
            if classification and (np.any(values < 0) or np.any(values > 1)):
                raise ValueError("Partial ensemble produced invalid class probabilities")
            live_sum += values
            live_sum_squares += np.square(values)
            member_count = member_index + 1
            mean = live_sum / member_count
            variance = np.maximum(live_sum_squares / member_count - np.square(mean), 0.0)
            _write_live_predictions(
                live_prediction_path,
                sample_ids=prediction_ids,
                y_pred=mean,
                y_std=np.sqrt(variance),
                member_count=member_count,
                total_members=n_models,
                prediction_role=self.split.prediction_role,
                outer_fold=self.request.outer_fold,
                inner_fold=self.request.inner_fold,
            )
            heartbeat.update(
                "live_partial_ensemble",
                detail=(
                    f"members={member_count}/{n_models} role={self.split.prediction_role}; "
                    "target-free snapshot published"
                ),
                completed=member_count * epochs,
                total=n_models * epochs,
                ensemble_member=member_count,
                ensemble_total_members=n_models,
                live_prediction_path=str(live_prediction_path),
                live_prediction_policy="read_only_monitor_never_training_feedback",
                fit_elapsed_seconds=round(time.monotonic() - fit_started, 3),
            )

        for member_index in range(n_models):
            member_number = member_index + 1
            member_path = member_root / f"member_{member_number:02d}.pkl"
            member_receipt_path = member_root / f"member_{member_number:02d}.json"
            stratify_targets = (
                np.asarray(prepared.train_data.df_targets[target], dtype=np.int64)
                if classification
                else None
            )
            bootstrap_indices = np.asarray(
                resample(
                    np.arange(n_train),
                    replace=True,
                    n_samples=n_train,
                    random_state=int(self.request.seed) + member_index,
                    stratify=stratify_targets,
                ),
                dtype=np.int64,
            )
            bootstrap_sha256 = hashlib.sha256(
                bootstrap_indices.tobytes(order="C")
            ).hexdigest()
            member_signature = hashlib.sha256(
                f"{recipe_signature}:{member_index}:{bootstrap_sha256}".encode()
            ).hexdigest()
            reusable: dict[str, Any] | None = None
            resume_mode: str | None = None
            if allow_resume and member_path.is_file() and member_receipt_path.is_file():
                try:
                    candidate = json.loads(member_receipt_path.read_text(encoding="utf-8"))
                    candidate_recipe_signature = str(candidate.get("recipe_signature", ""))
                    accepted_member_signature = hashlib.sha256(
                        (
                            f"{candidate_recipe_signature}:{member_index}:"
                            f"{bootstrap_sha256}"
                        ).encode()
                    ).hexdigest()
                    if (
                        candidate.get("status") == "complete"
                        and int(candidate.get("member_index", -1)) == member_index
                        and candidate.get("bootstrap_indices_sha256") == bootstrap_sha256
                        and candidate_recipe_signature == recipe_signature
                        and candidate.get("member_signature") == accepted_member_signature
                        and candidate.get("checkpoint_sha256") == _sha256(member_path)
                    ):
                        reusable = candidate
                        resume_mode = "final_recipe_signature"
                    else:
                        resume_rejections.append(
                            {
                                "member": member_number,
                                "reason": "recipe_or_checkpoint_signature_mismatch",
                            }
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    reusable = None
                    resume_rejections.append(
                        {"member": member_number, "reason": "member_receipt_unreadable"}
                    )
            if reusable is not None:
                print(
                    "HPSAFE_MEMBER_RESUME "
                    + json.dumps(
                        {
                            "event": "member_resume",
                            "member": member_number,
                            "total_members": n_models,
                            "mode": resume_mode,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                heartbeat.update(
                    "modnet_member_resume",
                    detail=f"member={member_number}/{n_models} mode={resume_mode}",
                    completed=member_number * epochs,
                    total=n_models * epochs,
                )
                model.models[member_index] = MODNetModel.load(str(member_path))
                resumed_members += 1
                member_receipts.append(reusable)
                publish_live_prediction(member_index)
                continue

            if member_path.is_file() or member_receipt_path.is_file():
                rejection_reason = (
                    resume_rejections[-1]["reason"]
                    if resume_rejections
                    and resume_rejections[-1]["member"] == member_number
                    else "atomic_member_pair_incomplete"
                )
                heartbeat.update(
                    "modnet_member_resume_rejected",
                    detail=f"member={member_number}/{n_models} reason={rejection_reason}",
                    completed=member_index * epochs,
                    total=n_models * epochs,
                )
                print(
                    "HPSAFE_MEMBER_RESUME "
                    + json.dumps(
                        {
                            "event": "member_resume_rejected",
                            "member": member_number,
                            "total_members": n_models,
                            "reason": rejection_reason,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            bootstrap_data, _ = prepared.train_data.split((bootstrap_indices.tolist(), []))
            heartbeat.update(
                "modnet_ensemble_fit",
                detail=f"model={member_number}/{n_models} epoch=0/{epochs}",
                completed=member_index * epochs,
                total=n_models * epochs,
            )
            model.models[member_index].fit(
                bootstrap_data,
                **fit_values,
                callbacks=[EpochProgress(member_number)],
            )
            with tempfile.NamedTemporaryFile(
                dir=member_root,
                suffix=".pkl.tmp",
                delete=False,
            ) as handle:
                temporary_member = Path(handle.name)
            try:
                model.models[member_index].save(str(temporary_member))
                os.replace(temporary_member, member_path)
            finally:
                if temporary_member.exists():
                    temporary_member.unlink()
            member_receipt = {
                "schema_version": 1,
                "status": "complete",
                "member_index": member_index,
                "member_signature": member_signature,
                "recipe_signature": recipe_signature,
                "bootstrap_indices_sha256": bootstrap_sha256,
                "checkpoint": member_path.name,
                "checkpoint_sha256": _sha256(member_path),
            }
            _atomic_json(member_receipt_path, member_receipt)
            member_receipts.append(member_receipt)
            publish_live_prediction(member_index)
            heartbeat.update(
                "modnet_member_checkpoint",
                detail=f"member={member_number}/{n_models} saved",
                completed=member_number * epochs,
                total=n_models * epochs,
            )

        prepared.heartbeat.update(
            "checkpoint_save",
            detail=f"serializing {n_models}-model ensemble",
        )
        checkpoint = self.request.output_path / "mattervial_modnet.pkl"
        with tempfile.NamedTemporaryFile(
            dir=self.request.output_path,
            suffix=".pkl.tmp",
            delete=False,
        ) as handle:
            temporary_checkpoint = Path(handle.name)
        try:
            model.save(str(temporary_checkpoint))
            os.replace(temporary_checkpoint, checkpoint)
        finally:
            if temporary_checkpoint.exists():
                temporary_checkpoint.unlink()
        if not checkpoint.is_file():
            raise RuntimeError(f"MODNet did not create checkpoint {checkpoint}")
        summary = {
            "schema_version": 1,
            "scientific_recipe": str(
                self.request.extra.get(
                    "scientific_recipe", f"MatterVial {self.request.task_name} published preset"
                )
            ),
            "task_name": self.request.task_name,
            "task_type": task_type,
            "scientific_target_name": scientific_target,
            "model_internal_target_name": target,
            "model_internal_target_name_policy": _MODEL_TARGET_NAME_POLICY,
            "smoke_test": smoke_test,
            "n_models": n_models,
            "configured_n_models": n_models,
            "epochs_per_model": epochs,
            "n_feat": n_feat,
            "fit_parameters": fit_values,
            "recipe_signature": recipe_signature,
            "recipe_signature_schema_version": 2,
            "preprocessing_signature_policy": (
                "scientific_recipe_fields"
            ),
            "member_checkpoint_policy": "atomic_per_member_signature_validated_resume",
            "live_prediction_policy": (
                "target_free_atomic_partial_then_final_ensemble_reseal_monitor_read_only"
            ),
            "resumed_members": resumed_members,
            "member_resume_rejections": resume_rejections,
            "member_checkpoint_sha256": [
                receipt["checkpoint_sha256"] for receipt in member_receipts
            ],
            "checkpoint_sha256": _sha256(checkpoint),
        }
        _atomic_json(self.request.output_path / "native_training_summary.json", summary)
        return FittedMatterVialModel(model=model, checkpoint=checkpoint)

    def predict(
        self,
        fitted: FittedMatterVialModel,
        prepared: PreparedMatterVialData,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        try:
            prepared.heartbeat.update("prediction", detail="outer or inner held-out rows")
            definition = self.manifest.metadata["definition"]
            target = prepared.model_target_name
            classification = str(definition["task_type"]) == "classification"
            predict_values = {
                "return_unc": True,
                "return_prob": classification,
                "remap_out_of_bounds": not classification,
            }
            result = fitted.model.predict(
                prepared.prediction_data,
                **_supported(fitted.model.predict, predict_values),
            )
            if isinstance(result, tuple):
                predictions, uncertainty_result = result
            else:
                predictions, uncertainty_result = result, None
            values = _prediction_values(predictions, target, classification=classification)
            if len(values) != len(self.split.prediction_positions):
                raise ValueError("MatterVial prediction row count mismatch")
            if not np.isfinite(values).all():
                raise ValueError("MatterVial predictions contain NaN or infinity")
            if classification and (np.any(values < 0) or np.any(values > 1)):
                raise ValueError("MatterVial classification predictions are not probabilities")
            uncertainty = None
            if uncertainty_result is not None:
                uncertainty = _uncertainty_values(
                    uncertainty_result,
                    len(values),
                    target=target,
                    classification=classification,
                )
                if not np.isfinite(uncertainty).all() or (uncertainty < 0).any():
                    raise ValueError("MatterVial uncertainty must be finite and non-negative")
            training_summary = json.loads(
                (self.request.output_path / "native_training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            effective_members = int(training_summary.get("n_models", -1))
            member_hashes = training_summary.get("member_checkpoint_sha256", [])
            if effective_members < 1 or len(member_hashes) != effective_members:
                raise ValueError("Cannot seal live predictions from an incomplete ensemble")
            live_seal = _seal_final_live_predictions(
                self.request.output_path / "live_predictions.npz",
                sample_ids=self.manifest.sample_ids[self.split.prediction_positions],
                y_pred=values,
                y_uncertainty=uncertainty,
                member_count=effective_members,
                prediction_role=self.split.prediction_role,
                outer_fold=self.request.outer_fold,
                inner_fold=self.request.inner_fold,
            )
            prepared.heartbeat.update(
                "live_prediction_sealed",
                detail=(
                    f"members={effective_members}/{effective_members} "
                    f"role={self.split.prediction_role}; ensemble prediction sealed"
                ),
                ensemble_member=effective_members,
                ensemble_total_members=effective_members,
                live_prediction_path=str(self.request.output_path / "live_predictions.npz"),
                live_prediction_policy=(
                    "live_monitor_then_ensemble_seal"
                ),
                live_prediction_sha256=live_seal["sealed_sha256"],
            )
            prepared.heartbeat.update("artifact_validation", detail="writing fold contract")
            return values, uncertainty
        except Exception as exc:
            prepared.heartbeat.__exit__(type(exc), exc, exc.__traceback__)
            raise

    def export_latent(
        self,
        fitted: FittedMatterVialModel,
        prepared: PreparedMatterVialData,
    ) -> tuple[np.ndarray, np.ndarray]:
        return prepared.latent_prediction, np.ones(len(prepared.latent_prediction), dtype=np.bool_)

    def checkpoint_path(self, fitted: FittedMatterVialModel) -> Path:
        return fitted.checkpoint

    def preprocessing_manifest(self, prepared: PreparedMatterVialData) -> Path:
        return prepared.receipt

    def finalize_progress(self, error: BaseException | None = None) -> None:
        heartbeat = getattr(self, "_heartbeat", None)
        if heartbeat is None:
            return
        if error is None:
            heartbeat.__exit__(None, None, None)
        else:
            heartbeat.__exit__(type(error), error, error.__traceback__)

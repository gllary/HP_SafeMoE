from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hpsafe_sota.data.manifest import TaskManifest

SCHEMA_VERSION = "1.0"
SPLIT_ROLES = {"outer_test", "holdout_val", "inner_oof", "inner_val"}


class ArtifactValidationError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class FoldArtifact:
    root: Path
    metadata: dict[str, Any]
    sample_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_uncertainty: np.ndarray | None
    latents: np.ndarray | None
    available: np.ndarray | None


def expected_positions(
    manifest: TaskManifest, split_role: str, outer_fold: int, inner_fold: int | None
) -> np.ndarray:
    if split_role == "outer_test":
        return manifest.positions("outer_test", outer_fold)
    if split_role == "holdout_val":
        return manifest.positions("holdout_val", outer_fold)
    if split_role == "inner_oof":
        return manifest.positions("inner_oof", outer_fold)
    if split_role == "inner_val":
        if inner_fold is None:
            raise ArtifactValidationError("inner_fold is required for inner_val")
        return manifest.positions("inner_val", outer_fold, inner_fold)
    raise ArtifactValidationError(f"Unsupported split_role: {split_role}")


def write_fold_artifact(
    output_dir: Path,
    *,
    manifest: TaskManifest,
    model_name: str,
    model_family: str,
    outer_fold: int,
    split_role: str,
    y_pred: np.ndarray,
    sample_ids: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    y_uncertainty: np.ndarray | None = None,
    latents: np.ndarray | None = None,
    available: np.ndarray | None = None,
    inner_fold: int | None = None,
    run_id: str | None = None,
    run_kind: str = "production",
    checkpoint_sha256: str | None = None,
    environment_lock_sha256: str | None = None,
    preprocessing_manifest_sha256: str | None = None,
    pretrained_disclosure: dict[str, Any] | None = None,
    selection: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    if split_role not in SPLIT_ROLES:
        raise ValueError(f"Unknown split role: {split_role}")
    positions = expected_positions(manifest, split_role, outer_fold, inner_fold)
    expected_ids = manifest.sample_ids[positions]
    expected_targets = manifest.targets[positions]
    ids = expected_ids if sample_ids is None else np.asarray(sample_ids).astype(str)
    targets = expected_targets if y_true is None else np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    n_samples = len(expected_ids)
    for label, values in (("sample_ids", ids), ("y_true", targets), ("y_pred", predictions)):
        if len(values) != n_samples:
            raise ValueError(f"{label} has {len(values)} rows, expected {n_samples}")
    prediction_arrays = {
        "sample_ids": ids,
        "y_true": targets,
        "y_pred": predictions,
    }
    if y_uncertainty is not None:
        prediction_arrays["y_uncertainty"] = np.asarray(y_uncertainty, dtype=np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.npz"
    _atomic_npz(prediction_path, prediction_arrays)

    latent_record: dict[str, Any] | None = None
    if latents is not None:
        latent_values = np.asarray(latents, dtype=np.float32)
        if latent_values.ndim != 2 or latent_values.shape[0] != n_samples:
            raise ValueError(f"latents must have shape ({n_samples}, d)")
        availability = (
            np.ones(n_samples, dtype=np.bool_) if available is None else np.asarray(available, dtype=np.bool_)
        )
        if availability.shape != (n_samples,):
            raise ValueError(f"available must have shape ({n_samples},)")
        latent_path = output_dir / "latents.npz"
        _atomic_npz(
            latent_path,
            {"sample_ids": ids, "latents": latent_values, "available": availability},
        )
        latent_record = {
            "filename": latent_path.name,
            "sha256": _sha256(latent_path),
            "shape": list(latent_values.shape),
        }

    definition = manifest.metadata["definition"]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or output_dir.name,
        "run_kind": run_kind,
        "task": {
            key: definition[key]
            for key in (
                "task_name",
                "matbench_name",
                "input_type",
                "task_type",
                "target",
                "unit",
                "metric",
                "direction",
            )
        },
        "model": {
            "name": model_name,
            "family": model_family,
            "pretrained_disclosure": pretrained_disclosure or {"uses_pretraining": False},
        },
        "split": {
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "role": split_role,
            "sample_order": "exact_split_manifest_order",
        },
        "provenance": {
            "task_manifest_digest": manifest.digest,
            "raw_sha256": definition["raw_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "environment_lock_sha256": environment_lock_sha256,
            "preprocessing_manifest_sha256": preprocessing_manifest_sha256,
        },
        "selection": selection
        or {
            "source": "predeclared_without_outer_test_access",
            "outer_test_used": False,
        },
        "predictions": {
            "filename": prediction_path.name,
            "sha256": _sha256(prediction_path),
            "n_samples": n_samples,
            "value_kind": (
                "positive_class_probability"
                if definition["task_type"] == "classification"
                else "regression_value"
            ),
            "uncertainty_present": y_uncertainty is not None,
        },
        "latents": latent_record,
    }
    if extra_metadata:
        metadata["extra"] = extra_metadata
    _atomic_json(output_dir / "metadata.json", metadata)
    validate_fold_artifact(output_dir, manifest=manifest)
    return output_dir


def _required_production_hash(value: Any, label: str, run_kind: str) -> None:
    if run_kind != "production":
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ArtifactValidationError(f"Production artifact requires a 64-character {label}")


def validate_fold_artifact(
    artifact_dir: Path,
    *,
    manifest: TaskManifest,
    expected_latent_dim: int | None = None,
) -> FoldArtifact:
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactValidationError(f"Missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError("Artifact schema version mismatch")
    if metadata["task"]["task_name"] != manifest.task_name:
        raise ArtifactValidationError("Artifact task does not match manifest")
    if metadata["provenance"]["task_manifest_digest"] != manifest.digest:
        raise ArtifactValidationError("Artifact references the wrong task manifest")
    if metadata["selection"].get("outer_test_used") is not False:
        raise ArtifactValidationError("Outer-test data was marked as used for model selection")
    run_kind = str(metadata.get("run_kind", "production"))
    provenance = metadata["provenance"]
    _required_production_hash(provenance.get("checkpoint_sha256"), "checkpoint hash", run_kind)
    _required_production_hash(provenance.get("environment_lock_sha256"), "environment-lock hash", run_kind)
    _required_production_hash(provenance.get("preprocessing_manifest_sha256"), "preprocessing hash", run_kind)

    prediction_path = artifact_dir / metadata["predictions"]["filename"]
    if _sha256(prediction_path) != metadata["predictions"]["sha256"]:
        raise ArtifactValidationError("Prediction file hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        sample_ids = loaded["sample_ids"].astype(str)
        y_true = np.asarray(loaded["y_true"], dtype=np.float64)
        y_pred = np.asarray(loaded["y_pred"], dtype=np.float64)
        y_uncertainty = (
            np.asarray(loaded["y_uncertainty"], dtype=np.float64) if "y_uncertainty" in loaded.files else None
        )
    split = metadata["split"]
    positions = expected_positions(
        manifest, str(split["role"]), int(split["outer_fold"]), split.get("inner_fold")
    )
    expected_ids = manifest.sample_ids[positions]
    expected_targets = manifest.targets[positions]
    if not np.array_equal(sample_ids, expected_ids):
        raise ArtifactValidationError("Sample IDs or ordering differ from the split manifest")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ArtifactValidationError("Duplicate sample IDs in predictions")
    if not np.array_equal(y_true, expected_targets, equal_nan=True):
        raise ArtifactValidationError("Targets differ from the immutable task manifest")
    if y_pred.shape != y_true.shape or not np.all(np.isfinite(y_pred)):
        raise ArtifactValidationError("Predictions have the wrong shape or non-finite values")
    if metadata["task"]["task_type"] == "classification" and not np.all((0.0 <= y_pred) & (y_pred <= 1.0)):
        raise ArtifactValidationError("Classification predictions must be probabilities in [0, 1]")
    if y_uncertainty is not None:
        if y_uncertainty.shape != y_true.shape or not np.all(np.isfinite(y_uncertainty)):
            raise ArtifactValidationError("Uncertainty has the wrong shape or non-finite values")
        if np.any(y_uncertainty < 0):
            raise ArtifactValidationError("Uncertainty cannot be negative")

    latents = None
    available = None
    latent_record = metadata.get("latents")
    if latent_record is not None:
        latent_path = artifact_dir / latent_record["filename"]
        if _sha256(latent_path) != latent_record["sha256"]:
            raise ArtifactValidationError("Latent file hash mismatch")
        with np.load(latent_path, allow_pickle=False) as loaded:
            latent_ids = loaded["sample_ids"].astype(str)
            latents = np.asarray(loaded["latents"], dtype=np.float32)
            available = np.asarray(loaded["available"], dtype=np.bool_)
        if not np.array_equal(latent_ids, sample_ids):
            raise ArtifactValidationError("Latent IDs/order differ from predictions")
        if latents.ndim != 2 or latents.shape[0] != len(sample_ids):
            raise ArtifactValidationError("Latents must be a two-dimensional row-aligned array")
        if available.shape != (len(sample_ids),):
            raise ArtifactValidationError("Latent availability mask has the wrong shape")
        if not np.all(np.isfinite(latents[available])):
            raise ArtifactValidationError("Available latent rows contain non-finite values")
        if expected_latent_dim is not None and latents.shape[1] != expected_latent_dim:
            raise ArtifactValidationError(
                f"Latent dimension {latents.shape[1]} != expected {expected_latent_dim}"
            )

    diagnostics_hash = metadata.get("extra", {}).get("routing_diagnostics_sha256")
    if diagnostics_hash is not None:
        diagnostics_path = artifact_dir / "routing_diagnostics.npz"
        if not diagnostics_path.is_file() or _sha256(diagnostics_path) != diagnostics_hash:
            raise ArtifactValidationError("Routing diagnostics file is missing or its hash changed")

    return FoldArtifact(
        root=artifact_dir,
        metadata=metadata,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
        y_uncertainty=y_uncertainty,
        latents=latents,
        available=available,
    )

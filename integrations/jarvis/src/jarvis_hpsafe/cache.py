from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import JARVIS_TASKS, LATENT_DIM
from .io import array_file_record, atomic_json, atomic_npy, read_json, sha256_file
from .manifest import JarvisManifest

ARRAY_FILES = {
    "sample_ids": "sample_ids.npy",
    "hidden": "hidden.npy",
    "prediction": "prediction.npy",
    "uncertainty": "uncertainty.npy",
    "available": "available.npy",
}


@dataclass(frozen=True)
class ExpertRoleCache:
    root: Path
    metadata: dict[str, Any]
    sample_ids: np.ndarray
    hidden: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    available: np.ndarray


def cache_directory(cache_root: Path, source_task: str, target_task: str, role: str) -> Path:
    return cache_root.expanduser().resolve() / source_task / target_task / role


def write_role_cache(
    output: Path,
    *,
    source_task: str,
    target_manifest: JarvisManifest,
    role: str,
    sample_ids: np.ndarray,
    hidden: np.ndarray,
    prediction: np.ndarray,
    available: np.ndarray,
    checkpoint_sha256: str,
    producer: dict[str, Any],
) -> Path:
    if source_task not in JARVIS_TASKS or role not in {"train", "val", "test"}:
        raise ValueError("Unknown source task or role")
    ids = np.asarray(sample_ids).astype(str)
    expected_ids = target_manifest.ids(role)
    latent = np.asarray(hidden, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
    mask = np.asarray(available, dtype=np.bool_).reshape(-1)
    uncertainty = np.zeros(len(ids), dtype=np.float32)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{source_task}/{target_manifest.task}/{role}: sample order drift")
    if latent.shape != (len(ids), LATENT_DIM):
        raise ValueError(f"Unexpected hidden shape: {latent.shape}")
    if pred.shape != (len(ids),) or mask.shape != (len(ids),):
        raise ValueError("Prediction/availability arrays are not row aligned")
    finite = np.isfinite(pred) & np.isfinite(latent).all(axis=1)
    mask &= finite
    latent = latent.copy()
    pred = pred.copy()
    latent[~mask] = 0.0
    pred[~mask] = 0.0
    if len(checkpoint_sha256) != 64:
        raise ValueError("A frozen Stage1 source requires a SHA-256 checkpoint receipt")
    output = output.expanduser().resolve()
    arrays = {
        "sample_ids": ids.astype(f"<U{max(1, max(map(len, ids.tolist()), default=1))}"),
        "hidden": latent,
        "prediction": pred,
        "uncertainty": uncertainty,
        "available": mask,
    }
    records: dict[str, Any] = {}
    for key, filename in ARRAY_FILES.items():
        path = output / filename
        atomic_npy(path, arrays[key])
        records[key] = array_file_record(path, arrays[key])
    metadata = {
        "schema_version": "jarvis-hpsafemoe-cache",
        "status": "complete",
        "source_task": source_task,
        "target_task": target_manifest.task,
        "role": role,
        "sample_order": "exact_pinned_jarvis_manifest_order",
        "manifest_sha256": target_manifest.file_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "latent_dim": LATENT_DIM,
        "n_samples": len(ids),
        "available_rows": int(mask.sum()),
        "unavailable_rows": int((~mask).sum()),
        "labels_stored": False,
        "uncertainty_policy": "deterministic_single_checkpoint_zero_then_train_standardized",
        "files": records,
        "producer": producer,
    }
    atomic_json(output / "metadata.json", metadata)
    validate_role_cache(
        output,
        source_task=source_task,
        target_manifest=target_manifest,
        role=role,
        verify_hashes=True,
    )
    return output


def validate_role_cache(
    path: Path,
    *,
    source_task: str,
    target_manifest: JarvisManifest,
    role: str,
    verify_hashes: bool = False,
) -> ExpertRoleCache:
    path = path.expanduser().resolve()
    metadata = read_json(path / "metadata.json")
    expected = {
        "schema_version": "jarvis-hpsafemoe-cache",
        "status": "complete",
        "source_task": source_task,
        "target_task": target_manifest.task,
        "role": role,
        "manifest_sha256": target_manifest.file_sha256,
        "latent_dim": LATENT_DIM,
        "labels_stored": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cache metadata mismatch {key}: {path}")
    arrays: dict[str, np.ndarray] = {}
    for key, filename in ARRAY_FILES.items():
        file_path = path / filename
        record = metadata.get("files", {}).get(key, {})
        if record.get("filename") != filename:
            raise ValueError(f"Missing cache file record {key}: {path}")
        if verify_hashes and sha256_file(file_path) != record.get("sha256"):
            raise ValueError(f"Cache hash mismatch: {file_path}")
        arrays[key] = np.load(
            file_path,
            mmap_mode="r" if key == "hidden" else None,
            allow_pickle=False,
        )
    ids = np.asarray(arrays["sample_ids"]).astype(str)
    if not np.array_equal(ids, target_manifest.ids(role)):
        raise ValueError(f"Cache ID/order drift: {path}")
    n = len(ids)
    if arrays["hidden"].shape != (n, LATENT_DIM):
        raise ValueError(f"Cache hidden shape drift: {path}")
    if any(arrays[key].shape != (n,) for key in ("prediction", "uncertainty", "available")):
        raise ValueError(f"Cache vector shape drift: {path}")
    available = np.asarray(arrays["available"], dtype=np.bool_)
    if not np.isfinite(np.asarray(arrays["hidden"])[available]).all():
        raise ValueError(f"Cache hidden contains non-finite values: {path}")
    return ExpertRoleCache(
        root=path,
        metadata=metadata,
        sample_ids=arrays["sample_ids"],
        hidden=arrays["hidden"],
        prediction=arrays["prediction"],
        uncertainty=arrays["uncertainty"],
        available=arrays["available"],
    )


def validate_role_metadata_only(
    path: Path,
    *,
    source_task: str,
    target_manifest: JarvisManifest,
    role: str,
) -> dict[str, Any]:
    """Validate a cache receipt without opening feature arrays.

    This is used before the OOF safety profile is locked.  In particular, a
    preflight may establish that target-free test exports exist without
    materializing their prediction or hidden-representation contents.
    """
    path = path.expanduser().resolve()
    metadata = read_json(path / "metadata.json")
    expected = {
        "schema_version": "jarvis-hpsafemoe-cache",
        "status": "complete",
        "source_task": source_task,
        "target_task": target_manifest.task,
        "role": role,
        "manifest_sha256": target_manifest.file_sha256,
        "latent_dim": LATENT_DIM,
        "labels_stored": False,
        "n_samples": len(target_manifest.ids(role)),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cache metadata mismatch {key}: {path}")
    if set(metadata.get("files", {})) != set(ARRAY_FILES):
        raise ValueError(f"Incomplete cache file inventory: {path}")
    for key, filename in ARRAY_FILES.items():
        record = metadata["files"][key]
        file_path = path / filename
        if record.get("filename") != filename or not file_path.is_file():
            raise FileNotFoundError(file_path)
    return metadata

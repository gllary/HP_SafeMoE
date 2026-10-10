from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.stage2_common.io import atomic_json, read_json, sha256_file
from hpsafe_sota.stage2_common.protocol import (
    SPLIT_ROLE_TO_MANIFEST_ROLE,
    ExpertPool,
    ExpertSpec,
)

ARRAY_FILES = {
    "sample_ids": "sample_ids.npy",
    "hidden": "hidden.npy",
    "prediction": "prediction.npy",
    "uncertainty": "uncertainty.npy",
    "available": "available.npy",
}

# Only the high-dimensional hidden matrix needs to remain memory mapped during
# Stage2.  Keeping IDs and three one-dimensional arrays as memmaps would retain
# five file descriptors per expert export.  A 13 task x 3 role x 13 expert
# joint fold then exceeds the common 1024-descriptor soft limit before training
# starts.  Loading the small arrays into memory leaves one descriptor per
# export while preserving mmap access for the expensive hidden representation.
MEMMAP_ARRAY_KEYS = frozenset({"hidden"})


class ExpertExportError(ValueError):
    pass


def export_directory(spec: ExpertSpec, target_task: str, outer_fold: int, role: str) -> Path:
    return spec.export_root / target_task / f"outer_{outer_fold}" / role


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temporary = handle.name
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _expected_ids(manifest: TaskManifest, role: str, outer_fold: int) -> np.ndarray:
    try:
        manifest_role = SPLIT_ROLE_TO_MANIFEST_ROLE[role]
    except KeyError as exc:
        raise ExpertExportError(f"Unknown expert-export split role: {role}") from exc
    return manifest.ids(manifest_role, outer_fold).astype(str)


def write_expert_export(
    output_dir: Path,
    *,
    manifest: TaskManifest,
    spec: ExpertSpec,
    outer_fold: int,
    role: str,
    sample_ids: np.ndarray,
    hidden: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray | None,
    available: np.ndarray | None,
    checkpoint_sha256: str,
    producer: dict[str, Any] | None = None,
) -> Path:
    """Write a label-free, memory-mappable frozen-expert export.

    Missing/non-finite expert rows are represented only by ``available=False``.
    The arrays for those rows are zero-filled so downstream numerical kernels
    cannot accidentally propagate NaNs before masking.
    """

    manifest.require_valid()
    expected_ids = _expected_ids(manifest, role, outer_fold)
    ids = np.asarray(sample_ids).astype(str).reshape(-1)
    latent = np.asarray(hidden, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32).reshape(-1)
    unc = (
        np.zeros(len(ids), dtype=np.float32)
        if uncertainty is None
        else np.asarray(uncertainty, dtype=np.float32).reshape(-1)
    )
    mask = (
        np.ones(len(ids), dtype=np.bool_)
        if available is None
        else np.asarray(available, dtype=np.bool_).reshape(-1)
    )
    if not np.array_equal(ids, expected_ids):
        raise ExpertExportError("Expert export sample IDs/order differ from the Stage1 split manifest")
    if latent.shape != (len(ids), spec.hidden_dim):
        raise ExpertExportError(
            f"{spec.expert_id} hidden shape {latent.shape} != ({len(ids)}, {spec.hidden_dim})"
        )
    if pred.shape != (len(ids),) or unc.shape != (len(ids),) or mask.shape != (len(ids),):
        raise ExpertExportError("Prediction, uncertainty, and availability must be row aligned")
    if len(checkpoint_sha256) != 64:
        raise ExpertExportError("A frozen expert export requires its 64-character checkpoint SHA-256")
    finite = np.isfinite(latent).all(axis=1) & np.isfinite(pred) & np.isfinite(unc) & (unc >= 0)
    mask &= finite
    latent = latent.copy()
    pred = pred.copy()
    unc = unc.copy()
    latent[~mask] = 0.0
    pred[~mask] = 0.0
    unc[~mask] = 0.0

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sample_ids": ids.astype(f"<U{max(1, max(map(len, ids.tolist()), default=1))}"),
        "hidden": latent,
        "prediction": pred,
        "uncertainty": unc,
        "available": mask,
    }
    for key, value in arrays.items():
        _atomic_npy(output_dir / ARRAY_FILES[key], value)
    files = {
        key: {
            "filename": filename,
            "sha256": sha256_file(output_dir / filename),
            "shape": list(arrays[key].shape),
            "dtype": str(arrays[key].dtype),
        }
        for key, filename in ARRAY_FILES.items()
    }
    metadata = {
        "schema_version": 12,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "frozen_task_trained_hidden_export",
        "expert": {
            "expert_id": spec.expert_id,
            "source_task": spec.source_task,
            "provider": spec.provider,
            "hidden_dim": spec.hidden_dim,
            "input_modality": spec.input_modality,
            "representation_kind": spec.representation_kind,
            "checkpoint_sha256": checkpoint_sha256,
            "frozen": True,
        },
        "target_task": manifest.task_name,
        "split": {
            "outer_fold": outer_fold,
            "role": role,
            "manifest_role": SPLIT_ROLE_TO_MANIFEST_ROLE[role],
            "sample_order": "exact_stage1_manifest_order",
        },
        "manifest_digest": manifest.digest,
        "n_samples": len(ids),
        "available_rows": int(mask.sum()),
        "unavailable_rows": int((~mask).sum()),
        "availability_policy": "technical_inability_only",
        "label_values_stored": False,
        "relation_screening": "none",
        "files": files,
        "producer": producer or {},
    }
    atomic_json(output_dir / "metadata.json", metadata)
    validate_expert_export(
        output_dir,
        manifest=manifest,
        spec=spec,
        outer_fold=outer_fold,
        role=role,
        verify_hashes=True,
    )
    return output_dir


@dataclass(frozen=True)
class ExpertExport:
    root: Path
    metadata: dict[str, Any]
    sample_ids: np.ndarray
    hidden: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    available: np.ndarray


def validate_expert_export(
    path: Path,
    *,
    manifest: TaskManifest,
    spec: ExpertSpec,
    outer_fold: int,
    role: str,
    verify_hashes: bool = False,
) -> ExpertExport:
    path = path.expanduser().resolve()
    metadata = read_json(path / "metadata.json")
    if metadata.get("schema_version") != 12:
        raise ExpertExportError(f"Expert-export schema mismatch: {path}")
    expert = metadata.get("expert", {})
    split = metadata.get("split", {})
    expected = {
        "expert_id": spec.expert_id,
        "source_task": spec.source_task,
        "provider": spec.provider,
        "hidden_dim": spec.hidden_dim,
        "representation_kind": "task_trained_hidden",
        "frozen": True,
    }
    for key, value in expected.items():
        if expert.get(key) != value:
            raise ExpertExportError(f"{spec.expert_id} export has wrong {key}: {expert.get(key)!r}")
    if metadata.get("target_task") != manifest.task_name:
        raise ExpertExportError("Expert export target task mismatch")
    if metadata.get("manifest_digest") != manifest.digest:
        raise ExpertExportError("Expert export references a different immutable task manifest")
    if split.get("outer_fold") != outer_fold or split.get("role") != role:
        raise ExpertExportError("Expert export split mismatch")
    if split.get("manifest_role") != SPLIT_ROLE_TO_MANIFEST_ROLE[role]:
        raise ExpertExportError("Expert export does not preserve the Stage1 split role")
    if metadata.get("label_values_stored") is not False:
        raise ExpertExportError("Expert cache must not store target labels")
    if metadata.get("relation_screening") != "none":
        raise ExpertExportError("Expert exports must not use relation screening")
    producer = metadata.get("producer", {})
    if str(spec.provider).startswith("mattervial_modnet"):
        checkpoint_consistent = (
            spec.provider == "mattervial_modnet_0.4.5_checkpoint_consistent"
        )
        own_task = spec.source_task == manifest.task_name
        if checkpoint_consistent and own_task and role in {"train", "val"}:
            expected_checkpoint_role = "inner_oof"
            expected_fit_role = "inner_train_crossfit"
            expected_adapter = "mattervial_modnet_inner_oof_penultimate_concat"
        elif checkpoint_consistent:
            expected_checkpoint_role = "outer"
            expected_fit_role = "outer_train"
            expected_adapter = "mattervial_modnet_outer_consistent_penultimate_concat"
        else:
            expected_checkpoint_role = "holdout" if role in {"train", "val"} else "outer"
            expected_fit_role = "holdout_train" if role in {"train", "val"} else "outer_train"
            expected_adapter = "mattervial_modnet_true_penultimate_member_concat"
        if producer.get("adapter") != expected_adapter:
            raise ExpertExportError("MatterVial export is not a genuine trained-hidden replay")
        if producer.get("checkpoint_role") != expected_checkpoint_role:
            raise ExpertExportError("MatterVial export uses the wrong checkpoint for this split role")
        if producer.get("checkpoint_fit_role") != expected_fit_role:
            raise ExpertExportError("MatterVial checkpoint fitting role differs from the release contract")
        if checkpoint_consistent:
            if own_task and role in {"train", "val"}:
                if producer.get("outer_test_accessed") is not False:
                    raise ExpertExportError("Anchor inner-OOF export accessed outer-test")
                if producer.get("inner_validation_partition_exact") is not True:
                    raise ExpertExportError("Anchor inner-OOF export is not an exact partition")
            elif role in {"train", "val"} and own_task:
                raise ExpertExportError("Own-task train/val may not use the outer checkpoint")
        if producer.get("target_values_accessed_by_exporter") is not False:
            raise ExpertExportError("MatterVial replay may not consume target values")
        if producer.get("relation_screening") != "none":
            raise ExpertExportError("MatterVial replay may not apply relation screening")
    if spec.provider == "jmp_l_official_checkpoint_consistent":
        expected_adapter = "jmp_l_outer_checkpoint_replay"
        expected_checkpoint_role = "outer"
        expected_fit_role = "outer_train"
        if producer.get("adapter") != expected_adapter:
            raise ExpertExportError("JMP-L export uses an unexpected adapter")
        if producer.get("checkpoint_role") != expected_checkpoint_role:
            raise ExpertExportError("JMP-L export uses the wrong checkpoint role")
        if producer.get("checkpoint_fit_role") != expected_fit_role:
            raise ExpertExportError("JMP-L checkpoint fitting role differs from the declared protocol")
        if producer.get("source_outer_fold") != outer_fold:
            raise ExpertExportError("JMP-L source checkpoint is not mapped one-to-one by outer fold")
        if producer.get("target_outer_fold") != outer_fold:
            raise ExpertExportError("JMP-L target export outer fold differs from its source checkpoint")
        if producer.get("checkpoint_parameters_updated") is not False:
            raise ExpertExportError("JMP-L Stage2 replay must not update checkpoint parameters")
        if producer.get("official_commit") != "937b14874381d9b80809582e323ef82f2d4e1291":
            raise ExpertExportError("JMP-L official source commit drifted")
        if float(producer.get("learning_rate", -1.0)) != 8.0e-5:
            raise ExpertExportError("JMP-L learning rate drifted")
        if producer.get("latent_source") != (
            "gemnet_oc_final_atom_energy_embedding_before_task_head"
        ):
            raise ExpertExportError("JMP-L hidden-state source drifted")
        if producer.get("uncertainty_kind") != "unavailable_zero_sentinel":
            raise ExpertExportError("JMP-L must not fabricate an uncertainty branch")
        if producer.get("target_values_accessed_by_exporter") is not False:
            raise ExpertExportError("JMP-L Stage2 exporter may not consume target values")
        if producer.get("outer_test_accessed") is not False:
            raise ExpertExportError("JMP-L Stage2 exporter accessed outer-test labels")
        if producer.get("relation_screening") != "none":
            raise ExpertExportError("JMP-L replay may not apply relation screening")
    if spec.provider == "jmp_l_frozen_outer_checkpoint_replay":
        if producer.get("adapter") != (
            "jmp_l_fast32_fp32_outer_checkpoint_replay"
        ):
            raise ExpertExportError("JMP-L Fast-32 export uses an unexpected adapter")
        if producer.get("checkpoint_role") != "outer":
            raise ExpertExportError("JMP-L Fast-32 replay must use an outer checkpoint")
        if producer.get("checkpoint_fit_role") != "complete_outer_train":
            raise ExpertExportError("JMP-L Fast-32 checkpoint was not fit on outer-train")
        if producer.get("source_outer_fold") != outer_fold:
            raise ExpertExportError("JMP-L Fast-32 source fold is not mapped one-to-one")
        if producer.get("target_outer_fold") != outer_fold:
            raise ExpertExportError("JMP-L Fast-32 target fold differs from the source fold")
        if producer.get("checkpoint_parameters_updated") is not False:
            raise ExpertExportError("JMP-L Fast-32 frozen replay updated model parameters")
        if producer.get("official_commit") != "937b14874381d9b80809582e323ef82f2d4e1291":
            raise ExpertExportError("JMP-L Fast-32 official source commit drifted")
        if float(producer.get("learning_rate", -1.0)) != 8.0e-5:
            raise ExpertExportError("JMP-L Fast-32 Stage1 learning rate drifted")
        if producer.get("latent_source") != (
            "gemnet_oc_final_atom_energy_embedding_before_task_head"
        ):
            raise ExpertExportError("JMP-L Fast-32 hidden-state source drifted")
        if producer.get("uncertainty_kind") != "unavailable_zero_sentinel":
            raise ExpertExportError("JMP-L Fast-32 may not fabricate uncertainty")
        if producer.get("target_values_accessed_by_exporter") is not False:
            raise ExpertExportError("JMP-L Fast-32 replay consumed target values")
        if producer.get("outer_test_used_for_training") is not False:
            raise ExpertExportError("JMP-L Fast-32 replay used outer-test for training")
        if producer.get("relation_screening") != "none":
            raise ExpertExportError("JMP-L Fast-32 replay applied relation screening")
    if spec.provider == "tpot_mat_official_pipeline":
        if producer.get("adapter") != "tpot_mat_outer_checkpoint_replay":
            raise ExpertExportError("TPOT-Mat export uses the wrong TPOT-Mat adapter")
        if producer.get("checkpoint_role") != "outer":
            raise ExpertExportError("TPOT-Mat export must replay a frozen outer checkpoint")
        if producer.get("checkpoint_fit_role") != "outer_train":
            raise ExpertExportError("TPOT-Mat checkpoint was not fitted on outer_train")
        if producer.get("source_outer_fold") != outer_fold:
            raise ExpertExportError("TPOT-Mat source checkpoint is not mapped one-to-one by outer fold")
        if producer.get("target_outer_fold") != outer_fold:
            raise ExpertExportError("TPOT-Mat target export outer fold differs from its checkpoint")
        if producer.get("checkpoint_parameters_updated") is not False:
            raise ExpertExportError("TPOT-Mat Stage2 replay must not update checkpoint parameters")
        if producer.get("official_commit") != "936176db18ca4cd7b38cbd957c017a5bac770c6b":
            raise ExpertExportError("TPOT-Mat official source commit drifted")
        if producer.get("official_pipeline_sha256") != (
            "21a3f2e07020ce2437310e23aa8d8051542ed785949cf38b0a4b7b08a795bec0"
        ):
            raise ExpertExportError("TPOT-Mat official selected-pipeline hash drifted")
        if producer.get("hidden_source") != (
            "three_stacking_predictions_plus_xgboost_leaf_path"
        ):
            raise ExpertExportError("TPOT-Mat task-trained hidden source drifted")
        if producer.get("target_values_accessed_by_exporter") is not False:
            raise ExpertExportError("TPOT-Mat Stage2 exporter may not consume target values")
        if producer.get("outer_test_labels_accessed_by_exporter") is not False:
            raise ExpertExportError("TPOT-Mat Stage2 exporter accessed outer-test labels")
        if producer.get("relation_screening") != "none":
            raise ExpertExportError("TPOT-Mat replay may not apply relation screening")
    checkpoint_hash = str(expert.get("checkpoint_sha256", ""))
    if len(checkpoint_hash) != 64:
        raise ExpertExportError("Expert checkpoint hash is missing")

    loaded: dict[str, np.ndarray] = {}
    file_records = metadata.get("files", {})
    for key, filename in ARRAY_FILES.items():
        file_path = path / filename
        record = file_records.get(key, {})
        if record.get("filename") != filename:
            raise ExpertExportError(f"Missing file record for {key}")
        if verify_hashes and sha256_file(file_path) != record.get("sha256"):
            raise ExpertExportError(f"Hash mismatch for {file_path}")
        loaded[key] = np.load(
            file_path,
            mmap_mode="r" if key in MEMMAP_ARRAY_KEYS else None,
            allow_pickle=False,
        )
    expected_ids = _expected_ids(manifest, role, outer_fold)
    ids = np.asarray(loaded["sample_ids"]).astype(str)
    if not np.array_equal(ids, expected_ids):
        raise ExpertExportError("Expert export IDs/order differ from the Stage1 split")
    n = len(ids)
    if loaded["hidden"].shape != (n, spec.hidden_dim):
        raise ExpertExportError("Expert hidden shape mismatch")
    for key in ("prediction", "uncertainty", "available"):
        if loaded[key].shape != (n,):
            raise ExpertExportError(f"Expert {key} shape mismatch")
    available = np.asarray(loaded["available"], dtype=np.bool_)
    if not np.all(np.isfinite(np.asarray(loaded["hidden"])[available])):
        raise ExpertExportError("Available hidden rows contain non-finite values")
    if not np.all(np.isfinite(np.asarray(loaded["prediction"])[available])):
        raise ExpertExportError("Available predictions contain non-finite values")
    if not np.all(np.asarray(loaded["uncertainty"])[available] >= 0):
        raise ExpertExportError("Available uncertainty contains negative values")
    return ExpertExport(
        root=path,
        metadata=metadata,
        sample_ids=loaded["sample_ids"],
        hidden=loaded["hidden"],
        prediction=loaded["prediction"],
        uncertainty=loaded["uncertainty"],
        available=loaded["available"],
    )


@dataclass(frozen=True)
class TaskRoleCache:
    task: str
    role: str
    outer_fold: int
    manifest_role: str
    sample_ids: np.ndarray
    y_true: np.ndarray
    anchor_expert_id: str
    exports: dict[str, ExpertExport | None]

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)


def load_task_role_cache(
    *,
    project_root: Path,
    pool: ExpertPool,
    task: str,
    outer_fold: int,
    role: str,
    verify_hashes: bool = False,
) -> TaskRoleCache:
    manifest = TaskManifest(project_root / "data/manifests" / f"{task}.npz")
    manifest.require_valid()
    manifest_role = SPLIT_ROLE_TO_MANIFEST_ROLE[role]
    exports: dict[str, ExpertExport | None] = {}
    for spec in pool.experts:
        path = export_directory(spec, task, outer_fold, role)
        exports[spec.expert_id] = (
            validate_expert_export(
                path,
                manifest=manifest,
                spec=spec,
                outer_fold=outer_fold,
                role=role,
                verify_hashes=verify_hashes,
            )
            if (path / "metadata.json").is_file()
            else None
        )
    anchor_id = pool.anchor_for(task)
    anchor = exports[anchor_id]
    if anchor is None:
        raise ExpertExportError(f"Required target anchor export is missing: {task}/{anchor_id}/{role}")
    anchor_available = np.asarray(anchor.available, dtype=np.bool_)
    if not np.all(anchor_available):
        missing = np.flatnonzero(~anchor_available)[:10].tolist()
        raise ExpertExportError(f"Target anchor is unavailable for {task}/{role} rows {missing}")
    if manifest.metadata["definition"]["task_type"] == "classification":
        anchor_prediction = np.asarray(anchor.prediction, dtype=np.float64)
        if np.any((anchor_prediction < 0.0) | (anchor_prediction > 1.0)):
            raise ExpertExportError(
                f"Classification anchor predictions are outside [0, 1]: {task}/{role}"
            )
    return TaskRoleCache(
        task=task,
        role=role,
        outer_fold=outer_fold,
        manifest_role=manifest_role,
        sample_ids=manifest.ids(manifest_role, outer_fold),
        y_true=manifest.y(manifest_role, outer_fold),
        anchor_expert_id=anchor_id,
        exports=exports,
    )

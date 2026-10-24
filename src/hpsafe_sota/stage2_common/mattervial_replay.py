from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.stage2_common.io import read_json, sha256_file


class MatterVialReplayError(ValueError):
    pass


def validate_stage1_feature_receipt(
    receipt: Mapping[str, Any],
    *,
    manifest: TaskManifest,
) -> dict[str, Any]:
    """Validate the supplemental feature-cache receipt against the release contract."""

    schema_version = receipt.get("schema_version")
    if schema_version != 2:
        raise MatterVialReplayError(
            f"Unsupported Stage1 supplemental feature-cache schema: {schema_version!r}"
        )
    required_false = (
        "target_values_stored",
        "source_target_columns_read",
        "source_target_values_accessed_by_cache_builder",
    )
    invalid = [key for key in required_false if receipt.get(key) is not False]
    if receipt.get("status") != "complete":
        invalid.append("status")
    if receipt.get("task") != manifest.task_name:
        invalid.append("task")
    if receipt.get("manifest_sha256") != manifest.digest:
        invalid.append("manifest_sha256")
    blocks = receipt.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        invalid.append("blocks")
    if invalid:
        raise MatterVialReplayError(
            "Stage1 supplemental feature cache violates its target-free contract: "
            + ", ".join(sorted(set(invalid)))
        )
    return {
        "schema_version": int(schema_version),
        "validation_policy": "identity_and_target_access_contract",
        "target_firewall_fields": list(required_false),
    }


def _payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


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


def configured_id_alias(
    source_ids: Iterable[str],
    alias: Mapping[str, Any] | None,
    *,
    validate_total_count: bool = True,
) -> list[str]:
    values = [str(value) for value in source_ids]
    if alias is None:
        return values
    required = {"policy", "source_regex", "target_template", "expected_mapping_count"}
    if set(alias) != required or alias.get("policy") != "configured_regex_full_bijection":
        raise MatterVialReplayError(
            "sample_id_alias must be an exact configured_regex_full_bijection"
        )
    pattern = re.compile(str(alias["source_regex"]))
    template = str(alias["target_template"])
    mapped: list[str] = []
    for value in values:
        match = pattern.fullmatch(value)
        if match is None:
            raise MatterVialReplayError(f"sample_id_alias does not match {value!r}")
        mapped.append(template.format(**match.groupdict()))
    if validate_total_count and len(mapped) != int(alias["expected_mapping_count"]):
        raise MatterVialReplayError("sample_id_alias row count differs from its frozen contract")
    if len(set(mapped)) != len(mapped):
        raise MatterVialReplayError("sample_id_alias is not one-to-one")
    return mapped


@dataclass(frozen=True)
class ReplayFeatureCache:
    root: Path
    receipt: dict[str, Any]
    sample_ids: np.ndarray
    columns: tuple[str, ...]
    values: np.ndarray
    supplemental_columns: Mapping[str, tuple[int, int]] | None = None
    supplemental_blocks: tuple[np.ndarray, ...] = ()
    supplemental_receipt_sha256: str | None = None
    supplemental_receipt_schema_version: int | None = None
    supplemental_validation_policy: str | None = None

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        manifest: TaskManifest,
        registry_sha256: str,
        verify_hashes: bool = False,
        supplemental_root: Path | None = None,
        supplemental_receipt_sha256: str | None = None,
    ) -> ReplayFeatureCache:
        root = root.expanduser().resolve()
        receipt = read_json(root / "receipt.json")
        if receipt.get("schema_version") != 12 or receipt.get("status") != "complete":
            raise MatterVialReplayError(f"Incomplete replay feature cache: {root}")
        if receipt.get("task") != manifest.task_name:
            raise MatterVialReplayError("Replay feature cache task mismatch")
        if receipt.get("manifest_digest") != manifest.digest:
            raise MatterVialReplayError("Replay feature cache manifest mismatch")
        if receipt.get("checkpoint_registry_sha256") != registry_sha256:
            raise MatterVialReplayError("Replay feature cache was built for another registry")
        sample_path = root / "sample_ids.npy"
        columns_path = root / "columns.npy"
        values_path = root / "features.npy"
        if verify_hashes:
            expected = {
                sample_path: receipt.get("sample_ids_sha256"),
                columns_path: receipt.get("columns_sha256"),
                values_path: receipt.get("features_sha256"),
            }
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise MatterVialReplayError(f"Replay feature cache hash mismatch: {path}")
        sample_ids = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        columns_array = np.load(columns_path, allow_pickle=False).astype(str)
        values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        if not np.array_equal(sample_ids.astype(str), manifest.sample_ids.astype(str)):
            raise MatterVialReplayError("Replay feature cache sample order mismatch")
        if values.shape != (len(sample_ids), len(columns_array)):
            raise MatterVialReplayError("Replay feature cache shape mismatch")
        if len(set(columns_array.tolist())) != len(columns_array):
            raise MatterVialReplayError("Replay feature cache contains duplicate columns")
        supplemental_columns: dict[str, tuple[int, int]] = {}
        supplemental_blocks: list[np.ndarray] = []
        observed_supplemental_receipt_sha256: str | None = None
        supplemental_contract: dict[str, Any] | None = None
        if supplemental_root is not None:
            supplemental_root = supplemental_root.expanduser().resolve()
            supplemental_receipt_path = supplemental_root / "receipt.json"
            supplemental_receipt = read_json(supplemental_receipt_path)
            observed_supplemental_receipt_sha256 = sha256_file(
                supplemental_receipt_path
            )
            if (
                supplemental_receipt_sha256 is not None
                and observed_supplemental_receipt_sha256
                != supplemental_receipt_sha256
            ):
                raise MatterVialReplayError(
                    "Stage1 supplemental feature-cache receipt changed"
                )
            supplemental_contract = validate_stage1_feature_receipt(
                supplemental_receipt,
                manifest=manifest,
            )
            supplemental_sample_path = supplemental_root / str(
                supplemental_receipt["sample_ids_path"]
            )
            supplemental_sample_ids = np.load(
                supplemental_sample_path, mmap_mode="r", allow_pickle=False
            )
            if not np.array_equal(
                supplemental_sample_ids.astype(str), manifest.sample_ids.astype(str)
            ):
                raise MatterVialReplayError(
                    "Stage1 supplemental feature-cache sample order mismatch"
                )
            if verify_hashes and sha256_file(supplemental_sample_path) != supplemental_receipt.get(
                "sample_ids_sha256"
            ):
                raise MatterVialReplayError(
                    "Stage1 supplemental feature-cache sample hash mismatch"
                )
            supplemental_schema_path = supplemental_root / "feature_schema.json"
            if verify_hashes and (
                not supplemental_schema_path.is_file()
                or sha256_file(supplemental_schema_path)
                != supplemental_receipt.get("feature_schema_sha256")
            ):
                raise MatterVialReplayError(
                    "Stage1 supplemental feature-cache schema hash mismatch"
                )
            for block_index, metadata in enumerate(
                supplemental_receipt.get("blocks", {}).values()
            ):
                block_path = supplemental_root / str(metadata["path"])
                if verify_hashes and sha256_file(block_path) != metadata.get("sha256"):
                    raise MatterVialReplayError(
                        f"Stage1 supplemental feature-cache hash mismatch: {block_path}"
                    )
                block = np.load(block_path, mmap_mode="r", allow_pickle=False)
                names = [str(value) for value in metadata.get("feature_names", [])]
                if list(block.shape) != list(metadata.get("shape", [])) or block.shape != (
                    len(sample_ids),
                    len(names),
                ):
                    raise MatterVialReplayError(
                        f"Stage1 supplemental feature-cache shape mismatch: {block_path}"
                    )
                supplemental_blocks.append(block)
                for column_index, name in enumerate(names):
                    if name in supplemental_columns:
                        raise MatterVialReplayError(
                            f"Stage1 supplemental feature cache duplicates {name!r}"
                        )
                    supplemental_columns[name] = (block_index, column_index)
        return cls(
            root=root,
            receipt=receipt,
            sample_ids=sample_ids,
            columns=tuple(columns_array.tolist()),
            values=values,
            supplemental_columns=supplemental_columns or None,
            supplemental_blocks=tuple(supplemental_blocks),
            supplemental_receipt_sha256=observed_supplemental_receipt_sha256,
            supplemental_receipt_schema_version=(
                None
                if supplemental_contract is None
                else int(supplemental_contract["schema_version"])
            ),
            supplemental_validation_policy=(
                None
                if supplemental_contract is None
                else str(supplemental_contract["validation_policy"])
            ),
        )

    def member_matrix(
        self,
        descriptors: Iterable[str],
        *,
        row_positions: np.ndarray | None = None,
    ) -> np.ndarray | None:
        lookup = {value: index for index, value in enumerate(self.columns)}
        requested = [str(value) for value in descriptors]
        supplemental = self.supplemental_columns or {}
        if any(value not in lookup and value not in supplemental for value in requested):
            return None
        rows: np.ndarray | None
        if row_positions is None:
            rows = None
            n_rows = len(self.sample_ids)
        else:
            rows = np.asarray(row_positions, dtype=np.int64).reshape(-1)
            if np.any(rows < 0) or np.any(rows >= len(self.sample_ids)):
                raise MatterVialReplayError(
                    "Replay row positions are outside the feature cache"
                )
            n_rows = len(rows)
        if all(value in lookup for value in requested):
            columns = np.asarray([lookup[value] for value in requested], dtype=np.int64)
            source = self.values if rows is None else self.values[rows]
            return np.asarray(source[:, columns], dtype=np.float32)

        # The primary cache contains the union required by outer-refit checkpoints.
        # Inner checkpoints may select additional columns from the certified,
        # target-free Stage-1 cache. The requested descriptor order remains exactly
        # the checkpoint order and no target values are accessed.
        matrix = np.empty((n_rows, len(requested)), dtype=np.float32)
        base_output = [index for index, value in enumerate(requested) if value in lookup]
        if base_output:
            base_columns = np.asarray(
                [lookup[requested[index]] for index in base_output], dtype=np.int64
            )
            base = self.values if rows is None else self.values[rows]
            matrix[:, base_output] = np.asarray(base[:, base_columns], dtype=np.float32)
        by_block: dict[int, list[tuple[int, int]]] = {}
        for output_index, value in enumerate(requested):
            if value in lookup:
                continue
            block_index, column_index = supplemental[value]
            by_block.setdefault(block_index, []).append((output_index, column_index))
        for block_index, selections in by_block.items():
            output_columns = [value[0] for value in selections]
            source_columns = np.asarray([value[1] for value in selections], dtype=np.int64)
            block = self.supplemental_blocks[block_index]
            selected = block if rows is None else block[rows]
            matrix[:, output_columns] = np.asarray(
                selected[:, source_columns], dtype=np.float32
            )
        return matrix


def _keras_output_layer(network: Any) -> Any:
    outputs = list(network.outputs)
    if len(outputs) != 1:
        raise MatterVialReplayError("MatterVial adapter requires a single-target network")
    history = getattr(outputs[0], "_keras_history", None)
    if history is None:
        raise MatterVialReplayError("Cannot identify the trained MODNet output layer")
    if hasattr(history, "layer"):
        return history.layer
    try:
        return history[0]
    except (IndexError, TypeError) as exc:
        raise MatterVialReplayError("Cannot resolve the trained MODNet output layer") from exc


def trained_hidden_width(member: Any) -> int:
    output_layer = _keras_output_layer(member.model)
    hidden_tensor = output_layer.input
    if isinstance(hidden_tensor, list | tuple):
        if len(hidden_tensor) != 1:
            raise MatterVialReplayError("MODNet output has multiple hidden inputs")
        hidden_tensor = hidden_tensor[0]
    shape = tuple(hidden_tensor.shape)
    if len(shape) != 2 or shape[-1] is None or int(shape[-1]) < 1:
        raise MatterVialReplayError(f"Unexpected task-trained hidden shape: {shape}")
    return int(shape[-1])


def inspect_ensemble_checkpoint(checkpoint: Path) -> dict[str, Any]:
    from modnet.models import MODNetModel

    checkpoint = checkpoint.expanduser().resolve()
    ensemble = MODNetModel.load(str(checkpoint))
    members = list(getattr(ensemble, "models", []))
    if not members:
        raise MatterVialReplayError(f"Checkpoint is not a non-empty MODNet ensemble: {checkpoint}")
    widths = [trained_hidden_width(member) for member in members]
    if len(set(widths)) != 1:
        raise MatterVialReplayError(f"Ensemble hidden widths differ: {widths}")
    descriptor_hashes: list[str] = []
    descriptors: set[str] = set()
    for member in members:
        selected = [str(value) for value in member.optimal_descriptors[: member.n_feat]]
        if len(selected) != int(member.n_feat):
            raise MatterVialReplayError("MODNet checkpoint has an incomplete descriptor contract")
        descriptor_hashes.append(_payload_sha256({"descriptors": selected}))
        descriptors.update(selected)
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "ensemble_members": len(members),
        "member_hidden_dim": widths[0],
        "hidden_dim": widths[0] * len(members),
        "hidden_aggregation": "ordered_member_penultimate_concatenation",
        "descriptor_contract_sha256": sorted(set(descriptor_hashes)),
        "required_descriptors": sorted(descriptors),
        "modnet_version": str(getattr(ensemble, "__modnet_version__", "unknown")),
    }


def validate_checkpoint_split(
    artifact_dir: Path,
    *,
    manifest: TaskManifest,
    outer_fold: int,
    checkpoint_role: str,
    inner_fold: int | None = None,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.expanduser().resolve()
    request = read_json(artifact_dir / "request.json")
    preprocessing = read_json(artifact_dir / "preprocessing_manifest.json")
    summary = read_json(artifact_dir / "native_training_summary.json")
    checkpoint = artifact_dir / "mattervial_modnet.pkl"
    metadata = read_json(artifact_dir / "metadata.json")
    expected_phase = {
        "inner": "inner_oof",
        "holdout": "holdout_validation",
        "outer": "outer_refit",
    }.get(checkpoint_role)
    if expected_phase is None:
        raise MatterVialReplayError(f"Unknown checkpoint role: {checkpoint_role}")
    if request.get("phase") != expected_phase:
        raise MatterVialReplayError(
            f"{artifact_dir}: phase {request.get('phase')!r} != {expected_phase!r}"
        )
    if request.get("task_name") != manifest.task_name or request.get("outer_fold") != outer_fold:
        raise MatterVialReplayError(f"{artifact_dir}: request task/fold mismatch")
    if checkpoint_role == "inner":
        if inner_fold is None or inner_fold not in range(manifest.n_stacking_splits):
            raise MatterVialReplayError("inner checkpoint validation requires inner_fold")
        if int(request.get("inner_fold", -1)) != inner_fold:
            raise MatterVialReplayError(f"{artifact_dir}: request inner fold mismatch")
        fit_role = "inner_train"
        expected_fit = manifest.positions(fit_role, outer_fold, inner_fold)
        expected_prediction_role = "inner_val"
    else:
        if inner_fold is not None:
            raise MatterVialReplayError("inner_fold is only valid for an inner checkpoint")
        fit_role = "holdout_train" if checkpoint_role == "holdout" else "outer_train"
        expected_fit = manifest.positions(fit_role, outer_fold)
        expected_prediction_role = (
            "holdout_val" if checkpoint_role == "holdout" else "outer_test"
        )
    observed_fit = np.asarray(preprocessing.get("fit_positions", []), dtype=np.int64)
    if not np.array_equal(observed_fit, expected_fit):
        raise MatterVialReplayError(
            f"{artifact_dir}: checkpoint fitting rows differ scientifically"
        )
    if metadata.get("split", {}).get("role") != expected_prediction_role:
        raise MatterVialReplayError(f"{artifact_dir}: formal artifact role mismatch")
    checkpoint_hash = sha256_file(checkpoint)
    if summary.get("checkpoint_sha256") != checkpoint_hash:
        raise MatterVialReplayError(f"{artifact_dir}: checkpoint hash differs from training receipt")
    return {
        "artifact_dir": str(artifact_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_role": checkpoint_role,
        "inner_fold": inner_fold,
        "fit_role": fit_role,
        "fit_positions_sha256": _array_sha256(expected_fit.astype(np.int64)),
        "formal_prediction_role": expected_prediction_role,
        "formal_predictions": str(artifact_dir / "predictions.npz"),
        "selection_receipt": request.get("selected_hyperparameters"),
        "feature_cache_dir": request.get("extra", {}).get("feature_cache_dir"),
        "effective_members": int(summary.get("n_models", -1)),
        "configured_members": int(summary.get("configured_n_models", -1)),
        "preprocessing_manifest": str(artifact_dir / "preprocessing_manifest.json"),
        "preprocessing_manifest_sha256": sha256_file(
            artifact_dir / "preprocessing_manifest.json"
        ),
    }


def _member_outputs(
    member: Any,
    matrix: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    output_layer = _keras_output_layer(member.model)
    hidden_tensor = output_layer.input
    if isinstance(hidden_tensor, list | tuple):
        hidden_tensor = hidden_tensor[0]
    joint_model = getattr(member, "_hpsafe_joint_model", None)
    if joint_model is None:
        joint_model = tf.keras.Model(
            inputs=member.model.inputs,
            outputs=[hidden_tensor, member.model.outputs[0]],
        )
        member._hpsafe_joint_model = joint_model
    transform_input = np.array(matrix, dtype=np.float64, copy=True)
    transform_input[np.isinf(transform_input)] = np.nan
    transformed = member._scale_impute.transform(transform_input)
    n = len(transformed)
    hidden_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    classification = max(member.num_classes.values()) >= 2
    for start in range(0, n, batch_size):
        batch = transformed[start : start + batch_size]
        hidden, raw = joint_model(batch, training=False)
        hidden = np.asarray(hidden, dtype=np.float32)
        prediction = np.asarray(raw)
        if classification:
            if prediction.ndim != 2 or prediction.shape[1] < 2:
                raise MatterVialReplayError("Classification MODNet output is not class probability")
            prediction = prediction[:, 1]
        else:
            prediction = prediction.reshape(len(batch), -1)
            if prediction.shape[1] != 1:
                raise MatterVialReplayError("Regression MODNet output is not scalar")
            prediction = prediction[:, 0]
        hidden_parts.append(hidden)
        prediction_parts.append(np.asarray(prediction, dtype=np.float32))
    return np.concatenate(hidden_parts, axis=0), np.concatenate(prediction_parts, axis=0)


def load_ensemble_checkpoint(checkpoint: Path) -> Any:
    from modnet.models import MODNetModel

    ensemble = MODNetModel.load(str(checkpoint))
    if not list(getattr(ensemble, "models", [])):
        raise MatterVialReplayError("MatterVial replay requires a non-empty ensemble")
    return ensemble


def replay_loaded_ensemble(
    ensemble: Any,
    cache: ReplayFeatureCache,
    *,
    batch_size: int,
    row_positions: np.ndarray | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    members = list(getattr(ensemble, "models", []))
    if not members:
        raise MatterVialReplayError("MatterVial replay requires a non-empty ensemble")
    rows = (
        None
        if row_positions is None
        else np.asarray(row_positions, dtype=np.int64).reshape(-1)
    )
    n = len(cache.sample_ids) if rows is None else len(rows)
    expected_width = trained_hidden_width(members[0])
    hidden_blocks: list[np.ndarray] = []
    prediction_values: list[np.ndarray] = []
    missing_members: list[int] = []
    matrix_cache: dict[tuple[str, ...], np.ndarray | None] = {}
    for member_index, member in enumerate(members):
        if trained_hidden_width(member) != expected_width:
            raise MatterVialReplayError("MatterVial member hidden dimensions differ")
        descriptors = tuple(
            str(value) for value in member.optimal_descriptors[: member.n_feat]
        )
        if descriptors not in matrix_cache:
            matrix_cache[descriptors] = (
                cache.member_matrix(descriptors)
                if rows is None
                else cache.member_matrix(descriptors, row_positions=rows)
            )
        matrix = matrix_cache[descriptors]
        if matrix is None:
            missing_members.append(member_index)
            hidden_blocks.append(np.zeros((n, expected_width), dtype=np.float32))
            continue
        hidden, prediction = _member_outputs(member, matrix, batch_size=batch_size)
        hidden_blocks.append(hidden)
        prediction_values.append(prediction)
    hidden = np.concatenate(hidden_blocks, axis=1)
    if not prediction_values:
        return {
            "hidden": hidden,
            "prediction": np.zeros(n, dtype=np.float32),
            "uncertainty": np.zeros(n, dtype=np.float32),
            "available": np.zeros(n, dtype=np.bool_),
            "used_members": 0,
            "total_members": len(members),
            "missing_members": missing_members,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    prediction_stack = np.stack(prediction_values, axis=0)
    prediction = np.mean(prediction_stack, axis=0, dtype=np.float64).astype(np.float32)
    uncertainty = np.std(prediction_stack, axis=0, dtype=np.float64).astype(np.float32)
    available = (
        np.isfinite(hidden).all(axis=1)
        & np.isfinite(prediction)
        & np.isfinite(uncertainty)
    )
    return {
        "hidden": hidden,
        "prediction": prediction,
        "uncertainty": uncertainty,
        "available": available,
        "used_members": len(prediction_values),
        "total_members": len(members),
        "missing_members": missing_members,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def replay_ensemble(
    checkpoint: Path,
    cache: ReplayFeatureCache,
    *,
    batch_size: int,
) -> dict[str, Any]:
    return replay_loaded_ensemble(
        load_ensemble_checkpoint(checkpoint), cache, batch_size=batch_size
    )


def formal_predictions(artifact_dir: Path) -> dict[str, np.ndarray]:
    with np.load(artifact_dir / "predictions.npz", allow_pickle=False) as loaded:
        result = {
            "sample_ids": loaded["sample_ids"].astype(str),
            "prediction": np.asarray(loaded["y_pred"], dtype=np.float32),
        }
        if "y_uncertainty" in loaded.files:
            result["uncertainty"] = np.asarray(loaded["y_uncertainty"], dtype=np.float32)
    return result


def replace_role_with_formal_predictions(
    *,
    full_sample_ids: np.ndarray,
    role_positions: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    formal = formal_predictions(artifact_dir)
    expected_ids = np.asarray(full_sample_ids)[role_positions].astype(str)
    if not np.array_equal(formal["sample_ids"], expected_ids):
        raise MatterVialReplayError("Stage-1 prediction IDs differ from the declared split role")
    output_prediction = np.asarray(prediction, dtype=np.float32).copy()
    output_uncertainty = np.asarray(uncertainty, dtype=np.float32).copy()
    output_prediction[role_positions] = formal["prediction"]
    if "uncertainty" in formal:
        output_uncertainty[role_positions] = formal["uncertainty"]
    return output_prediction, output_uncertainty

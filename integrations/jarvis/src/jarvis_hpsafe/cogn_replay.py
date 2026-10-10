from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cache import cache_directory, write_role_cache
from .constants import JARVIS_TASKS, LATENT_DIM
from .io import atomic_json, read_json, sha256_file
from .manifest import JarvisManifest, load_manifests
from .stage1_artifact import align_artifact, load_stage1_artifact


CACHE_PROTOCOL = "cross_task_full_source_checkpoint"


def _stage1_paths(stage1_root: Path, task: str) -> dict[str, Path]:
    base = stage1_root / "outputs" / "exact" / "experts" / "kgcnn_cogn" / task / "outer_0"
    return {
        "oof": base / "inner_oof",
        "val": base / "official_validation",
        "test": base / "outer_test",
        "checkpoint": base / "official_validation" / "model.weights.h5",
        "scaler": base / "official_validation" / "target_scaler.json",
    }


def _authoritative_own_values(
    paths: dict[str, Path], manifest: JarvisManifest, role: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    artifact_role = "oof" if role == "train" else role
    artifact = load_stage1_artifact(paths[artifact_role], expected_latent_dim=LATENT_DIM)
    prediction, hidden, available = align_artifact(artifact, manifest.ids(role))
    return prediction, hidden, available, artifact.checkpoint_sha256


def _load_cogn_runtime(stage1_root: Path) -> dict[str, Any]:
    source_path = str((stage1_root / "src").resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from hpsafe_cogn_stage1.cache import verify_cache
    from hpsafe_cogn_stage1.graph_store import load_graphs_by_ids
    from hpsafe_cogn_stage1.model import build_latent_model, build_model, configure_tensorflow
    from hpsafe_cogn_stage1.tensors import input_tensors

    return {
        "verify_cache": verify_cache,
        "load_graphs_by_ids": load_graphs_by_ids,
        "build_latent_model": build_latent_model,
        "build_model": build_model,
        "configure_tensorflow": configure_tensorflow,
        "input_tensors": input_tensors,
    }


def _all_needed_materials(manifests: dict[str, JarvisManifest]) -> np.ndarray:
    values: set[str] = set()
    for manifest in manifests.values():
        for role in ("train", "val", "test"):
            values.update(manifest.materials(role).tolist())
    return np.asarray(sorted(values), dtype=str)


def _assemble_full_checkpoint_values(
    requested_materials: np.ndarray,
    *,
    material_lookup: dict[str, int],
    full_prediction: np.ndarray,
    full_hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.asarray(
        [material_lookup[value] for value in requested_materials.tolist()], dtype=np.int64
    )
    return (
        full_prediction[order],
        full_hidden[order],
        np.ones(len(order), dtype=np.bool_),
    )


def _predict_missing_materials(
    *,
    stage1_root: Path,
    source_task: str,
    material_ids: np.ndarray,
    graph_cache: Path,
    batch_size: int,
    progress_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    paths = _stage1_paths(stage1_root, source_task)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    val_metadata = read_json(paths["val"] / "metadata.json")
    declared = str(val_metadata.get("provenance", {}).get("checkpoint_sha256", ""))
    if checkpoint_hash != declared:
        raise ValueError(f"Frozen coGN checkpoint hash mismatch for {source_task}")
    graph_receipt_path = graph_cache.with_suffix(".receipt.json")
    graph_receipt = read_json(graph_receipt_path)
    graph_hash = str(graph_receipt.get("cache_sha256", ""))
    if len(graph_hash) != 64:
        raise ValueError(f"Frozen coGN graph receipt lacks SHA-256: {graph_receipt_path}")
    started = time.monotonic()
    if len(material_ids) == 0:
        atomic_json(
            progress_path,
            {
                "status": "running",
                "stage": "cache_complete",
                "source_task": source_task,
                "n_missing_materials": 0,
                "percent": 70.0,
            },
        )
        return (
            material_ids,
            np.empty(0, dtype=np.float64),
            np.empty((0, LATENT_DIM), dtype=np.float32),
            checkpoint_hash,
            graph_hash,
        )

    runtime = _load_cogn_runtime(stage1_root)
    receipt = runtime["verify_cache"](graph_cache, graph_receipt_path)
    verified_graph_hash = str(receipt.get("cache_sha256", ""))
    if verified_graph_hash != graph_hash:
        raise ValueError("Frozen coGN graph cache receipt changed during replay")
    atomic_json(
        progress_path,
        {
            "status": "running",
            "stage": "load_incremental_replay_graphs",
            "source_task": source_task,
            "n_missing_materials": len(material_ids),
            "percent": 2.0,
        },
    )
    runtime["configure_tensorflow"]()
    graphs = runtime["load_graphs_by_ids"](graph_cache, material_ids)
    model = runtime["build_model"]()
    model.load_weights(str(paths["checkpoint"]))
    latent_model = runtime["build_latent_model"](model, expected_dim=LATENT_DIM)
    import tensorflow as tf
    from kgcnn.data.transform.scaler.standard import StandardScaler

    combined = tf.keras.Model(model.inputs, [model.output, latent_model.output])
    tensors = runtime["input_tensors"](model, graphs)
    atomic_json(
        progress_path,
        {
            "status": "running",
            "stage": "incremental_full_checkpoint_replay",
            "source_task": source_task,
            "n_missing_materials": len(material_ids),
            "percent": 10.0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    total_batches = max(1, math.ceil(len(material_ids) / batch_size))

    class ReplayProgress(tf.keras.callbacks.Callback):
        def on_predict_batch_end(self, batch: int, logs: dict[str, Any] | None = None) -> None:
            completed = int(batch) + 1
            if completed == 1 or completed % 20 == 0 or completed == total_batches:
                elapsed = time.monotonic() - started
                atomic_json(
                    progress_path,
                    {
                        "status": "running",
                        "stage": "incremental_full_checkpoint_replay",
                        "source_task": source_task,
                        "completed_batches": completed,
                        "total_batches": total_batches,
                        "percent": round(10.0 + 60.0 * completed / total_batches, 2),
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds": round(
                            (total_batches - completed) * elapsed / max(completed, 1), 3
                        ),
                    },
                )

    scaled, hidden = combined.predict(
        tensors,
        batch_size=batch_size,
        verbose=1,
        callbacks=[ReplayProgress()],
    )
    scaler = StandardScaler()
    loaded_scaler = scaler.load(str(paths["scaler"]))
    # kgcnn releases differ on whether ``load`` mutates in place or also
    # returns ``self``. Accept both while keeping the frozen scaler unchanged.
    if loaded_scaler is not None:
        scaler = loaded_scaler
    prediction = scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1)).reshape(-1)
    hidden = np.asarray(hidden, dtype=np.float32)
    if prediction.shape != (len(material_ids),) or hidden.shape != (len(material_ids), LATENT_DIM):
        raise ValueError("Combined coGN replay returned unexpected shapes")
    if not np.isfinite(prediction).all() or not np.isfinite(hidden).all():
        raise ValueError("Combined coGN replay returned non-finite values")
    atomic_json(
        progress_path,
        {
            "status": "running",
            "stage": "write_role_caches",
            "source_task": source_task,
            "n_unique_materials": len(material_ids),
            "percent": 75.0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
    )
    del tensors, graphs, combined, latent_model, model
    tf.keras.backend.clear_session()
    gc.collect()
    return material_ids, prediction, hidden, checkpoint_hash, graph_hash


def prepare_source_cache(
    *,
    stage1_root: Path,
    output_root: Path,
    source_task: str,
    graph_cache: Path | None = None,
    batch_size: int = 128,
) -> Path:
    if source_task not in JARVIS_TASKS:
        raise ValueError(f"Unknown source task: {source_task}")
    stage1_root = stage1_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    completion = output_root / source_task / "source_completion.json"
    if completion.is_file():
        value = read_json(completion)
        if (
            value.get("status") == "complete"
            and value.get("source_task") == source_task
            and value.get("cache_protocol") == CACHE_PROTOCOL
        ):
            if Path(str(value.get("stage1_root", ""))).resolve() != stage1_root:
                raise RuntimeError(
                    f"Completed cache belongs to a different Stage1 root: {completion}"
                )
            for record in value.get("roles", []):
                metadata_path = Path(str(record["path"])) / "metadata.json"
                if (
                    not metadata_path.is_file()
                    or sha256_file(metadata_path) != record.get("metadata_sha256")
                ):
                    raise RuntimeError(f"Completed role cache changed: {metadata_path}")
            return completion
        raise RuntimeError(
            f"Cache completion is inconsistent: {completion}. "
            "Use the cross-task full-source run root."
        )
    # Cache construction is strictly label-free, including the official test.
    manifests = load_manifests(stage1_root, load_targets=False)
    graph_cache = (
        stage1_root / "data" / "jarvis" / "features" / "cogn" / "cogn_knn24.h5"
        if graph_cache is None
        else graph_cache.expanduser().resolve()
    )
    progress_path = output_root / source_task / "progress.json"
    paths = _stage1_paths(stage1_root, source_task)
    checkpoint_hash = sha256_file(paths["checkpoint"])
    material_ids = _all_needed_materials(manifests)
    replay_materials, full_prediction, full_hidden, replay_hash, graph_hash = (
        _predict_missing_materials(
            stage1_root=stage1_root,
            source_task=source_task,
            material_ids=material_ids,
            graph_cache=graph_cache,
            batch_size=batch_size,
            progress_path=progress_path,
        )
    )
    if replay_hash != checkpoint_hash:
        raise ValueError("Frozen full source checkpoint changed during cache construction")
    if (
        not np.array_equal(replay_materials, material_ids)
        or not np.isfinite(full_prediction).all()
        or not np.isfinite(full_hidden).all()
    ):
        raise ValueError("Full source checkpoint feature bank is incomplete")
    material_lookup = {value: index for index, value in enumerate(material_ids.tolist())}
    role_records: list[dict[str, Any]] = []
    total_roles = len(JARVIS_TASKS) * 3
    completed_roles = 0
    for target_task in JARVIS_TASKS:
        manifest = manifests[target_task]
        for role in ("train", "val", "test"):
            if target_task == source_task:
                prediction, hidden, available, own_hash = _authoritative_own_values(
                    _stage1_paths(stage1_root, source_task), manifest, role
                )
                role_checkpoint_hash = own_hash
                producer = {
                    "kind": "own_task_oof" if role == "train" else "own_task_frozen_full_train",
                    "cross_task_replay": False,
                    "target_labels_accessed_by_cache_writer": False,
                    "official_test_used_for_selection": False,
                }
            else:
                requested_materials = manifest.materials(role)
                prediction, hidden, available = _assemble_full_checkpoint_values(
                    requested_materials,
                    material_lookup=material_lookup,
                    full_prediction=full_prediction,
                    full_hidden=full_hidden,
                )
                role_checkpoint_hash = checkpoint_hash
                producer = {
                    "kind": "cross_task_frozen_full_source_checkpoint",
                    "cross_task_replay": True,
                    "cache_protocol": CACHE_PROTOCOL,
                    "same_source_checkpoint_for_train_validation_test": True,
                    "same_material_seen_in_source_task_is_allowed": True,
                    "source_task_supervision_is_allowed": True,
                    "target_task_labels_accessed_by_cache_writer": False,
                    "availability_mask_policy": "technical_inability_only",
                    "entity_exposure_masking": False,
                    "official_test_used_for_selection": False,
                }
            destination = cache_directory(output_root, source_task, target_task, role)
            write_role_cache(
                destination,
                source_task=source_task,
                target_manifest=manifest,
                role=role,
                sample_ids=manifest.ids(role),
                hidden=hidden,
                prediction=prediction,
                available=available,
                checkpoint_sha256=role_checkpoint_hash,
                producer=producer,
            )
            role_records.append(
                {
                    "target_task": target_task,
                    "role": role,
                    "path": str(destination),
                    "metadata_sha256": sha256_file(destination / "metadata.json"),
                    "available_rows": int(np.sum(available)),
                }
            )
            completed_roles += 1
            atomic_json(
                progress_path,
                {
                    "status": "running",
                    "stage": "write_role_caches",
                    "source_task": source_task,
                    "completed_roles": completed_roles,
                    "total_roles": total_roles,
                    "percent": round(75.0 + 25.0 * completed_roles / total_roles, 2),
                    "current_target_task": target_task,
                    "current_role": role,
                },
            )
    payload = {
        "schema_version": "jarvis-hpsafemoe-source-cache",
        "status": "complete",
        "cache_protocol": CACHE_PROTOCOL,
        "source_task": source_task,
        "stage1_root": str(stage1_root),
        "stage1_checkpoint_sha256": checkpoint_hash,
        "graph_cache": str(graph_cache),
        "graph_cache_sha256": graph_hash,
        "n_union_materials": len(material_ids),
        "n_replayed_materials": len(material_ids),
        "same_material_seen_in_source_task_is_allowed": True,
        "roles": role_records,
        "labels_stored": False,
        "test_used_for_parameter_selection": False,
    }
    atomic_json(completion, payload)
    atomic_json(progress_path, {"status": "complete", "source_task": source_task, "percent": 100.0})
    return completion

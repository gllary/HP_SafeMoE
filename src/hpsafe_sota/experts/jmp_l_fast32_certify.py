from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from hpsafe_sota.artifacts import validate_fold_artifact
from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.experts.jmp_l_bridge import (
    FAST32_MAX_EPOCHS,
    FAST32_MAX_TIME_DAYS,
    FAST32_PROTOCOL,
    FAST32_TASK_SETTINGS,
    LATENT_CONTRACT_VERSION,
    LATENT_DIM,
    LATENT_SOURCE,
    LEARNING_RATE,
    OFFICIAL_COMMIT,
)
from hpsafe_sota.metrics import FoldMetric, score_predictions, summarize_folds


def _yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certify_jmp_l_fast32_stage1(
    project_root: Path,
    expert_root: Path,
    task_name: str,
) -> dict[str, Any]:
    """Certify the H20 FP32 fivefold Stage-1 run."""

    if task_name not in FAST32_TASK_SETTINGS:
        raise ValueError(task_name)
    manifest = TaskManifest(project_root / "data/manifests" / f"{task_name}.npz")
    task = _yaml(project_root / "configs/sota_registry.yaml")["tasks"][task_name]
    settings = FAST32_TASK_SETTINGS[task_name]
    folds: list[FoldMetric] = []
    receipts: list[dict[str, Any]] = []
    for outer_fold in range(5):
        artifact_root = expert_root / "jmp_l" / task_name / f"outer_{outer_fold}" / "outer_test"
        artifact = validate_fold_artifact(artifact_root, manifest=manifest)
        if artifact.metadata["split"]["role"] != "outer_test":
            raise ValueError(f"Outer {outer_fold}: expected outer_test predictions")
        if artifact.latents is None or artifact.available is None:
            raise ValueError(f"Outer {outer_fold}: hidden embeddings are missing")
        if artifact.latents.shape != (len(artifact.y_true), LATENT_DIM):
            raise ValueError(f"Outer {outer_fold}: invalid latent shape {artifact.latents.shape}")
        if not np.all(artifact.available) or not np.all(np.isfinite(artifact.latents)):
            raise ValueError(f"Outer {outer_fold}: incomplete/non-finite embeddings")

        description = artifact.metadata.get("extra", {}).get("expert_description", {})
        expected_latent = {
            "schema_version": LATENT_CONTRACT_VERSION,
            "source": LATENT_SOURCE,
            "dimension": LATENT_DIM,
            "pooling": settings["reduction"],
            "split": "official_matbench_outer_test",
            "checkpoint_state": "fp32_fitted_fold_checkpoint",
        }
        if description.get("finetune_protocol") != FAST32_PROTOCOL:
            raise ValueError(f"Outer {outer_fold}: Fast-32 protocol receipt drifted")
        if description.get("latent_contract") != expected_latent:
            raise ValueError(f"Outer {outer_fold}: latent contract drifted")
        selection = artifact.metadata["selection"]
        if selection.get("source") != "fixed_jmp_l_h20_fast32_recipe":
            raise ValueError(f"Outer {outer_fold}: unexpected selection branch")
        if selection.get("selected_hyperparameters") is not None:
            raise ValueError(f"Outer {outer_fold}: selection receipt is forbidden")

        preprocessing_path = artifact_root / "preprocessing_manifest.json"
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        expected_preprocessing = {
            "official_commit": OFFICIAL_COMMIT,
            "learning_rate": LEARNING_RATE,
            "finetune_protocol": FAST32_PROTOCOL,
            "maximum_epochs": FAST32_MAX_EPOCHS,
            "maximum_wall_time_days": FAST32_MAX_TIME_DAYS,
            "split_protocol": "official_matbench_outer_train_to_outer_test",
            "target_normalization_scope": "complete_official_outer_train_only",
            "hyperparameter_policy": "fixed_release_values",
            "fit_protocol": "single_outer_fold_fit",
            "workflow_stage": "stage1",
            "automatic_full_state_resume": True,
        }
        if any(preprocessing.get(key) != value for key, value in expected_preprocessing.items()):
            raise ValueError(f"Outer {outer_fold}: preprocessing receipt drifted")
        if preprocessing.get("task_settings") != settings:
            raise ValueError(f"Outer {outer_fold}: batch/precision/readout drifted")

        training_path = artifact_root / "native_training_summary.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        fixed = training.get("fixed_official_parameters", {})
        expected_fixed = {
            "learning_rate": LEARNING_RATE,
            "maximum_epochs": FAST32_MAX_EPOCHS,
            "maximum_wall_time_days": FAST32_MAX_TIME_DAYS,
            "batch_size": settings["batch_size"],
            "precision": settings["precision"],
            "graph_reduction": settings["reduction"],
            "num_workers": settings["num_workers"],
        }
        if training.get("finetune_protocol") != FAST32_PROTOCOL or fixed != expected_fixed:
            raise ValueError(f"Outer {outer_fold}: training recipe drifted")
        trained_epochs = int(training.get("trained_epochs", -1))
        if trained_epochs != FAST32_MAX_EPOCHS or not training.get(
            "trainer_fit_returned_normally", False
        ):
            raise ValueError(
                f"Outer {outer_fold}: Fast-32 did not complete all 32 epochs "
                f"(trained_epochs={trained_epochs})"
            )

        score = score_predictions(task["task_type"], artifact.y_true, artifact.y_pred)
        folds.append(FoldMetric(fold=outer_fold, score=score, n_samples=len(artifact.y_true)))
        receipts.append(
            {
                "outer_fold": outer_fold,
                "n_samples": len(artifact.y_true),
                "score": score,
                "trained_epochs": trained_epochs,
                "global_optimizer_steps": int(training["global_optimizer_steps"]),
                "metadata_sha256": _sha256(artifact_root / "metadata.json"),
                "preprocessing_sha256": _sha256(preprocessing_path),
                "native_training_summary_sha256": _sha256(training_path),
                "latents_sha256": _sha256(artifact_root / "latents.npz"),
                "checkpoint_sha256": artifact.metadata["provenance"]["checkpoint_sha256"],
            }
        )

    summary = summarize_folds(folds)
    values = np.asarray([fold.score for fold in folds], dtype=np.float64)
    payload = {
        "schema_version": 1,
        "status": "certified_jmp_l_h20_fast32_on_project_stage1_matbench_fivefold",
        "protocol": FAST32_PROTOCOL,
        "training_profile": "fp32_32epoch",
        "task_name": task_name,
        "expert_name": "jmp_l",
        "metric": task["metric"],
        "direction": task["direction"],
        "fold_scores": values.tolist(),
        "mean": float(summary["mean"]),
        "std": float(np.std(values)),
        "official_commit": OFFICIAL_COMMIT,
        "official_pretrained_checkpoint": True,
        "learning_rate": LEARNING_RATE,
        "latent_contract": {
            **expected_latent,
            "all_rows_available": True,
        },
        "protocol": {
            "outer_folds": list(range(5)),
            "fit_scope": "complete_outer_train",
            "hyperparameter_policy": "fixed_release_values",
            "fit_protocol": "single_outer_fold_fit",
            "workflow_stage": "stage1",
            "schedule": "fixed_32_epoch_primary_cosine",
            "hidden_embedding": "native_inference_on_official_outer_test",
        },
        "fold_receipts": receipts,
    }
    output_path = expert_root / "jmp_l" / task_name / "certification.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload

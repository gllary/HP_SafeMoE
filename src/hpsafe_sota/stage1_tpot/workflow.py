from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from hpsafe_sota.artifacts import validate_fold_artifact, write_fold_artifact
from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.metrics import score_predictions
from hpsafe_sota.stage1_tpot.features import load_feature_cache
from hpsafe_sota.stage1_tpot.official import (
    ELEMENTS,
    EXPERT_ID,
    HIDDEN_DIM,
    OFFICIAL_COMMIT,
    OFFICIAL_PIPELINE_SHA256,
    PROVIDER,
    load_official_pipeline_template,
    sha256_file,
    verify_official_assets,
)
from hpsafe_sota.stage1_tpot.pipeline import (
    fit_pipeline,
    fresh_official_pipeline,
    replay_pipeline,
)
from hpsafe_sota.stage2_common.cache import validate_expert_export, write_expert_export
from hpsafe_sota.stage2_common.io import atomic_json
from hpsafe_sota.stage2_common.protocol import ExpertSpec

TASK = "steels"
MODEL_NAME = "tpot_mat_official_pipeline"


def read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "tpot-mat-stage1"
        or value.get("status") != "release"
        or value.get("task") != TASK
        or int(value.get("recipe", {}).get("hidden_dim", -1)) != HIDDEN_DIM
        or value.get("official_source", {}).get("commit") != OFFICIAL_COMMIT
        or value.get("official_source", {}).get("pipeline_sha256")
        != OFFICIAL_PIPELINE_SHA256
        or int(value.get("recipe", {}).get("expected_input_dim", -1)) != len(ELEMENTS)
        or tuple(value.get("recipe", {}).get("composition_elements", ())) != ELEMENTS
    ):
        raise ValueError("Invalid Stage-1 TPOT-Mat release configuration")
    protocol = value.get("fixed_protocol", {})
    if int(protocol.get("random_seed", -1)) != 18012019:
        raise ValueError("The TPOT-Mat protocol fixes random_seed=18012019")
    if protocol.get("seed_scope") != "all_outer_folds":
        raise ValueError("The TPOT-Mat protocol uses one seed across all outer folds")
    return value


def preflight(*, project_root: Path, config_path: Path, cache_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = read_config(config_path)
    assets = verify_official_assets(project_root)
    template = load_official_pipeline_template(project_root)
    official_input_dim = int(
        template.named_steps["stackingestimator-1"].estimator.n_features_in_
    )
    if official_input_dim != len(ELEMENTS):
        raise ValueError("TPOT-Mat official pipeline/input schema mismatch")
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    if len(manifest.sample_ids) != 312:
        raise ValueError("TPOT-Mat official replay expects 312 steels samples")
    cache_parent = cache_root.expanduser().resolve().parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    free = os.statvfs(cache_parent)
    available = free.f_bavail * free.f_frsize / 1024**3
    required = float(config["execution"]["minimum_free_disk_gb"])
    if available < required:
        raise RuntimeError(f"TPOT-Mat requires {required:.1f} GiB free, found {available:.1f}")
    versions: dict[str, str] = {}
    for module in ("numpy", "pandas", "sklearn", "xgboost", "tpot", "joblib"):
        imported = __import__(module)
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    required_versions = {
        "numpy": "1.23.5",
        "pandas": "1.5.1",
        "sklearn": "1.2.2",
        "xgboost": "1.7.6",
        "tpot": "0.11.7",
        "joblib": "1.2.0",
    }
    mismatches = {
        key: {"expected": expected, "observed": versions.get(key)}
        for key, expected in required_versions.items()
        if versions.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"TPOT-Mat isolated environment version drift: {mismatches}")
    return {
        "schema_version": 1,
        "status": "pass",
        "task": TASK,
        "manifest_digest": manifest.digest,
        "official_assets": assets,
        "versions": versions,
        "fixed_seed": int(config["fixed_protocol"]["random_seed"]),
        "seed_scope": "all_outer_folds",
        "outer_folds": list(range(5)),
        "fit_jobs": 5,
        "gpu_jobs": 0,
        "cpu_parallelism": int(config["execution"]["cpu_concurrency"]),
        "hidden_dim": HIDDEN_DIM,
        "official_input_dim": official_input_dim,
        "official_input_elements": list(ELEMENTS),
        "fit_scope": "outer_train",
    }


def _atomic_joblib(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".joblib", delete=False) as handle:
            temporary = handle.name
        joblib.dump(payload, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _progress(path: Path, **values: Any) -> None:
    atomic_json(
        path,
        {
            "schema_version": 1,
            "event": "stage1_tpot_mat_progress",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        },
    )
    print("HPSAFE_TPOT " + json.dumps(values, sort_keys=True), flush=True)


def train_fixed_fold(
    *,
    project_root: Path,
    config_path: Path,
    cache_root: Path,
    artifact_root: Path,
    outer_fold: int,
    progress_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    project_root = project_root.resolve()
    config = read_config(config_path)
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    seed = int(config["fixed_protocol"]["random_seed"])
    features, _, cache_receipt = load_feature_cache(project_root, cache_root)
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    outer_train = manifest.positions("outer_train", outer_fold)
    outer_test = manifest.positions("outer_test", outer_fold)
    _progress(
        progress_path,
        stage="loading_official_recipe",
        seed=seed,
        outer_fold=outer_fold,
        completed=0,
        total=3,
        percent=0.0,
    )
    pipeline, seed_updates = fresh_official_pipeline(
        project_root,
        seed=seed,
        threads=int(config["execution"]["threads_per_fit"]),
    )
    pipeline, hidden_recipe = fit_pipeline(
        pipeline, features[outer_train], manifest.targets[outer_train]
    )
    _progress(
        progress_path,
        stage="fitted_complete_outer_train",
        seed=seed,
        outer_fold=outer_fold,
        completed=1,
        total=3,
        percent=33.33,
    )
    replay = replay_pipeline(pipeline, hidden_recipe, features[outer_test])
    output = artifact_root.resolve() / "folds" / f"outer_{outer_fold}" / "outer_test"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "tpot_mat_pipeline.joblib"
    checkpoint_payload = {
        "schema_version": 1,
        "status": "fitted_complete_outer_train",
        "task": TASK,
        "outer_fold": outer_fold,
        "seed": seed,
        "pipeline": pipeline,
        "hidden_recipe": hidden_recipe,
        "seed_parameter_updates": seed_updates,
        "official_commit": OFFICIAL_COMMIT,
        "official_pipeline_sha256": OFFICIAL_PIPELINE_SHA256,
        "fit_positions_sha256": hashlib.sha256(
            np.asarray(outer_train, dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    _atomic_joblib(checkpoint, checkpoint_payload)
    checkpoint_hash = sha256_file(checkpoint)
    preprocessing = {
        "schema_version": 1,
        "status": "complete",
        "task": TASK,
        "outer_fold": outer_fold,
        "seed": seed,
        "feature_cache": str(cache_root.resolve()),
        "feature_cache_receipt_sha256": sha256_file(cache_root.resolve() / "receipt.json"),
        "feature_recipe": cache_receipt["feature_recipe"],
        "feature_schema": cache_receipt["elements"],
        "feature_data_scope": "composition_only",
        "fit_scope": "complete_outer_train",
        "seed_scope": "all_outer_folds",
        "pipeline_initialization": "official_topology_fresh_outer_fold_fit",
    }
    atomic_json(output / "preprocessing_manifest.json", preprocessing)
    env_hash = sha256_file(project_root / "envs/tpot_mat.yml")
    preprocessing_hash = sha256_file(output / "preprocessing_manifest.json")
    _progress(
        progress_path,
        stage="checkpoint_committed_before_outer_test_scoring",
        seed=seed,
        outer_fold=outer_fold,
        completed=2,
        total=3,
        percent=66.67,
        checkpoint_sha256=checkpoint_hash,
    )
    write_fold_artifact(
        output,
        manifest=manifest,
        model_name=MODEL_NAME,
        model_family="TPOT_selected_pipeline_replay",
        outer_fold=outer_fold,
        split_role="outer_test",
        y_pred=replay.prediction,
        y_uncertainty=replay.uncertainty,
        latents=replay.hidden,
        available=np.ones(len(outer_test), dtype=np.bool_),
        run_id=f"tpot_mat_fixed_seed_{seed}_outer_{outer_fold}",
        checkpoint_sha256=checkpoint_hash,
        environment_lock_sha256=env_hash,
        preprocessing_manifest_sha256=preprocessing_hash,
        selection={"source": "fixed_release_seed", "outer_test_used": False},
        extra_metadata={
            "official_commit": OFFICIAL_COMMIT,
            "official_pipeline_sha256": OFFICIAL_PIPELINE_SHA256,
            "hidden_source": hidden_recipe["hidden_source"],
            "fixed_seed": seed,
            "checkpoint_committed_before_outer_test_score": True,
        },
    )
    artifact = validate_fold_artifact(output, manifest=manifest, expected_latent_dim=HIDDEN_DIM)
    score = float(score_predictions("regression", artifact.y_true, artifact.y_pred))
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "task": TASK,
        "seed": seed,
        "outer_fold": outer_fold,
        "score_mae": score,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "artifact": str(output),
        "metadata_sha256": sha256_file(output / "metadata.json"),
        "hidden_dim": HIDDEN_DIM,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "fit_scope": "outer_train",
        "seed_policy": "fixed_release_seed",
    }
    atomic_json(output / "completed.json", receipt)
    _progress(
        progress_path,
        stage="complete",
        seed=seed,
        outer_fold=outer_fold,
        completed=3,
        total=3,
        percent=100.0,
        score_mae=score,
        elapsed_seconds=receipt["elapsed_seconds"],
    )
    return receipt


def expert_spec(export_root: Path) -> ExpertSpec:
    return ExpertSpec(
        expert_id=EXPERT_ID,
        source_task=TASK,
        provider=PROVIDER,
        hidden_dim=HIDDEN_DIM,
        export_root=(export_root.resolve() / EXPERT_ID),
        input_modality="steels_14_element_composition",
        representation_kind="task_trained_hidden",
    )


def export_fixed_fold(
    *,
    project_root: Path,
    config_path: Path,
    cache_root: Path,
    artifact_root: Path,
    export_root: Path,
    outer_fold: int,
) -> dict[str, Any]:
    if outer_fold not in range(5):
        raise ValueError("outer_fold must be 0..4")
    project_root = project_root.resolve()
    config = read_config(config_path)
    seed = int(config["fixed_protocol"]["random_seed"])
    source = artifact_root.resolve() / "folds" / f"outer_{outer_fold}" / "outer_test"
    payload = joblib.load(source / "tpot_mat_pipeline.joblib")
    if (
        payload.get("status") != "fitted_complete_outer_train"
        or payload.get("outer_fold") != outer_fold
        or payload.get("seed") != seed
        or payload.get("official_pipeline_sha256") != OFFICIAL_PIPELINE_SHA256
    ):
        raise ValueError("Fixed TPOT-Mat checkpoint identity drifted")
    checkpoint_hash = sha256_file(source / "tpot_mat_pipeline.joblib")
    features, _, cache_receipt = load_feature_cache(project_root, cache_root)
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    spec = expert_spec(export_root)
    roles = {"train": "holdout_train", "val": "holdout_val", "test": "outer_test"}
    outputs: dict[str, str] = {}
    replay_hashes: dict[str, str] = {}
    for role, manifest_role in roles.items():
        positions = manifest.positions(manifest_role, outer_fold)
        replay = replay_pipeline(payload["pipeline"], payload["hidden_recipe"], features[positions])
        if role == "test":
            scored = validate_fold_artifact(source, manifest=manifest, expected_latent_dim=HIDDEN_DIM)
            if not np.allclose(replay.prediction, scored.y_pred, rtol=1.0e-10, atol=1.0e-10):
                raise ValueError("Fixed TPOT-Mat checkpoint replay differs from scored artifact")
            if not np.allclose(replay.hidden, scored.latents, rtol=1.0e-7, atol=1.0e-7):
                raise ValueError("Fixed TPOT-Mat hidden replay differs from scored artifact")
        destination = spec.export_root / TASK / f"outer_{outer_fold}" / role
        producer = {
            "adapter": "tpot_mat_outer_checkpoint_replay",
            "checkpoint_role": "outer",
            "checkpoint_fit_role": "outer_train",
            "source_outer_fold": outer_fold,
            "target_outer_fold": outer_fold,
            "checkpoint_usage": "frozen_outer_fold_checkpoint_replay",
            "official_commit": OFFICIAL_COMMIT,
            "official_pipeline_sha256": OFFICIAL_PIPELINE_SHA256,
            "hidden_source": "three_stacking_predictions_plus_xgboost_leaf_path",
            "uncertainty_kind": "within_pipeline_component_prediction_std",
            "fixed_seed": seed,
            "seed_policy": "fixed_release_seed",
            "export_input_scope": "features_and_sample_ids",
            "label_access": "scoring_after_prediction",
            "availability_scope": "own_task",
            "feature_cache_receipt_sha256": sha256_file(cache_root.resolve() / "receipt.json"),
            "feature_recipe": cache_receipt["feature_recipe"],
        }
        write_expert_export(
            destination,
            manifest=manifest,
            spec=spec,
            outer_fold=outer_fold,
            role=role,
            sample_ids=manifest.ids(manifest_role, outer_fold),
            hidden=replay.hidden,
            prediction=replay.prediction,
            uncertainty=replay.uncertainty,
            available=np.ones(len(positions), dtype=np.bool_),
            checkpoint_sha256=checkpoint_hash,
            producer=producer,
        )
        outputs[role] = str(destination)
        replay_hashes[role] = sha256_file(destination / "metadata.json")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "task": TASK,
        "outer_fold": outer_fold,
        "fixed_seed": seed,
        "checkpoint": str(source / "tpot_mat_pipeline.joblib"),
        "checkpoint_sha256": checkpoint_hash,
        "outputs": outputs,
        "metadata_sha256_by_role": replay_hashes,
        "hidden_dim": HIDDEN_DIM,
        "checkpoint_replay_scope": "train_val_test_within_outer_fold",
    }
    atomic_json(spec.export_root / TASK / f"outer_{outer_fold}/export_receipt.json", receipt)
    return receipt


def validate_exports(
    *,
    project_root: Path,
    config_path: Path,
    artifact_root: Path,
    export_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = read_config(config_path)
    seed = int(config["fixed_protocol"]["random_seed"])
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    spec = expert_spec(export_root)
    verified: list[dict[str, Any]] = []
    fold_scores: list[float] = []
    for outer in range(5):
        scored = validate_fold_artifact(
            artifact_root.resolve() / "folds" / f"outer_{outer}" / "outer_test",
            manifest=manifest,
            expected_latent_dim=HIDDEN_DIM,
        )
        fold_scores.append(float(score_predictions("regression", scored.y_true, scored.y_pred)))
        for role in ("train", "val", "test"):
            value = validate_expert_export(
                spec.export_root / TASK / f"outer_{outer}/{role}",
                manifest=manifest,
                spec=spec,
                outer_fold=outer,
                role=role,
                verify_hashes=True,
            )
            producer = value.metadata["producer"]
            if (
                producer.get("adapter") != "tpot_mat_outer_checkpoint_replay"
                or producer.get("source_outer_fold") != outer
                or producer.get("target_outer_fold") != outer
                or producer.get("export_input_scope") != "features_and_sample_ids"
                or producer.get("seed_policy") != "fixed_release_seed"
            ):
                raise ValueError(f"TPOT-Mat Stage-2 producer contract drifted: o{outer}/{role}")
            verified.append(
                {
                    "outer_fold": outer,
                    "role": role,
                    "rows": len(value.sample_ids),
                    "metadata_sha256": sha256_file(value.root / "metadata.json"),
                }
            )
    payload = {
        "schema_version": 1,
        "status": "pass",
        "task": TASK,
        "fixed_seed": seed,
        "seed_scope": "all_outer_folds",
        "fold_mae": fold_scores,
        "mean_fold_mae": float(np.mean(fold_scores)),
        "verified_stage2_exports": len(verified),
        "verified_export_rows": int(sum(value["rows"] for value in verified)),
        "hidden_dim": HIDDEN_DIM,
        "hidden_source": "three_stacking_predictions_plus_xgboost_leaf_path",
        "provider": PROVIDER,
        "expert_id": EXPERT_ID,
        "official_assets": verify_official_assets(project_root),
        "fit_scope": "outer_train",
        "certification_status": "certified_fixed_tpot_mat_steels_for_stage2_pool",
        "exports": verified,
    }
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "stage1_tpot_mat_validation.json", payload)
    lines = [
        "# Stage-1 TPOT-Mat steels validation",
        "",
        f"- Fixed random seed: `{seed}`",
        "- Seed selection uses outer-test labels: `false`",
        "- The published TPOT-Mat pipeline recipe is cloned and refitted on each outer-train split.",
        f"- Mean five-fold MAE: `{payload['mean_fold_mae']:.8g}`",
        f"- Verified Stage-2 exports: `{payload['verified_stage2_exports']}/15`",
        "",
        "| Fold | MAE |",
        "|---:|---:|",
    ]
    lines.extend(f"| {outer} | {value:.8g} |" for outer, value in enumerate(fold_scores))
    (output_root / "stage1_tpot_mat_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return payload

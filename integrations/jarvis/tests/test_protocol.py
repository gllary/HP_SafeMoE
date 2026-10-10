from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from jarvis_hpsafe import core as core
from jarvis_hpsafe.cache import cache_directory, write_role_cache
from jarvis_hpsafe.constants import JARVIS_TASKS
from jarvis_hpsafe.io import sha256_file
from jarvis_hpsafe.manifest import JarvisManifest
from jarvis_hpsafe.protocol import (
    SCHEMA_VERSION,
    calibrate_validation,
    commit_barrier,
    load_config,
    predict_test,
    score_test,
    summarize,
)

PROJECT = Path(__file__).resolve().parents[1]
STAGE2_CONFIG = PROJECT / "configs/jarvis_stage2.yaml"
PROPOSAL_CONFIG = PROJECT / "configs/jarvis_proposal.yaml"


def _artifact(path: Path, ids: np.ndarray, y: np.ndarray, pred: np.ndarray, hidden: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    prediction_path, latent_path = path / "predictions.npz", path / "latents.npz"
    np.savez_compressed(prediction_path, sample_ids=ids, y_true=y, y_pred=pred)
    np.savez_compressed(
        latent_path,
        sample_ids=ids,
        latents=hidden.astype(np.float32),
        available=np.ones(len(ids), dtype=np.bool_),
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "predictions": {"filename": prediction_path.name, "sha256": sha256_file(prediction_path)},
                "latents": {"filename": latent_path.name, "sha256": sha256_file(latent_path)},
                "provenance": {"checkpoint_sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )


def _synthetic_proposal(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(29)
    stage1, cache, proposal_output = tmp_path / "stage1", tmp_path / "cache", tmp_path / "proposal"
    n = 40
    positions = {"train": np.arange(30), "val": np.arange(30, 35), "test": np.arange(35, 40)}
    manifests: dict[str, JarvisManifest] = {}
    targets: dict[str, np.ndarray] = {}
    for task_index, task in enumerate(JARVIS_TASKS):
        ids = np.asarray([f"{task}-{row}" for row in range(n)])
        material_ids = np.asarray([f"JVASP-{row}" for row in range(n)])
        y = np.linspace(-1.0, 1.0, n) + task_index * 0.2
        targets[task] = y
        manifest_path = stage1 / "data/manifests" / f"{task}.npz"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            manifest_path,
            sample_ids=ids,
            material_ids=material_ids,
            targets=y,
            official_train_positions=positions["train"],
            official_val_positions=positions["val"],
            official_test_positions=positions["test"],
            metadata_json=np.asarray(
                [
                    json.dumps(
                        {
                            "definition": {"task_name": task},
                            "n_outer_folds": 1,
                            "split_protocol": "jarvis_leaderboard_holdout",
                        }
                    )
                ]
            ),
        )
        manifests[task] = JarvisManifest.load(manifest_path, load_targets=False)
        base = stage1 / "outputs/exact/experts/kgcnn_cogn" / task / "outer_0"
        for role, directory in (("train", "inner_oof"), ("val", "official_validation"), ("test", "outer_test")):
            selected = positions[role]
            error = 0.15 if role == "train" else 0.03
            _artifact(
                base / directory,
                ids[selected],
                y[selected],
                y[selected] + error,
                rng.normal(size=(len(selected), 128)),
            )
    for source_index, source in enumerate(JARVIS_TASKS):
        for target in JARVIS_TASKS:
            manifest = manifests[target]
            y = targets[target]
            for role in ("train", "val", "test"):
                selected = positions[role]
                error = 0.15 if role == "train" else 0.03
                write_role_cache(
                    cache_directory(cache, source, target, role),
                    source_task=source,
                    target_manifest=manifest,
                    role=role,
                    sample_ids=manifest.ids(role),
                    hidden=rng.normal(size=(len(selected), 128)),
                    prediction=y[selected] + error + 0.01 * source_index,
                    available=np.ones(len(selected), dtype=np.bool_),
                    checkpoint_sha256="a" * 64,
                    producer={"synthetic": True, "labels_accessed": False},
                )
    proposal_config = yaml.safe_load(PROPOSAL_CONFIG.read_text(encoding="utf-8"))
    proposal_config.update(
        {
            "token_dim": 8,
            "task_dim": 4,
            "residual_hidden": 12,
            "residual_rank": 3,
            "dropout": 0.0,
            "proposal_training_steps": 1,
            "progress_every": 1,
            "regression_batch_size": 8,
            "benefit_gate_steps": 2,
            "benefit_gate_max_rows": 100,
            "bootstrap_resamples": 8,
            "bootstrap_max_rows": 64,
            "minimum_free_disk_gb": 0.0,
        }
    )
    config_path = tmp_path / "source_proposal_config.yaml"
    config_path.write_text(yaml.safe_dump(proposal_config, sort_keys=False), encoding="utf-8")
    core.freeze_sources(
        stage1_root=stage1,
        cache_root=cache,
        output_root=proposal_output,
        config_path=config_path,
    )
    policy = proposal_output / "policy/frozen_stage1_sources.json"
    core.train_calibrate_candidate(
        source_policy_path=policy,
        output_root=proposal_output,
        outer_fold=0,
        candidate="proposal_full",
        device=torch.device("cpu"),
    )
    core.lock_profile_for_outer_fold(source_policy_path=policy, output_root=proposal_output, outer_fold=0)
    core.predict_candidate(
        source_policy_path=policy,
        profile_lock_path=proposal_output / "policy/profile_locks/outer_0.json",
        output_root=proposal_output,
        outer_fold=0,
        candidate="proposal_full",
        device=torch.device("cpu"),
    )
    return proposal_output, stage1


def test_hpsafemoe_is_uniform_validation_dual_certificate() -> None:
    config = load_config(STAGE2_CONFIG)
    assert config["stage1_frozen"] is True
    assert config["proposal_checkpoints_frozen"] is True
    assert config["validation_meta_crossfit_partitions"] == 5
    assert config["calibration_data_roles"] == ["train_oof", "val"]
    assert config["scoring_data_role"] == "test"
    assert config["prediction_barrier_before_test_scoring"] is True


def test_hpsafemoe_end_to_end_locks_before_test(tmp_path: Path) -> None:
    proposal_root, _ = _synthetic_proposal(tmp_path)
    output = tmp_path / "final"
    receipt = calibrate_validation(
        proposal_run_root=proposal_root,
        output_root=output,
        config_path=STAGE2_CONFIG,
        device=torch.device("cpu"),
    )
    assert receipt["proposal_model_count"] == 10
    lock_path = output / "policy/deployment_validation_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert set(lock["task_calibrations"]) == set(JARVIS_TASKS)
    assert lock["test_labels_accessed"] is False
    predict_test(proposal_run_root=proposal_root, output_root=output, lock_path=lock_path)
    barrier = commit_barrier(proposal_run_root=proposal_root, output_root=output, lock_path=lock_path)
    assert barrier["prediction_file_count"] == 5
    score_test(
        proposal_run_root=proposal_root,
        output_root=output,
        barrier_path=output / "policy/prediction_barrier.json",
    )
    summary = summarize(output_root=output)
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["calibration_data_roles"] == ["train_oof", "official_validation"]

#!/usr/bin/env python3
"""Train the Matbench No-cross-expert control with the shared OOF safety profiles.

Source hashes and evaluation records are documented in results/matbench/experiment/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hpsafe_sota.metrics import score_predictions  # noqa: E402
from hpsafe_sota.stage2.experiment import (  # noqa: E402
    SCHEMA_VERSION,
    _canonical_digest,
    atomic_json,
    freeze_sources,
    normalized_gain,
    predict_candidate,
    train_calibrate_candidate,
    validate_profile_lock,
    validate_source_policy,
)
from hpsafe_sota.stage2_common.protocol import CANONICAL_TASKS  # noqa: E402

CANDIDATE = "ablation_no_cross_expert"
VARIANT = CANDIDATE + "__oof_locked"
CLASSIFICATION = {"glass", "expt_is_metal", "mp_is_metal"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bind_profile(record: Path, policy: Path, outer: int, destination: Path) -> dict:
    """Bind released Full OOF evidence to a fresh, validated input policy."""
    lock = read(record)
    if lock.get("selection_evidence_scope") != "matching_official_outer_train_true_inner_oof_only":
        raise ValueError("Profile evidence must come from matching outer-training OOF predictions")
    lock.update(schema_version=SCHEMA_VERSION, source_policy=str(policy.resolve()),
                profile_selection_data_scope="outer_train_oof",
                profile_evidence_record=str(record.resolve()),
                profile_evidence_record_sha256=sha256(record))
    lock["selection_digest"] = _canonical_digest(lock)
    if destination.exists() and read(destination) != lock:
        raise RuntimeError("Refusing to replace a different profile lock")
    atomic_json(destination, lock)
    return validate_profile_lock(destination, source_policy_path=policy, outer_fold=outer)


def collect_predictions(output: Path, policy: Path) -> dict:
    """Validate all prediction files without accessing manifest targets."""
    source = validate_source_policy(policy)
    manifests = Path(source["project_root"]) / "data/manifests"
    files, dependencies, profiles = {}, {}, {}
    for outer in range(5):
        lock_path = output / f"policy/profile_locks/outer_{outer}.json"
        lock = validate_profile_lock(lock_path, source_policy_path=policy, outer_fold=outer)
        profiles[str(outer)] = lock["selected_safety_profile"]
        run = output / f"proposal_training/outer_{outer}/{CANDIDATE}"
        for name in ("calibration_receipt.json", "prediction_receipt.json"):
            receipt_path = run / name
            receipt = read(receipt_path)
            if name == "calibration_receipt.json":
                if (receipt.get("status") != "true_inner_oof_calibration_complete_before_outer_test_access"
                        or receipt.get("outer_test_labels_accessed") is not False):
                    raise ValueError("Calibration must precede outer-test label access")
            elif (receipt.get("profile_lock_digest") != lock["selection_digest"]
                  or receipt.get("proposal_model_count") != 10):
                raise ValueError("Prediction receipt does not match the locked ten-member ensemble")
            dependencies[str(receipt_path.relative_to(output))] = sha256(receipt_path)
        dependencies[str(lock_path.relative_to(output))] = sha256(lock_path)
        for task in CANONICAL_TASKS:
            path = output / f"outer_prediction/outer_{outer}/{VARIANT}/{task}.npz"
            with np.load(manifests / f"{task}.npz", allow_pickle=False) as manifest:
                expected = manifest["sample_ids"][manifest[f"outer_{outer}_test"]].astype(str)
            with np.load(path, allow_pickle=False) as prediction:
                if any(key in prediction for key in ("targets", "target", "y_true", "labels")):
                    raise ValueError("Prediction archives must be target-free")
                if not np.array_equal(prediction["sample_ids"].astype(str), expected):
                    raise ValueError(f"Prediction order differs: outer {outer}, {task}")
                if not np.isfinite(prediction["prediction"]).all():
                    raise ValueError("Non-finite prediction")
            files[str(path.relative_to(output))] = sha256(path)
    manifest_hashes = {task: sha256(manifests / f"{task}.npz") for task in CANONICAL_TASKS}
    return {"predictions": files, "dependencies": dependencies, "profiles": profiles,
            "input_manifests_sha256": manifest_hashes,
            "source_policy_sha256": sha256(policy)}


def commit(output: Path, policy: Path) -> dict:
    path = output / "policy/no_cross_prediction_commit.json"
    inventory = collect_predictions(output, policy)
    if path.exists():
        committed = read(path)
        if any(committed[key] != value for key, value in inventory.items()):
            raise ValueError("Committed predictions or dependencies have changed")
        return committed
    record = dict(inventory, status="all_65_predictions_committed_before_scoring",
                  committed_unix=time.time(), outer_test_labels_accessed=False)
    atomic_json(path, record)
    return record


def score(output: Path, policy: Path) -> dict:
    destination = output / "reports/no_cross_summary.json"
    if destination.exists():
        raise RuntimeError("This run has already been scored. Use offline metric verification for the release.")
    barrier = read(output / "policy/no_cross_prediction_commit.json")
    if barrier.get("status") != "all_65_predictions_committed_before_scoring":
        raise ValueError("A complete prediction commit is required before scoring")
    inventory = collect_predictions(output, policy)
    if any(barrier[key] != value for key, value in inventory.items()):
        raise ValueError("Committed predictions or dependencies have changed")
    source = validate_source_policy(policy)
    manifests = Path(source["project_root"]) / "data/manifests"
    marker = output / "policy/scoring_started.json"
    with marker.open("x") as handle:
        json.dump({"scoring_started_unix": time.time(), "prediction_commit_sha256":
                   sha256(output / "policy/no_cross_prediction_commit.json")}, handle)
    tasks = {}
    # Test targets are first read only after the complete inventory has been checked.
    for task in CANONICAL_TASKS:
        task_type = "classification" if task in CLASSIFICATION else "regression"
        rows = []
        with np.load(manifests / f"{task}.npz", allow_pickle=False) as manifest:
            for outer in range(5):
                target = manifest["targets"][manifest[f"outer_{outer}_test"]]
                path = output / f"outer_prediction/outer_{outer}/{VARIANT}/{task}.npz"
                with np.load(path, allow_pickle=False) as prediction:
                    anchor = score_predictions(task_type, target, prediction["anchor_prediction"])
                    value = score_predictions(task_type, target, prediction["prediction"])
                rows.append({"outer_fold": outer, "anchor_score": anchor, "model_score": value,
                             "normalized_gain_percent": normalized_gain(task_type, anchor, value)})
        tasks[task] = {"metric": "ROC-AUC" if task_type == "classification" else "MAE", "folds": rows,
                       "model_score_mean": float(np.mean([r["model_score"] for r in rows])),
                       "model_score_std": float(np.std([r["model_score"] for r in rows], ddof=0)),
                       "normalized_gain_percent": float(np.mean([r["normalized_gain_percent"] for r in rows]))}
    result = {"status": "complete", "variant": VARIANT, "tasks": tasks,
              "scored_unix": time.time(), "prediction_commit_sha256": sha256(output / "policy/no_cross_prediction_commit.json"),
              "selected_safety_profile_by_outer_fold": barrier["profiles"]}
    atomic_json(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "predict", "commit", "score"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/stage2.yaml")
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    policy = output / "policy/frozen_stage1_sources.json"
    locked = output / "policy/no_cross_prediction_commit.json"
    if args.action in {"prepare", "train", "predict"} and locked.exists():
        raise RuntimeError("This run is committed. Use a fresh output directory for another fit.")
    torch.set_num_threads(args.cpu_threads)
    if args.action == "prepare":
        if (output / "outer_prediction").exists():
            raise RuntimeError("Profiles must be bound before prediction")
        freeze_sources(project_root=ROOT, feature_root=args.feature_root, output_root=output, config_path=args.config)
        for outer in range(5):
            bind_profile(ROOT / f"results/matbench/experiment/profiles/outer_{outer}.json", policy, outer,
                         output / f"policy/profile_locks/outer_{outer}.json")
        result = {"status": "prepared", "profiles": ["balanced"] * 3 + ["conservative"] * 2}
    elif args.action in {"train", "predict"}:
        if args.outer_fold is None:
            parser.error("--outer-fold is required for train and predict")
        kwargs = dict(source_policy_path=policy, output_root=output, outer_fold=args.outer_fold,
                      candidate=CANDIDATE, device=torch.device(args.device))
        if args.action == "train":
            if (output / "outer_prediction").exists():
                raise RuntimeError("Complete all training before starting prediction")
            result = train_calibrate_candidate(**kwargs)
        else:
            result = predict_candidate(**kwargs, profile_lock_path=output / f"policy/profile_locks/outer_{args.outer_fold}.json")
    elif args.action == "commit":
        result = commit(output, policy)
    else:
        result = score(output, policy)
    print(json.dumps({"action": args.action, "status": result.get("status", "complete")}, indent=2))


if __name__ == "__main__":
    main()

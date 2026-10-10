from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from . import core as core
from .constants import JARVIS_TASKS, OFFICIAL_SPLIT
from .data import _role_targets
from .io import atomic_json, read_json, sha256_file
from .manifest import load_manifests

SCHEMA_VERSION = "jarvis-hpsafemoe"
SOURCE_CANDIDATE = "proposal_full"


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": SCHEMA_VERSION,
        "source_candidate": SOURCE_CANDIDATE,
        "stage1_frozen": True,
        "proposal_checkpoints_frozen": True,
        "use_crossfit_benefit_gate": True,
        "validation_role": "val",
        "validation_meta_crossfit_partitions": 5,
        "require_training_oof_certificate": True,
        "require_validation_crossfit_certificate": True,
        "use_same_oof_locked_global_profile": True,
        "calibration_data_roles": ["train_oof", "val"],
        "scoring_data_role": "test",
        "prediction_barrier_before_test_scoring": True,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"JARVIS HP-SafeMoE protocol drifted: {key}")
    scales = list(map(float, value["residual_scale_grid"]))
    thresholds = list(map(float, value["benefit_probability_threshold_grid"]))
    if scales != sorted(set(scales)) or scales[0] != 0.0 or scales[-1] != 1.0:
        raise ValueError("HP-SafeMoE residual scale grid must span [0, 1]")
    if thresholds != sorted(set(thresholds)) or min(thresholds) <= 0 or max(thresholds) >= 1:
        raise ValueError("HP-SafeMoE probability thresholds must lie inside (0, 1)")
    return value


def _paths(proposal_run_root: Path) -> tuple[Path, Path]:
    root = proposal_run_root.expanduser().resolve(strict=True)
    return root / "policy/frozen_stage1_sources.json", root / "policy/profile_locks/outer_0.json"


def _source(proposal_run_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    policy_path, lock_path = _paths(proposal_run_root)
    policy = core.validate_source_policy(policy_path)
    lock = core.validate_profile_lock(
        lock_path,
        source_policy_path=policy_path,
        outer_fold=OFFICIAL_SPLIT,
    )
    if lock.get("profile_selection_data_scope") != "official_train_oof":
        raise ValueError("The source proposal profile requires official-train OOF evidence")
    return policy, lock, policy_path, lock_path


def _stats_payload(stats: dict[str, core.TaskStats]) -> dict[str, Any]:
    return {task: value.to_dict() for task, value in stats.items()}


def _validation_meta_calibration(
    *,
    values: dict[str, np.ndarray],
    y: np.ndarray,
    training_calibration: dict[str, Any],
    selected_profile: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    if training_calibration.get("fit_labels") != "jarvis_official_train_true_inner_oof_only":
        raise ValueError("Expected the training_calibration train-OOF benefit calibration")
    features = core._gate_features(values)
    probability = core._benefit_probability(training_calibration["benefit_gate"], features)
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    raw = np.asarray(values["raw_prediction"], dtype=np.float64)
    assignment = core._meta_partitions(
        y,
        "regression",
        int(config["validation_meta_crossfit_partitions"]),
        seed=seed,
    )
    meta_prediction = anchor.copy()
    meta_accepted = np.zeros(len(y), dtype=np.bool_)
    partitions: list[dict[str, Any]] = []
    for held in range(int(config["validation_meta_crossfit_partitions"])):
        tune, certify = assignment != held, assignment == held
        selected = core._select_parameters(
            y=y[tune],
            anchor=anchor[tune],
            raw=raw[tune],
            probability=probability[tune],
            task_type="regression",
            config=config,
            sample_gate=True,
        )
        prediction, accepted = core._apply_parameters(
            anchor=anchor[certify],
            raw=raw[certify],
            probability=probability[certify],
            alpha=float(selected["alpha"]),
            threshold=selected["threshold"],
        )
        meta_prediction[certify] = prediction
        meta_accepted[certify] = accepted
        gain = core.normalized_gain(
            "regression",
            core.score_predictions("regression", y[certify], anchor[certify]),
            core.score_predictions("regression", y[certify], prediction),
        )
        partitions.append(
            {
                **selected,
                "held_partition": held,
                "held_gain_percent": gain,
                "held_coverage": float(accepted.mean()),
            }
        )
    alpha = float(np.median([float(row["alpha"]) for row in partitions]))
    chosen_thresholds = [
        float(row["threshold"])
        for row in partitions
        if row["threshold"] is not None and float(row["alpha"]) > 0.0
    ]
    threshold = float(np.median(chosen_thresholds)) if chosen_thresholds else None
    gains = [float(row["held_gain_percent"]) for row in partitions]
    certificate = core._bootstrap_gain(
        task_type="regression",
        y=y,
        anchor=anchor,
        model=meta_prediction,
        config=config,
        seed=seed + 1777,
    )
    validation_profile_acceptance = core._profile_acceptance(
        alpha=alpha,
        certificate=certificate,
        partition_gains=gains,
    )
    oof_accepted = bool(training_calibration["profile_acceptance"][selected_profile])
    validation_accepted = bool(validation_profile_acceptance[selected_profile])
    accepted = bool(oof_accepted and validation_accepted)
    full_prediction, full_accepted = core._apply_parameters(
        anchor=anchor,
        raw=raw,
        probability=probability,
        alpha=alpha,
        threshold=threshold,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "deployment_aligned_validation_meta_crossfit_complete",
        "selected_global_profile": selected_profile,
        "alpha": alpha,
        "benefit_probability_threshold": threshold,
        "training_oof_certificate_accepted": oof_accepted,
        "validation_profile_acceptance": validation_profile_acceptance,
        "validation_certificate_accepted": validation_accepted,
        "task_deployment_accepted": accepted,
        "validation_meta_crossfit_partition_records": partitions,
        "validation_meta_crossfit_partition_gains_percent": gains,
        "validation_meta_crossfit_positive_partitions": int(sum(value > 0.0 for value in gains)),
        "validation_meta_crossfit_coverage": float(meta_accepted.mean()),
        "validation_certificate": certificate,
        "validation_full_parameter_gain_percent_diagnostic_only": core.normalized_gain(
            "regression",
            core.score_predictions("regression", y, anchor),
            core.score_predictions("regression", y, full_prediction),
        ),
        "validation_full_parameter_coverage_diagnostic_only": float(full_accepted.mean()),
        "benefit_gate_source": "training_oof_benefit_gate",
        "fit_labels": "jarvis_official_validation_only",
        "test_labels_accessed": False,
    }


def calibrate_validation(
    *,
    proposal_run_root: Path,
    output_root: Path,
    config_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    receipt_path = output / "validation/calibration_receipt.json"
    lock_path = output / "policy/deployment_validation_lock.json"
    if receipt_path.is_file() and lock_path.is_file():
        lock = validate_lock(lock_path, proposal_run_root=proposal_run_root)
        receipt = read_json(receipt_path)
        if receipt.get("deployment_lock_digest") != lock["selection_digest"]:
            raise ValueError("Completed HP-SafeMoE calibration receipt/lock mismatch")
        return receipt
    if any((output / "test_prediction").glob("*.npz")):
        raise RuntimeError("Validation safety must be locked before HP-SafeMoE test prediction")
    config = load_config(config_path)
    policy, source_lock, policy_path, source_lock_path = _source(proposal_run_root)
    selected_profile = str(source_lock["selected_safety_profile"])
    proposal_root = proposal_run_root.expanduser().resolve(strict=True)
    proposal_training = proposal_root / "proposal_training/outer_0/proposal_full"
    source_receipt = read_json(proposal_training / "calibration_receipt.json")
    if source_receipt.get("outer_test_labels_accessed") is not False:
        raise ValueError("Source proposal calibration is not target-free")
    train_fold = core._fold(policy, OFFICIAL_SPLIT, ("train",))
    inner_folds = int(policy["config_payload"]["inner_oof_folds"])
    seeds = list(map(int, policy["config_payload"]["model_seeds"]))
    assignments = core._inner_partitions(
        train_fold,
        inner_folds,
        seed=int(policy["config_payload"]["inner_partition_seed"]),
    )
    plans = []
    for inner in range(inner_folds):
        eligible = core._partition_indices(train_fold, assignments, inner, training=True)
        plans.append(core._masked_stats(train_fold, eligible))
    records: dict[str, list[dict[str, np.ndarray]]] = {task: [] for task in JARVIS_TASKS}
    y_true: dict[str, np.ndarray] = {}
    total_models = inner_folds * len(seeds)
    completed = 0
    for inner, stats in enumerate(plans):
        validation_fold = core._fold(
            policy,
            OFFICIAL_SPLIT,
            ("val",),
            frozen_stats=_stats_payload(stats),
            label_roles=("val",),
        )
        for task in JARVIS_TASKS:
            current = np.asarray(validation_fold.roles[task]["val"].y_true, dtype=np.float64)
            if task in y_true and not np.array_equal(y_true[task], current):
                raise ValueError("Validation labels changed across normalization folds")
            y_true[task] = current
        for seed in seeds:
            checkpoint = proposal_training / f"checkpoints/inner_{inner}/seed_{seed}.pt"
            model = core._load_proposal(
                checkpoint=checkpoint,
                fold=validation_fold,
                candidate=SOURCE_CANDIDATE,
                config=policy["config_payload"],
                device=device,
            )
            for task in JARVIS_TASKS:
                records[task].append(
                    core._predict(
                        fold=validation_fold,
                        model=model,
                        task=task,
                        role="val",
                        device=device,
                    )
                )
            completed += 1
            atomic_json(
                output / "validation/progress.json",
                {
                    "status": "running",
                    "stage": "official_validation_crossfit_proposal",
                    "completed_models": completed,
                    "total_models": total_models,
                    "percent": 100.0 * completed / total_models,
                    "test_labels_accessed": False,
                },
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    calibrations: dict[str, dict[str, Any]] = {}
    for task_number, task in enumerate(JARVIS_TASKS):
        values = core._ensemble_records(records[task])
        training_calibration = read_json(proposal_training / f"calibration/{task}.json")
        calibrations[task] = _validation_meta_calibration(
            values=values,
            y=y_true[task],
            training_calibration=training_calibration,
            selected_profile=selected_profile,
            config=config,
            seed=int(config["validation_partition_seed"]) + task_number * 1009,
        )
        atomic_json(output / f"validation/tasks/{task}.json", calibrations[task])
        core._atomic_npz(
            output / f"validation/proposals/{task}.npz",
            {
                "sample_ids": values["sample_ids"],
                "anchor_prediction": values["anchor_prediction"],
                "raw_prediction": values["raw_prediction"],
                "benefit_probability": core._benefit_probability(
                    training_calibration["benefit_gate"], core._gate_features(values)
                ),
            },
        )
    lock = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked_from_train_oof_and_deployment_aligned_official_validation",
        "selected_global_profile": selected_profile,
        "task_order": list(JARVIS_TASKS),
        "task_calibrations": calibrations,
        "accepted_tasks_count": int(
            sum(value["task_deployment_accepted"] for value in calibrations.values())
        ),
        "selection_scope": "uniform_dual_certificate_rule_applied_independently_to_all_tasks",
        "proposal_policy": str(policy_path),
        "proposal_profile_lock": str(source_lock_path),
        "proposal_profile_lock_digest": source_lock["selection_digest"],
        "config": str(config_path.expanduser().resolve(strict=True)),
        "config_sha256": sha256_file(config_path.expanduser().resolve(strict=True)),
        "test_features_accessed_by_calibration": False,
        "test_labels_accessed": False,
        "locked_before_test_prediction": True,
    }
    lock["selection_digest"] = core._canonical_digest(lock)
    atomic_json(lock_path, lock)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "validation_calibration_complete_and_locked_before_test_prediction",
        "proposal_model_count": total_models,
        "task_calibration_files": [str(output / f"validation/tasks/{task}.json") for task in JARVIS_TASKS],
        "deployment_lock": str(lock_path),
        "deployment_lock_digest": lock["selection_digest"],
        "test_labels_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def validate_lock(path: Path, *, proposal_run_root: Path) -> dict[str, Any]:
    lock = read_json(path.expanduser().resolve(strict=True))
    policy, source_lock, policy_path, source_lock_path = _source(proposal_run_root)
    del policy
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "locked_from_train_oof_and_deployment_aligned_official_validation",
        "task_order": list(JARVIS_TASKS),
        "proposal_policy": str(policy_path),
        "proposal_profile_lock": str(source_lock_path),
        "proposal_profile_lock_digest": source_lock["selection_digest"],
        "test_features_accessed_by_calibration": False,
        "test_labels_accessed": False,
        "locked_before_test_prediction": True,
    }
    for key, expected in required.items():
        if lock.get(key) != expected:
            raise ValueError(f"Malformed HP-SafeMoE validation lock: {key}")
    unsigned = dict(lock)
    supplied = unsigned.pop("selection_digest", None)
    if supplied != core._canonical_digest(unsigned):
        raise ValueError("HP-SafeMoE validation lock digest mismatch")
    if set(lock.get("task_calibrations", {})) != set(JARVIS_TASKS):
        raise ValueError("HP-SafeMoE validation lock task set drifted")
    return lock


def predict_test(
    *,
    proposal_run_root: Path,
    output_root: Path,
    lock_path: Path,
) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    receipt_path = output / "test_prediction/prediction_receipt.json"
    if receipt_path.is_file():
        lock = validate_lock(lock_path, proposal_run_root=proposal_run_root)
        receipt = read_json(receipt_path)
        if receipt.get("validation_lock_digest") != lock["selection_digest"]:
            raise ValueError("Completed HP-SafeMoE prediction receipt/lock mismatch")
        if any(not Path(path).is_file() for path in receipt.get("prediction_files", [])):
            raise FileNotFoundError("Completed HP-SafeMoE prediction receipt has missing files")
        return receipt
    if (output / "policy/prediction_barrier.json").exists():
        raise RuntimeError("Cannot write predictions after the HP-SafeMoE barrier")
    lock = validate_lock(lock_path, proposal_run_root=proposal_run_root)
    policy, _, _, _ = _source(proposal_run_root)
    manifests = load_manifests(Path(policy["stage1_root"]), load_targets=False)
    files = []
    for task in JARVIS_TASKS:
        source_path = (
            proposal_run_root.expanduser().resolve(strict=True)
            / f"outer_prediction/outer_0/hpsafemoe__oof_locked/{task}.npz"
        )
        with np.load(source_path, allow_pickle=False) as loaded:
            if "y_true" in loaded.files:
                raise ValueError("Source target-free proposal proposal unexpectedly stores labels")
            sample_ids = np.asarray(loaded["sample_ids"]).astype(str)
            anchor = np.asarray(loaded["anchor_prediction"], dtype=np.float64)
            raw = np.asarray(loaded["raw_prediction"], dtype=np.float64)
            probability = np.asarray(loaded["benefit_probability"], dtype=np.float64)
        if not np.array_equal(sample_ids, manifests[task].ids("test")):
            raise ValueError(f"proposal source proposal order drift: {task}")
        calibration = lock["task_calibrations"][task]
        prediction, accepted = core._apply_parameters(
            anchor=anchor,
            raw=raw,
            probability=probability,
            alpha=float(calibration["alpha"]),
            threshold=calibration["benefit_probability_threshold"],
        )
        if not calibration["task_deployment_accepted"]:
            prediction = anchor.copy()
            accepted[:] = False
        destination = output / f"test_prediction/{task}.npz"
        core._atomic_npz(
            destination,
            {
                "sample_ids": sample_ids,
                "anchor_prediction": anchor,
                "raw_prediction": raw,
                "prediction": prediction,
                "benefit_probability": probability,
                "accepted": accepted,
                "residual_scale": np.full(len(anchor), float(calibration["alpha"]), dtype=np.float32),
                "task_deployment_accepted": np.full(
                    len(anchor), bool(calibration["task_deployment_accepted"]), dtype=np.bool_
                ),
            },
        )
        files.append(str(destination))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "five_target_free_test_predictions_complete_after_validation_lock",
        "proposal_source": "hpsafemoe__oof_locked/raw_prediction_only",
        "validation_lock": str(lock_path.expanduser().resolve(strict=True)),
        "validation_lock_digest": lock["selection_digest"],
        "prediction_files": files,
        "test_labels_accessed": False,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def commit_barrier(*, proposal_run_root: Path, output_root: Path, lock_path: Path) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    lock = validate_lock(lock_path, proposal_run_root=proposal_run_root)
    receipt = read_json(output / "test_prediction/prediction_receipt.json")
    if receipt.get("validation_lock_digest") != lock["selection_digest"]:
        raise ValueError("HP-SafeMoE prediction receipt/lock mismatch")
    files = []
    for task in JARVIS_TASKS:
        path = output / f"test_prediction/{task}.npz"
        with np.load(path, allow_pickle=False) as loaded:
            if "y_true" in loaded.files:
                raise ValueError("HP-SafeMoE prediction barrier found test labels")
        files.append({"path": str(path), "sha256": sha256_file(path)})
    barrier = {
        "schema_version": SCHEMA_VERSION,
        "status": "all_five_predictions_committed_before_scoring",
        "validation_lock": str(lock_path.expanduser().resolve(strict=True)),
        "validation_lock_digest": lock["selection_digest"],
        "prediction_file_count": len(files),
        "prediction_files": files,
        "test_labels_accessed": False,
    }
    atomic_json(output / "policy/prediction_barrier.json", barrier)
    return barrier


def score_test(*, proposal_run_root: Path, output_root: Path, barrier_path: Path) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    barrier = read_json(barrier_path.expanduser().resolve(strict=True))
    if (
        barrier.get("status") != "all_five_predictions_committed_before_scoring"
        or barrier.get("prediction_file_count") != 5
        or barrier.get("test_labels_accessed") is not False
    ):
        raise ValueError("HP-SafeMoE prediction barrier is incomplete")
    for record in barrier.get("prediction_files", []):
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"HP-SafeMoE prediction changed after barrier: {path}")
    lock = validate_lock(Path(barrier["validation_lock"]), proposal_run_root=proposal_run_root)
    if lock["selection_digest"] != barrier["validation_lock_digest"]:
        raise ValueError("HP-SafeMoE barrier/lock digest mismatch")
    policy, _, _, _ = _source(proposal_run_root)
    stage1 = Path(policy["stage1_root"])
    manifests = load_manifests(stage1, load_targets=False)
    results: dict[str, Any] = {}
    for task in JARVIS_TASKS:
        expected = manifests[task].ids("test")
        y = _role_targets(stage1, manifests[task], "test")
        with np.load(output / f"test_prediction/{task}.npz", allow_pickle=False) as loaded:
            ids = np.asarray(loaded["sample_ids"]).astype(str)
            anchor = np.asarray(loaded["anchor_prediction"], dtype=np.float64)
            prediction = np.asarray(loaded["prediction"], dtype=np.float64)
        with np.load(
            proposal_run_root.expanduser().resolve(strict=True)
            / f"outer_prediction/outer_0/hpsafemoe__oof_locked/{task}.npz",
            allow_pickle=False,
        ) as loaded:
            proposal = np.asarray(loaded["prediction"], dtype=np.float64)
        if not np.array_equal(ids, expected):
            raise ValueError(f"HP-SafeMoE test IDs drifted: {task}")
        anchor_mae = core.score_predictions("regression", y, anchor)
        proposal_mae = core.score_predictions("regression", y, proposal)
        final_mae = core.score_predictions("regression", y, prediction)
        results[task] = {
            "n_samples": len(y),
            "anchor_mae": anchor_mae,
            "proposal_mae": proposal_mae,
            "final_mae": final_mae,
            "proposal_gain_percent": core.normalized_gain("regression", anchor_mae, proposal_mae),
            "hpsafemoe_gain_percent": core.normalized_gain("regression", anchor_mae, final_mae),
            "task_deployment_accepted": bool(lock["task_calibrations"][task]["task_deployment_accepted"]),
            "alpha": float(lock["task_calibrations"][task]["alpha"]),
            "threshold": lock["task_calibrations"][task]["benefit_probability_threshold"],
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "test_scored_once_after_validation_lock_and_prediction_barrier",
        "scoring_data_role": "test",
        "selection_stage": "training_oof_and_official_validation",
        "results": results,
    }
    atomic_json(output / "results/test_results.json", receipt)
    return receipt


def summarize(*, output_root: Path) -> dict[str, Any]:
    output = output_root.expanduser().resolve()
    score = read_json(output / "results/test_results.json")
    lock = read_json(output / "policy/deployment_validation_lock.json")
    if score.get("selection_stage") != "training_oof_and_official_validation":
        raise ValueError("HP-SafeMoE scoring receipt has an unexpected selection stage")
    rows = score["results"]
    proposal_gains = [float(rows[task]["proposal_gain_percent"]) for task in JARVIS_TASKS]
    hpsafemoe_gains = [float(rows[task]["hpsafemoe_gain_percent"]) for task in JARVIS_TASKS]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "method_name": "HP-SafeMoE for JARVIS",
        "selected_global_profile": lock["selected_global_profile"],
        "accepted_tasks_count": lock["accepted_tasks_count"],
        "proposal_macro_gain_percent": float(np.mean(proposal_gains)),
        "hpsafemoe_macro_gain_percent": float(np.mean(hpsafemoe_gains)),
        "hpsafemoe_positive_neutral_negative": [
            int(sum(value > 1.0e-9 for value in hpsafemoe_gains)),
            int(sum(abs(value) <= 1.0e-9 for value in hpsafemoe_gains)),
            int(sum(value < -1.0e-9 for value in hpsafemoe_gains)),
        ],
        "tasks": rows,
        "calibration_data_roles": ["train_oof", "official_validation"],
    }
    atomic_json(output / "results/JARVIS_RESULTS.json", summary)
    lines = [
        "# JARVIS HP-SafeMoE results",
        "",
        "HP-SafeMoE combines frozen Stage-1 experts with a train-OOF benefit gate and "
        "five-part official-validation cross-fitting. The locked policy then generates the official test predictions.",
        "",
        f"- Selected OOF global profile: `{summary['selected_global_profile']}`",
        f"- Tasks accepted by the validation dual-certificate: {summary['accepted_tasks_count']}/5",
        f"- Stage-2 proposal macro gain: {summary['proposal_macro_gain_percent']:+.3f}%",
        f"- HP-SafeMoE macro gain: {summary['hpsafemoe_macro_gain_percent']:+.3f}%",
        "- HP-SafeMoE positive/neutral/negative tasks: "
        f"{'/'.join(map(str, summary['hpsafemoe_positive_neutral_negative']))}",
        "",
        "| Task | coGN Anchor MAE | Stage-2 proposal MAE | HP-SafeMoE MAE | HP-SafeMoE gain | val-certified | alpha | threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in JARVIS_TASKS:
        row = rows[task]
        threshold = "—" if row["threshold"] is None else f"{float(row['threshold']):.2f}"
        lines.append(
            f"| {task} | {row['anchor_mae']:.8g} | {row['proposal_mae']:.8g} | "
            f"{row['final_mae']:.8g} | {row['hpsafemoe_gain_percent']:+.3f}% | "
            f"{str(row['task_deployment_accepted'])} | {row['alpha']:.2f} | {threshold} |"
        )
    (output / "results/JARVIS_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from hpsafe_sota.metrics import score_predictions
from hpsafe_sota.stage2.model import (
    CrossExpertProposalModel,
    Stage2ProposalConfig,
)
from hpsafe_sota.stage2_common.cache import (
    ARRAY_FILES,
    ExpertExport,
    TaskRoleCache,
    export_directory,
)
from hpsafe_sota.stage2_common.data import FoldInputs, ScalarStats, TaskStats
from hpsafe_sota.stage2_common.io import atomic_json, read_json
from hpsafe_sota.stage2_common.protocol import CANONICAL_TASKS, ExpertPool, load_expert_pool

SCHEMA_VERSION = "hpsafemoe"
OUTER_FOLDS = (0, 1, 2, 3, 4)
PRIMARY_PROPOSAL = "proposal_full"

# Every entry is a proposal architecture. The downstream safety controller is
# learned from held-out inner-OOF predictions.
ARCHITECTURE_CANDIDATES: dict[str, dict[str, Any]] = {
    "proposal_full": {
        "role": "main_learned_data_relation_task_private_proposal",
        "use_cross_expert_input": True,
        "use_data_branch": True,
        "use_relation_branch": True,
        "use_learned_task_source_relation": True,
        "use_shared_low_rank_basis": False,
        "use_task_private_correction": True,
    },
    "ablation_no_cross_expert": {
        "role": "cross_expert_input_ablation",
        "use_cross_expert_input": False,
        "use_data_branch": True,
        "use_relation_branch": True,
        "use_learned_task_source_relation": False,
        "use_shared_low_rank_basis": False,
        "use_task_private_correction": True,
    },
}

# These global, task-agnostic certification rules act on each task's OOF evidence.
SAFETY_PROFILES: dict[str, dict[str, float | int]] = {
    "conservative": {
        "minimum_oof_gain_percent": 0.25,
        "bootstrap_lower_floor_percent": 0.00,
        "required_positive_partitions": 4,
        "maximum_partition_regret_percent": 0.50,
    },
    "balanced": {
        "minimum_oof_gain_percent": 0.10,
        "bootstrap_lower_floor_percent": -0.25,
        "required_positive_partitions": 3,
        "maximum_partition_regret_percent": 1.00,
    },
    "liberal": {
        "minimum_oof_gain_percent": 0.00,
        "bootstrap_lower_floor_percent": -1.00,
        "required_positive_partitions": 2,
        "maximum_partition_regret_percent": 3.00,
    },
}

REPORT_NAMES = {
    "proposal_full": "Full learned Data+Relation proposal",
    "ablation_no_cross_expert": "No-cross-expert",
}
PROFILE_NAMES = {
    "conservative": "Conservative OOF safety",
    "balanced": "Balanced OOF safety",
    "liberal": "Liberal OOF safety",
}


def _candidate_variant(candidate: str) -> str:
    prefix = "hpsafemoe" if candidate == PRIMARY_PROPOSAL else candidate
    return f"{prefix}__oof_locked"


SELECTABLE_PROFILES = tuple(SAFETY_PROFILES)
ARCHITECTURE_STORAGE_VARIANTS = tuple(_candidate_variant(candidate) for candidate in ARCHITECTURE_CANDIDATES)
BASELINE_VARIANTS = ("stage1_specialist_only", "ablation_no_safety")
ALL_STORAGE_VARIANTS = tuple(
    dict.fromkeys(ARCHITECTURE_STORAGE_VARIANTS + BASELINE_VARIANTS)
)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or str(value.get("schema_version")) != SCHEMA_VERSION:
        raise ValueError("Expected a Stage2 HP-SafeMoE configuration")
    required = {
        "task_order": list(CANONICAL_TASKS),
        "outer_folds": list(OUTER_FOLDS),
        "architecture_candidate_order": list(ARCHITECTURE_CANDIDATES),
        "architecture_candidate_specs": ARCHITECTURE_CANDIDATES,
        "safety_profiles": SAFETY_PROFILES,
        "primary_proposal": PRIMARY_PROPOSAL,
        "stage1_frozen": True,
        "joint_task_count": 13,
        "learned_task_source_relation": True,
        "task_private_residual_output": True,
        "evaluation_protocol": "official_matbench_five_fold",
        "same_outer_fold_stage1_stage2": True,
        "inner_oof_folds": 5,
        "true_inner_oof_proposals": True,
        "crossfit_ensemble_deployment": True,
        "proposal_fit_scope": "matching_outer_train_inner_crossfit",
        "stage2_artifact_source": "current_run",
        "task_routing_policy": "learned_shared_rules",
        "training_data_scope": "outer_train",
        "calibration_data_scope": "outer_train_oof",
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_lock_before_outer_test_prediction": True,
        "profile_selection_policy": "fold_local_global_profile",
        "all_outer_predictions_committed_before_scoring": True,
        "input_validation": "schema_alignment_and_manifest",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"HP-SafeMoE protocol drifted: {key}")
    seeds = list(map(int, value.get("model_seeds", [])))
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("HP-SafeMoE requires at least two fixed proposal seeds")
    if int(value["meta_crossfit_partitions"]) != 5:
        raise ValueError("HP-SafeMoE requires five meta-crossfit partitions")
    if float(value.get("oof_profile_selection_clear_gain_percent", -1.0)) < 0.0:
        raise ValueError("OOF clear-gain threshold must be non-negative")
    if float(value.get("oof_profile_selection_max_worst_loss_percent", -1.0)) < 0.0:
        raise ValueError("OOF worst-loss limit must be non-negative")
    tie_order = list(map(str, value.get("oof_profile_tie_break_order", [])))
    if set(tie_order) != set(SAFETY_PROFILES) or len(tie_order) != len(SAFETY_PROFILES):
        raise ValueError("oof_profile_tie_break_order must list every safety profile once")
    scales = list(map(float, value["residual_scale_grid"]))
    if scales != sorted(set(scales)) or not scales or scales[0] != 0.0 or scales[-1] != 1.0:
        raise ValueError("residual_scale_grid must be sorted, unique, and span [0, 1]")
    thresholds = list(map(float, value["benefit_probability_threshold_grid"]))
    if thresholds != sorted(set(thresholds)) or not thresholds:
        raise ValueError("benefit_probability_threshold_grid must be sorted and non-empty")
    if min(thresholds) <= 0.0 or max(thresholds) >= 1.0:
        raise ValueError("Benefit thresholds must lie strictly inside (0, 1)")
    return value


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def normalized_gain(task_type: str, anchor: float, model: float) -> float:
    if task_type == "classification":
        return 100.0 * (model - anchor) / max(1.0 - anchor, 1.0e-12)
    return 100.0 * (anchor - model) / max(abs(anchor), 1.0e-12)


def _source_paths(
    project_root: Path, feature_root: Path, config: dict[str, Any]
) -> tuple[Path, Path]:
    return (
        (project_root / str(config["pool_config_relative_path"])).resolve(strict=True),
        (feature_root / str(config["input_validation_relative_path"])).resolve(strict=True),
    )


def freeze_sources(
    *, project_root: Path, feature_root: Path, output_root: Path, config_path: Path
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve(strict=True)
    feature_root = feature_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve(strict=True)
    config = load_config(config_path)
    pool_path, validation_path = _source_paths(project_root, feature_root, config)
    pool = load_expert_pool(pool_path)
    input_validation = read_json(validation_path)
    if (
        input_validation.get("status") != "pass"
        or input_validation.get("target_values_accessed_by_validator") is not False
        or tuple(input_validation.get("task_order", ())) != CANONICAL_TASKS
    ):
        raise ValueError("Stage-1 input validation does not satisfy the release label-free contract")
    if tuple(spec.source_task for spec in pool.experts) != CANONICAL_TASKS:
        raise ValueError("Frozen provider pool is not ordered as the 13 canonical tasks")
    providers = Counter(spec.provider for spec in pool.experts)
    if providers.get("jmp_l_official_checkpoint_consistent") != 3:
        raise ValueError("HP-SafeMoE requires the three official JMP-L experts")
    if providers.get("jmp_l_frozen_outer_checkpoint_replay") != 2:
        raise ValueError("HP-SafeMoE requires the two Fast-32 JMP-L experts")
    if pool.anchor_for("steels") != "tpot_mat_steels_anchor":
        raise ValueError("HP-SafeMoE requires TPOT-Mat as the steels specialist")
    available_export_count = 0
    for task in CANONICAL_TASKS:
        anchor = next(spec for spec in pool.experts if spec.expert_id == pool.anchor_for(task))
        for spec in pool.experts:
            for outer in OUTER_FOLDS:
                for role in ("train", "val", "test"):
                    metadata = spec.export_root / task / f"outer_{outer}" / role / "metadata.json"
                    available_export_count += int(metadata.is_file())
        for outer in OUTER_FOLDS:
            for role in ("train", "val", "test"):
                required = anchor.export_root / task / f"outer_{outer}" / role / "metadata.json"
                if not required.is_file():
                    raise FileNotFoundError(required)
    policy = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_heterogeneous_stage1_foundation",
        "project_root": str(project_root),
        "feature_root": str(feature_root),
        "pool_path": str(pool_path),
        "input_validation_path": str(validation_path),
        "input_validation_schema_version": input_validation.get("schema_version"),
        "task_source_relation_initialization": "learned_from_neutral_zero_matrix",
        "config_path": str(config_path),
        "config_payload": config,
        "task_order": list(CANONICAL_TASKS),
        "expert_order": list(pool.expert_ids),
        "anchor_map": dict(pool.anchor_map),
        "provider_counts": dict(providers),
        "available_export_count": available_export_count,
        "outer_fold_checkpoint_mapping": {str(k): str(k) for k in OUTER_FOLDS},
        "stage1_stage2_outer_fold_alignment": "identity",
        "stage2_fit_scope": "matching_official_outer_train_inner_crossfit_ensemble",
        "stage2_prediction_scope": "matching_official_outer_test",
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_selection_data_scope": "outer_train_oof",
        "profile_lock_before_outer_test_prediction": True,
        "inner_oof_folds": int(config["inner_oof_folds"]),
        "deployment_model": "inner_crossfit_ensemble",
        "scoring_label_access": "after_prediction_barrier",
        "stage1_frozen": True,
        "stage2_artifact_source": "current_run",
        "input_validation": "schema_alignment_and_manifest",
    }
    path = output_root / "policy/frozen_stage1_sources.json"
    if path.is_file() and read_json(path) != policy:
        raise RuntimeError(f"Refusing to replace a different source policy: {path}")
    atomic_json(path, policy)
    return policy


def validate_source_policy(path: Path) -> dict[str, Any]:
    value = read_json(path.expanduser().resolve(strict=True))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_heterogeneous_stage1_foundation",
        "stage1_frozen": True,
        "task_source_relation_initialization": "learned_from_neutral_zero_matrix",
        "stage2_artifact_source": "current_run",
        "scoring_label_access": "after_prediction_barrier",
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_selection_data_scope": "outer_train_oof",
        "profile_lock_before_outer_test_prediction": True,
        "inner_oof_folds": 5,
        "deployment_model": "inner_crossfit_ensemble",
        "input_validation": "schema_alignment_and_manifest",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"Malformed HP-SafeMoE source policy: {key}")
    load_config(Path(str(value["config_path"])))
    return value


def _fold(policy: dict[str, Any], outer: int, roles: tuple[str, ...]) -> FoldInputs:
    return FoldInputs(
        project_root=Path(str(policy["project_root"])),
        pool=load_expert_pool(
            Path(str(policy["pool_path"])),
        ),
        outer_fold=outer,
        roles_to_load=roles,
        verify_hashes=False,
    )


def _attach_target_free_test(fold: FoldInputs) -> None:
    for task in CANONICAL_TASKS:
        manifest_path = fold.project_root / "data/manifests" / f"{task}.npz"
        with np.load(manifest_path, allow_pickle=False) as manifest:
            all_ids = manifest["sample_ids"].astype(str)
            positions = manifest[f"outer_{fold.outer_fold}_test"].astype(np.int64)
        sample_ids = all_ids[positions]
        exports: dict[str, ExpertExport | None] = {}
        for spec in fold.pool.experts:
            root = export_directory(spec, task, fold.outer_fold, "test")
            if not (root / "metadata.json").is_file():
                exports[spec.expert_id] = None
                continue
            metadata = read_json(root / "metadata.json")
            expert, split = metadata.get("expert", {}), metadata.get("split", {})
            if (
                expert.get("expert_id") != spec.expert_id
                or expert.get("source_task") != spec.source_task
                or expert.get("frozen") is not True
                or metadata.get("target_task") != task
                or metadata.get("label_values_stored") is not False
                or split.get("outer_fold") != fold.outer_fold
                or split.get("role") != "test"
            ):
                raise ValueError(f"Malformed target-free export metadata: {root}")
            arrays = {
                key: np.load(
                    root / filename,
                    mmap_mode="r" if key == "hidden" else None,
                    allow_pickle=False,
                )
                for key, filename in ARRAY_FILES.items()
            }
            if not np.array_equal(arrays["sample_ids"].astype(str), sample_ids):
                raise ValueError(f"Target-free export IDs differ from manifest: {root}")
            exports[spec.expert_id] = ExpertExport(
                root=root,
                metadata=metadata,
                sample_ids=arrays["sample_ids"],
                hidden=arrays["hidden"],
                prediction=arrays["prediction"],
                uncertainty=arrays["uncertainty"],
                available=arrays["available"],
            )
        anchor_id = fold.pool.anchor_for(task)
        anchor = exports[anchor_id]
        if anchor is None or not np.all(np.asarray(anchor.available, dtype=np.bool_)):
            raise ValueError(f"Target-free test Anchor unavailable for {task}")
        fold.roles[task]["test"] = TaskRoleCache(
            task=task,
            role="test",
            outer_fold=fold.outer_fold,
            manifest_role="outer_test",
            sample_ids=sample_ids,
            y_true=np.zeros(len(sample_ids), dtype=np.float64),
            anchor_expert_id=anchor_id,
            exports=exports,
        )


def _scalar_stats(values: np.ndarray) -> ScalarStats:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return ScalarStats(0.0, 1.0)
    return ScalarStats(float(array.mean()), max(float(array.std()), 1.0e-8))


def _inner_partitions(fold: FoldInputs, partitions: int, *, seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Create deterministic, balanced partitions inside each official outer-train."""
    result: dict[str, dict[str, np.ndarray]] = {}
    for task_number, task in enumerate(CANONICAL_TASKS):
        role_sizes = [fold.roles[task][role].n_samples for role in ("train", "val")]
        labels = np.concatenate([np.asarray(fold.roles[task][role].y_true) for role in ("train", "val")])
        assignment = np.empty(len(labels), dtype=np.int8)
        rng = np.random.default_rng(seed + task_number * 1009)
        if fold.task_types[task] == "classification":
            for label in (0, 1):
                selected = np.flatnonzero(labels == label)
                if len(selected) < partitions:
                    raise ValueError(f"{task} has too few class-{label} rows for inner OOF")
                rng.shuffle(selected)
                assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
        else:
            selected = np.arange(len(labels), dtype=np.int64)
            rng.shuffle(selected)
            assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
        if set(assignment.tolist()) != set(range(partitions)):
            raise ValueError(f"Incomplete inner partitions for {task}")
        result[task] = {}
        offset = 0
        for role, size in zip(("train", "val"), role_sizes, strict=True):
            result[task][role] = assignment[offset : offset + size]
            offset += size
    return result


def _partition_indices(
    fold: FoldInputs,
    assignments: dict[str, dict[str, np.ndarray]],
    held_out: int,
    *,
    training: bool,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for task in CANONICAL_TASKS:
        result[task] = {}
        for role in ("train", "val"):
            selected = (
                assignments[task][role] != held_out if training else assignments[task][role] == held_out
            )
            indices = np.flatnonzero(selected).astype(np.int64)
            if training and not len(indices):
                raise ValueError(f"Empty inner training slice for {task}/{role}")
            result[task][role] = indices
    return result


def _masked_stats(fold: FoldInputs, eligible: dict[str, dict[str, np.ndarray]]) -> dict[str, TaskStats]:
    result: dict[str, TaskStats] = {}
    for task in CANONICAL_TASKS:
        targets = np.concatenate(
            [np.asarray(fold.roles[task][role].y_true)[eligible[task][role]] for role in ("train", "val")]
        )
        target = (
            ScalarStats(0.0, 1.0) if fold.task_types[task] == "classification" else _scalar_stats(targets)
        )
        prediction: dict[str, ScalarStats] = {}
        uncertainty: dict[str, ScalarStats] = {}
        for expert_id in fold.pool.expert_ids:
            prediction_parts, uncertainty_parts = [], []
            for role in ("train", "val"):
                export = fold.roles[task][role].exports[expert_id]
                if export is None:
                    continue
                indices = eligible[task][role]
                available = np.asarray(export.available, dtype=np.bool_)[indices]
                if np.any(available):
                    prediction_parts.append(np.asarray(export.prediction)[indices][available])
                    uncertainty_parts.append(
                        np.log1p(np.asarray(export.uncertainty, dtype=np.float64)[indices][available])
                    )
            prediction[expert_id] = _scalar_stats(
                np.concatenate(prediction_parts) if prediction_parts else np.asarray([])
            )
            uncertainty[expert_id] = _scalar_stats(
                np.concatenate(uncertainty_parts) if uncertainty_parts else np.asarray([])
            )
        result[task] = TaskStats(target, prediction, uncertainty)
    return result


def _model_config(pool: ExpertPool, config: dict[str, Any], candidate: str) -> Stage2ProposalConfig:
    spec = ARCHITECTURE_CANDIDATES[candidate]
    return Stage2ProposalConfig(
        task_names=CANONICAL_TASKS,
        expert_ids=pool.expert_ids,
        hidden_dims=pool.hidden_dims,
        anchor_indices=tuple(pool.expert_ids.index(pool.anchor_for(task)) for task in CANONICAL_TASKS),
        token_dim=int(config["token_dim"]),
        task_dim=int(config["task_dim"]),
        residual_hidden=int(config["residual_hidden"]),
        residual_rank=int(config["residual_rank"]),
        dropout=float(config["dropout"]),
        attention_top_k=int(config["attention_top_k"]),
        residual_limit=float(config["residual_limit"]),
        route_temperature=float(config["route_temperature"]),
        use_cross_expert_input=bool(spec["use_cross_expert_input"]),
        use_data_branch=bool(spec["use_data_branch"]),
        use_relation_branch=bool(spec["use_relation_branch"]),
        use_learned_task_source_relation=bool(spec["use_learned_task_source_relation"]),
        use_shared_low_rank_basis=bool(spec["use_shared_low_rank_basis"]),
        use_task_private_correction=bool(spec["use_task_private_correction"]),
        use_sample_gate=True,
    )


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _sample_indices(
    *,
    fold: FoldInputs,
    task: str,
    eligible: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[str, np.ndarray]:
    roles = ("train", "val")
    choices = [eligible[task][role] for role in roles]
    sizes = np.asarray([len(value) for value in choices], dtype=np.float64)
    if fold.task_types[task] == "classification":
        for role_number, (role, valid) in enumerate(zip(roles, choices, strict=True)):
            y_role = np.asarray(fold.roles[task][role].y_true)
            if any(not np.any(y_role[valid] == label) for label in (0, 1)):
                sizes[role_number] = 0.0
    if sizes.sum() <= 0.0:
        raise ValueError(f"No eligible training shard for {task}")
    role_number = int(rng.choice(len(roles), p=sizes / sizes.sum()))
    role, valid = roles[role_number], choices[role_number]
    y = np.asarray(fold.roles[task][role].y_true)
    if fold.task_types[task] == "classification":
        count = int(config["classification_per_class"])
        pieces = []
        for label in (0, 1):
            label_rows = valid[y[valid] == label]
            if not len(label_rows):
                raise ValueError(f"Inner training slice lost class {label} for {task}/{role}")
            pieces.append(rng.choice(label_rows, size=count, replace=len(label_rows) < count))
        indices = np.concatenate(pieces).astype(np.int64)
        rng.shuffle(indices)
        return role, indices
    count = int(config["regression_batch_size"])
    return role, rng.choice(valid, size=count, replace=len(valid) < count).astype(np.int64)


def _prediction_loss(
    prediction: torch.Tensor,
    anchor: torch.Tensor,
    target: torch.Tensor,
    task_type: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task_type == "classification":
        positive = target > 0.5
        difference = prediction[positive][:, None] - prediction[~positive][None, :]
        anchor_difference = anchor[positive][:, None] - anchor[~positive][None, :]
        loss = F.softplus(-difference / temperature).mean()
        anchor_loss = F.softplus(-anchor_difference / temperature).mean().detach()
    else:
        loss = torch.abs(prediction - target).mean()
        anchor_loss = torch.abs(anchor - target).mean().detach()
    return loss, anchor_loss


def _fit_one_model(
    *,
    fold: FoldInputs,
    candidate: str,
    config: dict[str, Any],
    seed: int,
    eligible: dict[str, dict[str, np.ndarray]],
    device: torch.device,
    progress_path: Path,
    progress_offset: int,
    progress_total: int,
    phase: str,
) -> CrossExpertProposalModel:
    rng = _seed_everything(seed)
    model = CrossExpertProposalModel(_model_config(fold.pool, config, candidate)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    steps, started = int(config["proposal_training_steps"]), time.time()
    objective_trace: list[float] = []
    spec = ARCHITECTURE_CANDIDATES[candidate]
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses, anchor_losses, branch_auxiliary, entropies = [], [], [], []
        for task in CANONICAL_TASKS:
            role, indices = _sample_indices(fold=fold, task=task, eligible=eligible, config=config, rng=rng)
            batch = fold.batch(
                task=task,
                role=role,
                indices=indices,
                device=device,
                expert_dropout=float(config["expert_dropout"]),
                generator=rng,
            )
            output = model(
                task_index=batch["task_index"],
                anchor_standard=batch["anchor_standard"],
                hidden=batch["hidden"],
                prediction_standard=batch["prediction_standard"],
                uncertainty_standard=batch["uncertainty_standard"],
                available=batch["available"],
            )
            loss, anchor_loss = _prediction_loss(
                output.raw_standard,
                batch["anchor_standard"],
                batch["y_standard"],
                fold.task_types[task],
                float(config["pairwise_auc_temperature"]),
            )
            losses.append(loss)
            anchor_losses.append(anchor_loss)
            branch_losses = []
            if spec["use_data_branch"]:
                branch_losses.append(
                    _prediction_loss(
                        output.data_standard,
                        batch["anchor_standard"],
                        batch["y_standard"],
                        fold.task_types[task],
                        float(config["pairwise_auc_temperature"]),
                    )[0]
                )
            if spec["use_relation_branch"]:
                branch_losses.append(
                    _prediction_loss(
                        output.relation_standard,
                        batch["anchor_standard"],
                        batch["y_standard"],
                        fold.task_types[task],
                        float(config["pairwise_auc_temperature"]),
                    )[0]
                )
            branch_auxiliary.append(torch.stack(branch_losses).mean())
            probability = output.route_probabilities[:, 1:].clamp_min(1.0e-8)
            entropies.append(-(probability * probability.log()).sum(dim=1).mean())
        task_vector = torch.stack(losses)
        anchor_vector = torch.stack(anchor_losses)
        positive_regret = F.relu(task_vector - anchor_vector)
        tail_count = max(1, int(math.ceil(float(config["task_cvar_fraction"]) * len(CANONICAL_TASKS))))
        cvar = torch.topk(positive_regret, k=tail_count).values.mean()
        entropy = torch.stack(entropies).mean()
        entropy_target = float(config["route_entropy_target"])
        objective = (
            task_vector.mean()
            + float(config["mean_positive_task_regret_weight"]) * positive_regret.mean()
            + float(config["task_cvar_weight"]) * cvar
            + float(config["branch_auxiliary_weight"]) * torch.stack(branch_auxiliary).mean()
            + float(config["route_entropy_weight"]) * (entropy - entropy_target).square()
        )
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Non-finite HP-SafeMoE objective at step {step}")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
        optimizer.step()
        objective_trace.append(float(objective.detach().cpu()))
        if step == 0 or (step + 1) % int(config["progress_every"]) == 0 or step + 1 == steps:
            completed = progress_offset + step + 1
            elapsed = time.time() - started
            atomic_json(
                progress_path,
                {
                    "stage": phase,
                    "candidate": candidate,
                    "seed": seed,
                    "completed": completed,
                    "total": progress_total,
                    "percent": 100.0 * completed / progress_total,
                    "elapsed_seconds_current_model": elapsed,
                    "eta_seconds": elapsed / (step + 1) * (progress_total - completed),
                    "objective": float(np.mean(objective_trace[-10:])),
                    "positive_task_regret": float(positive_regret.mean().detach().cpu()),
                    "task_cvar": float(cvar.detach().cpu()),
                    "progress_unit_label": "joint-13-task-step",
                    "progress_note": "true inner-OOF proposal; no internal acceptance gate",
                },
            )
    return model


PREDICTION_KEYS = (
    "raw_prediction",
    "data_prediction",
    "relation_prediction",
    "route_probabilities",
    "route_index",
    "data_attention",
    "relation_attention",
    "reliability",
    "anchor_prediction",
)


def _predict(
    *,
    fold: FoldInputs,
    model: CrossExpertProposalModel,
    task: str,
    role: str,
    device: torch.device,
    indices: np.ndarray | None = None,
    chunk_size: int = 4096,
    shuffle_relation: bool = False,
    shuffle_seed: int = 0,
) -> dict[str, np.ndarray]:
    cache = fold.roles[task][role]
    selected = (
        np.arange(cache.n_samples, dtype=np.int64) if indices is None else np.asarray(indices, dtype=np.int64)
    )
    if selected.ndim != 1 or not len(selected):
        raise ValueError("Prediction indices must be a non-empty vector")
    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "raw",
            "data",
            "relation",
            "route_probabilities",
            "route_index",
            "data_attention",
            "relation_attention",
            "reliability",
            "anchor",
        )
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(selected), chunk_size):
            rows = selected[start : start + chunk_size]
            batch = fold.batch(task=task, role=role, indices=rows, device=device)
            output = model(
                task_index=batch["task_index"],
                anchor_standard=batch["anchor_standard"],
                hidden=batch["hidden"],
                prediction_standard=batch["prediction_standard"],
                uncertainty_standard=batch["uncertainty_standard"],
                available=batch["available"],
            )
            relation_standard = output.relation_standard
            if shuffle_relation and len(rows) > 1:
                rng = np.random.default_rng(shuffle_seed + start)
                order = torch.as_tensor(rng.permutation(len(rows)), device=device)
                relation_standard = relation_standard[order]
                route = output.route_one_hot
                raw = (
                    route[:, 0] * batch["anchor_standard"]
                    + route[:, 1] * output.data_standard
                    + route[:, 2] * relation_standard
                )
            else:
                raw = output.raw_standard
            parts["raw"].append(raw.detach().cpu().numpy())
            parts["data"].append(output.data_standard.detach().cpu().numpy())
            parts["relation"].append(relation_standard.detach().cpu().numpy())
            parts["route_probabilities"].append(output.route_probabilities.detach().cpu().numpy())
            parts["route_index"].append(output.route_one_hot.argmax(dim=1).detach().cpu().numpy())
            parts["data_attention"].append(output.data_attention.detach().cpu().numpy())
            parts["relation_attention"].append(output.relation_attention.detach().cpu().numpy())
            parts["reliability"].append(output.reliability.detach().cpu().numpy())
            parts["anchor"].append(np.asarray(batch["anchor_prediction"]))
    return {
        "sample_ids": np.asarray(cache.sample_ids[selected]).astype(str),
        "raw_prediction": fold.restore_prediction(task, np.concatenate(parts["raw"])),
        "data_prediction": fold.restore_prediction(task, np.concatenate(parts["data"])),
        "relation_prediction": fold.restore_prediction(task, np.concatenate(parts["relation"])),
        "route_probabilities": np.concatenate(parts["route_probabilities"]).astype(np.float32),
        "route_index": np.concatenate(parts["route_index"]).astype(np.int8),
        "data_attention": np.concatenate(parts["data_attention"]).astype(np.float32),
        "relation_attention": np.concatenate(parts["relation_attention"]).astype(np.float32),
        "reliability": np.concatenate(parts["reliability"]).astype(np.float32),
        "anchor_prediction": np.concatenate(parts["anchor"]).astype(np.float64),
    }


def _empty_oof_buffers(fold: FoldInputs) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    experts = len(fold.pool.expert_ids)
    for task in CANONICAL_TASKS:
        size = sum(fold.roles[task][role].n_samples for role in ("train", "val"))
        result[task] = {
            "sample_ids": np.empty(size, dtype="U256"),
            "y_true": np.full(size, np.nan, dtype=np.float64),
            "raw_prediction": np.full(size, np.nan, dtype=np.float64),
            "data_prediction": np.full(size, np.nan, dtype=np.float64),
            "relation_prediction": np.full(size, np.nan, dtype=np.float64),
            "route_probabilities": np.full((size, 3), np.nan, dtype=np.float32),
            "route_index": np.full(size, -1, dtype=np.int8),
            "data_attention": np.full((size, experts), np.nan, dtype=np.float32),
            "relation_attention": np.full((size, experts), np.nan, dtype=np.float32),
            "reliability": np.full(
                (size, CrossExpertProposalModel.RELIABILITY_DIM), np.nan, dtype=np.float32
            ),
            "anchor_prediction": np.full(size, np.nan, dtype=np.float64),
        }
    return result


def _write_oof_rows(
    *,
    destination: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    positions: np.ndarray,
    y_true: np.ndarray,
) -> None:
    destination["sample_ids"][positions] = prediction["sample_ids"]
    destination["y_true"][positions] = np.asarray(y_true, dtype=np.float64)
    for key in PREDICTION_KEYS:
        destination[key][positions] = prediction[key]


def _check_oof_complete(candidate: str, task: str, arrays: dict[str, np.ndarray]) -> None:
    for key, values in arrays.items():
        if key == "sample_ids":
            if np.any(np.asarray(values).astype(str) == ""):
                raise RuntimeError(f"Incomplete OOF sample IDs for {candidate}/{task}")
        elif key == "route_index":
            if np.any(np.asarray(values) < 0):
                raise RuntimeError(f"Incomplete OOF routes for {candidate}/{task}")
        elif not np.isfinite(np.asarray(values)).all():
            raise RuntimeError(f"Non-finite OOF {key} for {candidate}/{task}")


def _ensemble_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        raise ValueError("At least one proposal record is required")
    if any(not np.array_equal(records[0]["sample_ids"], record["sample_ids"]) for record in records[1:]):
        raise ValueError("Proposal ensemble sample IDs differ")
    anchor = np.asarray(records[0]["anchor_prediction"], dtype=np.float64)
    raw_members = np.stack([np.asarray(record["raw_prediction"], dtype=np.float64) for record in records])
    residual_members = raw_members - anchor[None, :]
    denominator = np.mean(np.abs(residual_members), axis=0)
    directional = np.divide(
        np.abs(np.mean(residual_members, axis=0)),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1.0e-12,
    )
    route_members = np.stack([record["route_index"] for record in records])
    route_consensus = np.max(np.stack([(route_members == route).mean(axis=0) for route in range(3)]), axis=0)
    return {
        "sample_ids": np.asarray(records[0]["sample_ids"]).astype(str),
        "anchor_prediction": anchor,
        "raw_prediction": raw_members.mean(axis=0),
        "member_raw_prediction": raw_members,
        "data_prediction": np.mean([record["data_prediction"] for record in records], axis=0),
        "relation_prediction": np.mean([record["relation_prediction"] for record in records], axis=0),
        "route_probabilities": np.mean([record["route_probabilities"] for record in records], axis=0).astype(
            np.float32
        ),
        "route_index": np.asarray(
            np.mean([record["route_probabilities"] for record in records], axis=0).argmax(axis=1),
            dtype=np.int8,
        ),
        "route_consensus": route_consensus.astype(np.float32),
        "directional_agreement": directional.astype(np.float32),
        "data_attention": np.mean([record["data_attention"] for record in records], axis=0),
        "relation_attention": np.mean([record["relation_attention"] for record in records], axis=0),
        "reliability": np.mean([record["reliability"] for record in records], axis=0),
    }


def _gate_features(values: dict[str, np.ndarray]) -> np.ndarray:
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    raw = np.asarray(values["raw_prediction"], dtype=np.float64)
    residual = raw - anchor
    members = np.asarray(values["member_raw_prediction"], dtype=np.float64)
    data = np.asarray(values["data_prediction"], dtype=np.float64)
    relation = np.asarray(values["relation_prediction"], dtype=np.float64)
    route = np.asarray(values["route_probabilities"], dtype=np.float64)
    data_attention = np.asarray(values["data_attention"], dtype=np.float64)
    relation_attention = np.asarray(values["relation_attention"], dtype=np.float64)
    reliability = np.asarray(values["reliability"], dtype=np.float64)

    def entropy(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, 1.0e-12, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=1)

    return np.column_stack(
        [
            anchor,
            raw,
            residual,
            np.abs(residual),
            members.std(axis=0),
            np.asarray(values["directional_agreement"]),
            np.asarray(values["route_consensus"]),
            data - relation,
            np.abs(data - relation),
            route,
            route.max(axis=1),
            entropy(route),
            data_attention.max(axis=1),
            relation_attention.max(axis=1),
            entropy(data_attention),
            entropy(relation_attention),
            reliability,
        ]
    ).astype(np.float64)


def _standardize_fit(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(features, dtype=np.float64).mean(axis=0)
    scale = np.asarray(features, dtype=np.float64).std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return (features - mean) / scale, mean, scale


def _fit_benefit_gate(
    *, features: np.ndarray, labels: np.ndarray, seed: int, config: dict[str, Any]
) -> dict[str, Any]:
    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.float64)
    maximum = int(config["benefit_gate_max_rows"])
    if len(label_array) > maximum:
        rng, pieces = np.random.default_rng(seed), []
        for label in (0.0, 1.0):
            candidates = np.flatnonzero(label_array == label)
            take = min(len(candidates), maximum // 2)
            pieces.append(rng.choice(candidates, size=take, replace=False))
        selected = np.concatenate(pieces)
        rng.shuffle(selected)
        feature_array, label_array = feature_array[selected], label_array[selected]
    x, mean, scale = _standardize_fit(feature_array)
    y = label_array
    if len(np.unique(y)) < 2:
        probability = float(np.mean(y))
        return {
            "kind": "constant",
            "constant_probability": probability,
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "weight": [0.0] * x.shape[1],
            "bias": float(math.log((probability + 1.0e-4) / (1.0001 - probability))),
            "fit_rows": len(y),
        }
    torch.manual_seed(seed)
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32)
    layer = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.zeros_(layer.bias)
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=float(config["benefit_gate_learning_rate"]),
        weight_decay=float(config["benefit_gate_weight_decay"]),
    )
    positive = float(y.sum())
    negative = float(len(y) - positive)
    positive_weight = torch.tensor(max(negative / max(positive, 1.0), 1.0))
    for _ in range(int(config["benefit_gate_steps"])):
        optimizer.zero_grad(set_to_none=True)
        logits = layer(tensor_x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, tensor_y, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()
    return {
        "kind": "linear_logistic",
        "constant_probability": None,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weight": layer.weight.detach().cpu().numpy().reshape(-1).astype(float).tolist(),
        "bias": float(layer.bias.detach().cpu()),
        "fit_rows": len(y),
    }


def _benefit_probability(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    if model["kind"] == "constant":
        return np.full(len(features), float(model["constant_probability"]), dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weight = np.asarray(model["weight"], dtype=np.float64)
    logits = ((np.asarray(features) - mean) / scale) @ weight + float(model["bias"])
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _meta_partitions(y: np.ndarray, task_type: str, partitions: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(y), dtype=np.int8)
    if task_type == "classification":
        for label in (0, 1):
            selected = np.flatnonzero(np.asarray(y) == label)
            rng.shuffle(selected)
            assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
    else:
        selected = np.arange(len(y), dtype=np.int64)
        rng.shuffle(selected)
        assignment[selected] = np.arange(len(selected), dtype=np.int64) % partitions
    return assignment


def _apply_parameters(
    *,
    anchor: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    alpha: float,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    proposal = np.asarray(anchor) + float(alpha) * (np.asarray(raw) - np.asarray(anchor))
    accepted = (
        np.ones(len(proposal), dtype=np.bool_)
        if threshold is None
        else np.asarray(probability) >= float(threshold)
    )
    accepted &= float(alpha) > 0.0
    return np.where(accepted, proposal, anchor), accepted


def _parameter_grid(
    task_type: str, config: dict[str, Any], *, sample_gate: bool
) -> list[tuple[float, float | None]]:
    scales = list(map(float, config["residual_scale_grid"]))
    if task_type == "classification" or not sample_gate:
        return [(alpha, None) for alpha in scales]
    thresholds = list(map(float, config["benefit_probability_threshold_grid"]))
    return [(0.0, None)] + [(alpha, threshold) for alpha in scales if alpha > 0.0 for threshold in thresholds]


def _select_parameters(
    *,
    y: np.ndarray,
    anchor: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    sample_gate: bool,
) -> dict[str, Any]:
    anchor_score = score_predictions(task_type, y, anchor)
    best: tuple[tuple[float, float, float, float], dict[str, Any]] | None = None
    for alpha, threshold in _parameter_grid(task_type, config, sample_gate=sample_gate):
        prediction, accepted = _apply_parameters(
            anchor=anchor,
            raw=raw,
            probability=probability,
            alpha=alpha,
            threshold=threshold,
        )
        coverage = float(accepted.mean())
        if alpha > 0.0 and coverage < float(config["minimum_sample_coverage"]):
            continue
        gain = normalized_gain(task_type, anchor_score, score_predictions(task_type, y, prediction))
        # OOF metric gain is primary. Exact ties prefer less correction and
        # broader coverage. This convention is identical for every task.
        rank = (gain, -alpha, coverage, -(threshold or 0.0))
        record = {
            "alpha": alpha,
            "threshold": threshold,
            "gain_percent": gain,
            "coverage": coverage,
        }
        if best is None or rank > best[0]:
            best = (rank, record)
    if best is None:
        raise RuntimeError("No valid OOF calibration parameters")
    return best[1]


def _bootstrap_gain(
    *,
    task_type: str,
    y: np.ndarray,
    anchor: np.ndarray,
    model: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    y, anchor, model = map(lambda value: np.asarray(value, dtype=np.float64), (y, anchor, model))
    point = normalized_gain(
        task_type,
        score_predictions(task_type, y, anchor),
        score_predictions(task_type, y, model),
    )
    rng, gains = np.random.default_rng(seed), []
    maximum = int(config["bootstrap_max_rows"])
    for _ in range(int(config["bootstrap_resamples"])):
        if task_type == "classification":
            pieces = []
            for label in (0, 1):
                candidates = np.flatnonzero(y == label)
                pieces.append(rng.choice(candidates, size=min(len(candidates), maximum // 2), replace=True))
            selected = np.concatenate(pieces)
        else:
            selected = rng.integers(0, len(y), size=min(len(y), maximum))
        gains.append(
            normalized_gain(
                task_type,
                score_predictions(task_type, y[selected], anchor[selected]),
                score_predictions(task_type, y[selected], model[selected]),
            )
        )
    values = np.asarray(gains, dtype=np.float64)
    confidence = float(config["bootstrap_confidence"])
    return {
        "point_gain_percent": point,
        "lower_gain_percent": float(np.quantile(values, 1.0 - confidence)),
        "upper_gain_percent": float(np.quantile(values, confidence)),
    }


def _profile_acceptance(
    *,
    alpha: float,
    certificate: dict[str, float],
    partition_gains: list[float],
) -> dict[str, bool]:
    positive = sum(gain > 0.0 for gain in partition_gains)
    return {
        name: bool(
            alpha > 0.0
            and certificate["point_gain_percent"] >= float(profile["minimum_oof_gain_percent"])
            and certificate["lower_gain_percent"] >= float(profile["bootstrap_lower_floor_percent"])
            and positive >= int(profile["required_positive_partitions"])
            and min(partition_gains) >= -float(profile["maximum_partition_regret_percent"])
        )
        for name, profile in SAFETY_PROFILES.items()
    }


def _calibrate_task(
    *,
    records: list[dict[str, np.ndarray]],
    y_true: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    values = _ensemble_records(records)
    features = _gate_features(values)
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    raw = np.asarray(values["raw_prediction"], dtype=np.float64)
    regression = task_type == "regression"
    local_benefit = (np.abs(anchor - y) > np.abs(raw - y)).astype(np.float32)
    assignment = _meta_partitions(y, task_type, int(config["meta_crossfit_partitions"]), seed=seed)
    meta_prediction = anchor.copy()
    meta_accepted = np.zeros(len(y), dtype=np.bool_)
    meta_probability = np.ones(len(y), dtype=np.float64)
    partition_records: list[dict[str, Any]] = []
    for held in range(int(config["meta_crossfit_partitions"])):
        tune, certify = assignment != held, assignment == held
        if regression:
            gate = _fit_benefit_gate(
                features=features[tune],
                labels=local_benefit[tune],
                seed=seed + held * 7919,
                config=config,
            )
            tune_probability = _benefit_probability(gate, features[tune])
            held_probability = _benefit_probability(gate, features[certify])
        else:
            tune_probability = np.ones(int(tune.sum()), dtype=np.float64)
            held_probability = np.ones(int(certify.sum()), dtype=np.float64)
        selected = _select_parameters(
            y=y[tune],
            anchor=anchor[tune],
            raw=raw[tune],
            probability=tune_probability,
            task_type=task_type,
            config=config,
            sample_gate=regression,
        )
        prediction, accepted = _apply_parameters(
            anchor=anchor[certify],
            raw=raw[certify],
            probability=held_probability,
            alpha=float(selected["alpha"]),
            threshold=selected["threshold"],
        )
        meta_prediction[certify] = prediction
        meta_accepted[certify] = accepted
        meta_probability[certify] = held_probability
        held_gain = normalized_gain(
            task_type,
            score_predictions(task_type, y[certify], anchor[certify]),
            score_predictions(task_type, y[certify], prediction),
        )
        partition_records.append(
            {
                **selected,
                "held_partition": held,
                "held_gain_percent": held_gain,
                "held_coverage": float(accepted.mean()),
            }
        )
    alpha = float(np.median([float(row["alpha"]) for row in partition_records]))
    selected_thresholds = [
        float(row["threshold"])
        for row in partition_records
        if row["threshold"] is not None and float(row["alpha"]) > 0.0
    ]
    threshold = float(np.median(selected_thresholds)) if selected_thresholds else None
    final_gate = (
        _fit_benefit_gate(features=features, labels=local_benefit, seed=seed + 99991, config=config)
        if regression
        else {
            "kind": "disabled_for_auc_task",
            "feature_mean": features.mean(axis=0).tolist(),
            "feature_scale": np.where(features.std(axis=0) > 1.0e-8, features.std(axis=0), 1.0).tolist(),
            "weight": [0.0] * features.shape[1],
            "bias": 0.0,
            "constant_probability": 1.0,
        }
    )
    partition_gains = [float(row["held_gain_percent"]) for row in partition_records]
    certificate = _bootstrap_gain(
        task_type=task_type,
        y=y,
        anchor=anchor,
        model=meta_prediction,
        config=config,
        seed=seed + 1777,
    )
    profiles = _profile_acceptance(alpha=alpha, certificate=certificate, partition_gains=partition_gains)
    return {
        "status": "true_inner_oof_meta_crossfit_benefit_calibration_complete",
        "task_type": task_type,
        "alpha": alpha,
        "benefit_probability_threshold": threshold,
        "sample_gate_enabled": regression,
        "benefit_gate": final_gate,
        "meta_crossfit_partition_records": partition_records,
        "meta_crossfit_partition_gains_percent": partition_gains,
        "meta_crossfit_positive_partitions": int(sum(gain > 0.0 for gain in partition_gains)),
        "meta_crossfit_coverage": float(meta_accepted.mean()),
        "benefit_label_positive_fraction": float(local_benefit.mean()),
        "meta_probability_mean": float(meta_probability.mean()),
        "certificate": certificate,
        "profile_acceptance": profiles,
        "fit_labels": "matching_official_outer_train_only",
        "outer_test_labels_accessed": False,
    }


def _apply_calibration(
    records: list[dict[str, np.ndarray]],
    *,
    calibration: dict[str, Any],
    profile: str,
    use_sample_gate: bool = True,
    use_task_certificate: bool = True,
) -> dict[str, np.ndarray]:
    values = _ensemble_records(records)
    features = _gate_features(values)
    task_type = str(calibration["task_type"])
    if task_type == "regression" and use_sample_gate:
        probability = _benefit_probability(calibration["benefit_gate"], features)
        threshold = calibration["benefit_probability_threshold"]
    else:
        probability = np.ones(len(features), dtype=np.float64)
        threshold = None
    prediction, accepted = _apply_parameters(
        anchor=values["anchor_prediction"],
        raw=values["raw_prediction"],
        probability=probability,
        alpha=float(calibration["alpha"]),
        threshold=threshold,
    )
    task_accepted = bool(calibration["profile_acceptance"][profile])
    if use_task_certificate and not task_accepted:
        prediction = np.asarray(values["anchor_prediction"], dtype=np.float64).copy()
        accepted = np.zeros(len(prediction), dtype=np.bool_)
    return {
        "sample_ids": np.asarray(values["sample_ids"]).astype(str),
        "anchor_prediction": np.asarray(values["anchor_prediction"], dtype=np.float64),
        "raw_prediction": np.asarray(values["raw_prediction"], dtype=np.float64),
        "prediction": np.asarray(prediction, dtype=np.float64),
        "accepted": np.asarray(accepted, dtype=np.bool_),
        "benefit_probability": np.asarray(probability, dtype=np.float32),
        "task_certificate_accepted": np.full(len(prediction), task_accepted, dtype=np.bool_),
        "residual_scale": np.full(len(prediction), float(calibration["alpha"]), dtype=np.float32),
        "route_index": np.asarray(values["route_index"], dtype=np.int8),
        "route_probabilities": np.asarray(values["route_probabilities"], dtype=np.float32),
        "directional_agreement": np.asarray(values["directional_agreement"], dtype=np.float32),
        "route_consensus": np.asarray(values["route_consensus"], dtype=np.float32),
        "data_attention": np.asarray(values["data_attention"], dtype=np.float32),
        "relation_attention": np.asarray(values["relation_attention"], dtype=np.float32),
    }


def _save_variant(root: Path, outer: int, variant: str, task: str, arrays: dict[str, np.ndarray]) -> None:
    safe = {key: np.asarray(value) for key, value in arrays.items() if key != "member_raw_prediction"}
    if "y_true" in safe:
        raise ValueError("Outer-test predictions must not contain labels")
    _atomic_npz(root / f"outer_prediction/outer_{outer}/{variant}/{task}.npz", safe)


def _load_proposal(
    *,
    checkpoint: Path,
    fold: FoldInputs,
    candidate: str,
    config: dict[str, Any],
    device: torch.device,
) -> CrossExpertProposalModel:
    payload = torch.load(checkpoint, map_location="cpu")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("candidate") != candidate
        or payload.get("outer_test_labels_accessed") is not False
    ):
        raise ValueError(f"Malformed proposal checkpoint: {checkpoint}")
    model = CrossExpertProposalModel(_model_config(fold.pool, config, candidate)).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _oof_records(
    *, root: Path, outer: int, candidate: str, task: str, seeds: list[int]
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    records: list[dict[str, np.ndarray]] = []
    target: np.ndarray | None = None
    for seed in seeds:
        with np.load(
            root / f"proposal_training/outer_{outer}/{candidate}/oof/seed_{seed}/{task}.npz",
            allow_pickle=False,
        ) as values:
            record = {key: np.asarray(values[key]) for key in ("sample_ids",) + PREDICTION_KEYS}
            y = np.asarray(values["y_true"], dtype=np.float64)
        if target is None:
            target = y
        elif not np.array_equal(target, y):
            raise ValueError("OOF targets differ across seeds")
        records.append(record)
    assert target is not None
    return records, target


def _baseline_arrays(values: dict[str, np.ndarray], *, use_raw: bool) -> dict[str, np.ndarray]:
    anchor = np.asarray(values["anchor_prediction"], dtype=np.float64)
    prediction = np.asarray(values["raw_prediction"], dtype=np.float64) if use_raw else anchor
    size = len(anchor)
    return {
        "sample_ids": np.asarray(values["sample_ids"]).astype(str),
        "anchor_prediction": anchor,
        "raw_prediction": np.asarray(values["raw_prediction"], dtype=np.float64),
        "prediction": prediction,
        "accepted": np.full(size, use_raw, dtype=np.bool_),
        "benefit_probability": np.ones(size, dtype=np.float32),
        "task_certificate_accepted": np.full(size, use_raw, dtype=np.bool_),
        "residual_scale": np.full(size, 1.0 if use_raw else 0.0, dtype=np.float32),
        "route_index": np.asarray(values["route_index"], dtype=np.int8),
        "route_probabilities": np.asarray(values["route_probabilities"], dtype=np.float32),
        "directional_agreement": np.asarray(values["directional_agreement"], dtype=np.float32),
        "route_consensus": np.asarray(values["route_consensus"], dtype=np.float32),
        "data_attention": np.asarray(values["data_attention"], dtype=np.float32),
        "relation_attention": np.asarray(values["relation_attention"], dtype=np.float32),
    }


def train_calibrate_candidate(
    *,
    source_policy_path: Path,
    output_root: Path,
    outer_fold: int,
    candidate: str,
    device: torch.device,
) -> dict[str, Any]:
    if outer_fold not in OUTER_FOLDS or candidate not in ARCHITECTURE_CANDIDATES:
        raise ValueError("Unknown HP-SafeMoE outer fold or proposal candidate")
    policy = validate_source_policy(source_policy_path)
    config = policy["config_payload"]
    root = output_root.expanduser().resolve()
    run_root = root / f"proposal_training/outer_{outer_fold}/{candidate}"
    fold = _fold(policy, outer_fold, ("train", "val"))
    inner_folds = int(config["inner_oof_folds"])
    seeds = list(map(int, config["model_seeds"]))
    assignments = _inner_partitions(fold, inner_folds, seed=int(config["inner_partition_seed"]) + outer_fold)
    plans = []
    for inner in range(inner_folds):
        eligible = _partition_indices(fold, assignments, inner, training=True)
        held = _partition_indices(fold, assignments, inner, training=False)
        plans.append((eligible, held, _masked_stats(fold, eligible)))
    buffers = {seed: _empty_oof_buffers(fold) for seed in seeds}
    checkpoint_records: list[tuple[Path, int, int]] = []
    total_models = len(seeds) * inner_folds
    total_steps = total_models * int(config["proposal_training_steps"])
    model_number = 0
    for inner, (eligible, held, stats) in enumerate(plans):
        for seed in seeds:
            fold.stats = stats
            fit_seed = seed + inner * 1_000_003
            model = _fit_one_model(
                fold=fold,
                candidate=candidate,
                config=config,
                seed=fit_seed,
                eligible=eligible,
                device=device,
                progress_path=run_root / "progress.json",
                progress_offset=model_number * int(config["proposal_training_steps"]),
                progress_total=total_steps,
                phase=f"true_inner_oof_proposal_{inner}",
            )
            checkpoint = run_root / f"checkpoints/inner_{inner}/seed_{seed}.pt"
            _atomic_checkpoint(
                checkpoint,
                {
                    "schema_version": SCHEMA_VERSION,
                    "candidate": candidate,
                    "outer_fold": outer_fold,
                    "inner_fold": inner,
                    "seed": seed,
                    "fit_seed": fit_seed,
                    "normalization": fold.stats_receipt(),
                    "model_config": model.config.__dict__,
                    "state_dict": model.state_dict(),
                    "fit_scope": "official_outer_train_minus_matching_inner_fold",
                    "input_validation": "schema_alignment_and_manifest",
                },
            )
            checkpoint_records.append((checkpoint, inner, seed))
            for task in CANONICAL_TASKS:
                offset = 0
                for role in ("train", "val"):
                    selected = held[task][role]
                    if len(selected):
                        prediction = _predict(
                            fold=fold,
                            model=model,
                            task=task,
                            role=role,
                            indices=selected,
                            device=device,
                        )
                        positions = offset + selected
                        _write_oof_rows(
                            destination=buffers[seed][task],
                            prediction=prediction,
                            positions=positions,
                            y_true=np.asarray(fold.roles[task][role].y_true)[selected],
                        )
                    offset += fold.roles[task][role].n_samples
            model_number += 1
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for seed in seeds:
        for task, arrays in buffers[seed].items():
            _check_oof_complete(candidate, task, arrays)
            _atomic_npz(run_root / f"oof/seed_{seed}/{task}.npz", arrays)
    del buffers

    calibrations: dict[str, dict[str, Any]] = {}
    for task_number, task in enumerate(CANONICAL_TASKS):
        records, target = _oof_records(
            root=root,
            outer=outer_fold,
            candidate=candidate,
            task=task,
            seeds=seeds,
        )
        calibration = _calibrate_task(
            records=records,
            y_true=target,
            task_type=fold.task_types[task],
            config=config,
            seed=int(config["meta_partition_seed"]) + outer_fold * 1009 + task_number * 97,
        )
        calibrations[task] = calibration
        atomic_json(run_root / f"calibration/{task}.json", calibration)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "true_inner_oof_calibration_complete_before_outer_test_access",
        "candidate": candidate,
        "outer_fold": outer_fold,
        "inner_oof_folds": inner_folds,
        "model_seeds": seeds,
        "proposal_model_count": total_models,
        "deployment_model": "inner_crossfit_ensemble",
        "fit_scope": "matching_matbench_official_outer_train_only",
        "calibration_data_scope": "matching_outer_train_inner_oof",
        "profile_selection_eligible": candidate == PRIMARY_PROPOSAL,
        "checkpoints": [str(item[0]) for item in checkpoint_records],
        "calibration_receipts": [str(run_root / f"calibration/{task}.json") for task in CANONICAL_TASKS],
        "input_validation": "schema_alignment_and_manifest",
    }
    path = run_root / "calibration_receipt.json"
    if path.is_file() and read_json(path) != receipt:
        raise RuntimeError(f"Refusing to replace a different calibration receipt: {path}")
    atomic_json(path, receipt)
    return receipt


def _canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _profile_lock_path(root: Path, outer_fold: int) -> Path:
    return root / f"policy/profile_locks/outer_{outer_fold}.json"


def _aggregate_oof_profile(
    *,
    root: Path,
    outer_fold: int,
    profile: str,
    clear_threshold: float,
) -> dict[str, Any]:
    task_rows: dict[str, dict[str, Any]] = {}
    gains: list[float] = []
    for task in CANONICAL_TASKS:
        calibration = read_json(
            root / f"proposal_training/outer_{outer_fold}/{PRIMARY_PROPOSAL}/calibration/{task}.json"
        )
        if (
            calibration.get("outer_test_labels_accessed") is not False
            or calibration.get("fit_labels") != "matching_official_outer_train_only"
        ):
            raise ValueError(f"Unsafe HP-SafeMoE calibration receipt: outer={outer_fold}, task={task}")
        accepted = bool(calibration["profile_acceptance"][profile])
        raw_gain = float(calibration["certificate"]["point_gain_percent"])
        deployed_gain = raw_gain if accepted else 0.0
        gains.append(deployed_gain)
        task_rows[task] = {
            "profile_accepted": accepted,
            "deployed_oof_gain_percent": deployed_gain,
            "raw_meta_crossfit_gain_percent": raw_gain,
            "bootstrap_lower_gain_percent": float(calibration["certificate"]["lower_gain_percent"]),
            "bootstrap_upper_gain_percent": float(calibration["certificate"]["upper_gain_percent"]),
            "positive_partitions": int(calibration["meta_crossfit_positive_partitions"]),
            "partition_gains_percent": list(map(float, calibration["meta_crossfit_partition_gains_percent"])),
            "coverage": float(calibration["meta_crossfit_coverage"]),
        }
    return {
        "macro_normalized_gain_percent": float(np.mean(gains)),
        "positive_tasks": int(sum(value > 1.0e-9 for value in gains)),
        "clear_positive_tasks": int(sum(value >= clear_threshold for value in gains)),
        "neutral_tasks": int(sum(abs(value) <= 1.0e-9 for value in gains)),
        "negative_tasks": int(sum(value < -1.0e-9 for value in gains)),
        "worst_task_gain_percent": float(min(gains)),
        "tasks": task_rows,
    }


def _oof_profile_selection_key(
    record: dict[str, Any],
    *,
    profile: str,
    loss_limit: float,
    tie_priority: dict[str, int],
) -> tuple[int, int, float, float, int, int]:
    return (
        int(record["worst_task_gain_percent"] >= -loss_limit),
        int(record["clear_positive_tasks"]),
        float(record["macro_normalized_gain_percent"]),
        float(record["worst_task_gain_percent"]),
        int(record["positive_tasks"]),
        int(tie_priority[profile]),
    )


def lock_profile_for_outer_fold(
    *,
    source_policy_path: Path,
    output_root: Path,
    outer_fold: int,
) -> dict[str, Any]:
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("Unknown HP-SafeMoE outer fold")
    policy = validate_source_policy(source_policy_path)
    root = output_root.expanduser().resolve()
    prediction_root = root / f"outer_prediction/outer_{outer_fold}"
    if prediction_root.is_dir() and any(prediction_root.rglob("*.npz")):
        raise RuntimeError("HP-SafeMoE profile must be locked before outer-test prediction")
    if (root / f"outer_scoring/outer_{outer_fold}/outer_receipt.json").exists():
        raise RuntimeError("HP-SafeMoE profile cannot be selected after outer-test scoring")

    calibration_receipt = read_json(
        root / f"proposal_training/outer_{outer_fold}/{PRIMARY_PROPOSAL}/calibration_receipt.json"
    )
    required_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "true_inner_oof_calibration_complete_before_outer_test_access",
        "candidate": PRIMARY_PROPOSAL,
        "outer_fold": outer_fold,
        "outer_test_features_accessed": False,
        "outer_test_labels_accessed": False,
        "profile_selection_eligible": True,
    }
    for key, expected in required_receipt.items():
        if calibration_receipt.get(key) != expected:
            raise ValueError(f"Malformed HP-SafeMoE calibration receipt: {key}")

    config = policy["config_payload"]
    clear_threshold = float(config["oof_profile_selection_clear_gain_percent"])
    loss_limit = float(config["oof_profile_selection_max_worst_loss_percent"])
    tie_order = list(map(str, config["oof_profile_tie_break_order"]))
    tie_priority = {name: len(tie_order) - index for index, name in enumerate(tie_order)}
    evidence = {
        profile: _aggregate_oof_profile(
            root=root,
            outer_fold=outer_fold,
            profile=profile,
            clear_threshold=clear_threshold,
        )
        for profile in SELECTABLE_PROFILES
    }
    selected_profile = max(
        SELECTABLE_PROFILES,
        key=lambda name: _oof_profile_selection_key(
            evidence[name],
            profile=name,
            loss_limit=loss_limit,
            tie_priority=tie_priority,
        ),
    )
    lock = {
        "schema_version": SCHEMA_VERSION,
        "status": "profile_locked_from_fold_local_outer_train_oof",
        "outer_fold": outer_fold,
        "selected_safety_profile": selected_profile,
        "selection_scope": "one_profile_for_all_13_tasks_within_this_outer_fold",
        "selection_evidence_scope": "matching_official_outer_train_true_inner_oof_only",
        "selection_rule": (
            "prefer profiles satisfying the OOF worst-task floor; then maximize "
            "clear-positive tasks, macro OOF gain, worst-task OOF gain, and positive "
            "tasks; exact ties prefer the stricter predeclared profile"
        ),
        "clear_gain_threshold_percent": clear_threshold,
        "maximum_worst_loss_percent": loss_limit,
        "tie_break_order": tie_order,
        "profile_evidence": evidence,
        "source_policy": str(source_policy_path.expanduser().resolve(strict=True)),
        "calibration_receipt": str(
            root / f"proposal_training/outer_{outer_fold}/{PRIMARY_PROPOSAL}/calibration_receipt.json"
        ),
        "locked_before_outer_test_prediction": True,
        "profile_selection_data_scope": "outer_train_oof",
    }
    lock["selection_digest"] = _canonical_digest(lock)
    path = _profile_lock_path(root, outer_fold)
    if path.is_file():
        existing = read_json(path)
        if existing != lock:
            raise RuntimeError(f"Refusing to replace a different HP-SafeMoE profile lock: {path}")
        return existing
    atomic_json(path, lock)
    return lock


def validate_profile_lock(
    path: Path,
    *,
    source_policy_path: Path,
    outer_fold: int,
) -> dict[str, Any]:
    lock = read_json(path.expanduser().resolve(strict=True))
    required = {
        "schema_version": SCHEMA_VERSION,
        "status": "profile_locked_from_fold_local_outer_train_oof",
        "outer_fold": outer_fold,
        "selection_scope": "one_profile_for_all_13_tasks_within_this_outer_fold",
        "selection_evidence_scope": "matching_official_outer_train_true_inner_oof_only",
        "locked_before_outer_test_prediction": True,
        "profile_selection_data_scope": "outer_train_oof",
        "source_policy": str(source_policy_path.expanduser().resolve(strict=True)),
    }
    for key, expected in required.items():
        if lock.get(key) != expected:
            raise ValueError(f"Malformed HP-SafeMoE profile lock: {key}")
    profile = str(lock.get("selected_safety_profile"))
    if profile not in SELECTABLE_PROFILES:
        raise ValueError("Unknown safety profile in HP-SafeMoE lock")
    supplied_digest = str(lock.get("selection_digest", ""))
    unsigned = dict(lock)
    unsigned.pop("selection_digest", None)
    if supplied_digest != _canonical_digest(unsigned):
        raise ValueError("HP-SafeMoE profile lock digest mismatch")
    policy = validate_source_policy(source_policy_path)
    config = policy["config_payload"]
    evidence = lock.get("profile_evidence")
    if not isinstance(evidence, dict) or set(evidence) != set(SELECTABLE_PROFILES):
        raise ValueError("HP-SafeMoE profile lock has incomplete OOF evidence")
    for name in SELECTABLE_PROFILES:
        tasks = evidence[name].get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(CANONICAL_TASKS):
            raise ValueError(f"HP-SafeMoE profile lock has incomplete task evidence: {name}")
    tie_order = list(map(str, config["oof_profile_tie_break_order"]))
    tie_priority = {name: len(tie_order) - index for index, name in enumerate(tie_order)}
    loss_limit = float(config["oof_profile_selection_max_worst_loss_percent"])
    expected_profile = max(
        SELECTABLE_PROFILES,
        key=lambda name: _oof_profile_selection_key(
            evidence[name],
            profile=name,
            loss_limit=loss_limit,
            tie_priority=tie_priority,
        ),
    )
    if profile != expected_profile:
        raise ValueError("HP-SafeMoE profile lock does not match its recorded OOF evidence")
    return lock


def predict_candidate(
    *,
    source_policy_path: Path,
    profile_lock_path: Path,
    output_root: Path,
    outer_fold: int,
    candidate: str,
    device: torch.device,
) -> dict[str, Any]:
    if outer_fold not in OUTER_FOLDS or candidate not in ARCHITECTURE_CANDIDATES:
        raise ValueError("Unknown HP-SafeMoE outer fold or proposal candidate")
    policy = validate_source_policy(source_policy_path)
    lock = validate_profile_lock(
        profile_lock_path,
        source_policy_path=source_policy_path,
        outer_fold=outer_fold,
    )
    profile = str(lock["selected_safety_profile"])
    root = output_root.expanduser().resolve()
    run_root = root / f"proposal_training/outer_{outer_fold}/{candidate}"
    training_receipt = read_json(run_root / "calibration_receipt.json")
    if (
        training_receipt.get("status") != "true_inner_oof_calibration_complete_before_outer_test_access"
        or training_receipt.get("outer_test_features_accessed") is not False
        or training_receipt.get("outer_test_labels_accessed") is not False
    ):
        raise ValueError("Prediction requires a target-free HP-SafeMoE calibration receipt")

    config = policy["config_payload"]
    fold = _fold(policy, outer_fold, ("train", "val"))
    inner_folds = int(config["inner_oof_folds"])
    seeds = list(map(int, config["model_seeds"]))
    assignments = _inner_partitions(fold, inner_folds, seed=int(config["inner_partition_seed"]) + outer_fold)
    plans: list[dict[str, TaskStats]] = []
    for inner in range(inner_folds):
        eligible = _partition_indices(fold, assignments, inner, training=True)
        plans.append(_masked_stats(fold, eligible))

    calibrations = {task: read_json(run_root / f"calibration/{task}.json") for task in CANONICAL_TASKS}
    _attach_target_free_test(fold)
    test_records: dict[str, list[dict[str, np.ndarray]]] = {task: [] for task in CANONICAL_TASKS}
    checkpoints: list[str] = []
    for inner in range(inner_folds):
        for seed in seeds:
            checkpoint = run_root / f"checkpoints/inner_{inner}/seed_{seed}.pt"
            checkpoints.append(str(checkpoint))
            fold.stats = plans[inner]
            model = _load_proposal(
                checkpoint=checkpoint,
                fold=fold,
                candidate=candidate,
                config=config,
                device=device,
            )
            for task in CANONICAL_TASKS:
                test_records[task].append(
                    _predict(fold=fold, model=model, task=task, role="test", device=device)
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    expected_checkpoints = set(map(str, training_receipt["checkpoints"]))
    if set(checkpoints) != expected_checkpoints:
        raise ValueError("HP-SafeMoE prediction checkpoint population differs from OOF calibration")

    prediction_files: list[str] = []

    def save(variant: str, task: str, arrays: dict[str, np.ndarray]) -> None:
        _save_variant(root, outer_fold, variant, task, arrays)
        prediction_files.append(str(root / f"outer_prediction/outer_{outer_fold}/{variant}/{task}.npz"))

    for task in CANONICAL_TASKS:
        calibration = calibrations[task]
        save(
            _candidate_variant(candidate),
            task,
            _apply_calibration(test_records[task], calibration=calibration, profile=profile),
        )
        if candidate != PRIMARY_PROPOSAL:
            continue
        base = _ensemble_records(test_records[task])
        save("stage1_specialist_only", task, _baseline_arrays(base, use_raw=False))
        save("ablation_no_safety", task, _baseline_arrays(base, use_raw=True))

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "oof_profile_locked_crossfit_ensemble_test_prediction_complete",
        "candidate": candidate,
        "outer_fold": outer_fold,
        "selected_safety_profile": profile,
        "profile_lock": str(profile_lock_path.expanduser().resolve(strict=True)),
        "profile_lock_digest": str(lock["selection_digest"]),
        "profile_selection_data_scope": "outer_train_oof",
        "profile_locked_before_outer_test_prediction": True,
        "inner_oof_folds": inner_folds,
        "model_seeds": seeds,
        "proposal_model_count": len(checkpoints),
        "deployment_model": "inner_crossfit_ensemble",
        "deployment": "ensemble_of_the_same_inner_models_used_for_oof_calibration",
        "fit_scope": "matching_matbench_official_outer_train_only",
        "prediction_scope": "matching_matbench_official_outer_test_features_only",
        "checkpoints": checkpoints,
        "calibration_receipts": [str(run_root / f"calibration/{task}.json") for task in CANONICAL_TASKS],
        "prediction_files": prediction_files,
        "input_validation": "schema_alignment_and_manifest",
    }
    path = run_root / "prediction_receipt.json"
    if path.is_file() and read_json(path) != receipt:
        raise RuntimeError(f"Refusing to replace a different prediction receipt: {path}")
    atomic_json(path, receipt)
    return receipt


def commit_prediction_barrier(*, source_policy_path: Path, output_root: Path) -> dict[str, Any]:
    validate_source_policy(source_policy_path)
    root, file_count = output_root.expanduser().resolve(), 0
    profile_locks: dict[str, dict[str, Any]] = {}
    for outer in OUTER_FOLDS:
        lock_path = _profile_lock_path(root, outer)
        lock = validate_profile_lock(
            lock_path,
            source_policy_path=source_policy_path,
            outer_fold=outer,
        )
        profile_locks[str(outer)] = {
            "path": str(lock_path),
            "selection_digest": str(lock["selection_digest"]),
            "selected_safety_profile": str(lock["selected_safety_profile"]),
        }
        for candidate in ARCHITECTURE_CANDIDATES:
            receipt = read_json(root / f"proposal_training/outer_{outer}/{candidate}/prediction_receipt.json")
            required = {
                "schema_version": SCHEMA_VERSION,
                "status": "oof_profile_locked_crossfit_ensemble_test_prediction_complete",
                "candidate": candidate,
                "outer_fold": outer,
                "selected_safety_profile": str(lock["selected_safety_profile"]),
                "profile_lock_digest": str(lock["selection_digest"]),
                "profile_selection_data_scope": "outer_train_oof",
                "profile_locked_before_outer_test_prediction": True,
                "prediction_scope": "matching_matbench_official_outer_test_features_only",
            }
            for key, expected in required.items():
                if receipt.get(key) != expected:
                    raise ValueError(f"Malformed HP-SafeMoE prediction receipt: {candidate}/o{outer}/{key}")
        for variant in ALL_STORAGE_VARIANTS:
            for task in CANONICAL_TASKS:
                path = root / f"outer_prediction/outer_{outer}/{variant}/{task}.npz"
                if not path.is_file():
                    raise FileNotFoundError(path)
                with np.load(path, allow_pickle=False) as values:
                    if "y_true" in values.files:
                        raise ValueError(f"Unsafe target-bearing prediction file: {path}")
                file_count += 1
    barrier = {
        "schema_version": SCHEMA_VERSION,
        "status": "all_oof_locked_outer_predictions_committed_before_scoring",
        "source_policy": str(source_policy_path.expanduser().resolve(strict=True)),
        "storage_variants": list(ALL_STORAGE_VARIANTS),
        "profile_locks": profile_locks,
        "prediction_file_count": file_count,
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_selection_data_scope": "outer_train_oof",
        "all_profile_locks_precede_outer_test_prediction": True,
        "scoring_label_access": "after_prediction_barrier",
        "input_validation": "schema_alignment_and_manifest",
    }
    path = root / "policy/global_prediction_barrier.json"
    if path.is_file() and read_json(path) != barrier:
        raise RuntimeError(f"Refusing to replace a different HP-SafeMoE prediction barrier: {path}")
    atomic_json(path, barrier)
    return barrier


def score_outer_fold(*, prediction_barrier_path: Path, output_root: Path, outer_fold: int) -> dict[str, Any]:
    if outer_fold not in OUTER_FOLDS:
        raise ValueError("Unknown HP-SafeMoE outer fold")
    barrier = read_json(prediction_barrier_path.expanduser().resolve(strict=True))
    if (
        barrier.get("status") != "all_oof_locked_outer_predictions_committed_before_scoring"
        or barrier.get("profile_selection_data_scope") != "outer_train_oof"
        or barrier.get("all_profile_locks_precede_outer_test_prediction") is not True
    ):
        raise ValueError("All HP-SafeMoE target-free predictions must be committed before scoring")
    source_policy_path = Path(str(barrier["source_policy"]))
    policy = validate_source_policy(source_policy_path)
    lock_meta = barrier["profile_locks"][str(outer_fold)]
    lock = validate_profile_lock(
        Path(str(lock_meta["path"])),
        source_policy_path=source_policy_path,
        outer_fold=outer_fold,
    )
    if str(lock["selection_digest"]) != str(lock_meta["selection_digest"]):
        raise ValueError("HP-SafeMoE barrier/profile-lock digest mismatch")
    fold = _fold(policy, outer_fold, ("train", "test"))
    root, results = output_root.expanduser().resolve(), {}
    storage_variants = tuple(map(str, barrier["storage_variants"]))
    if storage_variants != ALL_STORAGE_VARIANTS:
        raise ValueError("HP-SafeMoE storage variants drifted after the prediction barrier")
    for variant in storage_variants:
        results[variant] = {}
        for task in CANONICAL_TASKS:
            path = root / f"outer_prediction/outer_{outer_fold}/{variant}/{task}.npz"
            with np.load(path, allow_pickle=False) as values:
                sample_ids = np.asarray(values["sample_ids"]).astype(str)
                anchor_prediction = np.asarray(values["anchor_prediction"], dtype=np.float64)
                prediction = np.asarray(values["prediction"], dtype=np.float64)
            expected = np.asarray(fold.roles[task]["test"].sample_ids).astype(str)
            if not np.array_equal(sample_ids, expected):
                raise ValueError(f"Prediction/manifest IDs differ: {variant}/{task}/o{outer_fold}")
            y = np.asarray(fold.roles[task]["test"].y_true, dtype=np.float64)
            task_type = fold.task_types[task]
            anchor_score = score_predictions(task_type, y, anchor_prediction)
            model_score = score_predictions(task_type, y, prediction)
            results[variant][task] = {
                "task_type": task_type,
                "metric": "roc_auc" if task_type == "classification" else "mae",
                "n_samples": len(y),
                "anchor_score": anchor_score,
                "model_score": model_score,
                "normalized_gain_percent": normalized_gain(task_type, anchor_score, model_score),
            }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "scored_after_oof_profile_lock_and_global_prediction_barrier",
        "outer_fold": outer_fold,
        "selected_safety_profile": str(lock["selected_safety_profile"]),
        "profile_lock_digest": str(lock["selection_digest"]),
        "profile_selection_data_scope": "outer_train_oof",
        "scoring_data_role": "outer_test",
        "results": results,
    }
    atomic_json(root / f"outer_scoring/outer_{outer_fold}/outer_receipt.json", receipt)
    return receipt


def _fold_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))


def _aggregate_variant(
    variant: str, folds: list[dict[str, Any]], *, clear_threshold: float
) -> dict[str, Any]:
    task_results, gains = {}, []
    for task in CANONICAL_TASKS:
        rows = [record["results"][variant][task] for record in folds]
        fold_gains = [float(row["normalized_gain_percent"]) for row in rows]
        gain, half = _fold_ci(fold_gains)
        gains.append(gain)
        task_results[task] = {
            "metric": rows[0]["metric"],
            "anchor_score": float(np.mean([row["anchor_score"] for row in rows])),
            "model_score": float(np.mean([row["model_score"] for row in rows])),
            "normalized_gain_percent": gain,
            "fold_gain_95ci_half_width_percent": half,
            "fold_gains_percent": fold_gains,
        }
    return {
        "macro_normalized_gain_percent": float(np.mean(gains)),
        "positive_tasks": int(sum(value > 1.0e-9 for value in gains)),
        "clear_positive_tasks": int(sum(value >= clear_threshold for value in gains)),
        "neutral_tasks": int(sum(abs(value) <= 1.0e-9 for value in gains)),
        "negative_tasks": int(sum(value < -1.0e-9 for value in gains)),
        "worst_task_gain_percent": float(min(gains)),
        "tasks": task_results,
    }


def summarize(*, prediction_barrier_path: Path, output_root: Path) -> dict[str, Any]:
    barrier = read_json(prediction_barrier_path.expanduser().resolve(strict=True))
    if (
        barrier.get("status") != "all_oof_locked_outer_predictions_committed_before_scoring"
        or barrier.get("profile_selection_data_scope") != "outer_train_oof"
    ):
        raise ValueError("HP-SafeMoE OOF-locked prediction barrier is missing")
    root = output_root.expanduser().resolve()
    policy = validate_source_policy(Path(str(barrier["source_policy"])))
    config = policy["config_payload"]
    clear_threshold = float(config["oof_profile_selection_clear_gain_percent"])
    folds = [read_json(root / f"outer_scoring/outer_{outer}/outer_receipt.json") for outer in OUTER_FOLDS]
    for outer, receipt in zip(OUTER_FOLDS, folds, strict=True):
        if (
            receipt.get("status") != "scored_after_oof_profile_lock_and_global_prediction_barrier"
            or receipt.get("profile_selection_data_scope") != "outer_train_oof"
            or receipt.get("scoring_data_role") != "outer_test"
        ):
            raise ValueError(f"Malformed HP-SafeMoE outer scoring receipt: o{outer}")
    storage_results = {
        variant: _aggregate_variant(variant, folds, clear_threshold=clear_threshold)
        for variant in ALL_STORAGE_VARIANTS
    }
    publication_variants = {
        "stage1_specialist_only": storage_results["stage1_specialist_only"],
        "full_method": storage_results[_candidate_variant(PRIMARY_PROPOSAL)],
        "no_cross_expert": storage_results[_candidate_variant("ablation_no_cross_expert")],
        "no_safety": storage_results["ablation_no_safety"],
    }
    display_names = {
        "stage1_specialist_only": "Stage1 specialist-only",
        "full_method": "Full method",
        "no_cross_expert": "No-cross-expert",
        "no_safety": "No safety control",
    }
    for key, value in publication_variants.items():
        value["report_name"] = display_names[key]

    locks = {
        str(outer): validate_profile_lock(
            Path(str(barrier["profile_locks"][str(outer)]["path"])),
            source_policy_path=Path(str(barrier["source_policy"])),
            outer_fold=outer,
        )
        for outer in OUTER_FOLDS
    }
    selected_by_fold = {outer: str(lock["selected_safety_profile"]) for outer, lock in locks.items()}
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_test_scored_after_fold_local_oof_profile_lock",
        "method_name": "HP-SafeMoE Stage2 HP-SafeMoE",
        "selected_global_variant": _candidate_variant(PRIMARY_PROPOSAL),
        "selected_safety_profile_by_outer_fold": selected_by_fold,
        "profile_selection_evidence_by_outer_fold": {
            outer: lock["profile_evidence"] for outer, lock in locks.items()
        },
        "profile_selection_rule": next(iter(locks.values()))["selection_rule"],
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_selection_data_scope": "outer_train_oof",
        "profile_locked_before_outer_test_prediction": True,
        "publication_variants": publication_variants,
        "training_data_scope": "outer_train",
        "scoring_label_access": "after_prediction_barrier",
        "selection_policy": "fold_local_global_profile",
        "deployment_model": "inner_crossfit_ensemble",
    }
    atomic_json(root / "reports/stage2_summary.json", summary)
    _write_markdown(root / "reports/stage2_summary.md", summary)
    _write_mechanism_report(root)
    return summary


def _fmt(value: float) -> str:
    return f"{value:.8g}"


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    variants = summary["publication_variants"]
    selected = variants["full_method"]
    columns = tuple(variants)
    selected_by_fold = summary["selected_safety_profile_by_outer_fold"]
    profile_text = ", ".join(
        f"outer {outer}={profile}"
        for outer, profile in sorted(selected_by_fold.items(), key=lambda item: int(item[0]))
    )
    lines = [
        "# HP-SafeMoE Stage 2: fold-local OOF-locked sharing",
        "",
        "- Stage 2 only uses frozen Stage-1 inputs aligned to the current official outer fold.",
        "- Five-fold inner-OOF proposals are trained within each outer-training fold.",
        "- One safety profile is selected per outer fold across all 13 tasks from that fold's "
        "outer-training inner-OOF evidence.",
        "- The profile lock is written before outer-test prediction, followed by scoring.",
        "- Deployment ensembles the same inner models used by the locked protocol.",
        "- The Sample-conditioned branch scores sources using the current material's "
        "expert tokens, target-task embedding and reliability summaries. The "
        "Relation-augmented branch additionally uses a target-source relation "
        "learned from outer-training data.",
        "- The primary model uses task-private residual heads.",
        f"- Fold-local OOF profiles: {profile_text}.",
        f"- Primary-model macro gain: **{selected['macro_normalized_gain_percent']:+.3f}%**; "
        f"positive/neutral/negative tasks: **{selected['positive_tasks']}/"
        f"{selected['neutral_tasks']}/{selected['negative_tasks']}**; worst-task gain: "
        f"**{selected['worst_task_gain_percent']:+.3f}%**.",
        "",
        "## Main result and task-level ablations",
        "",
        "Each cell reports the five-fold metric and the mean normalized gain in parentheses. "
        "Each fold uses its OOF-selected safety profile.",
        "",
        "| Matbench task | " + " | ".join(variants[name]["report_name"] for name in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    order = (
        "expt_is_metal",
        "glass",
        "mp_is_metal",
        "dielectric",
        "expt_gap",
        "jdft2d",
        "log_gvrh",
        "log_kvrh",
        "mp_e_form",
        "mp_gap",
        "perovskites",
        "phonons",
        "steels",
    )
    for task in order:
        metric = variants["full_method"]["tasks"][task]["metric"]
        cells = []
        for variant in columns:
            row = variants[variant]["tasks"][task]
            cells.append(f"{_fmt(row['model_score'])} ({row['normalized_gain_percent']:+.3f}%)")
        lines.append(f"| {task} {'↑' if metric == 'roc_auc' else '↓'} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Aggregate ablations",
        "",
        "| Variant | Macro gain | Clear positive | Positive/Neutral/Negative | Worst task gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in columns:
        row = variants[key]
        lines.append(
            f"| {row['report_name']} | {row['macro_normalized_gain_percent']:+.3f}% | "
            f"{row['clear_positive_tasks']} | "
            f"{row['positive_tasks']}/{row['neutral_tasks']}/{row['negative_tasks']} | "
            f"{row['worst_task_gain_percent']:+.3f}% |"
        )

    lines += [
        "",
        "## Outer-training OOF profile evidence",
        "",
        "The values are generated from the corresponding outer-training OOF evidence.",
        "",
        "| Outer fold | OOF-locked profile | Conservative OOF macro / worst / clear+ | "
        "Balanced OOF macro / worst / clear+ | Liberal OOF macro / worst / clear+ |",
        "|---:|---|---:|---:|---:|",
    ]
    evidence_by_fold = summary["profile_selection_evidence_by_outer_fold"]
    for outer, selected_profile in sorted(selected_by_fold.items(), key=lambda item: int(item[0])):
        evidence = evidence_by_fold[outer]
        cells = []
        for profile in SELECTABLE_PROFILES:
            row = evidence[profile]
            cells.append(
                f"{row['macro_normalized_gain_percent']:+.3f}% / "
                f"{row['worst_task_gain_percent']:+.3f}% / "
                f"{row['clear_positive_tasks']}"
            )
        lines.append(f"| {outer} | {PROFILE_NAMES[selected_profile]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Released comparison definitions",
        "",
        "- **No-cross-expert** retains the target-task anchor token and coarse "
        "pool-level reliability summaries, but not separate non-target expert "
        "tokens or the learned target-source relation.",
        "- **No safety control** emits the cross-fitted proposal ensemble directly.",
        "- **Stage-1 specialist-only** reproduces the target task's Stage-1 anchor.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mechanism_report(root: Path) -> None:
    variant = _candidate_variant(PRIMARY_PROPOSAL)
    selected_by_fold = {
        str(outer): str(read_json(_profile_lock_path(root, outer))["selected_safety_profile"])
        for outer in OUTER_FOLDS
    }
    data_exposure = np.zeros((13, 13), dtype=np.float64)
    relation_exposure = np.zeros((13, 13), dtype=np.float64)
    route_count = np.zeros((13, 3), dtype=np.float64)
    accepted_count = np.zeros(13, dtype=np.float64)
    task_certified = np.zeros(13, dtype=np.float64)
    sample_count = np.zeros(13, dtype=np.float64)
    for outer in OUTER_FOLDS:
        for task_number, task in enumerate(CANONICAL_TASKS):
            with np.load(
                root / f"outer_prediction/outer_{outer}/{variant}/{task}.npz",
                allow_pickle=False,
            ) as values:
                data_exposure[task_number] += values["data_attention"].sum(axis=0)
                relation_exposure[task_number] += values["relation_attention"].sum(axis=0)
                for route in range(3):
                    route_count[task_number, route] += np.sum(values["route_index"] == route)
                accepted_count[task_number] += np.sum(values["accepted"])
                task_certified[task_number] += float(values["task_certificate_accepted"][0])
                sample_count[task_number] += len(values["accepted"])
    data = data_exposure / sample_count[:, None]
    relation = relation_exposure / sample_count[:, None]
    routes = route_count / sample_count[:, None]
    relation_models: list[np.ndarray] = []
    for checkpoint in sorted(
        root.glob("proposal_training/outer_*/proposal_full/checkpoints/inner_*/seed_*.pt")
    ):
        payload = torch.load(checkpoint, map_location="cpu")
        state = payload["state_dict"]
        raw_relation = np.asarray(state["task_source_relation"], dtype=np.float64)
        raw_strength = float(np.asarray(state["learned_relation_strength"]))
        relation_models.append(np.logaddexp(0.0, raw_strength) * np.tanh(raw_relation))
    if not relation_models:
        raise FileNotFoundError("No primary HP-SafeMoE learned-relation checkpoints found")
    relation_stack = np.stack(relation_models)
    relation_mean = relation_stack.mean(axis=0)
    relation_std = relation_stack.std(axis=0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "selected_safety_profile_by_outer_fold": selected_by_fold,
        "profile_selection_scope": "fold_local_outer_train_oof",
        "profile_selection_data_scope": "outer_train_oof",
        "selected_variant": variant,
        "task_order": list(CANONICAL_TASKS),
        "expert_source_task_order": list(CANONICAL_TASKS),
        "route_order": ["anchor_fallback", "data", "learned_relation"],
        "route_fraction": routes.tolist(),
        "acceptance_fraction": (accepted_count / sample_count).tolist(),
        "task_certified_outer_fold_fraction": (task_certified / len(OUTER_FOLDS)).tolist(),
        "data_attention_exposure": data.tolist(),
        "learned_relation_attention_exposure": relation.tolist(),
        "learned_task_source_relation_model_count": len(relation_models),
        "learned_task_source_relation_mean": relation_mean.tolist(),
        "learned_task_source_relation_std": relation_std.tolist(),
        "mechanism_data_scope": "committed_predictions_and_outer_train_oof",
        "task_routing_policy": "learned_shared_rules",
    }
    atomic_json(root / "reports/stage2_mechanism.json", report)
    lines = [
        "# HP-SafeMoE Stage 2: learned sharing mechanism",
        "",
        "Each outer fold uses its OOF-selected safety profile: "
        f"{selected_by_fold}. The mechanism statistics summarize the committed predictions.",
        "",
        "| Task | Accepted | Certified folds | Sample-conditioned route | Relation-augmented route | "
        "Top Sample-conditioned sources | Top Relation-augmented sources | Top learned relations |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for index, task in enumerate(CANONICAL_TASKS):
        data_order = np.argsort(-data[index])[:3]
        relation_order = np.argsort(-relation[index])[:3]
        data_text = ", ".join(
            f"{CANONICAL_TASKS[item]}={data[index, item]:.3f}"
            for item in data_order
            if data[index, item] > 0.0
        )
        relation_text = ", ".join(
            f"{CANONICAL_TASKS[item]}={relation[index, item]:.3f}"
            for item in relation_order
            if relation[index, item] > 0.0
        )
        relation_order = [item for item in np.argsort(-relation_mean[index]) if item != index][:3]
        relation_text = ", ".join(
            f"{CANONICAL_TASKS[item]}={relation_mean[index, item]:+.3f}" for item in relation_order
        )
        lines.append(
            f"| {task} | {accepted_count[index] / sample_count[index]:.3f} | "
            f"{task_certified[index]:.0f}/5 | {routes[index, 1]:.3f} | "
            f"{routes[index, 2]:.3f} | {data_text or '—'} | {relation_text or '—'} | "
            f"{relation_text or '—'} |"
        )
    (root / "reports/stage2_mechanism.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

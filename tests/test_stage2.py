from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from hpsafe_sota.stage2.experiment import (
    ALL_STORAGE_VARIANTS,
    ARCHITECTURE_CANDIDATES,
    SAFETY_PROFILES,
    SELECTABLE_PROFILES,
    _apply_calibration,
    _apply_parameters,
    _benefit_probability,
    _calibrate_task,
    _ensemble_records,
    _fit_benefit_gate,
    _oof_profile_selection_key,
    load_config,
    lock_profile_for_outer_fold,
    validate_profile_lock,
)
from hpsafe_sota.stage2.model import (
    CrossExpertProposalModel,
    Stage2ProposalConfig,
)
from hpsafe_sota.stage2_common.data import FoldInputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/stage2.yaml"


def _model(
    *,
    cross: bool = True,
    data: bool = True,
    relation: bool = True,
    learned_relation: bool = True,
):
    model = CrossExpertProposalModel(
        Stage2ProposalConfig(
            task_names=("a", "b"),
            expert_ids=("ea", "eb", "ec"),
            hidden_dims={"ea": 3, "eb": 4, "ec": 2},
            anchor_indices=(0, 1),
            token_dim=8,
            task_dim=4,
            residual_hidden=12,
            residual_rank=3,
            dropout=0.0,
            attention_top_k=2,
            residual_limit=1.0,
            route_temperature=0.75,
            use_cross_expert_input=cross,
            use_data_branch=data,
            use_relation_branch=relation,
            use_learned_task_source_relation=learned_relation,
            use_shared_low_rank_basis=False,
            use_task_private_correction=True,
            use_sample_gate=True,
        )
    )
    return model


def _batch(size: int = 8) -> dict[str, object]:
    return {
        "task_index": torch.zeros(size, dtype=torch.long),
        "anchor_standard": torch.linspace(-1, 1, size),
        "hidden": {
            "ea": torch.randn(size, 3),
            "eb": torch.randn(size, 4),
            "ec": torch.randn(size, 2),
        },
        "prediction_standard": {name: torch.randn(size) for name in ("ea", "eb", "ec")},
        "uncertainty_standard": {name: torch.zeros(size) for name in ("ea", "eb", "ec")},
        "available": torch.ones(size, 3, dtype=torch.bool),
    }


def _record(raw: np.ndarray, anchor: np.ndarray, offset: float = 0.0) -> dict[str, np.ndarray]:
    size = len(raw)
    probability = np.tile(np.asarray([[0.0, 0.55, 0.45]], dtype=np.float32), (size, 1))
    attention = np.tile(np.asarray([[0.0, 0.6, 0.4]], dtype=np.float32), (size, 1))
    return {
        "sample_ids": np.asarray([f"s{i}" for i in range(size)]),
        "anchor_prediction": anchor.astype(np.float64),
        "raw_prediction": (raw + offset).astype(np.float64),
        "data_prediction": (raw + 0.01 + offset).astype(np.float64),
        "relation_prediction": (raw - 0.01 + offset).astype(np.float64),
        "route_probabilities": probability,
        "route_index": np.ones(size, dtype=np.int8),
        "data_attention": attention,
        "relation_attention": attention[:, ::-1].copy(),
        "reliability": np.tile(np.linspace(0.0, 1.0, 8), (size, 1)).astype(np.float32),
    }


def test_hpsafemoe_protocol_is_true_oof_and_crossfit_ensemble() -> None:
    config = load_config(CONFIG_PATH)
    assert config["inner_oof_folds"] == 5
    assert config["true_inner_oof_proposals"] is True
    assert config["crossfit_ensemble_deployment"] is True
    assert config["proposal_fit_scope"] == "matching_outer_train_inner_crossfit"
    assert config["evaluation_protocol"] == "official_matbench_five_fold"
    assert config["same_outer_fold_stage1_stage2"] is True
    assert config["stage2_artifact_source"] == "current_run"
    assert config["learned_task_source_relation"] is True
    assert config["task_private_residual_output"] is True
    assert config["profile_selection_scope"] == "fold_local_outer_train_oof"
    assert config["profile_selection_policy"] == "fold_local_global_profile"
    assert config["profile_lock_before_outer_test_prediction"] is True
    primary = ARCHITECTURE_CANDIDATES["proposal_full"]
    assert primary["use_learned_task_source_relation"] is True
    assert primary["use_shared_low_rank_basis"] is False
    assert primary["use_task_private_correction"] is True
    assert config["input_validation"] == "schema_alignment_and_manifest"
    assert tuple(ARCHITECTURE_CANDIDATES) == ("proposal_full", "ablation_no_cross_expert")
    assert len(SAFETY_PROFILES) == 3
    assert len(SELECTABLE_PROFILES) == 3
    assert len(ALL_STORAGE_VARIANTS) == 4
    assert all(profile not in variant for variant in ALL_STORAGE_VARIANTS for profile in SAFETY_PROFILES)


def test_proposal_routes_cross_expert_inputs_through_learned_branches() -> None:
    model, batch = _model(), _batch()
    output = model(**batch)
    assert output.raw_standard.shape == (8,)
    assert torch.all(output.route_one_hot[:, 0] == 0)
    assert torch.allclose(output.route_one_hot.sum(dim=1), torch.ones(8))
    assert torch.all(output.data_attention[:, 0] == 0)
    assert torch.all(output.relation_attention[:, 0] == 0)


def test_anchor_route_handles_unavailable_cross_expert_input() -> None:
    model, batch = _model(), _batch()
    batch["available"][:, 1:] = False
    output = model(**batch)
    assert torch.all(output.route_one_hot[:, 0] == 1)
    assert torch.allclose(output.raw_standard, batch["anchor_standard"], atol=1.0e-7)


def test_no_cross_expert_control_attends_only_to_anchor() -> None:
    output = _model(cross=False)(**_batch())
    assert torch.all(output.data_attention[:, 1:] == 0)
    assert torch.all(output.relation_attention[:, 1:] == 0)
    assert torch.allclose(output.data_attention[:, 0], torch.ones(8))


def test_no_cross_expert_retains_coarse_pool_reliability_summaries() -> None:
    model = _model(cross=False)
    batch = _batch()
    original = model(**batch)
    shifted = {key: value.clone() for key, value in batch["prediction_standard"].items()}
    shifted["eb"] = shifted["eb"] + 10.0
    modified = model(**{**batch, "prediction_standard": shifted})
    assert torch.all(original.data_attention[:, 1:] == 0)
    assert torch.all(modified.data_attention[:, 1:] == 0)
    assert not torch.allclose(original.reliability, modified.reliability)


def test_branch_ablation_removes_route() -> None:
    assert torch.all(_model(relation=False)(**_batch()).route_one_hot[:, 2] == 0)
    assert torch.all(_model(data=False)(**_batch()).route_one_hot[:, 1] == 0)


def test_task_source_relation_is_learned_from_neutral_without_fixed_prior() -> None:
    learned = _model(learned_relation=True)
    assert learned.task_source_relation is not None
    assert torch.count_nonzero(learned.task_source_relation).item() == 0
    control = _model(learned_relation=False)
    assert control.task_source_relation is None


def test_benefit_gate_learns_target_free_separator() -> None:
    config = load_config(CONFIG_PATH)
    x = np.column_stack([np.linspace(-2.0, 2.0, 200), np.ones(200)])
    y = (x[:, 0] > 0.0).astype(float)
    model = _fit_benefit_gate(features=x, labels=y, seed=7, config=config)
    probability = _benefit_probability(model, x)
    assert probability[y == 1].mean() > probability[y == 0].mean() + 0.5


def test_true_oof_calibration_accepts_consistent_gain() -> None:
    config = load_config(CONFIG_PATH)
    y = np.linspace(-1.0, 1.0, 200)
    anchor = np.zeros(200)
    records = [_record(y, anchor, -0.01), _record(y, anchor, 0.01)]
    calibration = _calibrate_task(records=records, y_true=y, task_type="regression", config=config, seed=11)
    assert calibration["certificate"]["point_gain_percent"] > 50.0
    assert calibration["profile_acceptance"]["balanced"] is True
    result = _apply_calibration(records, calibration=calibration, profile="balanced")
    assert np.mean(np.abs(result["prediction"] - y)) < np.mean(np.abs(anchor - y))


def test_true_oof_calibration_rejects_harmful_gain() -> None:
    config = load_config(CONFIG_PATH)
    y = np.linspace(-1.0, 1.0, 200)
    anchor = y.copy()
    records = [_record(-y, anchor, -0.01), _record(-y, anchor, 0.01)]
    calibration = _calibrate_task(records=records, y_true=y, task_type="regression", config=config, seed=13)
    assert calibration["profile_acceptance"]["conservative"] is False
    assert calibration["profile_acceptance"]["balanced"] is False
    result = _apply_calibration(records, calibration=calibration, profile="balanced")
    assert np.array_equal(result["prediction"], anchor)
    assert not np.any(result["accepted"])


def test_residual_scale_is_applied_in_restored_output_space() -> None:
    anchor = np.asarray([0.1, 0.8], dtype=np.float64)
    raw = np.asarray([0.5, 0.2], dtype=np.float64)
    prediction, accepted = _apply_parameters(
        anchor=anchor,
        raw=raw,
        probability=np.ones(2),
        alpha=0.25,
        threshold=None,
    )
    assert np.all(accepted)
    assert np.allclose(prediction, anchor + 0.25 * (raw - anchor))


def test_classification_members_are_restored_before_ensemble_and_safety_scaling() -> None:
    member_logits = np.asarray([[-4.0, 0.0], [2.0, 4.0]], dtype=np.float64)
    fold = object.__new__(FoldInputs)
    fold.task_types = {"classification_task": "classification"}
    member_probabilities = fold.restore_prediction("classification_task", member_logits)
    anchor = np.asarray([0.2, 0.6], dtype=np.float64)
    records = [_record(member_probability, anchor) for member_probability in member_probabilities]

    ensemble = _ensemble_records(records)
    expected_raw = member_probabilities.mean(axis=0)
    sigmoid_after_logit_averaging = 1.0 / (1.0 + np.exp(-member_logits.mean(axis=0)))
    assert np.allclose(ensemble["raw_prediction"], expected_raw)
    assert not np.allclose(ensemble["raw_prediction"], sigmoid_after_logit_averaging)

    prediction, accepted = _apply_parameters(
        anchor=anchor,
        raw=ensemble["raw_prediction"],
        probability=np.ones(2),
        alpha=0.25,
        threshold=None,
    )
    assert np.all(accepted)
    assert np.allclose(prediction, anchor + 0.25 * (expected_raw - anchor))


def test_oof_selection_prefers_safe_profile_and_uses_strict_tie_break() -> None:
    unsafe = {
        "worst_task_gain_percent": -2.0,
        "clear_positive_tasks": 12,
        "macro_normalized_gain_percent": 8.0,
        "positive_tasks": 12,
    }
    safe_many = {
        "worst_task_gain_percent": -0.8,
        "clear_positive_tasks": 9,
        "macro_normalized_gain_percent": 4.0,
        "positive_tasks": 10,
    }
    safe_fewer = {
        "worst_task_gain_percent": 0.0,
        "clear_positive_tasks": 8,
        "macro_normalized_gain_percent": 6.0,
        "positive_tasks": 11,
    }
    priority = {"conservative": 3, "balanced": 2, "liberal": 1}
    assert _oof_profile_selection_key(
        safe_many, profile="balanced", loss_limit=1.0, tie_priority=priority
    ) > _oof_profile_selection_key(unsafe, profile="liberal", loss_limit=1.0, tie_priority=priority)
    assert _oof_profile_selection_key(
        safe_many, profile="balanced", loss_limit=1.0, tie_priority=priority
    ) > _oof_profile_selection_key(safe_fewer, profile="conservative", loss_limit=1.0, tie_priority=priority)
    assert _oof_profile_selection_key(
        safe_many, profile="conservative", loss_limit=1.0, tie_priority=priority
    ) > _oof_profile_selection_key(safe_many, profile="liberal", loss_limit=1.0, tie_priority=priority)


def _calibration_payload(*, conservative: bool, balanced: bool, liberal: bool) -> dict:
    return {
        "fit_labels": "matching_official_outer_train_only",
        "outer_test_labels_accessed": False,
        "profile_acceptance": {
            "conservative": conservative,
            "balanced": balanced,
            "liberal": liberal,
        },
        "certificate": {
            "point_gain_percent": 2.0,
            "lower_gain_percent": 0.5,
            "upper_gain_percent": 3.0,
        },
        "meta_crossfit_positive_partitions": 5,
        "meta_crossfit_partition_gains_percent": [2.0] * 5,
        "meta_crossfit_coverage": 0.5,
    }


def test_profile_lock_is_fold_local_oof_only_and_precedes_prediction(monkeypatch) -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_policy = root / "source_policy.json"
        source_policy.write_text("{}\n", encoding="utf-8")
        run_root = root / "proposal_training/outer_0/proposal_full"
        calibration_root = run_root / "calibration"
        calibration_root.mkdir(parents=True)
        receipt = {
            "schema_version": "hpsafemoe",
            "status": "true_inner_oof_calibration_complete_before_outer_test_access",
            "candidate": "proposal_full",
            "outer_fold": 0,
            "outer_test_features_accessed": False,
            "outer_test_labels_accessed": False,
            "profile_selection_eligible": True,
        }
        (run_root / "calibration_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        for index, task in enumerate(
            (
                "steels",
                "expt_gap",
                "glass",
                "expt_is_metal",
                "jdft2d",
                "dielectric",
                "log_kvrh",
                "log_gvrh",
                "perovskites",
                "phonons",
                "mp_gap",
                "mp_is_metal",
                "mp_e_form",
            )
        ):
            payload = _calibration_payload(
                conservative=index < 4,
                balanced=index < 8,
                liberal=True,
            )
            (calibration_root / f"{task}.json").write_text(json.dumps(payload), encoding="utf-8")
        config = {
            "oof_profile_selection_clear_gain_percent": 0.1,
            "oof_profile_selection_max_worst_loss_percent": 1.0,
            "oof_profile_tie_break_order": ["conservative", "balanced", "liberal"],
        }
        monkeypatch.setattr(
            "hpsafe_sota.stage2.experiment.validate_source_policy",
            lambda _: {"config_payload": config},
        )
        lock = lock_profile_for_outer_fold(
            source_policy_path=source_policy,
            output_root=root,
            outer_fold=0,
        )
        assert lock["selected_safety_profile"] == "liberal"
        assert lock["profile_selection_data_scope"] == "outer_train_oof"
        assert lock["locked_before_outer_test_prediction"] is True
        lock_path = root / "policy/profile_locks/outer_0.json"
        assert validate_profile_lock(lock_path, source_policy_path=source_policy, outer_fold=0) == lock

        prediction_dir = root / "outer_prediction/outer_0/hpsafemoe__oof_locked"
        prediction_dir.mkdir(parents=True)
        np.savez(prediction_dir / "fake.npz", prediction=np.asarray([0.0]))
        with pytest.raises(RuntimeError, match="before outer-test prediction"):
            lock_profile_for_outer_fold(
                source_policy_path=source_policy,
                output_root=root,
                outer_fold=0,
            )


def test_profile_lock_digest_rejects_post_selection_mutation(monkeypatch) -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_policy = root / "source_policy.json"
        source_policy.write_text("{}\n", encoding="utf-8")
        lock_path = root / "lock.json"
        payload = {
            "schema_version": "hpsafemoe",
            "status": "profile_locked_from_fold_local_outer_train_oof",
            "outer_fold": 0,
            "selected_safety_profile": "balanced",
            "selection_scope": "one_profile_for_all_13_tasks_within_this_outer_fold",
            "selection_evidence_scope": "matching_official_outer_train_true_inner_oof_only",
            "locked_before_outer_test_prediction": True,
            "profile_selection_data_scope": "outer_train_oof",
            "source_policy": str(source_policy.resolve()),
            "selection_digest": "tampered",
        }
        lock_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="digest mismatch"):
            validate_profile_lock(lock_path, source_policy_path=source_policy, outer_fold=0)


def test_hpsafemoe_source_uses_declared_cache_validation_policy() -> None:
    source = (PROJECT_ROOT / "src/hpsafe_sota/stage2/experiment.py").read_text()
    source += (PROJECT_ROOT / "src/hpsafe_sota/stage2/model.py").read_text()
    assert "verify_hashes=False" in source
    assert '"input_validation": "schema_alignment_and_manifest"' in source


def test_source_declares_learned_shared_task_routing() -> None:
    source = (PROJECT_ROOT / "src/hpsafe_sota/stage2/experiment.py").read_text()
    assert '"task_routing_policy": "learned_shared_rules"' in source


def test_summary_uses_locked_fold_local_profiles() -> None:
    source = (PROJECT_ROOT / "src/hpsafe_sota/stage2/experiment.py").read_text()
    summary_body = source.split("def summarize(*,", 1)[1].split("def _fmt(", 1)[0]
    assert '"selection_policy": "fold_local_global_profile"' in summary_body
    assert '"profile_selection_data_scope": "outer_train_oof"' in summary_body


def test_rendered_graph_locks_profiles_before_all_test_prediction_jobs() -> None:
    script = PROJECT_ROOT / "scripts/render_stage2.py"
    spec = importlib.util.spec_from_file_location("render_stage2", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        feature_root = root / "features"
        feature_root.mkdir()
        graph = module.render(
            feature_root=feature_root,
            output_root=root / "output",
            config_path=CONFIG_PATH,
            resource_config=PROJECT_ROOT / "configs/server_resources_h20_8gpu_stage2.yaml",
            gpu_lanes=list(range(8)),
            environment="hpsafe-stage2",
        )
    jobs = graph["jobs"]
    gpu_jobs = [job for job in jobs if job["gpu_lane"] is not None]
    barrier = next(
        job for job in jobs if job["job_id"].endswith("commit_all_oof_locked_target_free_predictions")
    )
    locks = [job for job in jobs if job["kind"] == "stage2_fold_local_oof_profile_lock"]
    predictions = [
        job for job in jobs if job["kind"] == "stage2_oof_locked_crossfit_ensemble_predict"
    ]
    assert len(jobs) == 33
    assert len(gpu_jobs) == 20
    assert len(locks) == 5
    assert len(predictions) == 10
    assert len(barrier["dependencies"]) == 10
    lock_ids = {job["outer_fold"]: job["job_id"] for job in locks}
    assert all(lock_ids[job["outer_fold"]] in job["dependencies"] for job in predictions)
    assert graph["scientific_protocol"]["inner_oof_folds"] == 5
    assert graph["scientific_protocol"]["deployment_model"] == "inner_crossfit_ensemble"
    assert graph["scientific_protocol"]["profile_selection_data_scope"] == "fold_local_outer_train_oof"
    assert graph["scientific_protocol"]["profile_lock_before_outer_test_prediction"] is True
    assert graph["scientific_protocol"]["input_validation"] == "schema_alignment_and_manifest"

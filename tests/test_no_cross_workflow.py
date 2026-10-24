from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hpsafe_sota.stage2.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("no_cross_controller", ROOT / "scripts/run_no_cross.py")
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def test_released_profiles_revalidate_from_full_oof_evidence(tmp_path, monkeypatch):
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    config = load_config(ROOT / "configs/experiments/stage2.yaml")
    monkeypatch.setattr("hpsafe_sota.stage2.experiment.validate_source_policy", lambda _: {"config_payload": config})
    for outer, expected in enumerate(["balanced"] * 3 + ["conservative"] * 2):
        record = ROOT / f"results/matbench/experiment/profiles/outer_{outer}.json"
        path = tmp_path / f"outer_{outer}.json"
        result = controller.bind_profile(record, policy, outer, path)
        assert result["selected_safety_profile"] == expected
        assert result["profile_evidence"] == json.loads(record.read_text())["profile_evidence"]
    tampered = json.loads(path.read_text())
    tampered["selected_safety_profile"] = "liberal"
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="digest mismatch"):
        controller.validate_profile_lock(path, source_policy_path=policy, outer_fold=4)


def test_commit_requires_all_predictions_and_score_rejects_mutation(tmp_path, monkeypatch):
    """The commit succeeds with target-free manifests and rejects altered outputs before scoring."""
    policy = tmp_path / "policy.json"
    policy.write_text("{}")
    monkeypatch.setattr(controller, "validate_source_policy", lambda _: {"project_root": str(tmp_path)})
    monkeypatch.setattr(controller, "validate_profile_lock", lambda *a, **k: {
        "selection_digest": "fixed", "selected_safety_profile": "balanced"})
    manifests = tmp_path / "data/manifests"
    manifests.mkdir(parents=True)
    for task in controller.CANONICAL_TASKS:
        np.savez(manifests / f"{task}.npz", sample_ids=np.array(["s0", "s1"]),
                 **{f"outer_{outer}_test": np.array([0, 1]) for outer in range(5)})
    for outer in range(5):
        path = tmp_path / f"policy/profile_locks/outer_{outer}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        run = tmp_path / f"proposal_training/outer_{outer}/{controller.CANDIDATE}"
        run.mkdir(parents=True)
        (run / "calibration_receipt.json").write_text(json.dumps({
            "status": "true_inner_oof_calibration_complete_before_outer_test_access",
            "outer_test_labels_accessed": False}))
        (run / "prediction_receipt.json").write_text(json.dumps({
            "profile_lock_digest": "fixed", "proposal_model_count": 10}))
        for task in controller.CANONICAL_TASKS:
            path = tmp_path / f"outer_prediction/outer_{outer}/{controller.VARIANT}/{task}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, sample_ids=np.array(["s0", "s1"]), prediction=np.array([0.1, 0.7]))
    saved = path.read_bytes()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        controller.commit(tmp_path, policy)
    assert not (tmp_path / "policy/no_cross_prediction_commit.json").exists()
    path.write_bytes(saved)
    record = controller.commit(tmp_path, policy)
    assert len(record["predictions"]) == 65
    np.savez(path, sample_ids=np.array(["s0", "s1"]), prediction=np.array([0.2, 0.7]))
    with pytest.raises(ValueError, match="have changed"):
        controller.score(tmp_path, policy)
    assert not (tmp_path / "policy/scoring_started.json").exists()

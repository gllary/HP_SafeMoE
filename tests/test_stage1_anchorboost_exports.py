from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hpsafe_sota.experts.native_models import NativeDescriptorMLP
from hpsafe_sota.stage1_anchorboost.anchor_replay import (
    HIDDEN_DIM,
    LabelFreeSplitManifest,
    _load_npz_without_targets,
    predict_checkpoint_payload,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DummyRegressor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.value, dtype=np.float64)


def _renderer():
    path = PROJECT_ROOT / "scripts/render_stage1_anchorboost_exports.py"
    spec = importlib.util.spec_from_file_location(
        "render_stage1_anchorboost_exports", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_anchorboost_export_graph_has_five_replays_fifteen_outputs_and_no_training(
    tmp_path: Path,
) -> None:
    graph = _renderer().render(
        output_root=tmp_path / "run",
        stage1_root=tmp_path / "stage1",
        export_root=tmp_path / "exports",
    )
    assert graph["protocol"] == "stage1_anchorboost_aligned_hidden_replay"
    assert graph["tasks"] == ["glass"]
    assert graph["expected_exports"] == 15
    assert graph["stage1_retraining"] is False
    assert graph["stage2_launch"] is False
    counts = Counter(job["kind"] for job in graph["jobs"])
    assert len(graph["jobs"]) == 7
    assert counts == {
        "stage1_anchorboost_export_preflight": 1,
        "stage1_anchorboost_checkpoint_replay": 5,
        "stage1_anchorboost_export_validation": 1,
    }
    replay = [
        job for job in graph["jobs"] if job["kind"] == "stage1_anchorboost_checkpoint_replay"
    ]
    assert {job["gpu_lane"] for job in replay} == {0, 1, 2, 3}
    assert all(job["estimated_gpu_memory_gb"] == 8 for job in replay)
    commands = "\n".join(" ".join(job["command"]) for job in graph["jobs"])
    assert "export_stage1_anchorboost_aligned.py" in commands
    assert "hpsafe_sota.experts.runner" not in commands
    assert "train_safe_moe.py" not in commands


def test_label_free_npz_loader_does_not_materialize_y_true(tmp_path: Path) -> None:
    path = tmp_path / "predictions.npz"
    np.savez_compressed(
        path,
        sample_ids=np.asarray(["a", "b"]),
        y_true=np.asarray([object(), object()], dtype=object),
        y_pred=np.asarray([0.1, 0.2]),
        y_uncertainty=np.asarray([0.01, 0.02]),
    )
    loaded = _load_npz_without_targets(path)
    assert set(loaded) == {"sample_ids", "y_pred", "y_uncertainty"}
    assert loaded["sample_ids"].tolist() == ["a", "b"]
    assert np.allclose(loaded["y_pred"], [0.1, 0.2])


def test_label_free_manifest_does_not_materialize_target_array(tmp_path: Path) -> None:
    path = tmp_path / "manifest.npz"
    values: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(
            [json.dumps({"definition": {"task_name": "tiny", "n_samples": 5}})]
        ),
        "semantic_digest": np.asarray(["a" * 64]),
        "sample_ids": np.asarray([f"sample-{index}" for index in range(5)]),
        "targets": np.asarray([object() for _ in range(5)], dtype=object),
    }
    for outer in range(5):
        test = np.asarray([outer], dtype=np.int64)
        train = np.asarray([value for value in range(5) if value != outer], dtype=np.int64)
        values[f"outer_{outer}_test"] = test
        values[f"outer_{outer}_train"] = train
        values[f"holdout_{outer}_val"] = train[:1]
    np.savez_compressed(path, **values)
    manifest = LabelFreeSplitManifest(path)
    manifest.require_valid()
    assert manifest.task_name == "tiny"
    assert manifest.digest == "a" * 64
    assert not hasattr(manifest, "targets")


def test_checkpoint_payload_reconstructs_native_hidden_without_refit() -> None:
    model = NativeDescriptorMLP(
        3,
        hidden_dim=4,
        latent_dim=HIDDEN_DIM,
        layers=1,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    payload = {
        "forest": _DummyRegressor(7.0),
        "xgb": _DummyRegressor(9.0),
        "hurdle": None,
        "mlp_state": model.state_dict(),
        "mlp_architecture": {
            "input_dim": 3,
            "hidden_dim": 4,
            "layers": 1,
            "dropout": 0.0,
            "latent_dim": HIDDEN_DIM,
        },
        "target_mean": 2.5,
        "target_scale": 3.0,
        "blend_weights": np.asarray([0.0, 0.0, 1.0]),
        "candidate_names": ["xgboost", "extra_trees", "residual_mlp"],
    }
    tree = np.zeros((5, 3), dtype=np.float32)
    neural = np.zeros((5, 3), dtype=np.float32)
    prediction, uncertainty, hidden = predict_checkpoint_payload(
        payload,
        tree_features=tree,
        neural_features=neural,
        classification=False,
        device=torch.device("cpu"),
    )
    assert np.allclose(prediction, 2.5)
    assert uncertainty.shape == (5,)
    assert hidden.shape == (5, HIDDEN_DIM)
    assert np.allclose(hidden, 0.0)

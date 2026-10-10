from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from hpsafe_sota.experts.native_models import NativeDescriptorMLP
from hpsafe_sota.stage2_common.cache import write_expert_export
from hpsafe_sota.stage2_common.io import read_json, sha256_file
from hpsafe_sota.stage2_common.protocol import ExpertSpec
from hpsafe_sota.stage1_anchorboost.cache import AnchorBoostDescriptorCache

TASKS = ("glass",)
HIDDEN_DIM = 512
PROVIDER = "anchorboost"


class LabelFreeSplitManifest:
    """Read immutable IDs/splits without materializing the manifest target array."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._arrays: dict[str, np.ndarray] = {}
        with np.load(self.path, allow_pickle=False) as loaded:
            files = set(loaded.files)
            required = {"metadata_json", "semantic_digest", "sample_ids"}
            required.update(
                f"outer_{outer}_{role}"
                for outer in range(5)
                for role in ("train", "test")
            )
            required.update(f"holdout_{outer}_val" for outer in range(5))
            missing = sorted(required - files)
            if missing:
                raise ValueError(
                    "Label-free replay requires pre-materialized Stage1 splits; "
                    f"missing={missing}"
                )
            self.metadata = json.loads(str(loaded["metadata_json"][0]))
            self.declared_digest = str(loaded["semantic_digest"][0])
            for key in sorted(required - {"metadata_json", "semantic_digest"}):
                self._arrays[key] = np.asarray(loaded[key]).copy()

    @property
    def task_name(self) -> str:
        return str(self.metadata["definition"]["task_name"])

    @property
    def sample_ids(self) -> np.ndarray:
        return self._arrays["sample_ids"].astype(str)

    @property
    def digest(self) -> str:
        return self.declared_digest

    def positions(self, role: str, outer: int, inner: int | None = None) -> np.ndarray:
        if inner is not None or outer not in range(5):
            raise ValueError("Label-free replay supports only outer folds 0..4")
        if role == "outer_train":
            return self._arrays[f"outer_{outer}_train"].copy()
        if role == "outer_test":
            return self._arrays[f"outer_{outer}_test"].copy()
        if role == "holdout_val":
            return self._arrays[f"holdout_{outer}_val"].copy()
        if role == "holdout_train":
            outer_train = self._arrays[f"outer_{outer}_train"]
            validation = set(self._arrays[f"holdout_{outer}_val"].tolist())
            return np.asarray(
                [value for value in outer_train if value not in validation],
                dtype=np.int64,
            )
        raise ValueError(f"Unsupported label-free split role: {role}")

    def ids(self, role: str, outer: int, inner: int | None = None) -> np.ndarray:
        return self.sample_ids[self.positions(role, outer, inner)]

    def require_valid(self) -> None:
        n_samples = int(self.metadata["definition"]["n_samples"])
        if len(self.sample_ids) != n_samples or len(set(self.sample_ids.tolist())) != n_samples:
            raise ValueError(f"{self.task_name}: invalid sample IDs")
        universe = set(range(n_samples))
        outer_tests: list[int] = []
        for outer in range(5):
            train = self.positions("outer_train", outer)
            test = self.positions("outer_test", outer)
            holdout_train = self.positions("holdout_train", outer)
            holdout_val = self.positions("holdout_val", outer)
            if set(train) & set(test) or set(train) | set(test) != universe:
                raise ValueError(f"{self.task_name}/outer_{outer}: invalid outer split")
            if (
                set(holdout_train) & set(holdout_val)
                or set(holdout_train) | set(holdout_val) != set(train)
                or set(holdout_val) & set(test)
            ):
                raise ValueError(f"{self.task_name}/outer_{outer}: invalid holdout split")
            outer_tests.extend(test.tolist())
        if sorted(outer_tests) != list(range(n_samples)):
            raise ValueError(f"{self.task_name}: outer tests are not an exact partition")


@dataclass(frozen=True)
class LabelFreeArtifact:
    root: Path
    sample_ids: np.ndarray
    prediction: np.ndarray
    uncertainty: np.ndarray
    hidden: np.ndarray
    available: np.ndarray
    checkpoint: Path
    checkpoint_sha256: str
    preprocessor: Path
    preprocessor_sha256: str
    preprocessing_manifest: Path
    preprocessing_manifest_sha256: str
    metadata_sha256: str
    selection_receipt_sha256: str | None


def expert_id(task: str) -> str:
    if task not in TASKS:
        raise ValueError(f"Unsupported AnchorBoost task: {task}")
    return f"anchorboost_{task}_anchor"


def expert_spec(export_root: Path, task: str) -> ExpertSpec:
    return ExpertSpec(
        expert_id=expert_id(task),
        source_task=task,
        provider=PROVIDER,
        hidden_dim=HIDDEN_DIM,
        export_root=(export_root / expert_id(task)).resolve(),
        input_modality="enriched_composition_descriptors",
        representation_kind="task_trained_hidden",
    )


def _expected_ids(
    manifest: LabelFreeSplitManifest, role: str, outer_fold: int
) -> np.ndarray:
    return manifest.ids(role, outer_fold).astype(str)


def _load_npz_without_targets(path: Path) -> dict[str, np.ndarray]:
    """Load only label-free arrays even when the source archive also stores y_true."""

    with np.load(path, allow_pickle=False) as loaded:
        result = {
            "sample_ids": loaded["sample_ids"].astype(str),
            "y_pred": np.asarray(loaded["y_pred"], dtype=np.float64),
        }
        if "y_uncertainty" in loaded.files:
            result["y_uncertainty"] = np.asarray(
                loaded["y_uncertainty"], dtype=np.float64
            )
    return result


def load_label_free_artifact(
    root: Path,
    *,
    manifest: LabelFreeSplitManifest,
    outer_fold: int,
    expected_role: str,
) -> LabelFreeArtifact:
    root = root.resolve()
    metadata_path = root / "metadata.json"
    metadata = read_json(metadata_path)
    if metadata.get("task", {}).get("task_name") != manifest.task_name:
        raise ValueError(f"Stage1 task identity mismatch: {root}")
    split = metadata.get("split", {})
    if split.get("outer_fold") != outer_fold or split.get("role") != expected_role:
        raise ValueError(f"Stage1 split identity mismatch: {root}")
    if metadata.get("provenance", {}).get("task_manifest_digest") != manifest.digest:
        raise ValueError(f"Stage1 manifest digest mismatch: {root}")
    if metadata.get("selection", {}).get("outer_test_used") is not False:
        raise ValueError(f"Stage1 artifact declares outer-test selection: {root}")

    prediction_record = metadata.get("predictions", {})
    prediction_path = root / str(prediction_record.get("filename", ""))
    if sha256_file(prediction_path) != prediction_record.get("sha256"):
        raise ValueError(f"Stage1 prediction hash mismatch: {root}")
    prediction_arrays = _load_npz_without_targets(prediction_path)
    expected_ids = _expected_ids(manifest, expected_role, outer_fold)
    if not np.array_equal(prediction_arrays["sample_ids"], expected_ids):
        raise ValueError(f"Stage1 prediction IDs/order mismatch: {root}")

    latent_record = metadata.get("latents")
    if not isinstance(latent_record, dict):
        raise ValueError(f"Stage1 artifact has no trained hidden representation: {root}")
    latent_path = root / str(latent_record.get("filename", ""))
    if sha256_file(latent_path) != latent_record.get("sha256"):
        raise ValueError(f"Stage1 latent hash mismatch: {root}")
    with np.load(latent_path, allow_pickle=False) as loaded:
        latent_ids = loaded["sample_ids"].astype(str)
        hidden = np.asarray(loaded["latents"], dtype=np.float32)
        available = np.asarray(loaded["available"], dtype=np.bool_)
    if not np.array_equal(latent_ids, expected_ids):
        raise ValueError(f"Stage1 latent IDs/order mismatch: {root}")
    if hidden.shape != (len(expected_ids), HIDDEN_DIM):
        raise ValueError(f"Stage1 latent shape mismatch: {hidden.shape}")
    if available.shape != (len(expected_ids),) or not available.all():
        raise ValueError(f"Promoted Stage1 expert has unavailable latent rows: {root}")
    prediction = prediction_arrays["y_pred"]
    uncertainty = prediction_arrays.get(
        "y_uncertainty", np.zeros(len(expected_ids), dtype=np.float64)
    )
    if (
        prediction.shape != (len(expected_ids),)
        or uncertainty.shape != (len(expected_ids),)
        or not np.all(np.isfinite(prediction))
        or not np.all(np.isfinite(uncertainty))
        or not np.all(np.isfinite(hidden))
    ):
        raise ValueError(f"Stage1 label-free arrays are invalid: {root}")

    checkpoint = root / "anchorboost.joblib"
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != metadata.get("provenance", {}).get("checkpoint_sha256"):
        raise ValueError(f"Stage1 checkpoint hash mismatch: {root}")
    preprocessing_manifest = root / "preprocessing_manifest.json"
    preprocessing_hash = sha256_file(preprocessing_manifest)
    if preprocessing_hash != metadata.get("provenance", {}).get(
        "preprocessing_manifest_sha256"
    ):
        raise ValueError(f"Stage1 preprocessing manifest hash mismatch: {root}")
    preprocessing = read_json(preprocessing_manifest)
    if preprocessing.get("target_used_for_features") is not False:
        raise ValueError(f"Stage1 preprocessing used targets for features: {root}")
    preprocessor = root / "feature_preprocessor.joblib"
    if not preprocessor.is_file():
        raise FileNotFoundError(preprocessor)

    return LabelFreeArtifact(
        root=root,
        sample_ids=expected_ids,
        prediction=prediction,
        uncertainty=uncertainty,
        hidden=hidden,
        available=available,
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        preprocessor=preprocessor,
        preprocessor_sha256=sha256_file(preprocessor),
        preprocessing_manifest=preprocessing_manifest,
        preprocessing_manifest_sha256=preprocessing_hash,
        metadata_sha256=sha256_file(metadata_path),
        selection_receipt_sha256=metadata.get("selection", {}).get(
            "selection_receipt_sha256"
        ),
    )


def _tree_prediction(model: Any, features: np.ndarray, classification: bool) -> np.ndarray:
    value = (
        model.predict_proba(features)[:, 1]
        if classification
        else model.predict(features)
    )
    return np.asarray(value, dtype=np.float64).reshape(-1)


def predict_checkpoint_payload(
    payload: dict[str, Any],
    *,
    tree_features: np.ndarray,
    neural_features: np.ndarray,
    classification: bool,
    batch_size: int = 2048,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    architecture = payload.get("mlp_architecture", {})
    if int(architecture.get("latent_dim", -1)) != HIDDEN_DIM:
        raise ValueError("AnchorBoost checkpoint does not contain a 512d native latent")
    model = NativeDescriptorMLP(
        int(architecture["input_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        latent_dim=int(architecture["latent_dim"]),
        layers=int(architecture["layers"]),
        dropout=float(architecture["dropout"]),
    )
    model.load_state_dict(payload["mlp_state"], strict=True)
    runtime_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(runtime_device)
    model.eval()
    predictions: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    target_mean = float(payload["target_mean"])
    target_scale = float(payload["target_scale"])
    with torch.no_grad():
        for start in range(0, len(neural_features), batch_size):
            raw, latent = model(
                torch.from_numpy(neural_features[start : start + batch_size]).to(
                    runtime_device
                )
            )
            value = (
                torch.sigmoid(raw)
                if classification
                else raw * target_scale + target_mean
            )
            predictions.append(value.float().cpu().numpy())
            hidden_rows.append(latent.float().cpu().numpy())
    mlp_prediction = np.concatenate(predictions).astype(np.float64)
    hidden = np.concatenate(hidden_rows).astype(np.float32)

    candidates: dict[str, np.ndarray] = {
        "xgboost": _tree_prediction(payload["xgb"], tree_features, classification),
        "extra_trees": _tree_prediction(
            payload["forest"], tree_features, classification
        ),
        "residual_mlp": mlp_prediction,
    }
    hurdle = payload.get("hurdle")
    if hurdle is not None:
        p_metal = np.asarray(
            hurdle["metal"].predict_proba(tree_features)[:, 1], dtype=np.float64
        )
        positive_gap = np.maximum(
            np.asarray(hurdle["gap"].predict(tree_features), dtype=np.float64), 0.0
        )
        candidates["gap_hurdle"] = (1.0 - p_metal) * positive_gap
    names = [str(value) for value in payload["candidate_names"]]
    if any(name not in candidates for name in names):
        raise ValueError(f"Unknown AnchorBoost checkpoint candidate names: {names}")
    matrix = np.stack([candidates[name] for name in names], axis=1)
    weights = np.asarray(payload["blend_weights"], dtype=np.float64)
    if weights.shape != (len(names),) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("AnchorBoost checkpoint blend weights are invalid")
    prediction = matrix @ weights
    if classification:
        prediction = np.clip(prediction, 0.0, 1.0)
    uncertainty = matrix.std(axis=1)
    return prediction.astype(np.float64), uncertainty.astype(np.float64), hidden


def _features(
    cache: AnchorBoostDescriptorCache,
    positions: np.ndarray,
    preprocessor_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    preprocessors = joblib.load(preprocessor_path)
    raw = cache.rows(positions)
    tree = preprocessors["imputer"].transform(raw).astype(np.float32)
    neural = preprocessors["scaler"].transform(tree).astype(np.float32)
    return tree, neural


def _replay(
    source: LabelFreeArtifact,
    *,
    cache: AnchorBoostDescriptorCache,
    positions: np.ndarray,
    classification: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree, neural = _features(cache, positions, source.preprocessor)
    payload = joblib.load(source.checkpoint)
    return predict_checkpoint_payload(
        payload,
        tree_features=tree,
        neural_features=neural,
        classification=classification,
    )


def _assert_replay_matches(
    *,
    label: str,
    replay_prediction: np.ndarray,
    replay_hidden: np.ndarray,
    source: LabelFreeArtifact,
) -> dict[str, float]:
    prediction_delta = np.abs(replay_prediction - source.prediction)
    hidden_delta = np.abs(replay_hidden.astype(np.float64) - source.hidden.astype(np.float64))
    max_prediction_delta = float(prediction_delta.max(initial=0.0))
    max_hidden_delta = float(hidden_delta.max(initial=0.0))
    if not np.allclose(replay_prediction, source.prediction, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{label}: frozen checkpoint replay predictions differ; max={max_prediction_delta}"
        )
    if not np.allclose(replay_hidden, source.hidden, rtol=1e-5, atol=1e-5):
        raise ValueError(
            f"{label}: frozen checkpoint replay hidden differs; max={max_hidden_delta}"
        )
    return {
        "maximum_prediction_delta": max_prediction_delta,
        "maximum_hidden_delta": max_hidden_delta,
    }


def _producer(
    source: LabelFreeArtifact,
    *,
    checkpoint_role: str,
    checkpoint_fit_role: str,
    replay_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "adapter": "anchorboost_native_512d_checkpoint_replay",
        "checkpoint_role": checkpoint_role,
        "checkpoint_fit_role": checkpoint_fit_role,
        "source_artifact": str(source.root),
        "source_artifact_metadata_sha256": source.metadata_sha256,
        "feature_preprocessor": str(source.preprocessor),
        "feature_preprocessor_sha256": source.preprocessor_sha256,
        "preprocessing_manifest": str(source.preprocessing_manifest),
        "preprocessing_manifest_sha256": source.preprocessing_manifest_sha256,
        "selection_receipt_sha256": source.selection_receipt_sha256,
        "target_values_accessed_by_exporter": False,
        "source_target_arrays_loaded": False,
        "relation_screening": "none",
        "replay_receipt": replay_receipt,
    }


def export_fold(
    *,
    project_root: Path,
    stage1_root: Path,
    export_root: Path,
    task: str,
    outer_fold: int,
) -> dict[str, Any]:
    if task not in TASKS or outer_fold not in range(5):
        raise ValueError("AnchorBoost export requires glass and outer 0..4")
    project_root = project_root.resolve()
    stage1_root = stage1_root.resolve()
    export_root = export_root.resolve()
    manifest = LabelFreeSplitManifest(
        project_root / "data/manifests" / f"{task}.npz"
    )
    manifest.require_valid()
    cache = AnchorBoostDescriptorCache(project_root, task)
    if cache.receipt.get("manifest_digest") != manifest.digest:
        raise ValueError(f"{task}: AnchorBoost descriptor cache references a different manifest")
    classification = manifest.metadata["definition"]["task_type"] == "classification"
    base = stage1_root / "anchorboost" / task / f"outer_{outer_fold}"
    holdout = load_label_free_artifact(
        base / "validation",
        manifest=manifest,
        outer_fold=outer_fold,
        expected_role="holdout_val",
    )
    outer = load_label_free_artifact(
        base / "outer_test",
        manifest=manifest,
        outer_fold=outer_fold,
        expected_role="outer_test",
    )

    holdout_val_positions = manifest.positions("holdout_val", outer_fold)
    holdout_val_prediction, _, holdout_val_hidden = _replay(
        holdout,
        cache=cache,
        positions=holdout_val_positions,
        classification=classification,
    )
    holdout_replay = _assert_replay_matches(
        label=f"{task}/outer_{outer_fold}/holdout",
        replay_prediction=holdout_val_prediction,
        replay_hidden=holdout_val_hidden,
        source=holdout,
    )
    outer_positions = manifest.positions("outer_test", outer_fold)
    outer_prediction, _, outer_hidden = _replay(
        outer,
        cache=cache,
        positions=outer_positions,
        classification=classification,
    )
    outer_replay = _assert_replay_matches(
        label=f"{task}/outer_{outer_fold}/outer",
        replay_prediction=outer_prediction,
        replay_hidden=outer_hidden,
        source=outer,
    )

    train_positions = manifest.positions("holdout_train", outer_fold)
    train_prediction, train_uncertainty, train_hidden = _replay(
        holdout,
        cache=cache,
        positions=train_positions,
        classification=classification,
    )
    if train_hidden.shape != (len(train_positions), HIDDEN_DIM):
        raise ValueError("Train replay produced the wrong hidden shape")

    spec = expert_spec(export_root, task)
    role_values = {
        "train": (
            manifest.ids("holdout_train", outer_fold),
            train_hidden,
            train_prediction,
            train_uncertainty,
            holdout,
            _producer(
                holdout,
                checkpoint_role="holdout",
                checkpoint_fit_role="holdout_train",
                replay_receipt={
                    "checkpoint_replayed_on_source_artifact": True,
                    "source_comparison_role": "holdout_val",
                    **holdout_replay,
                },
            ),
        ),
        "val": (
            holdout.sample_ids,
            holdout.hidden,
            holdout.prediction,
            holdout.uncertainty,
            holdout,
            _producer(
                holdout,
                checkpoint_role="holdout",
                checkpoint_fit_role="holdout_train",
                replay_receipt={
                    "checkpoint_replayed_on_source_artifact": True,
                    "source_comparison_role": "holdout_val",
                    **holdout_replay,
                },
            ),
        ),
        "test": (
            outer.sample_ids,
            outer.hidden,
            outer.prediction,
            outer.uncertainty,
            outer,
            _producer(
                outer,
                checkpoint_role="outer",
                checkpoint_fit_role="outer_train",
                replay_receipt={
                    "checkpoint_replayed_on_source_artifact": True,
                    "source_comparison_role": "outer_test",
                    **outer_replay,
                },
            ),
        ),
    }
    outputs: dict[str, str] = {}
    for role, (ids, hidden, prediction, uncertainty, source, producer) in role_values.items():
        destination = spec.export_root / task / f"outer_{outer_fold}" / role
        write_expert_export(
            destination,
            manifest=manifest,
            spec=spec,
            outer_fold=outer_fold,
            role=role,
            sample_ids=np.asarray(ids).astype(str),
            hidden=np.asarray(hidden, dtype=np.float32),
            prediction=np.asarray(prediction, dtype=np.float64),
            uncertainty=np.asarray(uncertainty, dtype=np.float64),
            available=np.ones(len(ids), dtype=np.bool_),
            checkpoint_sha256=source.checkpoint_sha256,
            producer=producer,
        )
        outputs[role] = str(destination)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "task": task,
        "outer_fold": outer_fold,
        "outputs": outputs,
        "rows": {
            "train": len(train_positions),
            "val": len(holdout.sample_ids),
            "test": len(outer.sample_ids),
        },
        "hidden_dim": HIDDEN_DIM,
        "holdout_checkpoint_sha256": holdout.checkpoint_sha256,
        "outer_checkpoint_sha256": outer.checkpoint_sha256,
        "holdout_replay": holdout_replay,
        "outer_replay": outer_replay,
        "feature_data_scope": "target_free",
        "checkpoint_source": "supplied_stage1_checkpoint",
    }
    receipt_path = spec.export_root / task / f"outer_{outer_fold}" / "export_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "anchorboost_aligned_export_complete", **receipt}, sort_keys=True))
    return receipt

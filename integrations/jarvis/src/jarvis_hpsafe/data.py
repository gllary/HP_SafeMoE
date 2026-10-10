from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import ExpertRoleCache, cache_directory, validate_role_cache
from .constants import JARVIS_TASKS, LATENT_DIM, OFFICIAL_SPLIT, expert_id
from .manifest import JarvisManifest, load_manifests
from .stage1_artifact import align_artifact, load_stage1_artifact


@dataclass(frozen=True)
class ScalarStats:
    mean: float
    scale: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TaskStats:
    target: ScalarStats
    expert_prediction: dict[str, ScalarStats]
    expert_uncertainty: dict[str, ScalarStats]

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "expert_prediction": {
                key: value.to_dict() for key, value in self.expert_prediction.items()
            },
            "expert_uncertainty": {
                key: value.to_dict() for key, value in self.expert_uncertainty.items()
            },
        }


@dataclass(frozen=True)
class TaskRole:
    task: str
    role: str
    sample_ids: np.ndarray
    y_true: np.ndarray
    exports: dict[str, ExpertRoleCache]
    anchor_expert_id: str

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True)
class FrozenExpertPool:
    """Expert-pool interface consumed by the HP-SafeMoE proposal model."""

    expert_ids: tuple[str, ...]
    hidden_dims: dict[str, int]
    anchor_map: dict[str, str]

    def anchor_for(self, task: str) -> str:
        return self.anchor_map[task]


def _stats(values: np.ndarray) -> ScalarStats:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return ScalarStats(0.0, 1.0)
    scale = float(np.std(array))
    return ScalarStats(float(np.mean(array)), max(scale, 1e-8))


def _artifact_path(stage1_root: Path, task: str, role: str) -> Path:
    base = stage1_root / "outputs" / "exact" / "experts" / "kgcnn_cogn" / task / "outer_0"
    return base / ({"train": "inner_oof", "val": "official_validation", "test": "outer_test"}[role])


def _role_targets(stage1_root: Path, manifest: JarvisManifest, role: str) -> np.ndarray:
    # This is the only target reader. Callers control which role is opened.
    artifact = load_stage1_artifact(
        _artifact_path(stage1_root, manifest.task, role),
        expected_latent_dim=LATENT_DIM,
        load_targets=True,
    )
    if artifact.y_true is None:
        raise AssertionError("Requested Stage1 targets were not loaded")
    _, _, _ = align_artifact(artifact, manifest.ids(role))
    lookup = {value: index for index, value in enumerate(artifact.sample_ids.tolist())}
    order = np.asarray([lookup[value] for value in manifest.ids(role).tolist()], dtype=np.int64)
    return artifact.y_true[order]


class JarvisInputs:
    """Role-scoped JARVIS inputs with explicit feature and label roles.

    The object mirrors the Matbench Stage-2 interface while retaining the
    official one-holdout JARVIS split. Prediction loads test expert arrays from
    the feature role, and scoring loads targets from the label role.
    """

    def __init__(
        self,
        *,
        stage1_root: Path,
        cache_root: Path,
        roles_to_load: tuple[str, ...],
        frozen_stats: dict[str, Any] | None = None,
        verify_cache_hashes: bool = False,
        label_roles: tuple[str, ...] | None = None,
    ) -> None:
        if not roles_to_load or any(role not in {"train", "val", "test"} for role in roles_to_load):
            raise ValueError("roles_to_load must be a non-empty train/val/test subset")
        if len(set(roles_to_load)) != len(roles_to_load):
            raise ValueError("roles_to_load contains duplicates")
        if label_roles is None:
            label_roles = roles_to_load
        if any(role not in roles_to_load for role in label_roles):
            raise ValueError("label_roles must be a subset of roles_to_load")
        self.stage1_root = stage1_root.expanduser().resolve()
        self.cache_root = cache_root.expanduser().resolve()
        self.tasks = JARVIS_TASKS
        self.outer_fold = OFFICIAL_SPLIT
        self.task_types = {task: "regression" for task in self.tasks}
        self.task_index = {value: index for index, value in enumerate(self.tasks)}
        self.expert_ids = tuple(expert_id(task) for task in self.tasks)
        self.pool = FrozenExpertPool(
            expert_ids=self.expert_ids,
            hidden_dims={value: LATENT_DIM for value in self.expert_ids},
            anchor_map={task: expert_id(task) for task in self.tasks},
        )
        self.manifests = load_manifests(self.stage1_root, load_targets=False)
        self.roles: dict[str, dict[str, TaskRole]] = {}
        for task in self.tasks:
            manifest = self.manifests[task]
            self.roles[task] = {}
            for role in roles_to_load:
                exports = {
                    expert_id(source): validate_role_cache(
                        cache_directory(self.cache_root, source, task, role),
                        source_task=source,
                        target_manifest=manifest,
                        role=role,
                        verify_hashes=verify_cache_hashes,
                    )
                    for source in self.tasks
                }
                anchor = expert_id(task)
                if not np.asarray(exports[anchor].available, dtype=np.bool_).all():
                    raise ValueError(f"Required own-task coGN anchor unavailable: {task}/{role}")
                self.roles[task][role] = TaskRole(
                    task=task,
                    role=role,
                    sample_ids=manifest.ids(role),
                    y_true=(
                        _role_targets(self.stage1_root, manifest, role)
                        if role in label_roles
                        else np.zeros(len(manifest.ids(role)), dtype=np.float64)
                    ),
                    exports=exports,
                    anchor_expert_id=anchor,
                )
        if frozen_stats is None:
            if "train" not in roles_to_load:
                raise ValueError("train role is required to fit normalization")
            self.stats = {task: self._fit_stats(task) for task in self.tasks}
        else:
            self.stats = self._parse_stats(frozen_stats)

    def _fit_stats(self, task: str) -> TaskStats:
        train = self.roles[task]["train"]
        prediction: dict[str, ScalarStats] = {}
        uncertainty: dict[str, ScalarStats] = {}
        for value in self.expert_ids:
            export = train.exports[value]
            available = np.asarray(export.available, dtype=np.bool_)
            prediction[value] = _stats(np.asarray(export.prediction)[available])
            uncertainty[value] = _stats(
                np.log1p(np.asarray(export.uncertainty, dtype=np.float64)[available])
            )
        return TaskStats(_stats(train.y_true), prediction, uncertainty)

    def _parse_stats(self, payload: dict[str, Any]) -> dict[str, TaskStats]:
        if set(payload) != set(self.tasks):
            raise ValueError("Frozen stats do not cover exactly five JARVIS tasks")

        def scalar(value: Any) -> ScalarStats:
            result = ScalarStats(float(value["mean"]), float(value["scale"]))
            if not np.isfinite([result.mean, result.scale]).all() or result.scale <= 0:
                raise ValueError("Invalid frozen normalization stats")
            return result

        result: dict[str, TaskStats] = {}
        for task in self.tasks:
            value = payload[task]
            if set(value["expert_prediction"]) != set(self.expert_ids):
                raise ValueError("Frozen prediction stats expert set drift")
            if set(value["expert_uncertainty"]) != set(self.expert_ids):
                raise ValueError("Frozen uncertainty stats expert set drift")
            result[task] = TaskStats(
                target=scalar(value["target"]),
                expert_prediction={
                    key: scalar(value["expert_prediction"][key]) for key in self.expert_ids
                },
                expert_uncertainty={
                    key: scalar(value["expert_uncertainty"][key]) for key in self.expert_ids
                },
            )
        return result

    def restore_prediction(self, task: str, value: np.ndarray) -> np.ndarray:
        stats = self.stats[task].target
        return np.asarray(value, dtype=np.float64) * stats.scale + stats.mean

    def batch(
        self,
        *,
        task: str,
        role: str,
        indices: np.ndarray,
        device: torch.device,
        expert_dropout: float = 0.0,
        generator: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        cache = self.roles[task][role]
        index = np.asarray(indices, dtype=np.int64)
        hidden: dict[str, torch.Tensor] = {}
        prediction: dict[str, torch.Tensor] = {}
        uncertainty: dict[str, torch.Tensor] = {}
        availability: list[np.ndarray] = []
        for value in self.expert_ids:
            export = cache.exports[value]
            latent = np.asarray(export.hidden[index], dtype=np.float32)
            pred_raw = np.asarray(export.prediction[index], dtype=np.float64)
            unc_raw = np.log1p(np.asarray(export.uncertainty[index], dtype=np.float64))
            pred_stats = self.stats[task].expert_prediction[value]
            unc_stats = self.stats[task].expert_uncertainty[value]
            pred = ((pred_raw - pred_stats.mean) / pred_stats.scale).astype(np.float32)
            unc = ((unc_raw - unc_stats.mean) / unc_stats.scale).astype(np.float32)
            hidden[value] = torch.as_tensor(latent, device=device)
            prediction[value] = torch.as_tensor(pred, device=device)
            uncertainty[value] = torch.as_tensor(unc, device=device)
            availability.append(np.asarray(export.available[index], dtype=np.bool_))
        available = np.stack(availability, axis=1)
        anchor_column = self.expert_ids.index(cache.anchor_expert_id)
        if expert_dropout > 0:
            if generator is None:
                raise ValueError("expert dropout requires deterministic RNG")
            available &= generator.random(available.shape) >= expert_dropout
            available[:, anchor_column] = True
        anchor_raw = np.asarray(cache.exports[cache.anchor_expert_id].prediction[index], dtype=np.float64)
        target_raw = np.asarray(cache.y_true[index], dtype=np.float64)
        target_stats = self.stats[task].target
        return {
            "task_index": torch.full(
                (len(index),), self.task_index[task], dtype=torch.long, device=device
            ),
            "anchor_standard": torch.as_tensor(
                ((anchor_raw - target_stats.mean) / target_stats.scale).astype(np.float32),
                device=device,
            ),
            "anchor_prediction": anchor_raw,
            "sample_ids": np.asarray(cache.sample_ids[index]).astype(str),
            "y_true": target_raw,
            "y_standard": torch.as_tensor(
                ((target_raw - target_stats.mean) / target_stats.scale).astype(np.float32),
                device=device,
            ),
            "hidden": hidden,
            "prediction_standard": prediction,
            "uncertainty_standard": uncertainty,
            "available": torch.as_tensor(available, dtype=torch.bool, device=device),
        }

    def stats_receipt(self) -> dict[str, Any]:
        return {task: value.to_dict() for task, value in self.stats.items()}

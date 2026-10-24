from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.stage2_common.cache import TaskRoleCache, load_task_role_cache
from hpsafe_sota.stage2_common.protocol import CANONICAL_TASKS, ExpertPool


def probability_logit(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


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


def _stats(values: np.ndarray) -> ScalarStats:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return ScalarStats(0.0, 1.0)
    scale = float(np.std(array))
    return ScalarStats(float(np.mean(array)), max(scale, 1e-8))


class FoldInputs:
    """Memory-mapped frozen-expert inputs for one outer fold."""

    def __init__(
        self,
        *,
        project_root: Path,
        pool: ExpertPool,
        outer_fold: int,
        tasks: tuple[str, ...] = CANONICAL_TASKS,
        verify_hashes: bool = False,
        roles_to_load: tuple[str, ...] = ("train", "val", "test"),
        frozen_stats: dict[str, Any] | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.pool = pool
        self.outer_fold = outer_fold
        self.tasks = tasks
        self.task_index = {value: index for index, value in enumerate(tasks)}
        if not roles_to_load or any(
            value not in {"train", "val", "test"} for value in roles_to_load
        ):
            raise ValueError("roles_to_load must be a non-empty subset of train/val/test")
        if len(set(roles_to_load)) != len(roles_to_load):
            raise ValueError("roles_to_load contains duplicate roles")
        self.roles_to_load = roles_to_load
        self.task_types: dict[str, str] = {}
        self.roles: dict[str, dict[str, TaskRoleCache]] = {}
        for task in tasks:
            manifest = TaskManifest(self.project_root / "data/manifests" / f"{task}.npz")
            self.task_types[task] = str(manifest.metadata["definition"]["task_type"])
            self.roles[task] = {
                role: load_task_role_cache(
                    project_root=self.project_root,
                    pool=pool,
                    task=task,
                    outer_fold=outer_fold,
                    role=role,
                    verify_hashes=verify_hashes,
                )
                for role in roles_to_load
            }
        if frozen_stats is None:
            if "train" not in roles_to_load:
                raise ValueError("train role is required when fitting normalization statistics")
            self.stats = {task: self._fit_stats(task) for task in tasks}
        else:
            if set(frozen_stats) != set(tasks):
                raise ValueError("Frozen normalization statistics do not cover the requested tasks")
            self.stats = {
                task: self._parse_stats(task, frozen_stats[task]) for task in tasks
            }

    def _parse_stats(self, task: str, value: Any) -> TaskStats:
        if not isinstance(value, dict):
            raise ValueError(f"Malformed frozen statistics for {task}")

        def scalar(record: Any) -> ScalarStats:
            if not isinstance(record, dict):
                raise ValueError(f"Malformed scalar statistics for {task}")
            result = ScalarStats(float(record["mean"]), float(record["scale"]))
            if not np.isfinite([result.mean, result.scale]).all() or result.scale <= 0:
                raise ValueError(f"Invalid frozen scalar statistics for {task}")
            return result

        prediction = value.get("expert_prediction", {})
        uncertainty = value.get("expert_uncertainty", {})
        if set(prediction) != set(self.pool.expert_ids) or set(uncertainty) != set(
            self.pool.expert_ids
        ):
            raise ValueError(f"Frozen expert statistics do not match the pool for {task}")
        return TaskStats(
            target=scalar(value["target"]),
            expert_prediction={key: scalar(prediction[key]) for key in self.pool.expert_ids},
            expert_uncertainty={key: scalar(uncertainty[key]) for key in self.pool.expert_ids},
        )

    def _fit_stats(self, task: str) -> TaskStats:
        train = self.roles[task]["train"]
        task_type = self.task_types[task]
        target_stats = _stats(train.y_true) if task_type == "regression" else ScalarStats(0.0, 1.0)
        prediction_stats: dict[str, ScalarStats] = {}
        uncertainty_stats: dict[str, ScalarStats] = {}
        for expert_id in self.pool.expert_ids:
            export = train.exports[expert_id]
            if export is None:
                prediction_stats[expert_id] = ScalarStats(0.0, 1.0)
                uncertainty_stats[expert_id] = ScalarStats(0.0, 1.0)
                continue
            available = np.asarray(export.available, dtype=np.bool_)
            prediction_stats[expert_id] = _stats(np.asarray(export.prediction)[available])
            uncertainty_stats[expert_id] = _stats(
                np.log1p(np.asarray(export.uncertainty, dtype=np.float64)[available])
            )
        return TaskStats(target_stats, prediction_stats, uncertainty_stats)

    def _standard_target(self, task: str, y: np.ndarray) -> np.ndarray:
        if self.task_types[task] == "classification":
            return np.asarray(y, dtype=np.float32)
        stats = self.stats[task].target
        return ((np.asarray(y, dtype=np.float64) - stats.mean) / stats.scale).astype(np.float32)

    def _standard_anchor(self, task: str, anchor: np.ndarray) -> np.ndarray:
        if self.task_types[task] == "classification":
            return probability_logit(anchor).astype(np.float32)
        stats = self.stats[task].target
        return ((np.asarray(anchor, dtype=np.float64) - stats.mean) / stats.scale).astype(np.float32)

    def restore_prediction(self, task: str, raw_standard: np.ndarray) -> np.ndarray:
        if self.task_types[task] == "classification":
            value = np.clip(np.asarray(raw_standard, dtype=np.float64), -40.0, 40.0)
            return 1.0 / (1.0 + np.exp(-value))
        stats = self.stats[task].target
        return np.asarray(raw_standard, dtype=np.float64) * stats.scale + stats.mean

    def batch(
        self,
        *,
        task: str,
        role: str,
        indices: np.ndarray,
        device: torch.device,
        own_only: bool = False,
        expert_dropout: float = 0.0,
        generator: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        cache = self.roles[task][role]
        index = np.asarray(indices, dtype=np.int64)
        n = len(index)
        task_number = self.task_index[task]
        anchor_id = cache.anchor_expert_id
        anchor_export = cache.exports[anchor_id]
        assert anchor_export is not None
        hidden: dict[str, torch.Tensor] = {}
        prediction: dict[str, torch.Tensor] = {}
        uncertainty: dict[str, torch.Tensor] = {}
        availability_columns: list[np.ndarray] = []
        for expert_id in self.pool.expert_ids:
            export = cache.exports[expert_id]
            dimension = self.pool.hidden_dims[expert_id]
            if export is None:
                latent = np.zeros((n, dimension), dtype=np.float32)
                pred = np.zeros(n, dtype=np.float32)
                unc = np.zeros(n, dtype=np.float32)
                available = np.zeros(n, dtype=np.bool_)
            else:
                latent = np.asarray(export.hidden[index], dtype=np.float32)
                pred_raw = np.asarray(export.prediction[index], dtype=np.float64)
                unc_raw = np.log1p(np.asarray(export.uncertainty[index], dtype=np.float64))
                pred_stats = self.stats[task].expert_prediction[expert_id]
                unc_stats = self.stats[task].expert_uncertainty[expert_id]
                pred = ((pred_raw - pred_stats.mean) / pred_stats.scale).astype(np.float32)
                unc = ((unc_raw - unc_stats.mean) / unc_stats.scale).astype(np.float32)
                available = np.asarray(export.available[index], dtype=np.bool_)
            if own_only and expert_id != anchor_id:
                available[:] = False
            hidden[expert_id] = torch.as_tensor(latent, device=device)
            prediction[expert_id] = torch.as_tensor(pred, device=device)
            uncertainty[expert_id] = torch.as_tensor(unc, device=device)
            availability_columns.append(available)
        available_matrix = np.stack(availability_columns, axis=1)
        if expert_dropout > 0.0 and not own_only:
            if generator is None:
                raise ValueError("expert dropout requires a deterministic generator")
            keep = generator.random(available_matrix.shape) >= expert_dropout
            available_matrix &= keep
            anchor_column = self.pool.expert_ids.index(anchor_id)
            available_matrix[:, anchor_column] = True
        anchor_raw = np.asarray(anchor_export.prediction[index], dtype=np.float64)
        y = np.asarray(cache.y_true[index], dtype=np.float64)
        return {
            "task_index": torch.full((n,), task_number, dtype=torch.long, device=device),
            "anchor_standard": torch.as_tensor(
                self._standard_anchor(task, anchor_raw), device=device
            ),
            "anchor_prediction": anchor_raw,
            "y_standard": torch.as_tensor(self._standard_target(task, y), device=device),
            "y_true": y,
            "sample_ids": np.asarray(cache.sample_ids[index]).astype(str),
            "hidden": hidden,
            "prediction_standard": prediction,
            "uncertainty_standard": uncertainty,
            "available": torch.as_tensor(available_matrix, dtype=torch.bool, device=device),
        }

    def stats_receipt(self) -> dict[str, Any]:
        return {task: value.to_dict() for task, value in self.stats.items()}

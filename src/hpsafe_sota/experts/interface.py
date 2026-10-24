from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hpsafe_sota.data.manifest import TaskManifest

PHASES = {
    "holdout_validation",
    "inner_oof",
    "outer_refit",
    "artifact_import",
    "official_finetune",
}


@dataclass(frozen=True)
class ExpertRequest:
    schema_version: int
    project_root: str
    expert_name: str
    task_name: str
    outer_fold: int
    phase: str
    output_dir: str
    seed: int
    inner_fold: int | None = None
    selected_hyperparameters: str | None = None
    source_artifact: str | None = None
    gpu_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported expert request schema {self.schema_version}")
        if self.phase not in PHASES:
            raise ValueError(f"Unsupported expert phase: {self.phase}")
        if self.outer_fold not in range(5):
            raise ValueError("outer_fold must be 0..4")
        if self.phase == "inner_oof" and self.inner_fold not in range(5):
            raise ValueError("inner_oof requires inner_fold 0..4")
        if self.phase == "outer_refit" and not self.selected_hyperparameters:
            raise ValueError("outer_refit requires holdout-selected hyperparameters")
        if self.phase == "artifact_import" and not self.source_artifact:
            raise ValueError("artifact_import requires source_artifact")

    @property
    def project_path(self) -> Path:
        return Path(self.project_root).resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).resolve()

    @classmethod
    def read(cls, path: Path) -> ExpertRequest:
        value = cls(**json.loads(path.read_text(encoding="utf-8")))
        value.validate()
        return value

    def write(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class ResolvedExpertSplit:
    train_positions: np.ndarray
    validation_positions: np.ndarray
    prediction_positions: np.ndarray
    prediction_role: str
    inner_fold: int | None


def resolve_request_split(request: ExpertRequest, manifest: TaskManifest) -> ResolvedExpertSplit:
    request.validate()
    if request.phase == "official_finetune":
        # JMP-L is evaluated on the exact same Stage1 Matbench five-fold
        # contract as the other experts. Do not reproduce JMP's optional
        # internal 90/10 preprocessing split here.
        return ResolvedExpertSplit(
            train_positions=manifest.positions("outer_train", request.outer_fold),
            validation_positions=np.empty(0, dtype=np.int64),
            prediction_positions=manifest.positions("outer_test", request.outer_fold),
            prediction_role="outer_test",
            inner_fold=None,
        )
    if request.phase == "holdout_validation":
        validation = manifest.positions("holdout_val", request.outer_fold)
        return ResolvedExpertSplit(
            train_positions=manifest.positions("holdout_train", request.outer_fold),
            validation_positions=validation,
            prediction_positions=validation,
            prediction_role="holdout_val",
            inner_fold=None,
        )
    if request.phase == "inner_oof":
        inner = int(request.inner_fold)
        fit_train = manifest.positions("inner_train", request.outer_fold, inner)
        prediction = manifest.positions("inner_val", request.outer_fold, inner)
        if np.intersect1d(fit_train, prediction).size:
            raise ValueError("Expert fitting rows overlap the held-out OOF prediction rows")
        return ResolvedExpertSplit(
            train_positions=fit_train,
            validation_positions=np.empty(0, dtype=np.int64),
            prediction_positions=prediction,
            prediction_role="inner_val",
            inner_fold=inner,
        )
    if request.phase in {"outer_refit", "artifact_import"}:
        return ResolvedExpertSplit(
            train_positions=manifest.positions("outer_train", request.outer_fold),
            validation_positions=np.empty(0, dtype=np.int64),
            prediction_positions=manifest.positions("outer_test", request.outer_fold),
            prediction_role="outer_test",
            inner_fold=None,
        )
    raise ValueError(request.phase)


@dataclass
class NativeExpertOutput:
    sample_ids: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    checkpoint_path: Path | None
    preprocessing_manifest: Path
    y_uncertainty: np.ndarray | None = None
    latents: np.ndarray | None = None
    available: np.ndarray | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class ExpertBridge(ABC):
    """Lifecycle implemented inside each expert-specific environment."""

    def __init__(self, request: ExpertRequest, manifest: TaskManifest, config: dict[str, Any]):
        self.request = request
        self.manifest = manifest
        self.config = config
        self.split = resolve_request_split(request, manifest)

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return versions, modalities, pretraining, and latent availability."""

    @abstractmethod
    def prepare(self) -> Any:
        """Materialize or load fold-specific inputs without fitting on prediction rows."""

    @abstractmethod
    def fit(self, prepared: Any) -> Any:
        """Fit using train and validation positions only."""

    @abstractmethod
    def predict(self, fitted: Any, prepared: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Return predictions and optional uncertainty in exact requested order."""

    @abstractmethod
    def export_latent(self, fitted: Any, prepared: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """Return row-aligned native latents and their availability mask."""

    @abstractmethod
    def checkpoint_path(self, fitted: Any) -> Path | None:
        """Return the immutable fitted checkpoint path, if any."""

    @abstractmethod
    def preprocessing_manifest(self, prepared: Any) -> Path:
        """Return the exact preprocessing receipt used by this job."""

    def run(self) -> NativeExpertOutput:
        prepared = self.prepare()
        fitted = self.fit(prepared)
        predictions, uncertainty = self.predict(fitted, prepared)
        latent_output = self.export_latent(fitted, prepared)
        latents, available = latent_output if latent_output is not None else (None, None)
        positions = self.split.prediction_positions
        return NativeExpertOutput(
            sample_ids=self.manifest.sample_ids[positions],
            y_true=self.manifest.targets[positions],
            y_pred=np.asarray(predictions),
            y_uncertainty=None if uncertainty is None else np.asarray(uncertainty),
            latents=None if latents is None else np.asarray(latents),
            available=None if available is None else np.asarray(available),
            checkpoint_path=self.checkpoint_path(fitted),
            preprocessing_manifest=self.preprocessing_manifest(prepared),
            extra_metadata={"expert_description": self.describe()},
        )

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import ijson
import numpy as np

from hpsafe_sota.data.raw import file_sha256, load_task_definitions
from hpsafe_sota.experts.interface import ExpertBridge

OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/JMP"
OFFICIAL_COMMIT = "937b14874381d9b80809582e323ef82f2d4e1291"
OFFICIAL_CHECKPOINT_URL = "https://jmp-iclr-datasets.s3.amazonaws.com/jmp-l.pt"
SUPPORTED_TASKS = (
    "phonons",
    "mp_gap",
    "perovskites",
    "dielectric",
    "jdft2d",
    "mp_e_form",
)
LEARNING_RATE = 8.0e-5
MAX_TIME_DAYS = 7
CACHE_SCHEMA_VERSION = 1
LATENT_DIM = 256
LATENT_CONTRACT_VERSION = 1
LATENT_SOURCE = "gemnet_oc_final_atom_energy_embedding_before_task_head"
FAST32_PROTOCOL = "jmp_l_fp32_32epoch"
FAST32_MAX_EPOCHS = 32
FAST32_MAX_TIME_DAYS = 3
FAST32_CHECKPOINT_EVERY_N_STEPS = 10_000

TASK_SETTINGS: dict[str, dict[str, Any]] = {
    "phonons": {"batch_size": 8, "precision": "16-mixed", "reduction": "max"},
    "mp_gap": {"batch_size": 2, "precision": "32-true", "reduction": "mean"},
    "perovskites": {"batch_size": 8, "precision": "16-mixed", "reduction": "mean"},
    "dielectric": {"batch_size": 8, "precision": "16-mixed", "reduction": "mean"},
    "jdft2d": {"batch_size": 3, "precision": "32-true", "reduction": "mean"},
    "mp_e_form": {"batch_size": 6, "precision": "16-mixed", "reduction": "mean"},
}

# Fixed 32-epoch FP32 recipe used for the two large Materials Project tasks.
FAST32_TASK_SETTINGS: dict[str, dict[str, Any]] = {
    "mp_gap": {
        "batch_size": 16,
        "precision": "32-true",
        "reduction": "mean",
        "num_workers": 8,
    },
    "mp_e_form": {
        "batch_size": 32,
        "precision": "32-true",
        "reduction": "mean",
        "num_workers": 0,
    },
}


def _fast32_enabled(extra: dict[str, Any]) -> bool:
    protocol = extra.get("finetune_protocol")
    if protocol is None:
        return False
    if protocol != FAST32_PROTOCOL:
        raise ValueError(f"Unsupported JMP-L finetune_protocol: {protocol!r}")
    return True


def _effective_task_settings(task_name: str, extra: dict[str, Any]) -> dict[str, Any]:
    if not _fast32_enabled(extra):
        return TASK_SETTINGS[task_name]
    if task_name not in FAST32_TASK_SETTINGS:
        raise ValueError(
            f"{FAST32_PROTOCOL} supports only {sorted(FAST32_TASK_SETTINGS)}, got {task_name!r}"
        )
    return FAST32_TASK_SETTINGS[task_name]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            mode="w",
            encoding="utf-8",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _source_root(project_root: Path, config: dict[str, Any], extra: dict[str, Any]) -> Path:
    value = extra.get("jmp_source_root", config.get("source_root", "external/JMP"))
    path = _resolve_project_path(project_root, str(value))
    if not (path / "src/jmp").is_dir():
        raise FileNotFoundError(
            f"Official JMP source is missing at {path}; run scripts/setup_jmp_l_runtime.py first"
        )
    return path


def _checkpoint_path(project_root: Path, config: dict[str, Any], extra: dict[str, Any]) -> Path:
    value = extra.get("jmp_checkpoint", config.get("checkpoint_path", "checkpoints/jmp-l.pt"))
    path = _resolve_project_path(project_root, str(value))
    if not path.is_file():
        raise FileNotFoundError(
            f"Official JMP-L checkpoint is missing at {path}; run scripts/setup_jmp_l_runtime.py first"
        )
    receipt = path.with_suffix(path.suffix + ".sha256")
    if not receipt.is_file():
        raise FileNotFoundError(f"Checkpoint checksum receipt is missing: {receipt}")
    expected = receipt.read_text(encoding="utf-8").strip().split()[0]
    stat = path.stat()
    verification_path = path.with_suffix(path.suffix + ".verified.json")
    lock_path = path.with_suffix(path.suffix + ".verify.lock")
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        verification = (
            json.loads(verification_path.read_text(encoding="utf-8"))
            if verification_path.is_file()
            else {}
        )
        verified_stat = (
            verification.get("sha256") == expected
            and int(verification.get("size_bytes", -1)) == stat.st_size
            and int(verification.get("mtime_ns", -1)) == stat.st_mtime_ns
        )
        if not verified_stat:
            observed = _sha256(path)
            if expected != observed:
                raise ValueError(f"JMP-L checkpoint SHA-256 mismatch: {observed} != {expected}")
            _atomic_json(
                verification_path,
                {
                    "schema_version": 1,
                    "sha256": observed,
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "verification": "full_file_sha256",
                },
            )
    return path


def _verify_source_commit(source_root: Path, expected: str = OFFICIAL_COMMIT) -> str:
    if (source_root / ".git").is_dir():
        observed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        receipt = source_root / "HPSAFE_SOURCE_COMMIT"
        if not receipt.is_file():
            raise FileNotFoundError(
                f"JMP source has neither .git metadata nor {receipt.name} commit receipt"
            )
        observed = receipt.read_text(encoding="utf-8").strip()
    if observed != expected:
        raise ValueError(f"JMP source commit mismatch: {observed} != {expected}")
    return observed


def _activate_source(source_root: Path) -> None:
    source_path = str((source_root / "src").resolve())
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def _cache_identity(task_name: str, raw_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "task_name": task_name,
        "raw_sha256": raw_sha256,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "representation": "target_free_pyg_atomic_numbers_cartesian_positions_cell",
    }


def jmp_cache_path(
    project_root: Path,
    task_name: str,
    *,
    cache_root: str | Path = "data/cache/jmp_l",
) -> Path:
    definitions = load_task_definitions(project_root)
    if task_name not in definitions or definitions[task_name].input_type != "structure":
        raise ValueError(f"JMP-L cache requires a registered structure task, got {task_name!r}")
    identity = _cache_identity(task_name, definitions[task_name].raw_sha256)
    root = _resolve_project_path(project_root, cache_root)
    return root / task_name / _payload_sha256(identity)


def materialize_jmp_cache(
    project_root: Path,
    task_name: str,
    *,
    source_root: Path,
    cache_root: str | Path = "data/cache/jmp_l",
) -> Path:
    """Build one immutable, target-free official-JMP LMDB in manifest row order."""

    _verify_source_commit(source_root)
    _activate_source(source_root)
    try:
        import torch
        from jmp.datasets.finetune.base import LmdbDataset
        from pymatgen.core import Structure
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover - exercised in the isolated GPU environment
        raise RuntimeError("The pinned JMP/PyG runtime is incomplete") from exc

    definitions = load_task_definitions(project_root)
    definition = definitions[task_name]
    if definition.input_type != "structure":
        raise ValueError(f"JMP-L cache requires a registered structure task: {task_name}")
    identity = _cache_identity(task_name, definition.raw_sha256)
    final_root = jmp_cache_path(project_root, task_name, cache_root=cache_root)
    receipt_path = final_root / "cache_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("identity") == identity and int(receipt.get("n_samples", -1)) == definition.n_samples:
            return final_root
        raise ValueError(f"Existing JMP-L cache receipt is incompatible: {receipt_path}")

    parent = final_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{final_root.name}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if receipt_path.is_file():
            return final_root
        actual_raw_sha256 = file_sha256(definition.raw_path)
        if actual_raw_sha256 != definition.raw_sha256:
            raise RuntimeError(
                f"Raw Matbench payload hash mismatch for {task_name}: "
                f"expected {definition.raw_sha256}, got {actual_raw_sha256}"
            )
        temporary = Path(tempfile.mkdtemp(prefix=f".{final_root.name}.", dir=parent))
        try:
            lmdb_root = temporary / "lmdb"
            lmdb_root.mkdir(parents=True)

            def records():
                with gzip.open(definition.raw_path, "rb") as handle:
                    rows = ijson.items(handle, "data.item", use_float=True)
                    for row_index, row in enumerate(rows):
                        if not isinstance(row, list) or len(row) != 2:
                            raise ValueError(f"Malformed Matbench row {row_index} for {task_name}")
                        structure = Structure.from_dict(row[0])
                        yield Data(
                            atomic_numbers=torch.tensor(
                                [site.specie.number for site in structure], dtype=torch.long
                            ),
                            pos=torch.tensor(
                                np.asarray(structure.cart_coords), dtype=torch.float32
                            ),
                            cell=torch.tensor(
                                np.asarray(structure.lattice.matrix), dtype=torch.float32
                            ).unsqueeze(0),
                        )

            LmdbDataset.dump_data(records(), count=definition.n_samples, path=lmdb_root)
            _atomic_json(
                temporary / "cache_receipt.json",
                {
                    "schema_version": 1,
                    "identity": identity,
                    "cache_key": final_root.name,
                    "n_samples": definition.n_samples,
                    "target_values_stored": False,
                    "row_order": "immutable_matbench_manifest_position",
                    "lmdb_metadata_sha256": _sha256(lmdb_root / "metadata.npz"),
                },
            )
            os.replace(temporary, final_root)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return final_root


@dataclass(frozen=True)
class JMPLPrepared:
    cache_dir: Path
    train_split: Path
    prediction_split: Path
    preprocessing_manifest: Path
    target_mean: float
    target_std: float
    source_root: Path
    checkpoint: Path
    checkpoint_sha256: str


@dataclass
class JMPLFitted:
    model: Any
    config: Any
    checkpoint: Path
    trained_epochs: int
    target_mean: float
    target_std: float
    prediction_latents: np.ndarray | None = None


class _IndexedTargetDataset:
    def __init__(self, base: Any, split_path: Path) -> None:
        with np.load(split_path, allow_pickle=False) as payload:
            positions = np.asarray(payload["positions"], dtype=np.int64)
            targets = (
                np.asarray(payload["targets"], dtype=np.float32)
                if "targets" in payload.files
                else None
            )
        self.base = base
        self.positions = positions
        self.targets = targets
        if self.targets is not None and self.targets.shape != self.positions.shape:
            raise ValueError(f"Target/position mismatch in {split_path}")

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> Any:
        import torch

        position = int(self.positions[index])
        data = self.base[position]
        value = 0.0 if self.targets is None else float(self.targets[index])
        data.y = torch.tensor(value, dtype=torch.float32)
        data.hpsafe_position = torch.tensor(position, dtype=torch.long)
        return data


def training_normalization(targets: np.ndarray) -> tuple[float, float]:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("JMP-L training targets must be one finite vector")
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1.0e-12:
        raise ValueError("JMP-L training target standard deviation is zero")
    return mean, std


def _indexed_model_class(matbench_model: type[Any]) -> type[Any]:
    class HPSafeIndexedMatbenchModel(matbench_model):
        def create_dataset(self, split: Literal["train", "val", "test"]):
            dataset = super().create_dataset(split)
            if dataset is None:
                return None
            split_path = Path(str(self.config.meta[f"hpsafe_{split}_split"]))
            return _IndexedTargetDataset(dataset, split_path)

    return HPSafeIndexedMatbenchModel


class JMPLStage1Bridge(ExpertBridge):
    """Stage1 bridge around Meta's pinned public JMP-L fine-tuning code."""

    def describe(self) -> dict[str, Any]:
        settings = _effective_task_settings(self.request.task_name, self.request.extra)
        fast32 = _fast32_enabled(self.request.extra)
        inner_oof = self.request.phase == "inner_oof"
        return {
            "bridge": (
                "jmp_l_h20_fast32_finetune" if fast32 else "jmp_l_official_finetune"
            ),
            "finetune_protocol": FAST32_PROTOCOL if fast32 else "jmp_l_official_500_or_7d",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
            "model": "JMP-L",
            "learning_rate": LEARNING_RATE,
            "batch_size": settings["batch_size"],
            "precision": settings["precision"],
            "graph_reduction": settings["reduction"],
            "workflow_stage": "stage1",
            "stage2_feature_generation": inner_oof,
            "latents_exported": True,
            "latent_contract": {
                "schema_version": LATENT_CONTRACT_VERSION,
                "source": LATENT_SOURCE,
                "dimension": LATENT_DIM,
                "pooling": settings["reduction"],
                "split": (
                    "untouched_inner_validation_partition"
                    if inner_oof
                    else "official_matbench_outer_test"
                ),
                "checkpoint_state": (
                    "final_inner_crossfit_checkpoint"
                    if inner_oof
                    else (
                        "fp32_fitted_fold_checkpoint"
                        if fast32
                        else "fitted_fold_checkpoint"
                    )
                ),
            },
            "split_protocol": (
                "stage2_inner_train_to_disjoint_inner_val"
                if inner_oof
                else "official_matbench_outer_train_to_outer_test"
            ),
            "hyperparameter_policy": "fixed_release_values",
            "maximum_epochs": FAST32_MAX_EPOCHS if fast32 else 500,
            "maximum_wall_time_days": FAST32_MAX_TIME_DAYS if fast32 else MAX_TIME_DAYS,
            "checkpoint_resume": (
                "full_trainer_state_periodic_auto_resume" if fast32 else "final_weights_only"
            ),
        }

    def prepare(self) -> JMPLPrepared:
        if self.request.task_name not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported JMP-L Stage1 task: {self.request.task_name}")
        if self.request.phase not in {"official_finetune", "inner_oof"}:
            raise ValueError(
                "JMP-L accepts only official_finetune or leakage-free inner_oof"
            )
        if self.request.selected_hyperparameters is not None:
            raise ValueError("JMP-L official_finetune forbids a hyperparameter-selection receipt")
        if self.request.seed != 42:
            raise ValueError("JMP-L public fine-tuning recipe uses seed 42")
        fast32 = _fast32_enabled(self.request.extra)
        if fast32 and self.request.phase != "official_finetune":
            raise ValueError(f"{FAST32_PROTOCOL} is Stage1 outer-fold fine-tuning only")
        settings = _effective_task_settings(self.request.task_name, self.request.extra)
        requested_lr = float(self.request.extra.get("learning_rate", LEARNING_RATE))
        if requested_lr != LEARNING_RATE:
            raise ValueError(f"JMP-L learning rate is pinned to {LEARNING_RATE}, got {requested_lr}")
        for key, expected in {
            "batch_size": settings["batch_size"],
            "precision": settings["precision"],
            "graph_reduction": settings["reduction"],
        }.items():
            if key in self.request.extra and self.request.extra[key] != expected:
                raise ValueError(
                    f"JMP-L {self.request.task_name} {key} is pinned to {expected!r}, "
                    f"got {self.request.extra[key]!r}"
                )
        inner_oof = self.request.phase == "inner_oof"
        latent_export_contract = {
            "schema_version": LATENT_CONTRACT_VERSION,
            "source": LATENT_SOURCE,
            "dimension": LATENT_DIM,
            "pooling": settings["reduction"],
            "row_order": (
                "exact_inner_val_manifest_order"
                if inner_oof
                else "exact_official_outer_test_manifest_order"
            ),
            "checkpoint_state": (
                "final_inner_crossfit_checkpoint"
                if inner_oof
                else (
                    "fp32_fitted_fold_checkpoint"
                    if fast32
                    else "fitted_fold_checkpoint"
                )
            ),
            "inference_only": True,
            "targets_accessed": False,
        }
        maximum_epochs = FAST32_MAX_EPOCHS if fast32 else 500
        maximum_wall_time_days = FAST32_MAX_TIME_DAYS if fast32 else MAX_TIME_DAYS
        fixed_request_values = {
            "maximum_epochs": maximum_epochs,
            "maximum_wall_time_days": maximum_wall_time_days,
            "hyperparameter_policy": "fixed_release_values",
            "fit_protocol": "single_outer_fold_fit",
            "workflow_stage": "stage1",
            "stage2_feature_generation": inner_oof,
            "latent_export": latent_export_contract,
        }
        for key, expected in fixed_request_values.items():
            if self.request.extra.get(key, expected) != expected:
                recipe = FAST32_PROTOCOL if fast32 else "JMP-L official recipe"
                raise ValueError(f"{recipe} requires {key}={expected!r}")
        if fast32:
            checkpoint_interval = int(
                self.request.extra.get(
                    "checkpoint_every_n_train_steps", FAST32_CHECKPOINT_EVERY_N_STEPS
                )
            )
            if checkpoint_interval != FAST32_CHECKPOINT_EVERY_N_STEPS:
                raise ValueError(
                    f"{FAST32_PROTOCOL} requires checkpoint_every_n_train_steps="
                    f"{FAST32_CHECKPOINT_EVERY_N_STEPS}"
                )
            if self.request.extra.get("auto_resume", True) is not True:
                raise ValueError(f"{FAST32_PROTOCOL} requires auto_resume=true")
        source_root = _source_root(self.request.project_path, self.config, self.request.extra)
        _verify_source_commit(source_root)
        checkpoint = _checkpoint_path(self.request.project_path, self.config, self.request.extra)
        cache_root = self.request.extra.get("jmp_cache_root", "data/cache/jmp_l")
        cache_dir = materialize_jmp_cache(
            self.request.project_path,
            self.request.task_name,
            source_root=source_root,
            cache_root=str(cache_root),
        )

        split_root = self.request.output_path / "jmp_splits"
        split_root.mkdir(parents=True, exist_ok=True)
        train_positions = self.split.train_positions
        prediction_positions = self.split.prediction_positions
        train_targets = self.manifest.targets[train_positions]
        mean, std = training_normalization(train_targets)
        train_split = split_root / "train.npz"
        prediction_split = split_root / "test.npz"
        np.savez_compressed(train_split, positions=train_positions, targets=train_targets)
        # Prediction rows intentionally contain no target array.
        np.savez_compressed(prediction_split, positions=prediction_positions)

        cache_receipt = cache_dir / "cache_receipt.json"
        preprocessing_manifest = self.request.output_path / "preprocessing_manifest.json"
        payload = {
            "schema_version": 1,
            "expert": "jmp_l",
            "task": self.request.task_name,
            "outer_fold": self.request.outer_fold,
            "phase": self.request.phase,
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_checkpoint": str(checkpoint),
            "official_checkpoint_sha256": _sha256(checkpoint),
            "cache_receipt": str(cache_receipt),
            "cache_receipt_sha256": _sha256(cache_receipt),
            "train_positions_sha256": hashlib.sha256(train_positions.tobytes()).hexdigest(),
            "prediction_positions_sha256": hashlib.sha256(
                prediction_positions.tobytes()
            ).hexdigest(),
            "prediction_targets_materialized": False,
            "target_normalization_scope": (
                "current_inner_train_only"
                if inner_oof
                else "complete_official_outer_train_only"
            ),
            "target_mean": mean,
            "target_std": std,
            "learning_rate": LEARNING_RATE,
            "finetune_protocol": FAST32_PROTOCOL if fast32 else "jmp_l_official_500_or_7d",
            "maximum_epochs": maximum_epochs,
            "maximum_wall_time_days": maximum_wall_time_days,
            "task_settings": settings,
            "periodic_full_state_checkpoint_steps": (
                FAST32_CHECKPOINT_EVERY_N_STEPS if fast32 else None
            ),
            "automatic_full_state_resume": fast32,
            "graph": {
                "cutoff_angstrom": 12.0,
                "periodic": True,
                "conditional_max_neighbors": {"default": 30, "gt_200": 10, "gt_300": 5},
            },
            "outer_test_used_for_selection": False,
            "split_protocol": (
                "stage2_inner_train_to_disjoint_inner_val"
                if inner_oof
                else "official_matbench_outer_train_to_outer_test"
            ),
            "hyperparameter_policy": "fixed_release_values",
            "fit_protocol": "single_outer_fold_fit",
            "workflow_stage": "stage1",
            "stage2_feature_generation": inner_oof,
            "latent_export": fixed_request_values["latent_export"],
        }
        _atomic_json(preprocessing_manifest, payload)
        _atomic_json(
            self.request.output_path / "target_access_receipt.json",
            {
                "schema_version": 1,
                "fit_target_count": int(len(train_positions)),
                "validation_target_count": 0,
                "prediction_target_access_count": 0,
                "prediction_targets_materialized": False,
                "outer_test_used_for_selection": False,
                "outer_test_accessed": False,
                "inner_fold": self.request.inner_fold,
            },
        )
        return JMPLPrepared(
            cache_dir=cache_dir / "lmdb",
            train_split=train_split,
            prediction_split=prediction_split,
            preprocessing_manifest=preprocessing_manifest,
            target_mean=mean,
            target_std=std,
            source_root=source_root,
            checkpoint=checkpoint,
            checkpoint_sha256=_sha256(checkpoint),
        )

    def _configuration(self, prepared: JMPLPrepared) -> tuple[Any, type[Any], dict[str, Any]]:
        _activate_source(prepared.source_root)
        try:
            from jmp.lightning import Trainer
            from jmp.models.gemnet.config import BackboneConfig
            from jmp.modules.transforms.normalize import NormalizationConfig
            from jmp.tasks.config import AdamWConfig
            from jmp.tasks.finetune.base import (
                FinetuneLmdbDatasetConfig,
                PrimaryMetricConfig,
                RLPConfig,
                WarmupCosRLPConfig,
            )
            from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel
            from jmp.utils.param_specific_util import make_parameter_specific_optimizer_config
        except ImportError as exc:  # pragma: no cover - isolated environment only
            raise RuntimeError("Unable to import the pinned official JMP implementation") from exc

        MatbenchConfig.set_seed(self.request.seed)
        config = MatbenchConfig.draft()
        fast32 = _fast32_enabled(self.request.extra)
        settings = _effective_task_settings(self.request.task_name, self.request.extra)
        config.dataset = self.request.task_name
        config.fold = self.request.outer_fold
        config.name = (
            f"hpsafe_jmp_l_{self.request.task_name}_outer{self.request.outer_fold}_"
            f"{self.request.phase}"
            + (
                f"_inner{self.request.inner_fold}"
                if self.request.inner_fold is not None
                else ""
            )
        )
        config.project = (
            "hpsafe_jmp_l_stage2_inner_oof"
            if self.request.phase == "inner_oof"
            else ("hpsafe_jmp_l_stage1_h20_fast32" if fast32 else "hpsafe_jmp_l_stage1")
        )
        config.mp_e_form_dev = False
        config.graph_scalar_reduction_default = settings["reduction"]
        config.conditional_max_neighbors = True
        config.backbone = BackboneConfig.large()
        config.embedding.embedding_size = config.backbone.emb_size_atom
        config.backbone.scale_basis = False
        config.backbone.scale_file = str(
            prepared.source_root / "src/jmp/models/gemnet/scale_files/large.pt"
        )
        config.backbone.regress_forces = False
        config.backbone.direct_forces = False
        config.optimizer = AdamWConfig(
            lr=LEARNING_RATE,
            amsgrad=False,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
        )
        config.trainer.gradient_clip_val = 1.0
        config.trainer.gradient_clip_algorithm = "value"
        config.lr_scheduler = WarmupCosRLPConfig(
            warmup_epochs=5,
            warmup_start_lr_factor=0.1,
            should_restart=False,
            max_epochs=32,
            min_lr_factor=0.1,
            rlp=RLPConfig(mode="min", patience=3, factor=0.8),
        )
        config.parameter_specific_optimizers = make_parameter_specific_optimizer_config(
            config,
            config.backbone.num_blocks,
            {
                "embedding": 0.3,
                "blocks_0": 0.55,
                "blocks_1": 0.40,
                "blocks_2": 0.30,
                "blocks_3": 0.40,
                "blocks_4": 0.55,
                "blocks_5": 0.625,
            },
        )
        config.batch_size = int(settings["batch_size"])
        config.eval_batch_size = int(settings["batch_size"])
        config.num_workers = int(
            self.request.extra.get("num_workers", settings.get("num_workers", 8))
        )
        config.normalization = {
            "y": NormalizationConfig(mean=prepared.target_mean, std=prepared.target_std)
        }
        dataset_config = FinetuneLmdbDatasetConfig(src=prepared.cache_dir)
        config.train_dataset = dataset_config.model_copy(deep=True)
        config.val_dataset = None
        config.test_dataset = dataset_config.model_copy(deep=True)
        config.primary_metric = PrimaryMetricConfig(name="y_mae", mode="min")
        config.trainer.max_time = (
            f"{FAST32_MAX_TIME_DAYS:02d}:00:00:00" if fast32 else "07:00:00:00"
        )
        config.trainer.precision = settings["precision"]
        config.trainer.set_float32_matmul_precision = "medium"
        config.trainer.devices = 1
        config.trainer.num_nodes = 1
        config.trainer.accelerator = "gpu"
        config.trainer.default_root_dir = self.request.output_path / "lightning"
        config.trainer.auto_set_default_root_dir = False
        config.trainer.logger = False
        config.trainer.auto_set_loggers = False
        config.trainer.limit_val_batches = 0
        config.trainer.num_sanity_val_steps = 0
        config.trainer.optimizer.log_grad_norm = True
        config.trainer.optimizer.log_grad_norm_per_param = False
        config.trainer.optimizer.log_param_norm = False
        config.trainer.optimizer.log_param_norm_per_param = False
        config.trainer.supports_parameter_hooks = False
        config.trainer.supports_skip_batch_exception = False
        config.trainer.logging.wandb.log_model = False
        config.runner.save_output = None
        config.meta.update(
            {
                "ckpt_path": str(prepared.checkpoint),
                "ema_backbone": True,
                "hpsafe_train_split": str(prepared.train_split),
                "hpsafe_test_split": str(prepared.prediction_split),
                "hpsafe_official_commit": OFFICIAL_COMMIT,
            }
        )
        config.trainer.max_epochs = int(self.request.extra.get("maximum_epochs", 500))
        config.trainer.enable_checkpointing = True
        config.early_stopping = None
        config.ckpt_best = None
        config.id = MatbenchConfig.generate_id()
        config = config.finalize()
        return config, _indexed_model_class(MatbenchModel), {"Trainer": Trainer}

    def fit(self, prepared: JMPLPrepared) -> JMPLFitted:
        config, model_class, api = self._configuration(prepared)
        fast32 = _fast32_enabled(self.request.extra)
        try:
            from jmp.utils.finetune_state_dict import (
                filter_state_dict,
                retreive_state_dict_for_finetuning,
            )
        except ImportError as exc:  # pragma: no cover - isolated environment only
            raise RuntimeError("Unable to load official JMP checkpoint utilities") from exc

        with api["Trainer"].runner_init(config):
            with api["Trainer"].context(config):
                model = model_class(config)
                resume_checkpoint: Path | None = None
                callbacks: list[Any] = []
                if fast32:
                    try:
                        from lightning.pytorch.callbacks import ModelCheckpoint
                    except ImportError as exc:  # pragma: no cover - isolated environment only
                        raise RuntimeError("Lightning checkpoint callback is unavailable") from exc
                    resume_root = self.request.output_path / "resume"
                    resume_root.mkdir(parents=True, exist_ok=True)
                    callbacks.append(
                        ModelCheckpoint(
                            dirpath=resume_root,
                            filename="step-{step}",
                            save_top_k=0,
                            save_last=True,
                            save_weights_only=False,
                            every_n_train_steps=FAST32_CHECKPOINT_EVERY_N_STEPS,
                            save_on_train_epoch_end=False,
                        )
                    )
                    candidate = resume_root / "last.ckpt"
                    if candidate.is_file():
                        resume_checkpoint = candidate

                if resume_checkpoint is None:
                    state_dict = retreive_state_dict_for_finetuning(
                        prepared.checkpoint,
                        load_emas=True,
                    )
                    embedding = filter_state_dict(state_dict, "embedding.atom_embedding.")
                    backbone = filter_state_dict(state_dict, "backbone.")
                    model.load_backbone_state_dict(
                        backbone=backbone, embedding=embedding, strict=False
                    )
                trainer = api["Trainer"](config, callbacks=callbacks)
                trainer.fit(
                    model,
                    ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
                )

                trained_epochs = int(trainer.current_epoch)
                checkpoint = self.request.output_path / (
                    "jmp_l_inner_oof.ckpt"
                    if self.request.phase == "inner_oof"
                    else "jmp_l_official_fold.ckpt"
                )
                trainer.save_checkpoint(str(checkpoint), weights_only=True)

        _atomic_json(
            self.request.output_path / "native_training_summary.json",
            {
                "schema_version": 1,
                "official_repository": OFFICIAL_REPOSITORY,
                "official_commit": OFFICIAL_COMMIT,
                "official_checkpoint_sha256": prepared.checkpoint_sha256,
                "phase": self.request.phase,
                "finetune_protocol": FAST32_PROTOCOL if fast32 else "jmp_l_official_500_or_7d",
                "learning_rate": LEARNING_RATE,
                "trained_epochs": trained_epochs,
                "global_optimizer_steps": int(trainer.global_step),
                "resumed_from_full_state_checkpoint": resume_checkpoint is not None,
                "fixed_official_parameters": {
                    "learning_rate": LEARNING_RATE,
                    "maximum_epochs": int(config.trainer.max_epochs),
                    "maximum_wall_time_days": (
                        FAST32_MAX_TIME_DAYS if fast32 else MAX_TIME_DAYS
                    ),
                    "batch_size": _effective_task_settings(
                        self.request.task_name, self.request.extra
                    )["batch_size"],
                    "precision": _effective_task_settings(
                        self.request.task_name, self.request.extra
                    )["precision"],
                    "graph_reduction": _effective_task_settings(
                        self.request.task_name, self.request.extra
                    )["reduction"],
                    "num_workers": int(
                        self.request.extra.get(
                            "num_workers",
                            _effective_task_settings(
                                self.request.task_name, self.request.extra
                            ).get("num_workers", 8),
                        )
                    ),
                },
                "hyperparameter_policy": "fixed_release_values",
                "trainer_fit_returned_normally": True,
                "official_stopping_rule": (
                    f"fixed_32_epoch_primary_cosine_horizon_or_"
                    f"{FAST32_MAX_TIME_DAYS}_day_safety_cap"
                    if fast32
                    else "first_of_500_epochs_or_7_days"
                ),
                "validation_split": None,
                "early_stopping": False,
                "validation_dependent_rlp_updates": False,
                "fit_protocol": "single_outer_fold_fit",
                "workflow_stage": "stage1",
                "stage2_feature_generation": self.request.phase == "inner_oof",
                "latent_export": {
                    "schema_version": LATENT_CONTRACT_VERSION,
                    "source": LATENT_SOURCE,
                    "dimension": LATENT_DIM,
                    "pooling": _effective_task_settings(
                        self.request.task_name, self.request.extra
                    )["reduction"],
                    "split": (
                        "untouched_inner_validation_partition"
                        if self.request.phase == "inner_oof"
                        else "official_matbench_outer_test"
                    ),
                    "inference_only": True,
                },
            },
        )
        return JMPLFitted(
            model=model,
            config=config,
            checkpoint=checkpoint,
            trained_epochs=trained_epochs,
            target_mean=prepared.target_mean,
            target_std=prepared.target_std,
        )

    def predict(self, fitted: JMPLFitted, prepared: JMPLPrepared) -> tuple[np.ndarray, None]:
        try:
            import torch
            from torch.utils.data import DataLoader
            from torch_scatter import scatter
        except ImportError as exc:  # pragma: no cover - isolated environment only
            raise RuntimeError("PyTorch is required for JMP-L inference") from exc

        dataset = fitted.model.test_dataset()
        settings = _effective_task_settings(self.request.task_name, self.request.extra)
        loader = DataLoader(
            dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=False,
            num_workers=int(
                self.request.extra.get("num_workers", settings.get("num_workers", 8))
            ),
            collate_fn=fitted.model.collate_fn,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("JMP-L prediction requires a CUDA device")
        # The scheduler exposes a single physical lane via CUDA_VISIBLE_DEVICES,
        # so it is always logical device 0 inside this subprocess.
        device = torch.device("cuda:0")
        fitted.model.to(device)
        fitted.model.eval()
        values: list[np.ndarray] = []
        latent_values: list[np.ndarray] = []
        captured: dict[str, Any] = {}

        def capture_backbone_output(_module: Any, _inputs: Any, output: Any) -> None:
            captured["energy"] = output["energy"]

        hook = fitted.model.backbone.register_forward_hook(capture_backbone_output)
        try:
            with torch.inference_mode():
                for batch in loader:
                    batch = batch.to(device)
                    captured.clear()
                    normalized = fitted.model(batch)["y"]
                    prediction = normalized * fitted.target_std + fitted.target_mean
                    values.append(prediction.detach().cpu().numpy().reshape(-1))

                    atom_latents = captured.get("energy")
                    if atom_latents is None:
                        raise RuntimeError("JMP-L backbone hook did not capture the hidden state")
                    if atom_latents.ndim != 2 or atom_latents.shape[1] != LATENT_DIM:
                        raise ValueError(
                            "Unexpected JMP-L atom latent shape "
                            f"{tuple(atom_latents.shape)}; expected (*, {LATENT_DIM})"
                        )
                    n_graphs = int(torch.max(batch.batch).item()) + 1
                    graph_latents = scatter(
                        atom_latents,
                        batch.batch,
                        dim=0,
                        dim_size=n_graphs,
                        reduce=settings["reduction"],
                    )
                    latent_values.append(graph_latents.float().cpu().numpy())
        finally:
            hook.remove()
        predictions = np.concatenate(values).astype(np.float64, copy=False)
        latents = np.concatenate(latent_values).astype(np.float32, copy=False)
        if predictions.shape != (len(self.split.prediction_positions),):
            raise ValueError(
                f"JMP-L returned {predictions.shape}, expected "
                f"({len(self.split.prediction_positions)},)"
            )
        if not np.all(np.isfinite(predictions)):
            raise ValueError("JMP-L produced non-finite predictions")
        expected_latent_shape = (len(self.split.prediction_positions), LATENT_DIM)
        if latents.shape != expected_latent_shape:
            raise ValueError(
                f"JMP-L returned latent shape {latents.shape}, expected {expected_latent_shape}"
            )
        if not np.all(np.isfinite(latents)):
            raise ValueError("JMP-L produced non-finite hidden embeddings")
        fitted.prediction_latents = latents
        return predictions, None

    def export_latent(
        self, fitted: JMPLFitted, prepared: JMPLPrepared
    ) -> tuple[np.ndarray, np.ndarray]:
        if fitted.prediction_latents is None:
            raise RuntimeError("JMP-L prediction must run before latent export")
        return fitted.prediction_latents, np.ones(
            len(fitted.prediction_latents), dtype=np.bool_
        )

    def checkpoint_path(self, fitted: JMPLFitted) -> Path:
        return fitted.checkpoint

    def preprocessing_manifest(self, prepared: JMPLPrepared) -> Path:
        return prepared.preprocessing_manifest

    def finalize_progress(self, exc: Exception | None = None) -> None:
        # Full optimizer-state checkpoints exist only to make interrupted Fast-32
        # jobs resumable. Once the immutable artifact is complete, retaining
        # them would roughly triple transfer/storage size.
        if exc is None and _fast32_enabled(self.request.extra):
            shutil.rmtree(self.request.output_path / "resume", ignore_errors=True)

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hpsafe_sota.data.manifest import TaskManifest


@dataclass(frozen=True)
class AuxiliarySelection:
    task_name: str
    positions: np.ndarray
    match_kind: str
    excluded_exact_entities: int
    excluded_by_cap: int


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _definition(manifest: TaskManifest) -> dict[str, Any]:
    return dict(manifest.metadata["definition"])


def _match_kind(target: TaskManifest, auxiliary: TaskManifest) -> str:
    target_input = str(_definition(target)["input_type"])
    auxiliary_input = str(_definition(auxiliary)["input_type"])
    return "structure" if target_input == auxiliary_input == "structure" else "composition"


def target_forbidden_positions(
    manifest: TaskManifest,
    *,
    outer_fold: int,
    prediction_positions: np.ndarray,
) -> np.ndarray:
    outer_test = manifest.positions("outer_test", outer_fold)
    return np.unique(
        np.concatenate(
            [outer_test.astype(np.int64), np.asarray(prediction_positions, dtype=np.int64)]
        )
    )


def build_auxiliary_training_positions(
    project_root: Path,
    *,
    target_manifest: TaskManifest,
    target_prediction_positions: np.ndarray,
    outer_fold: int,
    auxiliary_tasks: list[str],
    max_samples_per_task: int,
    seed: int,
    receipt_path: Path | None = None,
) -> dict[str, AuxiliarySelection]:
    forbidden_positions = target_forbidden_positions(
        target_manifest,
        outer_fold=outer_fold,
        prediction_positions=target_prediction_positions,
    )
    results: dict[str, AuxiliarySelection] = {}
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "policy": "target_centric_cross_task_entity_firewall",
        "target_task": target_manifest.task_name,
        "outer_fold": outer_fold,
        "target_outer_test_always_forbidden": True,
        "current_prediction_split_forbidden": True,
        "target_forbidden_position_count": int(len(forbidden_positions)),
        "target_forbidden_positions_sha256": _digest(forbidden_positions),
        "auxiliaries": {},
    }
    rng = np.random.default_rng(seed)
    for task_name in auxiliary_tasks:
        auxiliary = TaskManifest(project_root / "data/manifests" / f"{task_name}.npz")
        auxiliary.require_valid()
        kind = _match_kind(target_manifest, auxiliary)
        forbidden_hashes = set(target_manifest.entity_hashes(kind, forbidden_positions).tolist())
        auxiliary_hashes = auxiliary.entity_hashes(kind)
        keep_mask = np.asarray([value not in forbidden_hashes for value in auxiliary_hashes], dtype=np.bool_)
        allowed = np.flatnonzero(keep_mask).astype(np.int64)
        excluded_exact = int(len(auxiliary_hashes) - len(allowed))
        excluded_by_cap = 0
        if max_samples_per_task > 0 and len(allowed) > max_samples_per_task:
            selected = set(
                rng.choice(allowed, size=max_samples_per_task, replace=False).astype(np.int64).tolist()
            )
            allowed = np.asarray([value for value in allowed if int(value) in selected], dtype=np.int64)
            excluded_by_cap = int(len(auxiliary_hashes) - excluded_exact - len(allowed))
        remaining_overlap = set(auxiliary.entity_hashes(kind, allowed).tolist()) & forbidden_hashes
        if remaining_overlap:
            raise RuntimeError(f"{task_name}: cross-task firewall left forbidden entities")
        selection = AuxiliarySelection(
            task_name=task_name,
            positions=allowed,
            match_kind=kind,
            excluded_exact_entities=excluded_exact,
            excluded_by_cap=excluded_by_cap,
        )
        results[task_name] = selection
        receipt["auxiliaries"][task_name] = {
            "match_kind": kind,
            "source_samples": int(len(auxiliary_hashes)),
            "allowed_samples": int(len(allowed)),
            "excluded_exact_entities": excluded_exact,
            "excluded_by_cap": excluded_by_cap,
            "allowed_positions_sha256": _digest(allowed),
            "remaining_forbidden_overlap": 0,
            "auxiliary_manifest_digest": auxiliary.digest,
        }
    if receipt_path is not None:
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results

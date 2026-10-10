from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import JARVIS_TASKS, ROLE_TO_MANIFEST_KEY
from .io import sha256_file


@dataclass(frozen=True)
class JarvisManifest:
    path: Path
    task: str
    sample_ids: np.ndarray
    material_ids: np.ndarray
    targets: np.ndarray | None
    role_positions: dict[str, np.ndarray]
    metadata: dict
    file_sha256: str

    @classmethod
    def load(cls, path: Path, *, load_targets: bool = True) -> "JarvisManifest":
        path = path.expanduser().resolve()
        with np.load(path, allow_pickle=False) as loaded:
            metadata = json.loads(str(loaded["metadata_json"][0]))
            sample_ids = loaded["sample_ids"].astype(str)
            material_ids = loaded["material_ids"].astype(str)
            targets = (
                np.asarray(loaded["targets"], dtype=np.float64).copy()
                if load_targets
                else None
            )
            roles = {
                role: np.asarray(loaded[key], dtype=np.int64).copy()
                for role, key in ROLE_TO_MANIFEST_KEY.items()
            }
        task = str(metadata["definition"]["task_name"])
        if task not in JARVIS_TASKS:
            raise ValueError(f"Not one of the five pinned JARVIS tasks: {task}")
        if int(metadata.get("n_outer_folds", -1)) != 1:
            raise ValueError(f"{task}: expected the one-holdout JARVIS protocol")
        if metadata.get("split_protocol") != "jarvis_leaderboard_holdout":
            raise ValueError(f"{task}: wrong split protocol")
        if len(sample_ids) != len(material_ids) or (
            targets is not None and len(targets) != len(sample_ids)
        ):
            raise ValueError(f"{task}: row-aligned manifest arrays differ in length")
        if len(set(sample_ids.tolist())) != len(sample_ids):
            raise ValueError(f"{task}: artifact sample IDs must be unique")
        role_sets = {key: set(value.tolist()) for key, value in roles.items()}
        if role_sets["train"] & role_sets["val"] or role_sets["train"] & role_sets["test"]:
            raise ValueError(f"{task}: official train overlaps val/test")
        if role_sets["val"] & role_sets["test"]:
            raise ValueError(f"{task}: official val overlaps test")
        return cls(
            path=path,
            task=task,
            sample_ids=sample_ids,
            material_ids=material_ids,
            targets=targets,
            role_positions=roles,
            metadata=metadata,
            file_sha256=sha256_file(path),
        )

    def ids(self, role: str) -> np.ndarray:
        return self.sample_ids[self.role_positions[role]]

    def materials(self, role: str) -> np.ndarray:
        return self.material_ids[self.role_positions[role]]

    def y(self, role: str) -> np.ndarray:
        if self.targets is None:
            raise RuntimeError(f"Targets were deliberately not loaded for {self.task}")
        return self.targets[self.role_positions[role]]


def load_manifests(
    stage1_root: Path, *, load_targets: bool = True
) -> dict[str, JarvisManifest]:
    root = stage1_root.expanduser().resolve() / "data" / "manifests"
    return {
        task: JarvisManifest.load(root / f"{task}.npz", load_targets=load_targets)
        for task in JARVIS_TASKS
    }

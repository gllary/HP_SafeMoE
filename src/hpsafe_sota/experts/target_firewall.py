from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _index_digest(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass
class TargetAccessFirewall:
    """Permit label access only for the declared fitting rows."""

    manifest: Any
    allowed_fit_positions: np.ndarray
    prediction_positions: np.ndarray
    _accessed_positions: list[int] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.allowed_fit_positions = np.unique(
            np.asarray(self.allowed_fit_positions, dtype=np.int64)
        )
        self.prediction_positions = np.unique(
            np.asarray(self.prediction_positions, dtype=np.int64)
        )
        overlap = np.intersect1d(self.allowed_fit_positions, self.prediction_positions)
        if overlap.size:
            raise ValueError(
                "Target firewall refused overlapping fit/prediction rows: "
                f"{overlap[:10].tolist()}"
            )

    def targets_for_fit(self, positions: np.ndarray) -> np.ndarray:
        requested = np.asarray(positions, dtype=np.int64)
        if requested.ndim != 1:
            raise ValueError("Target positions must be one-dimensional")
        if np.setdiff1d(requested, self.allowed_fit_positions).size:
            raise PermissionError("Attempted to read targets outside the declared fitting rows")
        if np.intersect1d(requested, self.prediction_positions).size:
            raise PermissionError("Attempted to read prediction targets during fitting")
        self._accessed_positions.extend(int(value) for value in requested.tolist())
        return np.asarray(self.manifest.targets[requested]).copy()

    def receipt(self) -> dict[str, Any]:
        accessed = np.unique(np.asarray(self._accessed_positions, dtype=np.int64))
        illegal = np.setdiff1d(accessed, self.allowed_fit_positions)
        prediction_overlap = np.intersect1d(accessed, self.prediction_positions)
        if illegal.size or prediction_overlap.size:
            raise RuntimeError("Target firewall validation failed")
        return {
            "schema_version": 1,
            "policy": "fit_targets_only_prediction_targets_unavailable",
            "allowed_fit_count": int(self.allowed_fit_positions.size),
            "allowed_fit_positions_sha256": _index_digest(self.allowed_fit_positions),
            "prediction_count": int(self.prediction_positions.size),
            "prediction_positions_sha256": _index_digest(self.prediction_positions),
            "accessed_count": int(accessed.size),
            "accessed_positions_sha256": _index_digest(accessed),
            "access_is_subset_of_fit": True,
            "prediction_target_access_count": 0,
        }

    def write_receipt(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent,
                mode="w",
                encoding="utf-8",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.receipt(), handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return path

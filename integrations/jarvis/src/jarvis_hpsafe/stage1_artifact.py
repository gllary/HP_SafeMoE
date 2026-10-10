from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import read_json, sha256_file


@dataclass(frozen=True)
class Stage1Artifact:
    root: Path
    sample_ids: np.ndarray
    y_true: np.ndarray | None
    prediction: np.ndarray
    hidden: np.ndarray
    available: np.ndarray
    checkpoint_sha256: str


def load_stage1_artifact(
    path: Path,
    *,
    expected_latent_dim: int = 128,
    load_targets: bool = False,
) -> Stage1Artifact:
    path = path.expanduser().resolve()
    metadata = read_json(path / "metadata.json")
    prediction_path = path / str(metadata["predictions"]["filename"])
    if sha256_file(prediction_path) != metadata["predictions"]["sha256"]:
        raise ValueError(f"Stage1 prediction hash mismatch: {prediction_path}")
    latent_record = metadata.get("latents")
    if not isinstance(latent_record, dict):
        raise ValueError(f"Stage1 trained hidden is absent: {path}")
    latent_path = path / str(latent_record["filename"])
    if sha256_file(latent_path) != latent_record["sha256"]:
        raise ValueError(f"Stage1 latent hash mismatch: {latent_path}")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        sample_ids = loaded["sample_ids"].astype(str)
        y_true = (
            np.asarray(loaded["y_true"], dtype=np.float64) if load_targets else None
        )
        prediction = np.asarray(loaded["y_pred"], dtype=np.float64)
    with np.load(latent_path, allow_pickle=False) as loaded:
        latent_ids = loaded["sample_ids"].astype(str)
        hidden = np.asarray(loaded["latents"], dtype=np.float32)
        available = np.asarray(loaded["available"], dtype=np.bool_)
    if not np.array_equal(sample_ids, latent_ids):
        raise ValueError(f"Stage1 prediction/latent IDs differ: {path}")
    if hidden.shape != (len(sample_ids), expected_latent_dim):
        raise ValueError(f"Unexpected Stage1 hidden shape {hidden.shape}: {path}")
    if prediction.shape != (len(sample_ids),) or available.shape != (len(sample_ids),):
        raise ValueError(f"Malformed Stage1 arrays: {path}")
    if not np.isfinite(prediction).all() or not np.isfinite(hidden[available]).all():
        raise ValueError(f"Non-finite Stage1 values: {path}")
    checkpoint = str(metadata.get("provenance", {}).get("checkpoint_sha256", ""))
    if len(checkpoint) != 64:
        raise ValueError(f"Missing Stage1 checkpoint hash: {path}")
    return Stage1Artifact(
        root=path,
        sample_ids=sample_ids,
        y_true=y_true,
        prediction=prediction,
        hidden=hidden,
        available=available,
        checkpoint_sha256=checkpoint,
    )


def align_artifact(artifact: Stage1Artifact, expected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {value: index for index, value in enumerate(artifact.sample_ids.tolist())}
    missing = [value for value in expected_ids.tolist() if value not in lookup]
    if missing:
        raise ValueError(f"Stage1 artifact misses IDs: {missing[:10]}")
    order = np.asarray([lookup[value] for value in expected_ids.tolist()], dtype=np.int64)
    return artifact.prediction[order], artifact.hidden[order], artifact.available[order]

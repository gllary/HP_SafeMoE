from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import ijson
import numpy as np

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.data.raw import file_sha256, load_task_definitions
from hpsafe_sota.stage1_tpot.official import ELEMENTS, sha256_file

TOKEN = re.compile(r"([A-Z][a-z]?)([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def featurize_official_steel_compositions(values: list[Any]) -> np.ndarray:
    """Reproduce the official 14-column steels composition convention strictly."""

    rows: list[list[float]] = []
    allowed = set(ELEMENTS)
    for row_index, raw in enumerate(values):
        if not isinstance(raw, str):
            raise TypeError(f"steels row {row_index} is not a composition string")
        tokens = TOKEN.findall(raw.replace(" ", ""))
        if not tokens or "".join(f"{e}{v}" for e, v in tokens) != raw.replace(" ", ""):
            raise ValueError(f"steels row {row_index} is not in the official element/value syntax: {raw!r}")
        mapping: dict[str, float] = {}
        for element, amount in tokens:
            if element not in allowed:
                raise ValueError(
                    f"steels row {row_index} contains {element}, outside the official TPOT-Mat schema"
                )
            if element in mapping:
                raise ValueError(f"steels row {row_index} repeats {element}")
            mapping[element] = float(amount)
        vector = [mapping.get(element, 0.0) for element in ELEMENTS]
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"steels row {row_index} contains a non-finite amount")
        rows.append(vector)
    result = np.asarray(rows, dtype=np.float64)
    if result.shape != (len(values), len(ELEMENTS)):
        raise AssertionError("TPOT-Mat composition feature shape drifted")
    return result


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
            temporary = handle.name
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def build_feature_cache(project_root: Path, output_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.resolve()
    definition = load_task_definitions(project_root)["steels"]
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    if definition.input_type != "composition" or definition.n_samples != 312:
        raise ValueError("TPOT-Mat expects the canonical 312-row steels composition task")
    if file_sha256(definition.raw_path) != definition.raw_sha256:
        raise ValueError("steels raw payload hash drifted")
    compositions: list[str] = []
    with gzip.open(definition.raw_path, "rb") as handle:
        for index, row in enumerate(ijson.items(handle, "data.item", use_float=True)):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"Malformed steels raw row {index}")
            # The source container also has a target column.  Only row[0] is
            # assigned or passed to the feature builder; no target value is stored.
            compositions.append(row[0])
    features = featurize_official_steel_compositions(compositions)
    if len(features) != len(manifest.sample_ids):
        raise ValueError("TPOT-Mat cache row count differs from the immutable manifest")
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_npy(output_root / "features.npy", features)
    ids = manifest.sample_ids.astype(str)
    _atomic_npy(output_root / "sample_ids.npy", ids)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "task": "steels",
        "feature_recipe": "official_tpot_mat_14_element_vector_fe_retained",
        "elements": list(ELEMENTS),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "manifest_digest": manifest.digest,
        "raw_sha256": definition.raw_sha256,
        "features_sha256": sha256_file(output_root / "features.npy"),
        "sample_ids_sha256": sha256_file(output_root / "sample_ids.npy"),
        "source_target_column_present": True,
        "source_target_values_used_for_features": False,
        "target_values_stored": False,
        "fit_statistics_used": False,
    }
    _atomic_json(output_root / "receipt.json", receipt)
    return receipt


def load_feature_cache(project_root: Path, cache_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    project_root = project_root.resolve()
    cache_root = cache_root.resolve()
    receipt = json.loads((cache_root / "receipt.json").read_text(encoding="utf-8"))
    manifest = TaskManifest(project_root / "data/manifests/steels.npz")
    manifest.require_valid()
    if (
        receipt.get("status") != "complete"
        or receipt.get("manifest_digest") != manifest.digest
        or receipt.get("source_target_values_used_for_features") is not False
        or receipt.get("target_values_stored") is not False
        or receipt.get("elements") != list(ELEMENTS)
    ):
        raise ValueError("Invalid TPOT-Mat steels feature-cache receipt")
    if sha256_file(cache_root / "features.npy") != receipt.get("features_sha256"):
        raise ValueError("TPOT-Mat feature cache hash drifted")
    if sha256_file(cache_root / "sample_ids.npy") != receipt.get("sample_ids_sha256"):
        raise ValueError("TPOT-Mat feature-cache sample IDs hash drifted")
    features = np.load(cache_root / "features.npy", allow_pickle=False)
    ids = np.load(cache_root / "sample_ids.npy", allow_pickle=False).astype(str)
    if features.shape != (312, len(ELEMENTS)) or not np.array_equal(ids, manifest.sample_ids):
        raise ValueError("TPOT-Mat feature cache no longer aligns to the steels manifest")
    return np.asarray(features, dtype=np.float64), ids, receipt

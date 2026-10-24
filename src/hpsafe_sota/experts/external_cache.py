from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MATTERVIAL_FEATURE_PATTERNS: dict[str, str] = {
    "megnet_mm": r"^MEGNet_MatMinerEncoded_v1_",
    "megnet_ofm": r"^MEGNet_OFMEncoded_v1_",
    "mvl_all": r"^MVL(?:16|32)_",
    "orb_v3": r"^ORB_v3_",
}

DEFAULT_ID_COLUMNS = (
    "__moddata_index__",
    "material_id",
    "mbid",
    "sample_id",
    "id",
    "identifier",
)


def _is_moddata_pickle(path: Path) -> bool:
    return path.name.lower().endswith((".pkl", ".pkl.gz", ".pickle", ".pickle.gz"))


def _read_moddata_feature_frame(path: Path) -> pd.DataFrame:
    """Load the published feature frame from an official MODData pickle.

    The pickle is executable input and must therefore be checksum-verified by
    the caller before this function is reached.  MODData.load reconstructs the
    complete container; this adapter exports ``df_featurized`` together with
    its immutable row index.
    """
    try:
        from modnet.preprocessing import MODData
    except ImportError as exc:
        raise ValueError(
            "Reading MatterVial .pkl.gz features requires the pinned MODNet environment"
        ) from exc
    payload = MODData.load(path)
    frame = getattr(payload, "df_featurized", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"MODData does not contain a non-empty df_featurized: {path}")
    if "__moddata_index__" in frame.columns:
        raise ValueError("MODData feature frame already contains the reserved ID column")
    # Shallow copy: inserting the synthetic ID must not duplicate a potentially
    # multi-gigabyte published feature matrix in memory.
    result = frame.copy(deep=False)
    result.insert(0, "__moddata_index__", result.index.astype(str))
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy.tmp", delete=False) as handle:
        np.save(handle, array, allow_pickle=False)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _manifest_digest(manifest: Any) -> str:
    if hasattr(manifest, "digest"):
        digest = manifest.digest
        return str(digest() if callable(digest) else digest)
    if hasattr(manifest, "semantic_digest"):
        digest = manifest.semantic_digest
        return str(digest() if callable(digest) else digest)
    payload = {
        "task": str(manifest.task_name),
        "sample_ids": np.asarray(manifest.sample_ids).astype(str).tolist(),
    }
    return _json_sha256(payload)


def _read_header(path: Path) -> list[str]:
    if _is_moddata_pickle(path):
        return _read_moddata_feature_frame(path).columns.astype(str).tolist()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise ValueError("Parquet header inspection requires pyarrow") from exc
        return [str(value) for value in parquet.ParquetFile(path).schema.names]
    raise ValueError(f"Unsupported external feature file: {path}")


def read_feature_frame(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if _is_moddata_pickle(path):
        frame = _read_moddata_feature_frame(path)
        return frame if columns is None else frame.loc[:, list(columns)]
    suffix = path.suffix.lower()
    if suffix in {".csv", ".gz"}:
        return pd.read_csv(path, usecols=columns, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported external feature file: {path}")


def resolve_id_column(columns: Sequence[str], preferred: str | None = None) -> str:
    if preferred is not None:
        if preferred not in columns:
            raise ValueError(f"Requested id column {preferred!r} is absent")
        return preferred
    for candidate in DEFAULT_ID_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(f"No supported sample-id column found; tried {DEFAULT_ID_COLUMNS}")


def feature_columns(
    columns: Sequence[str], patterns: Mapping[str, str]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    used: set[str] = set()
    for block, pattern in patterns.items():
        # MatterVial's published trainer sorts the union of selected feature
        # names before fold preprocessing.  Sorting each disjoint prefix block
        # preserves that exact, deterministic order and freezes latent semantics.
        matched = sorted(column for column in columns if re.search(pattern, str(column)))
        if not matched:
            raise ValueError(f"No columns matched MatterVial block {block!r} ({pattern})")
        duplicates = used.intersection(matched)
        if duplicates:
            raise ValueError(f"Feature blocks overlap: {sorted(duplicates)[:5]}")
        used.update(matched)
        result[block] = matched
    return result


def discover_feature_source(
    source_root: Path,
    *,
    task_name: str,
    patterns: Mapping[str, str] = MATTERVIAL_FEATURE_PATTERNS,
    expected_basename: str | None = None,
) -> Path:
    if expected_basename:
        exact = sorted(
            path.resolve()
            for path in source_root.rglob(expected_basename)
            if path.is_file() and task_name.lower() in str(path).lower()
        )
        if len(exact) != 1:
            raise ValueError(
                f"Expected exactly one official {task_name} feature file named "
                f"{expected_basename!r}; found {len(exact)}: "
                + ", ".join(str(path) for path in exact[:8])
            )
        return exact[0]
    candidates: list[tuple[int, Path]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or not (
            path.name.lower().endswith(".csv")
            or path.name.lower().endswith(".csv.gz")
            or path.suffix.lower() in {".parquet", ".pq"}
        ):
            continue
        try:
            columns = _read_header(path)
            resolve_id_column(columns)
            feature_columns(columns, patterns)
        except (OSError, ValueError):
            continue
        score = 10 if task_name.lower() in str(path).lower() else 0
        score += sum(task_name.lower() in column.lower() for column in columns)
        candidates.append((score, path))
    if not candidates:
        raise FileNotFoundError(
            f"No file below {source_root} contains all required MatterVial blocks"
        )
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]
    if len(best) > 1:
        raise ValueError(
            "Ambiguous MatterVial source files; pass --source explicitly: "
            + ", ".join(str(path) for path in best[:8])
        )
    return best[0]


def _apply_sample_id_alias(
    source_ids: pd.Series,
    alias: Mapping[str, Any] | None,
) -> tuple[pd.Series, dict[str, Any]]:
    values = source_ids.astype(str)
    if alias is None:
        return values, {
            "policy": "exact_id_match",
            "alias_applied": False,
            "mapping_count": 0,
            "bijection_verified": True,
        }
    required = {
        "policy",
        "source_regex",
        "target_template",
        "expected_mapping_count",
    }
    if set(alias) != required or alias.get("policy") != "configured_regex_full_bijection":
        raise ValueError(
            "sample_id_alias must be a configured_regex_full_bijection with exactly "
            f"{sorted(required)}"
        )
    pattern = re.compile(str(alias["source_regex"]))
    template = str(alias["target_template"])
    mapped: list[str] = []
    unmatched: list[str] = []
    for value in values.tolist():
        match = pattern.fullmatch(value)
        if match is None:
            if len(unmatched) < 5:
                unmatched.append(value)
            continue
        try:
            mapped.append(template.format(**match.groupdict()))
        except (KeyError, ValueError) as exc:
            raise ValueError("Invalid sample_id_alias target_template") from exc
    if unmatched or len(mapped) != len(values):
        raise ValueError(
            "Configured sample ID alias did not match every source ID: "
            f"unmatched={unmatched} matched={len(mapped)} total={len(values)}"
        )
    expected_count = int(alias["expected_mapping_count"])
    if expected_count != len(mapped):
        raise ValueError(
            f"Configured sample ID alias mapped {len(mapped)} rows, expected {expected_count}"
        )
    if len(set(mapped)) != len(mapped):
        raise ValueError("Configured sample ID alias is not one-to-one")
    mapped_values = pd.Series(mapped, index=values.index, dtype="object")
    return mapped_values, {
        **dict(alias),
        "alias_applied": True,
        "mapping_count": len(mapped),
        "bijection_verified": True,
        "source_ids_sha256": hashlib.sha256(
            "\n".join(values.tolist()).encode("utf-8")
        ).hexdigest(),
        "mapped_ids_sha256": hashlib.sha256(
            "\n".join(mapped).encode("utf-8")
        ).hexdigest(),
    }


def _align_frame(
    frame: pd.DataFrame,
    id_column: str,
    sample_ids: np.ndarray,
    sample_id_alias: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_ids, alignment = _apply_sample_id_alias(frame[id_column], sample_id_alias)
    if source_ids.duplicated().any():
        examples = source_ids[source_ids.duplicated()].head().tolist()
        raise ValueError(f"Duplicate external feature sample IDs: {examples}")
    expected = np.asarray(sample_ids).astype(str)
    source_id_values = source_ids.to_numpy(dtype=str)
    source_set = set(source_id_values.tolist())
    missing = sorted(set(expected).difference(source_set))
    extra = sorted(source_set.difference(expected))
    if missing or extra:
        raise ValueError(
            f"External feature ID mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    lookup = {value: position for position, value in enumerate(source_id_values.tolist())}
    order = np.asarray([lookup[value] for value in expected.tolist()], dtype=np.int64)
    aligned = frame.iloc[order].copy(deep=False)
    aligned.index = expected
    return aligned, alignment


def build_external_feature_cache(
    *,
    output_dir: Path,
    source_path: Path,
    manifest: Any,
    provider: str,
    provider_commit: str,
    patterns: Mapping[str, str] = MATTERVIAL_FEATURE_PATTERNS,
    id_column: str | None = None,
    expected_dimensions: Mapping[str, int] | None = None,
    forbidden_feature_patterns: Sequence[str] | None = None,
    sample_id_alias: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    moddata_source = _is_moddata_pickle(source_path)
    loaded_frame = _read_moddata_feature_frame(source_path) if moddata_source else None
    source_columns = (
        loaded_frame.columns.astype(str).tolist()
        if loaded_frame is not None
        else _read_header(source_path)
    )
    generic_target_names = {"target", "targets", "label", "labels", "y", "y_true"}
    definition = getattr(manifest, "metadata", {}).get("definition", {})
    declared_target = str(definition.get("target", "")).strip().casefold()
    forbidden_target_names = {
        *generic_target_names,
        *([declared_target] if declared_target else []),
    }
    source_target_columns = [
        str(column)
        for column in source_columns
        if str(column).strip().casefold() in forbidden_target_names
    ]
    resolved_id = resolve_id_column(source_columns, id_column)
    matched_blocks = feature_columns(source_columns, patterns)
    forbidden_feature_patterns = [str(value) for value in (forbidden_feature_patterns or [])]
    forbidden_selected = sorted(
        {
            str(column)
            for columns in matched_blocks.values()
            for column in columns
            if any(re.search(pattern, str(column)) for pattern in forbidden_feature_patterns)
        }
    )
    if forbidden_selected:
        raise ValueError(
            "Frozen feature recipe selected explicitly forbidden columns: "
            f"{forbidden_selected[:10]}"
        )
    blocks = {
        name: [
            column
            for column in columns
            if str(column).strip().casefold() not in forbidden_target_names
        ]
        for name, columns in matched_blocks.items()
    }
    empty_blocks = sorted(name for name, columns in blocks.items() if not columns)
    if empty_blocks:
        raise ValueError(
            "Feature patterns select only forbidden target columns for blocks: "
            f"{empty_blocks}"
        )
    selected_columns = [resolved_id, *[column for values in blocks.values() for column in values]]
    frame = (
        loaded_frame.loc[:, selected_columns]
        if loaded_frame is not None
        else read_feature_frame(source_path, columns=selected_columns)
    )
    aligned, sample_id_alignment = _align_frame(
        frame,
        resolved_id,
        np.asarray(manifest.sample_ids),
        sample_id_alias,
    )
    expected_dimensions = dict(expected_dimensions or {})
    source_digest = sha256_file(source_path)
    cache_inputs = {
        "provider": provider,
        "provider_commit": provider_commit,
        "task": str(manifest.task_name),
        "manifest_sha256": _manifest_digest(manifest),
        "source_sha256": source_digest,
        "patterns": dict(patterns),
        "forbidden_feature_patterns": forbidden_feature_patterns,
        "sample_id_alias": None if sample_id_alias is None else dict(sample_id_alias),
    }
    cache_key = _json_sha256(cache_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_receipts: dict[str, Any] = {}
    for block_name, columns in blocks.items():
        array = aligned.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )
        expected = expected_dimensions.get(block_name)
        if expected is not None and array.shape[1] != expected:
            raise ValueError(
                f"{block_name} dimension {array.shape[1]} does not match expected {expected}"
            )
        path = output_dir / f"{block_name}.npy"
        _atomic_npy(path, array)
        block_receipts[block_name] = {
            "path": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": sha256_file(path),
            "feature_names": list(columns),
            "nonfinite_count": int((~np.isfinite(array)).sum()),
        }

    sample_ids = np.asarray(manifest.sample_ids).astype(str)
    sample_path = output_dir / "sample_ids.npy"
    _atomic_npy(sample_path, sample_ids)
    schema = {
        "schema_version": 1,
        "task": str(manifest.task_name),
        "sample_count": int(sample_ids.size),
        "id_column": resolved_id,
        "blocks": {name: value["feature_names"] for name, value in block_receipts.items()},
    }
    _atomic_json(output_dir / "feature_schema.json", schema)
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "provider": provider,
        "provider_commit": provider_commit,
        "task": str(manifest.task_name),
        "manifest_sha256": cache_inputs["manifest_sha256"],
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": source_digest,
        "sample_ids_path": sample_path.name,
        "sample_ids_sha256": sha256_file(sample_path),
        "feature_schema_sha256": sha256_file(output_dir / "feature_schema.json"),
        "blocks": block_receipts,
        "target_values_stored": False,
        "source_target_columns_read": False,
        "source_payload_may_contain_targets": bool(moddata_source or source_target_columns),
        "source_target_columns_present": source_target_columns,
        "declared_manifest_target_excluded": (
            None if not declared_target else str(definition.get("target"))
        ),
        "target_exclusion_policy": "generic_aliases_plus_manifest_declared_target_before_value_read",
        "source_target_values_accessed_by_cache_builder": False,
        "forbidden_feature_patterns": forbidden_feature_patterns,
        "forbidden_feature_columns_selected": [],
        "sample_id_alias": None if sample_id_alias is None else dict(sample_id_alias),
        "sample_id_alignment": sample_id_alignment,
        "source_format": (
            "moddata_pickle_gzip"
            if moddata_source
            else (
                "official_feature_csv"
                if provider == "MatterVial_SupportData" and source_path.suffix.lower() == ".csv"
                else source_path.suffix.lower()
            )
        ),
        "source_columns_read_count": len(selected_columns),
        "row_alignment": (
            "configured_alias_then_exact_manifest_sample_id_order"
            if sample_id_alias is not None
            else "exact_manifest_sample_id_order"
        ),
    }
    _atomic_json(output_dir / "receipt.json", receipt)
    return receipt


@dataclass(frozen=True)
class ExternalFeatureCache:
    root: Path
    receipt: dict[str, Any]
    sample_ids: np.ndarray

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        manifest: Any | None = None,
        verify_hashes: bool = False,
    ) -> ExternalFeatureCache:
        root = root.resolve()
        receipt_path = root / "receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Incomplete external feature cache: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("status") != "complete"
            or receipt.get("target_values_stored") is not False
            or receipt.get("source_target_columns_read") is not False
        ):
            raise ValueError("External cache receipt is incomplete or contains targets")
        sample_path = root / str(receipt["sample_ids_path"])
        sample_ids = np.load(sample_path, allow_pickle=False).astype(str)
        if manifest is not None:
            if receipt.get("task") != str(manifest.task_name):
                raise ValueError("External cache task does not match manifest")
            if receipt.get("manifest_sha256") != _manifest_digest(manifest):
                raise ValueError("External cache manifest digest mismatch")
            if not np.array_equal(sample_ids, np.asarray(manifest.sample_ids).astype(str)):
                raise ValueError("External cache sample order differs from manifest")
        if verify_hashes:
            expected = {sample_path: receipt["sample_ids_sha256"]}
            expected[root / "feature_schema.json"] = receipt["feature_schema_sha256"]
            for block in receipt["blocks"].values():
                expected[root / block["path"]] = block["sha256"]
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise ValueError(f"External cache checksum mismatch: {path}")
        return cls(root=root, receipt=receipt, sample_ids=sample_ids)

    @property
    def cache_key(self) -> str:
        return str(self.receipt["cache_key"])

    def block(self, name: str, positions: Iterable[int]) -> np.ndarray:
        if name not in self.receipt["blocks"]:
            raise KeyError(f"Unknown external feature block {name!r}")
        metadata = self.receipt["blocks"][name]
        array = np.load(self.root / metadata["path"], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"]:
            raise ValueError(f"Cached block shape changed for {name}")
        index = np.asarray(list(positions), dtype=np.int64)
        return np.asarray(array[index], dtype=np.float32)

    def frame(self, names: Sequence[str], positions: Iterable[int]) -> pd.DataFrame:
        index = np.asarray(list(positions), dtype=np.int64)
        arrays: list[np.ndarray] = []
        columns: list[str] = []
        for name in names:
            metadata = self.receipt["blocks"][name]
            arrays.append(self.block(name, index))
            columns.extend(str(value) for value in metadata["feature_names"])
        matrix = np.concatenate(arrays, axis=1)
        return pd.DataFrame(matrix, columns=columns, index=self.sample_ids[index])

from __future__ import annotations

import gzip
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import numpy as np
import orjson
import yaml


@dataclass(frozen=True)
class TaskDefinition:
    task_name: str
    matbench_name: str
    input_type: str
    task_type: str
    n_samples: int
    target: str
    unit: str | None
    metric: str
    direction: str
    raw_path: Path
    raw_sha256: str


@dataclass(frozen=True)
class RawSample:
    position: int
    raw_index: int
    sample_id: str
    input: Any
    target: float
    fingerprint: bytes
    composition_hash: bytes
    structure_hash: bytes


def _yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_task_definitions(project_root: Path) -> dict[str, TaskDefinition]:
    project_root = project_root.resolve()
    registry = _yaml(project_root / "configs/sota_registry.yaml")
    sources = _yaml(project_root / "configs/reference/data_sources.yaml")
    raw_root = project_root / str(sources["local_root"])
    definitions: dict[str, TaskDefinition] = {}
    for task_name, task in registry["tasks"].items():
        raw = sources["files"][task_name]
        definitions[task_name] = TaskDefinition(
            task_name=task_name,
            matbench_name=str(task["matbench_name"]),
            input_type=str(task["input_type"]),
            task_type=str(task["task_type"]),
            n_samples=int(task["n_samples"]),
            target=str(task["target"]),
            unit=None if task.get("unit") is None else str(task["unit"]),
            metric=str(task["metric"]),
            direction=str(task["direction"]),
            raw_path=raw_root / str(raw["filename"]),
            raw_sha256=str(raw["sha256"]),
        )
    return definitions


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_header(path: Path, *, max_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    """Read the small `index`/`columns` prefix without materializing structure rows."""

    marker = b'"data":'
    payload = bytearray()
    with gzip.open(path, "rb") as handle:
        while marker not in payload:
            chunk = handle.read(64 * 1024)
            if not chunk:
                raise ValueError(f"No top-level data field found in {path}")
            payload.extend(chunk)
            if len(payload) > max_bytes:
                raise ValueError(f"Header exceeds {max_bytes} bytes in {path}")
    offset = payload.index(marker)
    header = orjson.loads(bytes(payload[:offset]) + marker + b"[]}")
    if not isinstance(header, dict):
        raise ValueError(f"Malformed Matbench JSON header in {path}")
    return header


def _hash_payload(payload: Any) -> bytes:
    canonical = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(canonical).digest()


def _species_amounts(raw_input: Any) -> dict[str, float]:
    if isinstance(raw_input, str):
        try:
            from pymatgen.core import Composition
        except ImportError as exc:  # pragma: no cover - dependency error is server-specific
            raise RuntimeError("pymatgen is required to normalize composition formulas") from exc
        return {str(k): float(v) for k, v in Composition(raw_input).get_el_amt_dict().items()}
    if not isinstance(raw_input, dict) or "sites" not in raw_input:
        raise TypeError(f"Unsupported Matbench input for entity hashing: {type(raw_input)!r}")
    amounts: dict[str, float] = {}
    for site in raw_input["sites"]:
        for species in site.get("species", []):
            element = str(species.get("element") or species.get("name"))
            amounts[element] = amounts.get(element, 0.0) + float(species.get("occu", 1.0))
    return amounts


def composition_entity_hash(raw_input: Any) -> bytes:
    amounts = _species_amounts(raw_input)
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError("Composition has no positive species amount")
    normalized = [(element, round(amount / total, 10)) for element, amount in sorted(amounts.items())]
    return _hash_payload(normalized)


def structure_entity_hash(raw_input: Any) -> bytes:
    """Hash a normalized exact structure serialization for cross-task overlap validation.

    This intentionally removes site properties and ordering, wraps fractional
    coordinates into the unit cell, and rounds numerical noise. Composition
    hashes are also stored and provide the conservative cross-task guard.
    """

    if not isinstance(raw_input, dict) or "lattice" not in raw_input:
        return b""
    matrix = [[round(float(value), 8) for value in vector] for vector in raw_input["lattice"]["matrix"]]
    sites: list[tuple[Any, ...]] = []
    for site in raw_input["sites"]:
        species = tuple(
            sorted(
                (
                    str(item.get("element") or item.get("name")),
                    round(float(item.get("occu", 1.0)), 10),
                )
                for item in site.get("species", [])
            )
        )
        abc = tuple(round(float(value) % 1.0, 8) for value in site["abc"])
        sites.append((species, abc))
    sites.sort()
    return _hash_payload({"lattice": matrix, "sites": sites})


class RawTaskDataset:
    """Streaming reader for one immutable Matbench task payload."""

    def __init__(self, definition: TaskDefinition, sample_ids: np.ndarray) -> None:
        self.definition = definition
        self.sample_ids = np.asarray(sample_ids).astype(str)
        if len(self.sample_ids) != definition.n_samples:
            raise ValueError(
                f"{definition.task_name}: expected {definition.n_samples} IDs, got {len(self.sample_ids)}"
            )

    def validate_header(self, *, verify_file_hash: bool = True) -> list[int]:
        definition = self.definition
        if not definition.raw_path.is_file():
            raise FileNotFoundError(definition.raw_path)
        if verify_file_hash:
            observed = file_sha256(definition.raw_path)
            if observed != definition.raw_sha256:
                raise ValueError(
                    f"{definition.task_name}: raw SHA-256 mismatch: {observed} != {definition.raw_sha256}"
                )
        header = _read_json_header(definition.raw_path)
        expected_input = "composition" if definition.input_type == "composition" else "structure"
        expected_columns = [expected_input, definition.target]
        if list(header.get("columns", [])) != expected_columns:
            raise ValueError(f"{definition.task_name}: columns {header.get('columns')} != {expected_columns}")
        indices = [int(value) for value in header.get("index", [])]
        if len(indices) != definition.n_samples:
            raise ValueError(
                f"{definition.task_name}: raw sample count {len(indices)} != {definition.n_samples}"
            )
        if indices != list(range(definition.n_samples)):
            raise ValueError(f"{definition.task_name}: raw indices are not contiguous from zero")
        return indices

    def iter_samples(self, *, verify_file_hash: bool = True) -> Iterator[RawSample]:
        indices = self.validate_header(verify_file_hash=verify_file_hash)
        seen = 0
        with gzip.open(self.definition.raw_path, "rb") as handle:
            rows = ijson.items(handle, "data.item", use_float=True)
            for position, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"{self.definition.task_name}: malformed row at position {position}")
                raw_input, raw_target = row
                target = float(raw_target)
                if not np.isfinite(target):
                    raise ValueError(f"{self.definition.task_name}: non-finite target at position {position}")
                if self.definition.task_type == "classification" and target not in (0.0, 1.0):
                    raise ValueError(f"{self.definition.task_name}: classification target is not binary")
                raw_index = indices[position]
                fingerprint = _hash_payload([self.definition.task_name, raw_index, raw_input, raw_target])
                yield RawSample(
                    position=position,
                    raw_index=raw_index,
                    sample_id=str(self.sample_ids[position]),
                    input=raw_input,
                    target=target,
                    fingerprint=fingerprint,
                    composition_hash=composition_entity_hash(raw_input),
                    structure_hash=structure_entity_hash(raw_input),
                )
                seen += 1
        if seen != self.definition.n_samples:
            raise ValueError(
                f"{self.definition.task_name}: streamed {seen} rows, expected {self.definition.n_samples}"
            )

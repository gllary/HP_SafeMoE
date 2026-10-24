from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import yaml
from sklearn.model_selection import KFold, StratifiedKFold

from hpsafe_sota.data.raw import RawTaskDataset, TaskDefinition, file_sha256, load_task_definitions


def _yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _identifier_position(identifier: str) -> int:
    try:
        return int(identifier.rsplit("-", 1)[1]) - 1
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Malformed official Matbench identifier: {identifier!r}") from exc


def _array_digest_update(digest: Any, key: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(orjson.dumps(list(contiguous.shape)))
    digest.update(b"\0")
    if contiguous.dtype.kind == "U":
        for item in contiguous.reshape(-1):
            encoded = str(item).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    else:
        digest.update(contiguous.tobytes(order="C"))


def semantic_digest(metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(orjson.dumps(metadata, option=orjson.OPT_SORT_KEYS))
    for key in sorted(arrays):
        _array_digest_update(digest, key, arrays[key])
    return digest.hexdigest()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _load_official(source_path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    expected = protocol["official_validation"]
    observed_hash = file_sha256(source_path)
    if observed_hash != expected["sha256"]:
        raise ValueError(f"Official validation SHA-256 mismatch: {observed_hash} != {expected['sha256']}")
    with source_path.open("rb") as handle:
        payload = orjson.loads(handle.read())
    metadata = payload.get("metadata", {})
    expected_meta = {
        "n_splits": int(expected["n_outer_folds"]),
        "random_state": int(expected["random_state"]),
        "shuffle": bool(expected["shuffle"]),
    }
    if metadata != expected_meta:
        raise ValueError(f"Official validation metadata changed: {metadata} != {expected_meta}")
    return payload


def _official_task_arrays(
    definition: TaskDefinition, official_task: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_ids: set[str] = set()
    arrays: dict[str, np.ndarray] = {}
    test_positions_across_folds: list[int] = []
    for outer in range(5):
        fold = official_task[f"fold_{outer}"]
        train_ids = [str(value) for value in fold["train"]]
        test_ids = [str(value) for value in fold["test"]]
        if set(train_ids) & set(test_ids):
            raise ValueError(f"{definition.task_name} outer {outer}: train/test overlap")
        all_ids.update(train_ids)
        all_ids.update(test_ids)
        train = np.asarray([_identifier_position(value) for value in train_ids], dtype=np.int64)
        test = np.asarray([_identifier_position(value) for value in test_ids], dtype=np.int64)
        arrays[f"outer_{outer}_train"] = train
        arrays[f"outer_{outer}_test"] = test
        test_positions_across_folds.extend(test.tolist())
    sample_ids = np.asarray(sorted(all_ids, key=_identifier_position))
    if len(sample_ids) != definition.n_samples:
        raise ValueError(
            f"{definition.task_name}: official ID count {len(sample_ids)} != {definition.n_samples}"
        )
    positions = [_identifier_position(value) for value in sample_ids]
    if positions != list(range(definition.n_samples)):
        raise ValueError(f"{definition.task_name}: official IDs are not contiguous")
    if sorted(test_positions_across_folds) != list(range(definition.n_samples)):
        raise ValueError(f"{definition.task_name}: outer test folds do not partition the dataset")
    universe = set(range(definition.n_samples))
    for outer in range(5):
        if set(arrays[f"outer_{outer}_train"]) | set(arrays[f"outer_{outer}_test"]) != universe:
            raise ValueError(f"{definition.task_name} outer {outer}: incomplete train/test union")
    return sample_ids, arrays


def _inner_validation_arrays(
    definition: TaskDefinition,
    targets: np.ndarray,
    outer_arrays: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    config = protocol.get("stacking_oof", protocol.get("inner_oof"))
    if not isinstance(config, dict):
        raise ValueError("data protocol requires stacking_oof")
    n_splits = int(config["n_splits"])
    base_seed = int(config["base_random_state"])
    result: dict[str, np.ndarray] = {}
    for outer in range(5):
        outer_train = outer_arrays[f"outer_{outer}_train"]
        seed = base_seed + outer
        if definition.task_type == "classification":
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(np.zeros(len(outer_train)), targets[outer_train])
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(np.zeros(len(outer_train)))
        seen_validation: list[int] = []
        for inner, (_, local_validation) in enumerate(iterator):
            validation = outer_train[np.asarray(local_validation, dtype=np.int64)]
            result[f"inner_{outer}_{inner}_val"] = validation
            seen_validation.extend(validation.tolist())
        if sorted(seen_validation) != sorted(outer_train.tolist()):
            raise ValueError(
                f"{definition.task_name} outer {outer}: inner validation does not partition train"
            )
    return result


def _single_validation_positions(
    definition: TaskDefinition,
    targets: np.ndarray,
    outer_train: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Return one deterministic validation holdout while preserving manifest order."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"validation fraction must be in (0, 1), got {fraction}")
    rng = np.random.default_rng(seed)
    if definition.task_type == "classification":
        selected: list[int] = []
        labels = targets[outer_train]
        for label in np.unique(labels):
            members = outer_train[labels == label]
            count = min(len(members) - 1, max(1, int(round(fraction * len(members)))))
            selected.extend(rng.permutation(members)[:count].tolist())
    else:
        count = min(len(outer_train) - 1, max(1, int(round(fraction * len(outer_train)))))
        selected = rng.permutation(outer_train)[:count].tolist()
    selected_set = set(selected)
    return np.asarray([value for value in outer_train if int(value) in selected_set], dtype=np.int64)


def _holdout_validation_arrays(
    definition: TaskDefinition,
    targets: np.ndarray,
    outer_arrays: dict[str, np.ndarray],
    protocol: dict[str, Any],
) -> dict[str, np.ndarray]:
    config = protocol["model_validation"]
    fraction = float(config["fraction"])
    base_seed = int(config["base_random_state"])
    return {
        f"holdout_{outer}_val": _single_validation_positions(
            definition,
            targets,
            outer_arrays[f"outer_{outer}_train"],
            fraction=fraction,
            seed=base_seed + outer,
        )
        for outer in range(5)
    }


def _metadata_definition(definition: TaskDefinition) -> dict[str, Any]:
    value = asdict(definition)
    value["raw_path"] = definition.raw_path.name
    return value


def build_task_manifest(
    project_root: Path,
    definition: TaskDefinition,
    official_task: dict[str, Any],
    protocol: dict[str, Any],
    *,
    verify_file_hash: bool = True,
) -> tuple[Path, str]:
    sample_ids, split_arrays = _official_task_arrays(definition, official_task)
    dataset = RawTaskDataset(definition, sample_ids)
    raw_indices: list[int] = []
    targets: list[float] = []
    fingerprints: list[bytes] = []
    composition_hashes: list[bytes] = []
    structure_hashes: list[bytes] = []
    for sample in dataset.iter_samples(verify_file_hash=verify_file_hash):
        raw_indices.append(sample.raw_index)
        targets.append(sample.target)
        fingerprints.append(sample.fingerprint)
        composition_hashes.append(sample.composition_hash)
        structure_hashes.append(sample.structure_hash)

    arrays: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "raw_indices": np.asarray(raw_indices, dtype=np.int64),
        "targets": np.asarray(targets, dtype=np.float64),
        "fingerprints": np.asarray(fingerprints, dtype="S32"),
        "composition_hashes": np.asarray(composition_hashes, dtype="S32"),
        "structure_hashes": np.asarray(structure_hashes, dtype="S32"),
        **split_arrays,
    }
    arrays.update(_inner_validation_arrays(definition, arrays["targets"], split_arrays, protocol))
    arrays.update(_holdout_validation_arrays(definition, arrays["targets"], split_arrays, protocol))
    stacking_oof = protocol.get("stacking_oof", protocol.get("inner_oof"))
    metadata = {
        "schema_version": int(protocol["schema_version"]),
        "protocol_version": str(protocol["protocol_version"]),
        "definition": _metadata_definition(definition),
        "official_validation_sha256": str(protocol["official_validation"]["sha256"]),
        "model_validation": protocol["model_validation"],
        "stacking_oof": stacking_oof,
    }
    digest = semantic_digest(metadata, arrays)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray([json.dumps(metadata, sort_keys=True, separators=(",", ":"))])
    payload["semantic_digest"] = np.asarray([digest])
    destination = project_root / "data/manifests" / f"{definition.task_name}.npz"
    _atomic_npz(destination, payload)
    return destination, digest


def build_all_manifests(
    project_root: Path,
    official_validation_path: Path,
    *,
    task_names: list[str] | None = None,
    verify_file_hash: bool = True,
) -> dict[str, dict[str, Any]]:
    project_root = project_root.resolve()
    protocol = _yaml(project_root / "configs/reference/data_protocol.yaml")
    official = _load_official(official_validation_path.resolve(), protocol)
    definitions = load_task_definitions(project_root)
    selected = list(definitions) if task_names is None else task_names
    report: dict[str, dict[str, Any]] = {}
    for task_name in selected:
        definition = definitions[task_name]
        if definition.matbench_name not in official["splits"]:
            raise KeyError(f"Official validation missing {definition.matbench_name}")
        destination, digest = build_task_manifest(
            project_root,
            definition,
            official["splits"][definition.matbench_name],
            protocol,
            verify_file_hash=verify_file_hash,
        )
        report[task_name] = {
            "path": str(destination.relative_to(project_root)),
            "semantic_digest": digest,
            "n_samples": definition.n_samples,
        }
    receipt = {
        "schema_version": 1,
        "official_source": {
            key: protocol["official_validation"][key]
            for key in (
                "benchmark",
                "package_version",
                "filename",
                "sha256",
                "n_outer_folds",
                "random_state",
                "shuffle",
                "source_url",
            )
        },
        "manifests": report,
    }
    receipt_path = project_root / "data/manifests/receipt.json"
    receipt_path.write_bytes(orjson.dumps(receipt, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    return report


class TaskManifest:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as loaded:
            self._arrays = {key: loaded[key] for key in loaded.files}
        self.metadata = json.loads(str(self._arrays.pop("metadata_json")[0]))
        self.declared_digest = str(self._arrays.pop("semantic_digest")[0])

    @property
    def task_name(self) -> str:
        return str(self.metadata["definition"]["task_name"])

    @property
    def sample_ids(self) -> np.ndarray:
        return self._arrays["sample_ids"].astype(str)

    @property
    def targets(self) -> np.ndarray:
        return self._arrays["targets"]

    @property
    def digest(self) -> str:
        return semantic_digest(self.metadata, self._arrays)

    def require_valid(self) -> None:
        if self.digest != self.declared_digest:
            raise ValueError(
                f"{self.task_name}: manifest digest mismatch: {self.digest} != {self.declared_digest}"
            )
        n_samples = int(self.metadata["definition"]["n_samples"])
        if len(self.sample_ids) != n_samples or len(set(self.sample_ids)) != n_samples:
            raise ValueError(f"{self.task_name}: duplicate or missing sample IDs")
        universe = set(range(n_samples))
        outer_tests: list[int] = []
        n_inner = self.n_stacking_splits
        for outer in range(5):
            train = self.positions("outer_train", outer)
            test = self.positions("outer_test", outer)
            if set(train) & set(test) or set(train) | set(test) != universe:
                raise ValueError(f"{self.task_name} outer {outer}: invalid split")
            outer_tests.extend(test.tolist())
            inner_values = [self.positions("inner_val", outer, inner) for inner in range(n_inner)]
            combined = np.concatenate(inner_values)
            if sorted(combined.tolist()) != sorted(train.tolist()):
                raise ValueError(f"{self.task_name} outer {outer}: invalid inner partition")
            holdout_train = self.positions("holdout_train", outer)
            holdout_val = self.positions("holdout_val", outer)
            if (
                set(holdout_train) & set(holdout_val)
                or set(holdout_train) | set(holdout_val) != set(train)
                or set(holdout_val) & set(test)
            ):
                raise ValueError(f"{self.task_name} outer {outer}: invalid validation holdout")
        if sorted(outer_tests) != list(range(n_samples)):
            raise ValueError(f"{self.task_name}: outer tests are not a partition")

    def positions(self, role: str, outer: int, inner: int | None = None) -> np.ndarray:
        if outer not in range(5):
            raise ValueError(f"outer fold must be 0..4, got {outer}")
        if role == "outer_train":
            return self._arrays[f"outer_{outer}_train"].copy()
        if role == "outer_test":
            return self._arrays[f"outer_{outer}_test"].copy()
        if role == "holdout_val":
            key = f"holdout_{outer}_val"
            if key in self._arrays:
                return self._arrays[key].copy()
            definition = TaskDefinition(**{
                **self.metadata["definition"],
                "raw_path": self.path.parent / str(self.metadata["definition"]["raw_path"]),
            })
            config = self.metadata.get(
                "model_validation",
                {"fraction": 0.10, "base_random_state": 20260814},
            )
            return _single_validation_positions(
                definition,
                self.targets,
                self._arrays[f"outer_{outer}_train"],
                fraction=float(config["fraction"]),
                seed=int(config["base_random_state"]) + outer,
            )
        if role == "holdout_train":
            outer_train = self._arrays[f"outer_{outer}_train"]
            validation = set(self.positions("holdout_val", outer).tolist())
            return np.asarray([value for value in outer_train if value not in validation], dtype=np.int64)
        if role == "inner_val":
            if inner is None:
                raise ValueError("inner fold is required for inner_val")
            return self._arrays[f"inner_{outer}_{inner}_val"].copy()
        if role == "inner_train":
            if inner is None:
                raise ValueError("inner fold is required for inner_train")
            outer_train = self._arrays[f"outer_{outer}_train"]
            validation = set(self._arrays[f"inner_{outer}_{inner}_val"].tolist())
            return np.asarray([value for value in outer_train if value not in validation], dtype=np.int64)
        if role == "inner_oof":
            return self._arrays[f"outer_{outer}_train"].copy()
        raise ValueError(f"Unknown split role: {role}")

    def ids(self, role: str, outer: int, inner: int | None = None) -> np.ndarray:
        return self.sample_ids[self.positions(role, outer, inner)]

    def y(self, role: str, outer: int, inner: int | None = None) -> np.ndarray:
        return self.targets[self.positions(role, outer, inner)]

    def entity_hashes(self, kind: str, positions: np.ndarray | None = None) -> np.ndarray:
        key = {"composition": "composition_hashes", "structure": "structure_hashes"}[kind]
        values = self._arrays[key]
        return values.copy() if positions is None else values[positions]

    def iter_inner(self, outer: int) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        n_inner = self.n_stacking_splits
        for inner in range(n_inner):
            yield (
                inner,
                self.positions("inner_train", outer, inner),
                self.positions("inner_val", outer, inner),
            )

    @property
    def n_stacking_splits(self) -> int:
        config = self.metadata.get("stacking_oof", self.metadata.get("inner_oof"))
        if not isinstance(config, dict):
            raise ValueError(f"{self.task_name}: stacking OOF metadata is missing")
        return int(config["n_splits"])

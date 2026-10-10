from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.data.raw import RawTaskDataset, load_task_definitions
from hpsafe_sota.experts.native_cache import collate_crystal_graphs, structure_to_graph
from hpsafe_sota.experts.native_models import CrystalBatch
from hpsafe_sota.stage1_anchorboost.features import FEATURE_VERSION, enriched_descriptor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def cache_root(project_root: Path, kind: str, task_name: str) -> Path:
    return project_root.resolve() / "cache" / kind / task_name


def prepare_anchorboost_cache(
    project_root: Path,
    task_name: str,
    *,
    descriptors: bool = True,
    graphs: bool = True,
    cutoff: float = 8.0,
    max_neighbors: int = 24,
    shard_size: int = 512,
    verify_raw_hash: bool = True,
) -> dict[str, str]:
    project_root = project_root.resolve()
    definition = load_task_definitions(project_root)[task_name]
    manifest = TaskManifest(project_root / "data/manifests" / f"{task_name}.npz")
    manifest.require_valid()
    result: dict[str, str] = {}
    descriptor_root = cache_root(project_root, "anchorboost_descriptors", task_name)
    graph_root = cache_root(project_root, "anchorboost_graphs", task_name)
    descriptor_receipt = descriptor_root / "receipt.json"
    graph_receipt = graph_root / "receipt.json"
    descriptor_ready = descriptor_receipt.is_file()
    graph_ready = definition.input_type != "structure" or graph_receipt.is_file()
    if (not descriptors or descriptor_ready) and (not graphs or graph_ready):
        if descriptors:
            result["descriptors"] = str(descriptor_receipt)
        if graphs and definition.input_type == "structure":
            result["graphs"] = str(graph_receipt)
        return result

    descriptor_map: np.memmap | None = None
    descriptor_temp: Path | None = None
    descriptor_dimension: int | None = None
    if descriptors and not descriptor_ready:
        descriptor_root.mkdir(parents=True, exist_ok=True)
    if graphs and definition.input_type == "structure" and not graph_ready:
        graph_root.mkdir(parents=True, exist_ok=True)
    shard_names: list[str] = []
    shard_graphs: list[dict[str, torch.Tensor]] = []
    shard_positions: list[int] = []

    def flush_graphs() -> None:
        if not shard_graphs:
            return
        name = f"graphs_{len(shard_names):05d}.pt"
        temporary = graph_root / f".{name}.{os.getpid()}.tmp"
        torch.save({"positions": shard_positions.copy(), "graphs": shard_graphs.copy()}, temporary)
        os.replace(temporary, graph_root / name)
        shard_names.append(name)
        shard_positions.clear()
        shard_graphs.clear()

    dataset = RawTaskDataset(definition, manifest.sample_ids)
    for sample in dataset.iter_samples(verify_file_hash=verify_raw_hash):
        if descriptors and not descriptor_ready:
            vector = enriched_descriptor(sample.input)
            if descriptor_map is None:
                descriptor_dimension = len(vector)
                descriptor_temp = descriptor_root / f".features.{os.getpid()}.npy"
                descriptor_map = np.lib.format.open_memmap(
                    descriptor_temp,
                    mode="w+",
                    dtype=np.float32,
                    shape=(definition.n_samples, descriptor_dimension),
                )
            if len(vector) != descriptor_dimension:
                raise ValueError(f"{task_name}: descriptor dimension changed within one dataset")
            descriptor_map[sample.position] = vector
        if graphs and definition.input_type == "structure" and not graph_ready:
            shard_graphs.append(
                structure_to_graph(sample.input, cutoff=cutoff, max_neighbors=max_neighbors)
            )
            shard_positions.append(sample.position)
            if len(shard_graphs) >= shard_size:
                flush_graphs()
        if (sample.position + 1) % 500 == 0 or sample.position + 1 == definition.n_samples:
            print(
                json.dumps(
                    {
                        "event": "anchorboost_cache_progress",
                        "task": task_name,
                        "completed": sample.position + 1,
                        "total": definition.n_samples,
                        "percent": round(100.0 * (sample.position + 1) / definition.n_samples, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    flush_graphs()
    if descriptor_map is not None and descriptor_temp is not None and descriptor_dimension is not None:
        descriptor_map.flush()
        del descriptor_map
        final = descriptor_root / "features.npy"
        os.replace(descriptor_temp, final)
        _atomic_json(
            descriptor_receipt,
            {
                "schema_version": 1,
                "feature_version": FEATURE_VERSION,
                "task": task_name,
                "n_samples": definition.n_samples,
                "dimension": descriptor_dimension,
                "raw_sha256": definition.raw_sha256,
                "manifest_digest": manifest.digest,
                "features_sha256": _sha256(final),
                "target_used": False,
            },
        )
    if graphs and definition.input_type == "structure" and not graph_ready:
        _atomic_json(
            graph_receipt,
            {
                "schema_version": 1,
                "implementation": "hpsafe_periodic_graph",
                "task": task_name,
                "n_samples": definition.n_samples,
                "raw_sha256": definition.raw_sha256,
                "manifest_digest": manifest.digest,
                "cutoff": cutoff,
                "max_neighbors": max_neighbors,
                "shard_size": shard_size,
                "graph_shards": shard_names,
                "graph_shard_sha256": {name: _sha256(graph_root / name) for name in shard_names},
                "target_used": False,
            },
        )
    if descriptors:
        result["descriptors"] = str(descriptor_receipt)
    if graphs and definition.input_type == "structure":
        result["graphs"] = str(graph_receipt)
    return result


class AnchorBoostDescriptorCache:
    def __init__(self, project_root: Path, task_name: str) -> None:
        self.root = cache_root(project_root, "anchorboost_descriptors", task_name)
        self.receipt_path = self.root / "receipt.json"
        if not self.receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing AnchorBoost descriptor cache for {task_name}; run scripts/prepare_anchorboost_cache.py"
            )
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.features = np.load(self.root / "features.npy", mmap_mode="r")

    def rows(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.features[np.asarray(positions, dtype=np.int64)], dtype=np.float32)


class AnchorBoostGraphCache:
    def __init__(self, project_root: Path, task_name: str) -> None:
        self.root = cache_root(project_root, "anchorboost_graphs", task_name)
        self.receipt_path = self.root / "receipt.json"
        if not self.receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing AnchorBoost graph cache for {task_name}; run scripts/prepare_anchorboost_cache.py"
            )
        self.receipt = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        self.shard_size = int(self.receipt["shard_size"])
        self.shard_names = list(self.receipt["graph_shards"])

    def _load_shard(self, index: int) -> tuple[list[int], list[dict[str, torch.Tensor]]]:
        value = torch.load(self.root / self.shard_names[index], map_location="cpu", weights_only=False)
        return [int(item) for item in value["positions"]], value["graphs"]

    def graph_batches(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> Iterator[CrystalBatch]:
        requested = np.asarray(positions, dtype=np.int64)
        target_lookup = {int(p): float(y) for p, y in zip(requested, targets, strict=True)}
        by_shard: dict[int, list[int]] = defaultdict(list)
        for position in requested.tolist():
            by_shard[position // self.shard_size].append(position)
        rng = np.random.default_rng(seed)
        shard_order = list(by_shard)
        if shuffle:
            rng.shuffle(shard_order)
        for shard_index in shard_order:
            cached_positions, graphs = self._load_shard(shard_index)
            lookup = {value: index for index, value in enumerate(cached_positions)}
            local = np.asarray(by_shard[shard_index], dtype=np.int64)
            if shuffle:
                rng.shuffle(local)
            for start in range(0, len(local), batch_size):
                chosen = local[start : start + batch_size]
                selected = [graphs[lookup[int(value)]] for value in chosen]
                selected_targets = np.asarray([target_lookup[int(value)] for value in chosen])
                yield collate_crystal_graphs(
                    selected,
                    positions=chosen,
                    targets=selected_targets,
                    include_line_graph=True,
                )

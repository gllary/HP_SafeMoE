from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pymatgen.core import Element, Structure

from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.data.raw import RawTaskDataset, load_task_definitions
from hpsafe_sota.experts.fingerprint import composition_fingerprint, structure_fingerprint
from hpsafe_sota.experts.native_models import CrystalBatch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_cache_root(project_root: Path, task_name: str) -> Path:
    return project_root / "data/native_cache" / task_name


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _site_atomic_number(site: Any) -> int:
    weighted = sum(float(amount) * Element(str(element)).Z for element, amount in site.species.items())
    total = sum(float(amount) for amount in site.species.values())
    return int(np.clip(round(weighted / max(total, 1e-12)), 1, 118))


@lru_cache(maxsize=118)
def _element_features(atomic_number: int) -> np.ndarray:
    element = Element.from_Z(int(atomic_number))
    values = [
        atomic_number / 118.0,
        float(element.atomic_mass) / 300.0,
        float(element.X or 0.0) / 4.0,
        float(element.row or 0.0) / 7.0,
        float(element.group or 0.0) / 18.0,
        float(element.atomic_radius or 0.0) / 3.0,
        float(element.mendeleev_no or 0.0) / 103.0,
    ]
    configuration = list(element.full_electronic_structure)
    outer_shell = max((int(item[0]) for item in configuration), default=0)
    outer_electrons = sum(
        float(electrons) for shell, _, electrons in configuration if int(shell) == outer_shell
    )
    values.append(outer_electrons / 14.0)
    return np.asarray(values, dtype=np.float32)


def structure_to_graph(
    raw: dict[str, Any], *, cutoff: float = 8.0, max_neighbors: int = 12
) -> dict[str, torch.Tensor]:
    structure = Structure.from_dict(raw)
    atomic_numbers = np.asarray([_site_atomic_number(site) for site in structure], dtype=np.int64)
    atom_features = np.stack([_element_features(value) for value in atomic_numbers])
    centers, neighbors, offsets, distances = structure.get_neighbor_list(cutoff)
    selected: list[int] = []
    for center in range(len(structure)):
        candidates = np.flatnonzero((centers == center) & (distances > 1e-6))
        order = candidates[np.argsort(distances[candidates], kind="stable")]
        selected.extend(order[:max_neighbors].tolist())
    if selected:
        chosen = np.asarray(selected, dtype=np.int64)
        source = centers[chosen].astype(np.int64)
        target = neighbors[chosen].astype(np.int64)
        displacement_fractional = (
            structure.frac_coords[target] + offsets[chosen] - structure.frac_coords[source]
        )
        displacement = structure.lattice.get_cartesian_coords(displacement_fractional)
        distance = np.linalg.norm(displacement, axis=1)
        unit = displacement / np.maximum(distance[:, None], 1e-8)
    else:
        source = np.arange(len(structure), dtype=np.int64)
        target = source.copy()
        distance = np.zeros(len(structure), dtype=np.float64)
        unit = np.zeros((len(structure), 3), dtype=np.float64)
    lattice = structure.lattice
    lengths = np.asarray(lattice.abc, dtype=np.float64)
    angles = np.asarray(lattice.angles, dtype=np.float64)
    n_sites = max(len(structure), 1)
    graph_features = np.asarray(
        [
            *(lengths / 10.0).tolist(),
            *(angles / 180.0).tolist(),
            np.log1p(lattice.volume / n_sites) / 5.0,
            float(structure.density) / 20.0,
            np.log1p(n_sites) / 5.0,
            float(lengths.max() / max(lengths.min(), 1e-8) - 1.0),
        ],
        dtype=np.float32,
    )
    return {
        "atomic_numbers": torch.from_numpy(atomic_numbers),
        "atom_features": torch.from_numpy(atom_features),
        "edge_index": torch.from_numpy(np.stack([source, target])).long(),
        "edge_distance": torch.from_numpy(distance.astype(np.float32)),
        "edge_vector": torch.from_numpy(unit.astype(np.float32)),
        "graph_features": torch.from_numpy(graph_features),
    }


def _receipt_valid(root: Path, expected: dict[str, Any]) -> bool:
    path = root / "receipt.json"
    if not path.is_file():
        return False
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for key, value in expected.items():
        if observed.get(key) != value:
            return False
    if not (root / "descriptors.npy").is_file():
        return False
    if observed.get("descriptor_sha256") != _sha256(root / "descriptors.npy"):
        return False
    graph_hashes = observed.get("graph_shard_sha256", {})
    return all(
        (root / item).is_file() and graph_hashes.get(item) == _sha256(root / item)
        for item in observed.get("graph_shards", [])
    )


def prepare_native_cache(
    project_root: Path,
    task_name: str,
    *,
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    shard_size: int = 512,
    verify_raw_hash: bool = True,
) -> Path:
    project_root = project_root.resolve()
    definition = load_task_definitions(project_root)[task_name]
    manifest_path = project_root / "data/manifests" / f"{task_name}.npz"
    manifest = TaskManifest(manifest_path)
    manifest.require_valid()
    root = native_cache_root(project_root, task_name)
    expected = {
        "schema_version": 1,
        "implementation": "hpsafe_native_pytorch",
        "task_name": task_name,
        "input_type": definition.input_type,
        "n_samples": int(definition.n_samples),
        "raw_sha256": definition.raw_sha256,
        "manifest_digest": manifest.digest,
        "cutoff": float(cutoff),
        "max_neighbors": int(max_neighbors),
        "shard_size": int(shard_size),
    }
    if _receipt_valid(root, expected):
        return root / "receipt.json"
    root.mkdir(parents=True, exist_ok=True)
    if definition.input_type == "structure":
        probe = Structure(
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
            ["Si"],
            [[0.0, 0.0, 0.0]],
        ).as_dict()
        fingerprint_dim = len(structure_fingerprint(probe))
    else:
        fingerprint_dim = len(composition_fingerprint("Si"))
    descriptor_final = root / "descriptors.npy"
    descriptor_temporary = root / f".descriptors.{os.getpid()}.npy"
    descriptors = np.lib.format.open_memmap(
        descriptor_temporary,
        mode="w+",
        dtype=np.float32,
        shape=(definition.n_samples, fingerprint_dim),
    )
    shard_names: list[str] = []
    shard_graphs: list[dict[str, torch.Tensor]] = []
    shard_positions: list[int] = []

    def flush() -> None:
        if not shard_graphs:
            return
        name = f"graphs_{len(shard_names):05d}.pt"
        final = root / name
        temporary = root / f".{name}.{os.getpid()}.tmp"
        torch.save({"positions": shard_positions.copy(), "graphs": shard_graphs.copy()}, temporary)
        os.replace(temporary, final)
        shard_names.append(name)
        shard_graphs.clear()
        shard_positions.clear()

    dataset = RawTaskDataset(definition, manifest.sample_ids)
    for sample in dataset.iter_samples(verify_file_hash=verify_raw_hash):
        if definition.input_type == "structure":
            descriptors[sample.position] = structure_fingerprint(sample.input)
            shard_graphs.append(
                structure_to_graph(sample.input, cutoff=cutoff, max_neighbors=max_neighbors)
            )
            shard_positions.append(sample.position)
            if len(shard_graphs) >= shard_size:
                flush()
        else:
            descriptors[sample.position] = composition_fingerprint(sample.input)
        if (sample.position + 1) % 1000 == 0 or sample.position + 1 == definition.n_samples:
            print(
                json.dumps(
                    {
                        "event": "native_cache_progress",
                        "task": task_name,
                        "completed": sample.position + 1,
                        "total": definition.n_samples,
                        "percent": round(100.0 * (sample.position + 1) / definition.n_samples, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    flush()
    descriptors.flush()
    del descriptors
    os.replace(descriptor_temporary, descriptor_final)
    receipt = {
        **expected,
        "descriptor_shape": [definition.n_samples, fingerprint_dim],
        "descriptor_sha256": _sha256(descriptor_final),
        "graph_shards": shard_names,
        "graph_shard_sha256": {name: _sha256(root / name) for name in shard_names},
        "graph_kind": "radius_graph_with_periodic_images" if shard_names else None,
        "line_graph_policy": "constructed_per_batch_from_same_center_edge_pairs",
        "target_used": False,
    }
    _atomic_json(root / "receipt.json", receipt)
    return root / "receipt.json"


def collate_crystal_graphs(
    graphs: list[dict[str, torch.Tensor]],
    *,
    positions: np.ndarray,
    targets: np.ndarray | None,
    include_line_graph: bool = True,
) -> CrystalBatch:
    node_offset = 0
    edge_offset = 0
    atomic_numbers: list[torch.Tensor] = []
    atom_features: list[torch.Tensor] = []
    edges: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    vectors: list[torch.Tensor] = []
    node_batches: list[torch.Tensor] = []
    graph_features: list[torch.Tensor] = []
    line_pairs: list[torch.Tensor] = []
    line_cosines: list[torch.Tensor] = []
    for graph_index, graph in enumerate(graphs):
        z = graph["atomic_numbers"].long()
        edge_index = graph["edge_index"].long()
        atomic_numbers.append(z)
        atom_features.append(graph["atom_features"].float())
        edges.append(edge_index + node_offset)
        distances.append(graph["edge_distance"].float())
        vectors.append(graph["edge_vector"].float())
        node_batches.append(torch.full((len(z),), graph_index, dtype=torch.long))
        graph_features.append(graph["graph_features"].float())
        if include_line_graph:
            local_pairs: list[tuple[int, int]] = []
            source = edge_index[0].numpy()
            for center in np.unique(source):
                incident = np.flatnonzero(source == center)
                if len(incident) < 2:
                    continue
                for first in incident:
                    for second in incident:
                        if first != second:
                            local_pairs.append((int(first), int(second)))
            if local_pairs:
                pair = torch.as_tensor(local_pairs, dtype=torch.long).T
                vector = graph["edge_vector"].float()
                cosine = (vector[pair[0]] * vector[pair[1]]).sum(-1).clamp(-1.0, 1.0)
                line_pairs.append(pair + edge_offset)
                line_cosines.append(cosine)
        node_offset += len(z)
        edge_offset += edge_index.shape[1]
    empty_line = torch.empty((2, 0), dtype=torch.long)
    return CrystalBatch(
        atomic_numbers=torch.cat(atomic_numbers),
        atom_features=torch.cat(atom_features),
        edge_index=torch.cat(edges, dim=1),
        edge_distance=torch.cat(distances),
        edge_vector=torch.cat(vectors),
        node_batch=torch.cat(node_batches),
        graph_features=torch.stack(graph_features),
        line_index=torch.cat(line_pairs, dim=1) if line_pairs else empty_line,
        line_cosine=torch.cat(line_cosines) if line_cosines else torch.empty(0),
        targets=None if targets is None else torch.as_tensor(targets, dtype=torch.float32),
        positions=torch.as_tensor(positions, dtype=torch.long),
    )


class NativeFeatureCache:
    def __init__(self, project_root: Path, task_name: str) -> None:
        self.root = native_cache_root(project_root.resolve(), task_name)
        receipt_path = self.root / "receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"Missing Native cache for {task_name}. Run python -m "
                "hpsafe_sota.experts.native_cache --project-root ... --task ... first."
            )
        self.receipt_path = receipt_path
        self.receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.descriptors = np.load(self.root / "descriptors.npy", mmap_mode="r")
        self.shard_size = int(self.receipt["shard_size"])
        self.shard_names = list(self.receipt.get("graph_shards", []))

    def descriptor_rows(self, positions: np.ndarray) -> np.ndarray:
        return np.asarray(self.descriptors[np.asarray(positions, dtype=np.int64)], dtype=np.float32)

    def _load_shard(self, shard_index: int) -> tuple[list[int], list[dict[str, torch.Tensor]]]:
        payload = torch.load(
            self.root / self.shard_names[shard_index], map_location="cpu", weights_only=False
        )
        return [int(value) for value in payload["positions"]], payload["graphs"]

    def graph_batches(
        self,
        positions: np.ndarray,
        targets: np.ndarray,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
        include_line_graph: bool = False,
    ) -> Iterator[CrystalBatch]:
        requested = np.asarray(positions, dtype=np.int64)
        target_lookup = {int(p): float(y) for p, y in zip(requested, targets, strict=True)}
        by_shard: dict[int, list[int]] = defaultdict(list)
        for position in requested.tolist():
            by_shard[position // self.shard_size].append(position)
        generator = np.random.default_rng(seed)
        shard_order = list(by_shard)
        if shuffle:
            generator.shuffle(shard_order)
        for shard_index in shard_order:
            cached_positions, graphs = self._load_shard(shard_index)
            lookup = {value: index for index, value in enumerate(cached_positions)}
            local_positions = np.asarray(by_shard[shard_index], dtype=np.int64)
            if shuffle:
                generator.shuffle(local_positions)
            for start in range(0, len(local_positions), batch_size):
                chosen = local_positions[start : start + batch_size]
                missing = [int(value) for value in chosen if int(value) not in lookup]
                if missing:
                    raise ValueError(f"Native graph shard is missing positions {missing[:10]}")
                selected_graphs = [graphs[lookup[int(value)]] for value in chosen]
                selected_targets = np.asarray([target_lookup[int(value)] for value in chosen])
                yield collate_crystal_graphs(
                    selected_graphs,
                    positions=chosen,
                    targets=selected_targets,
                    include_line_graph=include_line_graph,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared pure-PyTorch Native expert inputs.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--skip-raw-hash", action="store_true")
    args = parser.parse_args()
    path = prepare_native_cache(
        args.project_root,
        args.task,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        shard_size=args.shard_size,
        verify_raw_hash=not args.skip_raw_hash,
    )
    print(path)


if __name__ == "__main__":
    main()

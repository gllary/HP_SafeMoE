from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
from pymatgen.core import Composition, Element, Structure

from hpsafe_sota.experts.fingerprint import PROPERTY_NAMES, composition_fingerprint

FEATURE_VERSION = "anchorboost_enriched"


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


@cache
def _property(symbol: str, name: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _finite(getattr(Element(symbol), name, 0.0))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    if cumulative[-1] <= 0:
        return 0.0
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, ordered))


def _composition(value: Any) -> Composition:
    return Composition(value) if isinstance(value, str) else Structure.from_dict(value).composition


def _composition_extension(value: Any) -> np.ndarray:
    composition = _composition(value).fractional_composition
    elements = [(Element(str(element)), float(amount)) for element, amount in composition.items()]
    weights = np.asarray([amount for _, amount in elements], dtype=np.float64)
    values: list[float] = []
    for name in PROPERTY_NAMES:
        observed = np.asarray([_property(str(element), name) for element, _ in elements])
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            values.append(_weighted_quantile(observed, weights, q))
        pair_weight = weights[:, None] * weights[None, :]
        difference = np.abs(observed[:, None] - observed[None, :])
        values.extend(
            [
                float(np.sum(pair_weight * difference)),
                float(np.sqrt(np.sum(pair_weight * difference**2))),
            ]
        )
    fractions = np.asarray([amount for _, amount in elements])
    values.extend(
        [
            float(np.sum(fractions**2)),
            float(np.sum(fractions**3)),
            float(np.prod(np.maximum(fractions, 1e-12)) ** (1.0 / max(len(fractions), 1))),
            float(max(element.Z for element, _ in elements) - min(element.Z for element, _ in elements)),
        ]
    )
    return np.asarray(values, dtype=np.float32)


def _stats(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0] * 7
    return [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(np.quantile(values, 0.10)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.90)),
    ]


def _structure_extension(raw: dict[str, Any], *, cutoff: float = 8.0) -> np.ndarray:
    structure = Structure.from_dict(raw)
    centers, neighbors, offsets, distances = structure.get_neighbor_list(cutoff)
    selected: list[int] = []
    coordination = np.zeros(len(structure), dtype=np.float64)
    for center in range(len(structure)):
        candidates = np.flatnonzero((centers == center) & (distances > 1e-7))
        ordered = candidates[np.argsort(distances[candidates], kind="stable")][:24]
        selected.extend(ordered.tolist())
        coordination[center] = len(ordered)
    chosen = np.asarray(selected, dtype=np.int64)
    selected_distances = distances[chosen] if len(chosen) else np.empty(0)
    distance_hist, _ = np.histogram(selected_distances, bins=96, range=(0.0, cutoff))
    distance_hist = distance_hist.astype(np.float64) / max(len(structure), 1)
    coord_hist, _ = np.histogram(coordination, bins=25, range=(-0.5, 24.5))
    coord_hist = coord_hist.astype(np.float64) / max(len(structure), 1)

    angle_cosines: list[float] = []
    bond_features: list[list[float]] = []
    if len(chosen):
        source = centers[chosen].astype(np.int64)
        target = neighbors[chosen].astype(np.int64)
        fractional = structure.frac_coords[target] + offsets[chosen] - structure.frac_coords[source]
        vectors = structure.lattice.get_cartesian_coords(fractional)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        for center in np.unique(source):
            incident = np.flatnonzero(source == center)
            if len(incident) > 1:
                dot = vectors[incident] @ vectors[incident].T
                upper = dot[np.triu_indices(len(incident), k=1)]
                angle_cosines.extend(np.clip(upper, -1.0, 1.0).tolist())
        symbols = [
            str(site.specie) if site.is_ordered else str(site.species.elements[0])
            for site in structure
        ]
        for edge, (i, j) in enumerate(zip(source, target, strict=True)):
            a, b = symbols[int(i)], symbols[int(j)]
            bond_features.append(
                [
                    abs(_property(a, "X") - _property(b, "X")),
                    abs(_property(a, "atomic_radius") - _property(b, "atomic_radius")),
                    abs(_property(a, "atomic_mass") - _property(b, "atomic_mass")),
                    _property(a, "Z") * _property(b, "Z"),
                    float(selected_distances[edge]),
                ]
            )
    angle_hist, _ = np.histogram(angle_cosines, bins=36, range=(-1.0, 1.0))
    angle_hist = angle_hist.astype(np.float64) / max(len(angle_cosines), 1)
    bond = np.asarray(bond_features, dtype=np.float64)
    bond_stats: list[float] = []
    for column in range(5):
        bond_stats.extend(_stats(bond[:, column] if len(bond) else np.empty(0)))

    lattice = structure.lattice
    lengths = np.asarray(lattice.abc, dtype=np.float64)
    reciprocal = np.asarray(lattice.reciprocal_lattice.abc, dtype=np.float64)
    radii = np.asarray(
        [_property(str(element), "atomic_radius") for element in structure.composition.elements]
    )
    amounts = np.asarray(
        [float(structure.composition[element]) for element in structure.composition.elements]
    )
    sphere_volume = float(np.sum(amounts * (4.0 / 3.0) * np.pi * np.maximum(radii, 0.0) ** 3))
    global_values = [
        *_stats(selected_distances),
        *_stats(coordination),
        *reciprocal.tolist(),
        float(sphere_volume / max(lattice.volume, 1e-8)),
        float(lengths.max() / max(lengths.min(), 1e-8)),
        float(np.sort(lengths)[-1] / max(np.sort(lengths)[-2], 1e-8)),
    ]
    return np.concatenate(
        [
            np.asarray(global_values, dtype=np.float32),
            distance_hist.astype(np.float32),
            coord_hist.astype(np.float32),
            angle_hist.astype(np.float32),
            np.asarray(bond_stats, dtype=np.float32),
        ]
    )


def enriched_descriptor(value: Any) -> np.ndarray:
    base = composition_fingerprint(value)
    composition_extra = _composition_extension(value)
    if isinstance(value, str):
        # Structural tasks have a fixed 213-dimensional extension. Keeping zero-filled fields
        # makes feature receipts explicit without fabricating geometry for composition tasks.
        return np.concatenate([base, composition_extra]).astype(np.float32)
    return np.concatenate([base, composition_extra, _structure_extension(value)]).astype(np.float32)

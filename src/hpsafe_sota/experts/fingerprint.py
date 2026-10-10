from __future__ import annotations

import warnings
from functools import cache
from typing import Any

import numpy as np
from pymatgen.core import Composition, Element, Structure

ELEMENT_DIM = 118
PROPERTY_NAMES = (
    "Z",
    "atomic_mass",
    "X",
    "row",
    "group",
    "atomic_radius",
    "atomic_radius_calculated",
    "van_der_waals_radius",
    "mendeleev_no",
    "molar_volume",
    "melting_point",
    "boiling_point",
    "thermal_conductivity",
    "electrical_resistivity",
    "electron_affinity",
)


def _composition(value: Any) -> Composition:
    if isinstance(value, str):
        return Composition(value)
    return Structure.from_dict(value).composition


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if np.isfinite(result) else 0.0


@cache
def _element_property(symbol: str, property_name: str) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _finite(getattr(Element(symbol), property_name))


@cache
def _outer_orbital(symbol: str) -> tuple[float, float, float, float]:
    configuration = list(Element(symbol).full_electronic_structure)
    if not configuration:
        return (0.0, 0.0, 0.0, 0.0)
    outer_shell = max(int(item[0]) for item in configuration)
    values = [0.0, 0.0, 0.0, 0.0]
    for shell, orbital_name, electrons in configuration:
        if int(shell) == outer_shell and orbital_name in "spdf":
            values["spdf".index(orbital_name)] += float(electrons)
    return tuple(values)


def composition_fingerprint(value: Any) -> np.ndarray:
    composition = _composition(value).fractional_composition
    fractions = np.zeros(ELEMENT_DIM, dtype=np.float32)
    elements: list[tuple[Element, float]] = []
    for element, amount in composition.items():
        item = Element(str(element))
        fraction = float(amount)
        fractions[item.Z - 1] = fraction
        elements.append((item, fraction))
    weights = np.asarray([fraction for _, fraction in elements])
    positive = weights[weights > 0]
    statistics: list[float] = [
        len(elements),
        float(-np.sum(positive * np.log(positive))),
        float(np.max(weights)),
        float(np.min(weights)),
        *[float(np.sum(weights**power) ** (1.0 / power)) for power in (2, 3, 5, 7, 10)],
    ]
    for property_name in PROPERTY_NAMES:
        values = np.asarray([_element_property(str(element), property_name) for element, _ in elements])
        mean = float(np.sum(values * weights))
        statistics.extend(
            [
                mean,
                float(np.sqrt(np.sum(weights * (values - mean) ** 2))),
                float(np.min(values)),
                float(np.max(values)),
                float(np.max(values) - np.min(values)),
                float(np.sum(weights * np.abs(values - mean))),
            ]
        )
    orbital = np.zeros(4, dtype=np.float64)
    for element, fraction in elements:
        orbital += fraction * np.asarray(_outer_orbital(str(element)))
    statistics.extend(orbital.tolist())
    return np.concatenate([fractions, np.asarray(statistics, dtype=np.float32)])


def structure_fingerprint(value: Any, *, distance_bins: int = 64, max_distance: float = 8.0) -> np.ndarray:
    base = composition_fingerprint(value)
    if isinstance(value, str):
        geometry = np.zeros(16 + distance_bins, dtype=np.float32)
        return np.concatenate([base, geometry])
    structure = Structure.from_dict(value)
    lattice = structure.lattice
    n_sites = max(len(structure), 1)
    geometry_values = [
        len(structure),
        len(structure.composition.elements),
        lattice.a,
        lattice.b,
        lattice.c,
        lattice.alpha,
        lattice.beta,
        lattice.gamma,
        lattice.volume,
        lattice.volume / n_sites,
        structure.density,
        min(lattice.abc),
        max(lattice.abc),
        np.mean(lattice.abc),
        np.std(lattice.abc),
        max(lattice.abc) / max(min(lattice.abc), 1e-8),
    ]
    distances: list[float] = []
    for neighbors in structure.get_all_neighbors(max_distance):
        distances.extend(float(neighbor.nn_distance) for neighbor in neighbors)
    histogram, _ = np.histogram(distances, bins=distance_bins, range=(0.0, max_distance))
    histogram = histogram.astype(np.float32) / n_sites
    return np.concatenate([base, np.asarray(geometry_values, dtype=np.float32), histogram])


def fingerprint_many(inputs: list[Any], *, structure: bool) -> np.ndarray:
    function = structure_fingerprint if structure else composition_fingerprint
    return np.stack([function(value) for value in inputs]).astype(np.float32)

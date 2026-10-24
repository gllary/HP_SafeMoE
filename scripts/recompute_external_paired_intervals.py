#!/usr/bin/env python3
"""Reproduce the reported DS2-248 and Core-23 paired MAE intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/external_paired_comparison.json"
N_BOOTSTRAP = 200_000
CHUNK_SIZE = 2_000
DATASETS = (("DS2-248", 248, 20260916), ("Core-23", 23, 20260917))


def compute() -> dict:
    rows = []
    for dataset, expected_n, seed in DATASETS:
        frame = pd.read_csv(ROOT / f"results/{dataset}/outputs/ensemble_predictions.csv")
        n = len(frame)
        if n != expected_n or frame["sample_id"].nunique() != n:
            raise AssertionError(f"{dataset}: material coverage differs")
        if frame["bootstrap_cluster"].nunique() != n:
            raise AssertionError(f"{dataset}: expected one unique cluster per material")
        values = frame[["target", "stage1_prediction", "stage2_prediction"]].to_numpy(float)
        if not np.isfinite(values).all():
            raise AssertionError(f"{dataset}: non-finite label or prediction")
        anchor_error = np.abs(values[:, 1] - values[:, 0])
        full_error = np.abs(values[:, 2] - values[:, 0])
        delta = full_error - anchor_error
        rng = np.random.Generator(np.random.PCG64(seed))
        samples = []
        # Preserve the released row order and draw the same indices for both models.
        for start in range(0, N_BOOTSTRAP, CHUNK_SIZE):
            count = min(CHUNK_SIZE, N_BOOTSTRAP - start)
            indices = rng.integers(0, n, size=(count, n))
            samples.append(full_error[indices].mean(axis=1) - anchor_error[indices].mean(axis=1))
        interval = np.quantile(np.concatenate(samples), [0.025, 0.975])
        rows.append({
            "dataset": dataset,
            "n_materials": n,
            "seed": seed,
            "mae_difference_full_minus_stage1": float(delta.mean()),
            "ci95_low": float(interval[0]),
            "ci95_high": float(interval[1]),
            "relative_mae_reduction_percent": float(-100.0 * delta.mean() / anchor_error.mean()),
            "materials_improved": int((delta < 0).sum()),
            "materials_worsened": int((delta > 0).sum()),
            "materials_tied": int((delta == 0).sum()),
        })
    return {
        "method": "paired nonparametric material bootstrap; one unique cluster per material",
        "difference": "Full HP-SafeMoE minus Stage 1; negative values favour Full",
        "replicates": N_BOOTSTRAP,
        "random_generator": "NumPy PCG64",
        "interval": "percentile 2.5% and 97.5%; linear quantile interpolation",
        "row_order": "ensemble_predictions.csv order; identical sampled indices for both models",
        "datasets": rows,
    }


def verify() -> dict:
    actual = compute()
    expected = json.loads(OUTPUT.read_text(encoding="utf-8"))
    if actual.keys() != expected.keys():
        raise AssertionError("external paired-comparison schema differs")
    for key in actual.keys() - {"datasets"}:
        if actual[key] != expected[key]:
            raise AssertionError(f"external paired-comparison metadata differs: {key}")
    if len(expected["datasets"]) != len(actual["datasets"]):
        raise AssertionError("external paired-comparison dataset count differs")
    for found, retained in zip(actual["datasets"], expected["datasets"]):
        if found.keys() != retained.keys():
            raise AssertionError("external paired-comparison result schema differs")
        for key, value in found.items():
            if isinstance(value, float):
                matches = np.isclose(value, retained[key], rtol=0.0, atol=1e-12)
            else:
                matches = value == retained[key]
            if not matches:
                raise AssertionError(f"{found['dataset']}: paired-comparison field differs: {key}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        verify()
        print("DS2-248 and Core-23 paired MAE intervals verified (200,000 replicates each)")
    else:
        OUTPUT.write_text(json.dumps(compute(), indent=2) + "\n", encoding="utf-8")
        print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()

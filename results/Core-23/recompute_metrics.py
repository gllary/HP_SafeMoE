#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
ROWS = list(csv.DictReader((ROOT / "outputs/ensemble_predictions.csv").open(encoding="utf-8")))
TARGET = np.asarray([float(row["target"]) for row in ROWS])
CLUSTERS = np.asarray([row["bootstrap_cluster"] for row in ROWS])
N_BOOTSTRAP = 200_000


def bootstrap_std(absolute_error, seed=20260918):
    names = np.asarray(list(dict.fromkeys(CLUSTERS.tolist())))
    sizes = np.asarray([np.sum(CLUSTERS == name) for name in names], dtype=float)
    sums = np.asarray([absolute_error[CLUSTERS == name].sum() for name in names], dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, N_BOOTSTRAP, 2000):
        count = min(2000, N_BOOTSTRAP - start)
        indices = rng.integers(0, len(names), size=(count, len(names)))
        values.append(sums[indices].sum(1) / sizes[indices].sum(1))
    return float(np.concatenate(values).std(ddof=1))


result = []
for algorithm, column in (("Stage 1", "stage1_prediction"), ("HP-SafeMoE", "stage2_prediction")):
    prediction = np.asarray([float(row[column]) for row in ROWS])
    error = prediction - TARGET
    absolute_error = np.abs(error)
    result.append({
        "algorithm": algorithm,
        "mean mae": float(absolute_error.mean()),
        "std mae": bootstrap_std(absolute_error),
        "mean rmse": float(np.sqrt(np.mean(error ** 2))),
        "max max_error": float(absolute_error.max()),
    })
print(json.dumps(result, indent=2))

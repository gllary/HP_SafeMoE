#!/usr/bin/env python3
"""Build the final benchmark comparison CSV from the released score records."""

from __future__ import annotations

import argparse
import csv
import io
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "full_benchmark_comparison"

METHODS = (
    "Frozen Matbench v0.1 reference",
    "MatterVial",
    "JMP-L",
    "coGN",
    "coNGN",
    "HP-SafeMoE",
)


def available(mean: float, sd: float | None, provider: str, evidence: str) -> dict[str, object]:
    return {
        "mean": mean,
        "sd": sd,
        "status": "available" if sd is not None else "reported_point_estimate",
        "provider": provider,
        "evidence": evidence,
    }


def not_evaluated() -> dict[str, object]:
    return {
        "mean": None,
        "sd": None,
        "status": "not_evaluated",
        "provider": "not_evaluated",
        "evidence": "not_evaluated",
    }


TASKS = (
    (
        "expt_is_metal",
        "Experimental metallicity",
        "ROC-AUC",
        "higher",
        {
            METHODS[0]: available(0.9598, 0.0041, "Darwin", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.9763, None, "MatterVial", "literature_reported"),
            METHODS[2]: not_evaluated(),
            METHODS[3]: not_evaluated(),
            METHODS[4]: not_evaluated(),
        },
    ),
    (
        "glass",
        "Glass-forming ability",
        "ROC-AUC",
        "higher",
        {
            METHODS[0]: available(0.9603, 0.0075, "MODNet v0.1.12", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.9370, None, "MatterVial", "literature_reported"),
            METHODS[2]: not_evaluated(),
            METHODS[3]: not_evaluated(),
            METHODS[4]: not_evaluated(),
        },
    ),
    (
        "mp_is_metal",
        "MP metallicity",
        "ROC-AUC",
        "higher",
        {
            METHODS[0]: available(0.9520, 0.0074, "CGCNN v2019", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.9780, None, "MatterVial", "literature_reported"),
            METHODS[2]: not_evaluated(),
        },
    ),
    (
        "dielectric",
        "Refractive index",
        "MAE",
        "lower",
        {
            METHODS[0]: available(0.2711, 0.0714, "MODNet v0.1.12", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.2337, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "expt_gap",
        "Experimental band gap",
        "MAE (eV)",
        "lower",
        {
            METHODS[0]: available(0.2865, 0.0083, "Darwin", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.2900, None, "MatterVial", "literature_reported"),
            METHODS[2]: not_evaluated(),
            METHODS[3]: not_evaluated(),
            METHODS[4]: not_evaluated(),
        },
    ),
    (
        "jdft2d",
        "2D exfoliation energy",
        "MAE (meV atom^-1)",
        "lower",
        {
            METHODS[0]: available(33.1918, 7.3428, "MODNet v0.1.12", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(28.8650, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "log_gvrh",
        "Log shear modulus",
        "MAE",
        "lower",
        {
            METHODS[0]: available(0.0670, 0.0006, "coNGN", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.0325, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "log_kvrh",
        "Log bulk modulus",
        "MAE",
        "lower",
        {
            METHODS[0]: available(0.0491, 0.0026, "coNGN", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.0270, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "mp_e_form",
        "MP formation energy",
        "MAE (eV atom^-1)",
        "lower",
        {
            METHODS[0]: available(0.0170, 0.0003, "coGN", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.0138, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "mp_gap",
        "MP band gap",
        "MAE (eV)",
        "lower",
        {
            METHODS[0]: available(0.1559, 0.0017, "coGN", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.1368, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "perovskites",
        "Perovskite formation energy",
        "MAE (eV unit cell^-1)",
        "lower",
        {
            METHODS[0]: available(0.0269, 0.0008, "coGN", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(0.0386, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "phonons",
        "Optical phonon peak",
        "MAE (cm^-1)",
        "lower",
        {
            METHODS[0]: available(28.7606, 2.5767, "MEGNet, kgcnn v2.1.0", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(30.0800, None, "MatterVial", "literature_reported"),
        },
    ),
    (
        "steels",
        "Steel yield strength",
        "MAE (MPa)",
        "lower",
        {
            METHODS[0]: available(79.9468, 13.5883, "TPOT-Mat", "frozen_matbench_v0.1_reference"),
            METHODS[1]: available(85.1200, None, "MatterVial", "literature_reported"),
            METHODS[2]: not_evaluated(),
            METHODS[3]: not_evaluated(),
            METHODS[4]: not_evaluated(),
        },
    ),
)


def comparison_tasks():
    """Derive fitted-model scores from retained records; round only for display."""
    path = ROOT / "results/matbench/task_variant_mean_std.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        full = {row["task"]: row for row in csv.DictReader(handle) if row["variant"] == "HP-SafeMoE"}
    if set(full) != {row[0] for row in TASKS}:
        raise ValueError("Full score inventory must contain all 13 Matbench tasks")
    baseline_folds = defaultdict(dict)
    path = OUTPUT_DIR / "reproduced_baseline_fold_scores.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"], row["task"])
            fold = int(row["outer_fold"])
            if fold in baseline_folds[key]:
                raise ValueError(f"Duplicate baseline fold: {key}/{fold}")
            baseline_folds[key][fold] = float(row["score"])
    if len(baseline_folds) != 26 or any(set(scores) != set(range(5)) for scores in baseline_folds.values()):
        raise ValueError("Expected 26 reproduced baseline task/method pairs with five folds each")
    for task_id, task, metric, direction, values in TASKS:
        score = full[task_id]
        entries = dict(values)
        for method in METHODS[2:5]:
            scores = baseline_folds.get((method, task_id))
            entries[method] = (available(statistics.mean(scores.values()), statistics.stdev(scores.values()),
                                         method, "archived_official_outer_fold_scores")
                               if scores else not_evaluated())
        entries["HP-SafeMoE"] = available(
            float(score["score_mean"]), float(score["score_std"]),
            "HP-SafeMoE", "official_outer_five_fold",
        )
        yield task_id, task, metric, direction, entries


def records() -> list[dict[str, object]]:
    output = []
    for task_id, task, metric, direction, values in comparison_tasks():
        for method in METHODS:
            entry = values[method]
            output.append(
                {
                    "task_id": task_id,
                    "task": task,
                    "metric": metric,
                    "direction": direction,
                    "method": method,
                    **entry,
                }
            )
    return output


def csv_text(rows: list[dict[str, object]]) -> str:
    fields = ("task_id", "task", "metric", "direction", "method", "mean", "sd", "status", "provider", "evidence")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed outputs without rewriting them")
    args = parser.parse_args()
    outputs = {
        OUTPUT_DIR / "full_benchmark_comparison.csv": csv_text(records()),
    }
    if args.check:
        for path, expected in outputs.items():
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                raise SystemExit(f"Full-benchmark comparison differs: {path}")
        print("Full-benchmark comparison verified: 13 tasks × 6 methods = 78 records")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

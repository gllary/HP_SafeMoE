#!/usr/bin/env python3
"""Build the complete machine-readable and human-readable benchmark comparison."""

from __future__ import annotations

import argparse
import csv
import io
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
            METHODS[5]: available(0.9805, 0.0038, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[5]: available(0.9559, 0.0078, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[3]: available(0.9519, 0.0021, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.9554, 0.0020, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.9871, 0.0011, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.2530, 0.0940, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.3017, 0.1009, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.3238, 0.1032, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.2434, 0.0938, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[5]: available(0.2461, 0.0033, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(29.6696, 10.2381, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(37.4127, 13.0932, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(39.7885, 13.5337, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(28.9400, 11.2200, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.0587, 0.0010, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.0691, 0.0015, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.0676, 0.0015, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.0346, 0.0012, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.0453, 0.0030, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.0528, 0.0029, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.0496, 0.0028, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.0266, 0.0018, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.0145, 0.0005, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.0169, 0.0003, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.0178, 0.0006, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.0112, 0.0003, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.0992, 0.0022, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.1558, 0.0024, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.1720, 0.0019, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.0976, 0.0022, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(0.0257, 0.0011, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(0.0271, 0.0006, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(0.0287, 0.0016, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(0.0265, 0.0013, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[2]: available(21.1923, 2.1026, "JMP-L", "official_outer_five_fold"),
            METHODS[3]: available(30.6247, 1.7338, "coGN", "official_outer_five_fold"),
            METHODS[4]: available(28.4937, 2.2786, "coNGN", "official_outer_five_fold"),
            METHODS[5]: available(20.1000, 1.5600, "HP-SafeMoE", "official_outer_five_fold"),
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
            METHODS[5]: available(78.5100, 14.3200, "HP-SafeMoE", "official_outer_five_fold"),
        },
    ),
)


def records() -> list[dict[str, object]]:
    output = []
    for task_id, task, metric, direction, values in TASKS:
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


def display(entry: dict[str, object]) -> str:
    if entry["status"] == "not_evaluated":
        return "—"
    mean = float(entry["mean"])
    if entry["sd"] is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {float(entry['sd']):.4f}"


def markdown_text() -> str:
    lines = [
        "# Full Matbench comparison",
        "",
        "Values are five-fold mean ± SD unless the entry is a literature-reported point estimate. A dash marks an unavailable task-level result.",
        "",
        "| Task and metric | Frozen Matbench v0.1 reference | MatterVial | JMP-L | coGN | coNGN | HP-SafeMoE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, task, metric, direction, values in TASKS:
        arrow = "↑" if direction == "higher" else "↓"
        cells = [display(values[method]) for method in METHODS]
        lines.append(f"| {task}, {metric} {arrow} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "The frozen reference column is task-wise and names its provider in `full_benchmark_comparison.csv`. MatterVial values are literature-reported. JMP-L, coGN, coNGN, and HP-SafeMoE values use the official Matbench v0.1 outer folds.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed outputs without rewriting them")
    args = parser.parse_args()
    outputs = {
        OUTPUT_DIR / "full_benchmark_comparison.csv": csv_text(records()),
        OUTPUT_DIR / "full_benchmark_comparison.md": markdown_text(),
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

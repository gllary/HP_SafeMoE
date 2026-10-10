#!/usr/bin/env python3
"""Recompute the released JARVIS metrics from row-level CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "per_sample_predictions"
METRICS = ROOT / "task_metrics.csv"
CERTIFICATE = ROOT / "protocol_certificate.json"
EXPECTED_FIELDS = {
    "sample_id",
    "y_true",
    "stage1_anchor_prediction",
    "raw_stage2_prediction",
    "hpsafemoe_prediction",
    "benefit_probability",
    "correction_accepted",
    "residual_scale",
    "task_safety_accepted",
    "abs_error_stage1_anchor",
    "abs_error_hpsafemoe",
    "proposal_route_index",
    "directional_agreement",
    "route_consensus",
    "route_probability_0",
    "route_probability_1",
    "route_probability_2",
    "data_attention_source_0",
    "data_attention_source_1",
    "data_attention_source_2",
    "data_attention_source_3",
    "data_attention_source_4",
    "learned_relation_attention_source_0",
    "learned_relation_attention_source_1",
    "learned_relation_attention_source_2",
    "learned_relation_attention_source_3",
    "learned_relation_attention_source_4",
}


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise AssertionError(f"invalid boolean value: {value!r}")


def close(actual: float, expected: float, *, tolerance: float = 5e-14) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{actual!r} differs from {expected!r}")


def recompute(task: str) -> dict[str, float | int | bool]:
    path = PREDICTIONS / f"{task}.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != EXPECTED_FIELDS:
            raise AssertionError(f"{path.name}: field set differs")
        rows = list(reader)

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise AssertionError(f"{path.name}: duplicate sample identifiers")

    stage1_errors: list[float] = []
    raw_errors: list[float] = []
    final_errors: list[float] = []
    accepted_rows = 0
    residual_scales: set[float] = set()
    task_acceptances: set[bool] = set()

    for row in rows:
        numeric = {
            key: float(value)
            for key, value in row.items()
            if key != "sample_id" and value not in {"True", "False"}
        }
        if not all(math.isfinite(value) for value in numeric.values()):
            raise AssertionError(f"{path.name}: non-finite numeric value")

        y_true = numeric["y_true"]
        anchor = numeric["stage1_anchor_prediction"]
        raw = numeric["raw_stage2_prediction"]
        final = numeric["hpsafemoe_prediction"]
        correction_accepted = parse_bool(row["correction_accepted"])
        task_accepted = parse_bool(row["task_safety_accepted"])
        residual_scale = numeric["residual_scale"]
        reconstructed = anchor + float(task_accepted and correction_accepted) * residual_scale * (raw - anchor)
        close(final, reconstructed, tolerance=1e-12)

        stage1_error = abs(anchor - y_true)
        raw_error = abs(raw - y_true)
        final_error = abs(final - y_true)
        close(numeric["abs_error_stage1_anchor"], stage1_error, tolerance=1e-12)
        close(numeric["abs_error_hpsafemoe"], final_error, tolerance=1e-12)
        route_sum = sum(numeric[f"route_probability_{index}"] for index in range(3))
        close(route_sum, 1.0, tolerance=1e-6)

        stage1_errors.append(stage1_error)
        raw_errors.append(raw_error)
        final_errors.append(final_error)
        accepted_rows += int(task_accepted and correction_accepted)
        residual_scales.add(residual_scale)
        task_acceptances.add(task_accepted)

    if len(residual_scales) != 1 or len(task_acceptances) != 1:
        raise AssertionError(f"{path.name}: task policy is not constant across rows")
    count = len(rows)
    stage1_mae = sum(stage1_errors) / count
    raw_mae = sum(raw_errors) / count
    final_mae = sum(final_errors) / count
    return {
        "n_samples": count,
        "stage1_mae": stage1_mae,
        "raw_stage2_mae": raw_mae,
        "hpsafemoe_mae": final_mae,
        "gain_percent": 100.0 * (stage1_mae - final_mae) / stage1_mae,
        "accepted_rows": accepted_rows,
        "accepted_fraction": accepted_rows / count,
        "residual_scale": residual_scales.pop(),
        "task_safety_accepted": task_acceptances.pop(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if recomputed values differ from the release summary")
    args = parser.parse_args()

    certificate = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
    required_boundary = {
        "locked_before_test_prediction": True,
        "task_scope": "five_predeclared_release_tasks",
        "task_policy": "one_global_profile",
        "calibration_data_scope": "official_training_oof_and_official_validation",
        "prediction_input_scope": "features_only",
        "prediction_barrier_before_test_scoring": True,
        "test_label_role": "evaluation",
        "profile_selection_timing": "before_test_prediction",
    }
    for key, expected in required_boundary.items():
        if certificate.get(key) != expected:
            raise AssertionError(f"protocol boundary differs for {key}")

    with METRICS.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    if len(summary_rows) != 5:
        raise AssertionError("expected five JARVIS task summaries")

    report: dict[str, dict[str, float | int | bool]] = {}
    total_rows = 0
    for summary in summary_rows:
        task = summary["task"]
        computed = recompute(task)
        total_rows += int(computed["n_samples"])
        report[task] = computed
        if args.check:
            integer_fields = ("n_samples", "accepted_rows")
            float_fields = (
                "stage1_mae",
                "raw_stage2_mae",
                "hpsafemoe_mae",
                "gain_percent",
                "accepted_fraction",
                "residual_scale",
            )
            for field in integer_fields:
                if int(summary[field]) != computed[field]:
                    raise AssertionError(f"{task} {field} differs")
            for field in float_fields:
                close(float(summary[field]), float(computed[field]))
            if parse_bool(summary["task_safety_accepted"]) is not computed["task_safety_accepted"]:
                raise AssertionError(f"{task} task safety decision differs")
            policy = certificate["tasks"][task]
            close(float(policy["residual_scale"]), float(computed["residual_scale"]), tolerance=2e-8)
            if policy["task_safety_accepted"] is not computed["task_safety_accepted"]:
                raise AssertionError(f"{task} certificate decision differs")

    if total_rows != 24068:
        raise AssertionError(f"expected 24,068 JARVIS rows, found {total_rows:,}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.check:
        print("JARVIS release verified: 5 tasks, 24,068 unique rows, exact safety equation and metrics")


if __name__ == "__main__":
    main()

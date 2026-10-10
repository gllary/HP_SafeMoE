#!/usr/bin/env python3
"""Recompute the four packaged external case-study results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HEA95_SPLIT_COUNT = 100
HEA95_CALIBRATION_COUNT = 5
HEA95_TEST_COUNT = 90
ALGORITHMS = {
    "Stage 1 + 5% output-bias calibration": "stage1_specialist_only__pred_log10_e_gpa",
    "HP-SafeMoE + 5% output-bias calibration": "hpsafemoe__oof_locked__pred_log10_e_gpa",
}


def metrics(error: np.ndarray) -> tuple[float, float, float]:
    absolute = np.abs(error)
    return float(absolute.mean()), float(np.sqrt(np.square(error).mean())), float(absolute.max())


def assert_frame_matches(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    if actual[keys].astype(str).to_dict("records") != expected[keys].astype(str).to_dict("records"):
        raise AssertionError(f"key columns differ: {keys}")
    numeric = [column for column in actual.columns if column not in keys]
    if not np.allclose(actual[numeric], expected[numeric], rtol=0.0, atol=1e-12):
        difference = np.max(np.abs(actual[numeric].to_numpy() - expected[numeric].to_numpy()))
        raise AssertionError(f"numeric output differs; maximum absolute difference={difference}")


def recompute_hea95() -> dict[str, dict[str, float]]:
    base = pd.read_csv(ROOT / "results/HEA-95/base_predictions.csv")
    definitions = pd.read_csv(ROOT / "results/HEA-95/split_definitions.csv")
    expected = pd.read_csv(ROOT / "results/HEA-95/paired_output_bias_calibration.csv")
    expected_split_ids = list(range(HEA95_SPLIT_COUNT))
    if len(definitions) != HEA95_SPLIT_COUNT or sorted(definitions["split_id"].tolist()) != expected_split_ids:
        raise AssertionError(f"HEA-95 requires exactly {HEA95_SPLIT_COUNT} fixed paired splits")
    if not (definitions["calibration_n"] == HEA95_CALIBRATION_COUNT).all():
        raise AssertionError(f"HEA-95 requires {HEA95_CALIBRATION_COUNT} calibration samples per split")
    if not (definitions["test_n"] == HEA95_TEST_COUNT).all():
        raise AssertionError(f"HEA-95 requires {HEA95_TEST_COUNT} test samples per split")
    if len(expected) != 2 * HEA95_SPLIT_COUNT or sorted(expected["split_id"].unique()) != expected_split_ids:
        raise AssertionError("HEA-95 paired outputs must contain both algorithms for every fixed split")
    truth = base["measured_log10_e_gpa"].to_numpy(dtype=float)
    rows = []
    for split in definitions.itertuples(index=False):
        calibration = np.asarray([int(value) for value in split.calibration_indices_zero_based.split(";")])
        test = np.ones(len(base), dtype=bool)
        test[calibration] = False
        for algorithm, column in ALGORITHMS.items():
            prediction = base[column].to_numpy(dtype=float)
            shift = float(np.median(truth[calibration] - prediction[calibration]))
            mae, rmse, max_error = metrics(prediction[test] + shift - truth[test])
            rows.append(
                {
                    "split_id": int(split.split_id),
                    "algorithm": algorithm,
                    "fitted_shift_log10_e_gpa": shift,
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_max_error": max_error,
                }
            )
    actual = pd.DataFrame(rows)
    assert_frame_matches(actual, expected, ["split_id", "algorithm"])
    return {
        algorithm: {
            "split_count": int(len(group)),
            "mean_test_mae": float(group["test_mae"].mean()),
            "test_mae_sd": float(group["test_mae"].std(ddof=1)),
        }
        for algorithm, group in actual.groupby("algorithm", sort=False)
    }


def recompute_b2() -> dict[str, dict[str, float]]:
    base = pd.read_csv(ROOT / "results/B2-18/base_predictions.csv")
    expected = pd.read_csv(ROOT / "results/B2-18/leave_one_out_output_bias_calibration.csv")
    truth = base["measured_log10_e_gpa"].to_numpy(dtype=float)
    rows = []
    for calibration_index in range(len(base)):
        test = np.arange(len(base)) != calibration_index
        for algorithm, column in ALGORITHMS.items():
            prediction = base[column].to_numpy(dtype=float)
            shift = float(truth[calibration_index] - prediction[calibration_index])
            mae, rmse, max_error = metrics(prediction[test] + shift - truth[test])
            rows.append(
                {
                    "calibration_index": calibration_index,
                    "calibration_sample_id": base.loc[calibration_index, "sample_id"],
                    "algorithm": algorithm,
                    "fitted_shift_log10_e_gpa": shift,
                    "test_n": int(test.sum()),
                    "test_mae": mae,
                    "test_rmse": rmse,
                    "test_max_error": max_error,
                }
            )
    actual = pd.DataFrame(rows)
    assert_frame_matches(actual, expected, ["calibration_index", "calibration_sample_id", "algorithm"])
    return {
        algorithm: {
            "split_count": int(len(group)),
            "mean_test_mae": float(group["test_mae"].mean()),
            "test_mae_sd": float(group["test_mae"].std(ddof=0)),
        }
        for algorithm, group in actual.groupby("algorithm", sort=False)
    }


def recompute_zero_shot(dataset: str) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(ROOT / f"results/{dataset}/outputs/ensemble_predictions.csv")
    truth = frame["target"].to_numpy(dtype=float)
    output = {}
    for algorithm, column in (("Stage 1", "stage1_prediction"), ("HP-SafeMoE", "stage2_prediction")):
        mae, rmse, max_error = metrics(frame[column].to_numpy(dtype=float) - truth)
        output[algorithm] = {"n": int(len(frame)), "mae": mae, "rmse": rmse, "max_error": max_error}
    return output


def main() -> None:
    result = {
        "HEA-95": recompute_hea95(),
        "B2-18": recompute_b2(),
        "Core-23": recompute_zero_shot("Core-23"),
        "DS2-248": recompute_zero_shot("DS2-248"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate same-composition HEA-95 pairs and reproduce the sensitivity analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "HEA-95"
RESULT_PATH = ROOT / "results" / "HEA-95" / "shared_label_structure_validation.csv"
PREDICTION_PATH = ROOT / "results" / "HEA-95" / "base_predictions.csv"
SPLIT_PATH = ROOT / "results" / "HEA-95" / "split_definitions.csv"
SENSITIVITY_PATH = ROOT / "source_data" / "hea95_shared_label_group_sensitivity.csv"
SENSITIVITY_SUMMARY_PATH = (
    ROOT / "source_data" / "hea95_shared_label_group_sensitivity_summary.csv"
)

PREDICTION_COLUMNS = {
    "Stage 1": "stage1_specialist_only__pred_log10_e_gpa",
    "HP-SafeMoE": "hpsafemoe__oof_locked__pred_log10_e_gpa",
}

# Supplementary Table S10 is one-based; released sample IDs are zero-based.
SHARED_LABEL_GROUPS = {
    "SLG-01": ("hea95-16", "hea95-29"),  # HEA-017 / HEA-030
    "SLG-02": ("hea95-17", "hea95-52"),  # HEA-018 / HEA-053
    "SLG-03": ("hea95-25", "hea95-81"),  # HEA-026 / HEA-082
    "SLG-04": ("hea95-26", "hea95-43"),  # HEA-027 / HEA-044
    "SLG-05": ("hea95-34", "hea95-37"),  # HEA-035 / HEA-038
}


def canonical_sha256(structure_json: str) -> str:
    payload = json.dumps(json.loads(structure_json), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_validation(materials: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = materials.merge(labels, on="sample_id", validate="one_to_one").set_index("sample_id")
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=False,
        scale=False,
        attempt_supercell=True,
    )
    rows: list[dict[str, object]] = []
    for group_id, (sample_a, sample_b) in SHARED_LABEL_GROUPS.items():
        record_a = merged.loc[sample_a]
        record_b = merged.loc[sample_b]
        raw_a = str(record_a["structure_json"])
        raw_b = str(record_b["structure_json"])
        structure_a = Structure.from_dict(json.loads(raw_a))
        structure_b = Structure.from_dict(json.loads(raw_b))
        rows.append(
            {
                "shared_label_group": group_id,
                "sample_id_a": sample_a,
                "sample_id_b": sample_b,
                "supplementary_id_a": f"HEA-{int(sample_a.split('-')[1]) + 1:03d}",
                "supplementary_id_b": f"HEA-{int(sample_b.split('-')[1]) + 1:03d}",
                "composition_a": record_a["composition"],
                "composition_b": record_b["composition"],
                "target_a": float(record_a["target"]),
                "target_b": float(record_b["target"]),
                "site_count_a": len(structure_a),
                "site_count_b": len(structure_b),
                "canonical_structure_sha256_a": canonical_sha256(raw_a),
                "canonical_structure_sha256_b": canonical_sha256(raw_b),
                "serialized_structures_identical": raw_a == raw_b,
                "structure_matcher_fit": bool(matcher.fit(structure_a, structure_b)),
                "max_abs_lattice_matrix_difference_angstrom": float(
                    np.max(np.abs(structure_a.lattice.matrix - structure_b.lattice.matrix))
                ),
            }
        )
    return pd.DataFrame(rows)


def build_group_sensitivity(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Exclude an evaluation partner when the other pair member calibrates the bias."""
    merged = labels.merge(predictions, on="sample_id", validate="one_to_one").set_index(
        "sample_id"
    )
    all_sample_ids = set(merged.index)
    group_by_sample = {
        sample_id: group_id
        for group_id, sample_ids in SHARED_LABEL_GROUPS.items()
        for sample_id in sample_ids
    }

    rows: list[dict[str, object]] = []
    for split in splits.itertuples(index=False):
        calibration_ids = set(str(split.calibration_sample_ids).split(";"))
        evaluation_ids = all_sample_ids - calibration_ids
        represented_groups = {
            group_by_sample[sample_id]
            for sample_id in calibration_ids
            if sample_id in group_by_sample
        }
        excluded_ids = sorted(
            sample_id
            for sample_id in evaluation_ids
            if group_by_sample.get(sample_id) in represented_groups
        )
        retained_ids = sorted(evaluation_ids - set(excluded_ids))
        record: dict[str, object] = {
            "split_id": int(split.split_id),
            "calibration_n": len(calibration_ids),
            "original_evaluation_n": len(evaluation_ids),
            "group_separated_evaluation_n": len(retained_ids),
            "excluded_evaluation_n": len(excluded_ids),
            "excluded_evaluation_sample_ids": ";".join(excluded_ids),
            "affected_split": bool(excluded_ids),
        }
        truth_cal = merged.loc[sorted(calibration_ids), "target"].to_numpy(float)
        truth_eval = merged.loc[retained_ids, "target"].to_numpy(float)
        for model_name, prediction_column in PREDICTION_COLUMNS.items():
            pred_cal = merged.loc[sorted(calibration_ids), prediction_column].to_numpy(float)
            shift = float(np.median(truth_cal - pred_cal))
            pred_eval = merged.loc[retained_ids, prediction_column].to_numpy(float) + shift
            error = pred_eval - truth_eval
            key = "stage1" if model_name == "Stage 1" else "full"
            record[f"{key}_output_bias"] = shift
            record[f"{key}_mae"] = float(np.mean(np.abs(error)))
            record[f"{key}_rmse"] = float(np.sqrt(np.mean(np.square(error))))
        record["full_minus_stage1_mae"] = float(record["full_mae"] - record["stage1_mae"])
        rows.append(record)
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)


def summarize_group_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    affected_splits = int(results["affected_split"].sum())
    rows: list[dict[str, object]] = []
    for label, column in (
        ("Stage 1", "stage1_mae"),
        ("HP-SafeMoE", "full_mae"),
        ("HP-SafeMoE - Stage 1", "full_minus_stage1_mae"),
    ):
        values = results[column].to_numpy(float)
        rows.append(
            {
                "model_or_comparison": label,
                "split_n": len(values),
                "affected_split_n": affected_splits,
                "mean": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "full_lower_n": int((results["full_minus_stage1_mae"] < 0).sum())
                if label == "HP-SafeMoE - Stage 1"
                else "",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    materials = pd.read_csv(DATA_DIR / "materials.csv")
    labels = pd.read_csv(DATA_DIR / "labels.csv")
    validation = build_validation(materials, labels)

    if validation["serialized_structures_identical"].any() or validation["structure_matcher_fit"].any():
        raise AssertionError("At least one declared distinct HEA-95 structure pair matched")
    if not (validation["composition_a"] == validation["composition_b"]).all():
        raise AssertionError("A declared pair has different compositions")
    if not np.allclose(validation["target_a"], validation["target_b"], rtol=0, atol=0):
        raise AssertionError("A declared pair has different displayed targets")

    validation.to_csv(RESULT_PATH, index=False)
    sensitivity = build_group_sensitivity(
        labels,
        pd.read_csv(PREDICTION_PATH),
        pd.read_csv(SPLIT_PATH),
    )
    summary = summarize_group_sensitivity(sensitivity)
    sensitivity.to_csv(SENSITIVITY_PATH, index=False)
    summary.to_csv(SENSITIVITY_SUMMARY_PATH, index=False)

    affected = int(sensitivity["affected_split"].sum())
    if affected != 36:
        raise AssertionError(f"Expected 36 affected splits, found {affected}")
    if int((sensitivity["full_minus_stage1_mae"] < 0).sum()) != 85:
        raise AssertionError("Expected HP-SafeMoE to have lower MAE in 85 of 100 splits")
    print(
        f"Validated {len(validation)} different-structure pairs and reproduced the "
        f"group-separated sensitivity analysis ({affected}/100 splits affected)."
    )


if __name__ == "__main__":
    main()

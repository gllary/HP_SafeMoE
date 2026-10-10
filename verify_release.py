#!/usr/bin/env python3
"""Validate the public release structure, results, terminology, and path hygiene."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent


def read_flat_yaml_scalars(path: Path) -> dict[str, object]:
    """Read top-level scalar YAML values needed by the release contract."""
    output: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.split("#", 1)[0].strip()
        if not value or value.startswith(("[", "{")):
            continue
        if value.lower() in {"true", "false"}:
            parsed: object = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip("\"'")
        output[key.strip()] = parsed
    return output


def require(path: str) -> Path:
    resolved = ROOT / path
    if not resolved.exists():
        raise AssertionError(f"required path is absent: {path}")
    return resolved


def check_datasets() -> None:
    expected = {"HEA-95": 95, "B2-18": 18, "Core-23": 23, "DS2-248": 248}
    for dataset, count in expected.items():
        materials = pd.read_csv(require(f"data/{dataset}/materials.csv"))
        labels = pd.read_csv(require(f"data/{dataset}/labels.csv"))
        if len(materials) != count or len(labels) != count:
            raise AssertionError(f"{dataset}: expected {count} material and label rows")
        if materials["sample_id"].tolist() != labels["sample_id"].tolist():
            raise AssertionError(f"{dataset}: sample identifiers are not aligned")


def check_external_results() -> None:
    hea = pd.read_csv(require("results/HEA-95/paired_output_bias_calibration.csv"))
    if len(hea) != 200 or sorted(hea["split_id"].unique()) != list(range(100)):
        raise AssertionError("HEA-95 must contain two algorithms for each of 100 fixed splits")
    definitions = pd.read_csv(require("results/HEA-95/split_definitions.csv"))
    if len(definitions) != 100 or not (definitions["calibration_n"] == 5).all() or not (definitions["test_n"] == 90).all():
        raise AssertionError("HEA-95 split definitions must be 5-calibration/90-test")
    group_sensitivity = pd.read_csv(require("source_data/hea95_shared_label_group_sensitivity.csv"))
    if len(group_sensitivity) != 100 or int(group_sensitivity["affected_split"].sum()) != 36:
        raise AssertionError("HEA-95 group sensitivity must cover 100 splits, 36 affected")
    if int((group_sensitivity["full_minus_stage1_mae"] < 0).sum()) != 85:
        raise AssertionError("HEA-95 group sensitivity must retain 85 Full-lower splits")
    observed_sensitivity = group_sensitivity[["stage1_mae", "full_mae"]].mean().to_numpy(float)
    if not np.allclose(observed_sensitivity, [0.11132079611369575, 0.11095420530556133], rtol=0.0, atol=1e-14):
        raise AssertionError("HEA-95 group-sensitivity means differ")
    b2 = pd.read_csv(require("results/B2-18/leave_one_out_output_bias_calibration.csv"))
    if len(b2) != 36 or sorted(b2["calibration_index"].unique()) != list(range(18)):
        raise AssertionError("B2-18 must contain two algorithms for every calibration choice")
    expected_mae = {
        ("Core-23", "stage1_prediction"): 0.05255817255886799,
        ("Core-23", "stage2_prediction"): 0.04788133843439546,
        ("DS2-248", "stage1_prediction"): 0.7481466895309896,
        ("DS2-248", "stage2_prediction"): 0.6805946898381403,
    }
    for (dataset, column), expected in expected_mae.items():
        frame = pd.read_csv(require(f"results/{dataset}/outputs/ensemble_predictions.csv"))
        actual = float(np.abs(frame[column] - frame["target"]).mean())
        if not np.isclose(actual, expected, rtol=0.0, atol=1e-14):
            raise AssertionError(f"{dataset} {column}: MAE differs")
    jarvis_expected = {
        "formation_energy_peratom": (5572, 0.027271755963579013, 0.028991179050936847, 0.027107601675993073, 473),
        "optb88vdw_bandgap": (5572, 0.11675258053088938, 0.11448570018800175, 0.11195475483275104, 1589),
        "optb88vdw_total_energy": (5572, 0.026278422272376038, 0.02670779054364262, 0.026278422272376038, 0),
        "mbj_bandgap": (1815, 0.25122337860437755, 0.20439151806768477, 0.19632140631478726, 844),
        "ehull": (5537, 0.04093309033034836, 0.041778346251906864, 0.04093309033034836, 0),
    }
    jarvis_summary = pd.read_csv(require("results/JARVIS/task_metrics.csv")).set_index("task")
    total_rows = 0
    for task, (count, expected_stage1, expected_raw, expected_final, expected_accepted) in jarvis_expected.items():
        frame = pd.read_csv(require(f"results/JARVIS/per_sample_predictions/{task}.csv"))
        total_rows += len(frame)
        if len(frame) != count or frame["sample_id"].nunique() != count:
            raise AssertionError(f"JARVIS {task}: row count or sample identifiers differ")
        correction_accepted = frame["correction_accepted"].map(lambda value: str(value).lower() == "true")
        task_accepted = frame["task_safety_accepted"].map(lambda value: str(value).lower() == "true")
        accepted = correction_accepted & task_accepted
        reconstructed = frame["stage1_anchor_prediction"] + accepted.astype(float) * frame["residual_scale"] * (
            frame["raw_stage2_prediction"] - frame["stage1_anchor_prediction"]
        )
        if not np.allclose(frame["hpsafemoe_prediction"], reconstructed, rtol=0.0, atol=1e-12):
            raise AssertionError(f"JARVIS {task}: safety-controlled residual equation differs")
        actual = tuple(
            float(np.mean(np.abs(frame[column] - frame["y_true"])))
            for column in ("stage1_anchor_prediction", "raw_stage2_prediction", "hpsafemoe_prediction")
        )
        expected = (expected_stage1, expected_raw, expected_final)
        if not np.allclose(actual, expected, rtol=0.0, atol=2e-14):
            raise AssertionError(f"JARVIS {task}: MAE differs")
        summary = jarvis_summary.loc[task]
        if int(summary["n_samples"]) != count or int(summary["accepted_rows"]) != expected_accepted:
            raise AssertionError(f"JARVIS {task}: summary count differs")
        if not np.allclose(
            summary[["stage1_mae", "raw_stage2_mae", "hpsafemoe_mae"]].astype(float),
            expected,
            rtol=0.0,
            atol=2e-14,
        ):
            raise AssertionError(f"JARVIS {task}: summary MAE differs")
    if total_rows != 24068:
        raise AssertionError("JARVIS must contain 24,068 row-level predictions")

    protocol = json.loads(require("results/JARVIS/protocol_certificate.json").read_text(encoding="utf-8"))
    boundary = {
        "locked_before_test_prediction": True,
        "task_scope": "five_predeclared_release_tasks",
        "task_policy": "one_global_profile",
        "calibration_data_scope": "official_training_oof_and_official_validation",
        "prediction_input_scope": "features_only",
        "prediction_barrier_before_test_scoring": True,
        "test_label_role": "evaluation",
        "profile_selection_timing": "before_test_prediction",
    }
    for key, expected in boundary.items():
        if protocol.get(key) != expected:
            raise AssertionError(f"JARVIS protocol boundary differs for {key}")

    comparison = pd.read_csv(require("results/JARVIS/benchmark_comparison.csv"))
    if len(comparison) != 20 or set(comparison["evidence_type"]) != {
        "matched_current_pipeline", "literature_reported_reference"
    }:
        raise AssertionError("JARVIS benchmark comparison is incomplete")


def check_matbench_predictions() -> None:
    root = require("results/matbench/per_sample_predictions")
    variants = (
        "stage1_specialist_only",
        "ablation_no_cross_expert__oof_locked",
        "final__oof_locked",
    )
    expected_counts = {
        "steels": 312,
        "expt_gap": 4604,
        "glass": 5680,
        "expt_is_metal": 4921,
        "jdft2d": 636,
        "dielectric": 4764,
        "log_kvrh": 10987,
        "log_gvrh": 10987,
        "perovskites": 18928,
        "phonons": 1265,
        "mp_gap": 106113,
        "mp_is_metal": 106113,
        "mp_e_form": 132752,
    }
    required_keys = {
        "sample_ids",
        "anchor_prediction",
        "raw_prediction",
        "prediction",
        "accepted",
        "benefit_probability",
        "task_certificate_accepted",
        "residual_scale",
        "route_index",
        "route_probabilities",
        "directional_agreement",
        "route_consensus",
        "data_attention",
        "learned_relation_attention",
    }
    collected: dict[str, list[np.ndarray]] = {task: [] for task in expected_counts}
    for fold in range(5):
        for task in expected_counts:
            reference_ids: np.ndarray | None = None
            reference_anchor: np.ndarray | None = None
            for variant in variants:
                path = root / f"outer_{fold}" / variant / f"{task}.npz"
                if not path.exists():
                    raise AssertionError(f"Matbench prediction archive is absent: {path.relative_to(ROOT)}")
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != required_keys:
                        raise AssertionError(f"Matbench prediction schema differs: {path.relative_to(ROOT)}")
                    sample_ids = np.asarray(archive["sample_ids"]).astype(str)
                    anchor = np.asarray(archive["anchor_prediction"], dtype=np.float64)
                    size = len(sample_ids)
                    if len(set(sample_ids.tolist())) != size:
                        raise AssertionError(f"duplicate Matbench sample IDs: {path.relative_to(ROOT)}")
                    vector_keys = required_keys - {
                        "route_probabilities",
                        "data_attention",
                        "learned_relation_attention",
                    }
                    for key in vector_keys:
                        if np.asarray(archive[key]).shape != (size,):
                            raise AssertionError(
                                f"Matbench row-array shape differs for {key}: {path.relative_to(ROOT)}"
                            )
                    if np.asarray(archive["route_probabilities"]).shape != (size, 3):
                        raise AssertionError(f"Matbench route shape differs: {path.relative_to(ROOT)}")
                    for key in ("data_attention", "learned_relation_attention"):
                        if np.asarray(archive[key]).shape != (size, 13):
                            raise AssertionError(
                                f"Matbench attention shape differs for {key}: {path.relative_to(ROOT)}"
                            )
                    if reference_ids is None:
                        reference_ids = sample_ids
                        reference_anchor = anchor
                    elif not np.array_equal(sample_ids, reference_ids) or not np.array_equal(
                        anchor, reference_anchor
                    ):
                        raise AssertionError(f"Matbench matched variants are not row-aligned: outer_{fold}/{task}")
                    if variant == "stage1_specialist_only":
                        if not np.array_equal(np.asarray(archive["prediction"]), anchor):
                            raise AssertionError(
                                f"Matbench Stage-1 prediction differs from anchor: outer_{fold}/{task}"
                            )
                        if np.asarray(archive["accepted"]).any():
                            raise AssertionError(
                                f"Matbench Stage-1 archive contains accepted corrections: outer_{fold}/{task}"
                            )
            assert reference_ids is not None
            collected[task].append(reference_ids)
    archives = list(root.glob("outer_*/*/*.npz"))
    if len(archives) != 195:
        raise AssertionError(f"expected 195 Matbench prediction archives, found {len(archives)}")
    for task, expected_count in expected_counts.items():
        sample_ids = np.concatenate(collected[task])
        if len(sample_ids) != expected_count or len(set(sample_ids.tolist())) != expected_count:
            raise AssertionError(f"Matbench five-fold coverage differs for {task}")


def check_full_benchmark_comparison() -> None:
    path = require("results/full_benchmark_comparison/full_benchmark_comparison.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    methods = {row["method"] for row in rows}
    tasks = {row["task_id"] for row in rows}
    if len(rows) != 78 or len(methods) != 6 or len(tasks) != 13:
        raise AssertionError("Full-benchmark comparison must contain 13 tasks × 6 methods")


def check_method_contract() -> None:
    config = read_flat_yaml_scalars(require("configs/experiments/stage2.yaml"))
    required = {
        "attention_top_k": 4,
        "stage1_frozen": True,
        "learned_task_source_relation": True,
        "task_private_residual_output": True,
        "stage2_artifact_source": "current_run",
        "training_data_scope": "outer_train",
        "calibration_data_scope": "outer_train_oof",
        "profile_lock_before_outer_test_prediction": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise AssertionError(f"Matbench method contract differs for {key}")
    if config.get("pool_config_relative_path") != "configs/reference/stage1_provider_pool.yaml":
        raise AssertionError("Matbench Stage-2 configuration does not use the released provider pool")
    stage2 = yaml.safe_load(require("configs/experiments/stage2.yaml").read_text(encoding="utf-8"))
    expected_candidates = ["proposal_full", "ablation_no_cross_expert"]
    if stage2.get("architecture_candidate_order") != expected_candidates:
        raise AssertionError("Matbench release contains unreported proposal variants")
    if list(stage2.get("architecture_candidate_specs", {})) != expected_candidates:
        raise AssertionError("Matbench proposal specification inventory differs")

    protocol = yaml.safe_load(require("configs/reference/data_protocol.yaml").read_text(encoding="utf-8"))
    official = protocol.get("official_validation", {})
    if (
        official.get("benchmark") != "matbench_v0.1"
        or int(official.get("n_outer_folds", -1)) != 5
        or int(official.get("random_state", -1)) != 18012019
        or official.get("shuffle") is not True
    ):
        raise AssertionError("Official Matbench split protocol differs")

    pool = yaml.safe_load(
        require("configs/reference/stage1_provider_pool.yaml").read_text(encoding="utf-8")
    )
    expected_pool = {
        "steels": ("tpot_mat_steels_anchor", 103),
        "expt_gap": ("mattervial_expt_gap_anchor", 40),
        "glass": ("anchorboost_glass_anchor", 512),
        "expt_is_metal": ("mattervial_expt_is_metal_anchor", 160),
        "jdft2d": ("jmp_l_jdft2d_anchor", 256),
        "dielectric": ("mattervial_dielectric_anchor", 160),
        "log_kvrh": ("mattervial_log_kvrh_anchor", 160),
        "log_gvrh": ("mattervial_log_gvrh_anchor", 160),
        "perovskites": ("jmp_l_perovskites_anchor", 256),
        "phonons": ("jmp_l_phonons_anchor", 256),
        "mp_gap": ("jmp_l_mp_gap_anchor", 256),
        "mp_is_metal": ("mattervial_mp_is_metal_anchor", 40),
        "mp_e_form": ("jmp_l_mp_e_form_anchor", 256),
    }
    experts = {str(item["source_task"]): item for item in pool.get("experts", [])}
    if set(experts) != set(expected_pool) or pool.get("anchor_map") != {
        task: expert_id for task, (expert_id, _) in expected_pool.items()
    }:
        raise AssertionError("Stage-1 provider assignment differs")
    for task, (expert_id, hidden_dim) in expected_pool.items():
        if experts[task].get("expert_id") != expert_id or int(experts[task].get("hidden_dim", -1)) != hidden_dim:
            raise AssertionError(f"Provider contract differs for {task}")

    runtime = yaml.safe_load(require("configs/expert_runtime.yaml").read_text(encoding="utf-8"))
    if set(runtime.get("experts", {})) != {"tpot_mat", "anchorboost", "mattervial_modnet", "jmp_l"}:
        raise AssertionError("Stage-1 runtime must declare all four provider families")
    jarvis = yaml.safe_load(require("integrations/jarvis/configs/jarvis_stage2.yaml").read_text(encoding="utf-8"))
    jarvis_required = {
        "calibration_data_roles": ["train_oof", "val"],
        "scoring_data_role": "test",
        "prediction_barrier_before_test_scoring": True,
    }
    for key, expected in jarvis_required.items():
        if jarvis.get(key) != expected:
            raise AssertionError(f"JARVIS label boundary differs for {key}")
    jarvis_proposal = yaml.safe_load(
        require("integrations/jarvis/configs/jarvis_proposal.yaml").read_text(encoding="utf-8")
    )
    if jarvis_proposal.get("architecture_candidate_order") != ["proposal_full"]:
        raise AssertionError("JARVIS release contains unreported proposal variants")
    tpot = read_flat_yaml_scalars(require("configs/experiments/stage1_tpot_mat_steels.yaml"))
    if tpot.get("status") != "release":
        raise AssertionError("TPOT-Mat configuration is not marked for release")
    text = require("configs/experiments/stage1_tpot_mat_steels.yaml").read_text(encoding="utf-8")
    if "random_seed: 18012019" not in text or "seed_scope: all_outer_folds" not in text:
        raise AssertionError("TPOT-Mat must use the fixed Matbench seed across all outer folds")

def check_analysis_assets() -> None:
    paths = (
        "analysis/representation/tables/representation_target_probes.csv",
        "analysis/representation/tables/representation_descriptor_cka.csv",
        "analysis/representation/tables/pca_coordinates.csv",
        "analysis/representation/input/physical_descriptors.csv",
        "analysis/descriptors/tables/raw_descriptor_linear_shap.csv",
        "analysis/descriptors/tables/raw_surrogate_fidelity.csv",
        "analysis/routing_and_benefit/tables/benefit_gate_linear_shap.csv",
        "analysis/routing_and_benefit/tables/benefit_gate_shap_task_feature.csv",
        "analysis/routing_and_benefit/tables/benefit_gate_shap_task_group.csv",
        "analysis/routing_and_benefit/tables/router_source_exposure_and_gain.csv",
        "analysis/routing_and_benefit/tables/task_correction_and_source_summary.csv",
        "results/matbench/task_matched_comparisons.csv",
        "analysis/routing_and_benefit/figures/benefit_gate_shap_part1.png",
        "analysis/routing_and_benefit/figures/benefit_gate_shap_part2.png",
        "analysis/representation/figures/representation_target_probes.pdf",
        "analysis/representation/figures/representation_target_probes.png",
        "analysis/representation/figures/representation_target_probes.svg",
        "analysis/representation/figures/representation_pca.pdf",
        "analysis/representation/figures/representation_pca.png",
        "analysis/representation/figures/representation_pca.svg",
        "analysis/representation/figures/descriptor_group_cka.pdf",
        "analysis/representation/figures/descriptor_group_cka.png",
        "analysis/representation/figures/descriptor_group_cka.svg",
        "analysis/representation/figures/descriptor_group_cka_permutation_z.pdf",
        "analysis/representation/figures/descriptor_group_cka_permutation_z.png",
        "analysis/representation/figures/descriptor_group_cka_permutation_z.svg",
        "analysis/descriptors/figures/descriptor_surrogate_attribution.pdf",
        "analysis/descriptors/figures/descriptor_surrogate_attribution.png",
        "analysis/descriptors/figures/descriptor_surrogate_attribution.svg",
        "analysis/routing_and_benefit/figures/safety_decisions_and_source_weights.pdf",
        "analysis/routing_and_benefit/figures/safety_decisions_and_source_weights.png",
        "analysis/routing_and_benefit/figures/safety_decisions_and_source_weights.svg",
        "analysis/routing_and_benefit/figures/electronic_transfer_evidence.pdf",
        "analysis/routing_and_benefit/figures/electronic_transfer_evidence.png",
        "analysis/routing_and_benefit/figures/electronic_transfer_evidence.svg",
        "results/JARVIS/README.md",
        "results/JARVIS/field_dictionary.csv",
        "results/JARVIS/protocol_certificate.json",
        "results/JARVIS/recompute_metrics.py",
        "results/JARVIS/benchmark_comparison.csv",
        "scripts/recompute_analysis_outputs.py",
        "scripts/prepare_matbench_manifests.py",
        "scripts/validate_stage1_provider_exports.py",
        "docs/REPRODUCIBILITY.md",
    )
    for path in paths:
        require(path)
    representations = list((ROOT / "analysis/representation/input/representations").glob("*.npz"))
    if len(representations) != 13:
        raise AssertionError("expected 13 representation archives")
    final_keys = {
        "stage1_anchor_prediction",
        "raw_stage2_prediction",
        "hpsafemoe_prediction",
        "correction_accepted",
        "benefit_probability",
        "route_index",
        "route_probabilities",
        "data_attention",
        "learned_relation_attention",
    }
    for path in representations:
        with np.load(path, allow_pickle=False) as archive:
            if not final_keys.issubset(archive.files):
                raise AssertionError(f"unsupported representation schema in {path.name}")
    sums = json.loads(require("analysis/representation/input/SHA256SUMS.json").read_text(encoding="utf-8"))
    for relative, record in sums.items():
        path = require(f"analysis/representation/input/{relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"] or path.stat().st_size != int(record["bytes"]):
            raise AssertionError(f"analysis input digest differs: {relative}")
    benefit = pd.read_csv(require("analysis/routing_and_benefit/tables/benefit_gate_shap_task_feature.csv"))
    if benefit[["task_display", "feature_display"]].astype(str).apply(
        lambda column: column.str.contains(r"[\u4e00-\u9fff]").any()
    ).any():
        raise AssertionError("benefit-gate display labels must use the release terminology")
    sharing = pd.read_csv(require("analysis/routing_and_benefit/tables/task_correction_and_source_summary.csv"))
    if len(sharing) != 13 or not np.allclose(
        sharing["anchor_retained_percent"] + sharing["stage2_correction_applied_percent"],
        100.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise AssertionError("task correction rates must partition evaluated samples")


def check_manuscript_source_data() -> None:
    required = (
        "source_data/README.md",
        "source_data/jarvis_paired_bootstrap.csv",
        "source_data/elastic_expert_fold_cka.csv",
        "source_data/gain_normalization_sensitivity.csv",
        "source_data/gain_normalization_sensitivity_method.json",
        "source_data/hea95_shared_label_group_sensitivity.csv",
        "source_data/hea95_shared_label_group_sensitivity_summary.csv",
        "source_data/low_label_calibration_distribution_summary.csv",
        "source_data/low_label_calibration_paired_splits.csv",
        "source_data/matbench_aggregate_uncertainty.csv",
        "source_data/mechanism_uncertainty.csv",
        "source_data/mechanism_uncertainty_method.json",
        "source_data/native_metric_changes.csv",
    )
    for path in required:
        require(path)

    aggregate = pd.read_csv(require("source_data/matbench_aggregate_uncertainty.csv"))
    expected_effects = {
        "Full HP-SafeMoE vs Stage 1": 7.157629369537986,
        "Full HP-SafeMoE vs No-cross-expert": 5.5820337444003805,
    }
    if set(aggregate["comparison"]) != set(expected_effects):
        raise AssertionError("Matbench aggregate comparison inventory differs")
    for comparison, expected in expected_effects.items():
        row = aggregate.loc[aggregate["comparison"] == comparison].iloc[0]
        if not np.isclose(float(row["macro_effect_percentage_points"]), expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"Matbench aggregate effect differs for {comparison}")
        if int(row["n_tasks"]) != 13 or int(row["n_folds_per_task"]) != 5:
            raise AssertionError(f"Matbench aggregate analysis scope differs for {comparison}")

    jarvis_bootstrap = pd.read_csv(require("source_data/jarvis_paired_bootstrap.csv"))
    if len(jarvis_bootstrap) != 6 or set(jarvis_bootstrap["database"]) != {"JARVIS"}:
        raise AssertionError("JARVIS paired-bootstrap inventory differs")

    sensitivity = pd.read_csv(require("source_data/gain_normalization_sensitivity.csv"))
    if len(sensitivity) != 6 or "Primary 13-task macro" not in set(sensitivity["analysis"]):
        raise AssertionError("gain-normalization sensitivity inventory differs")
    native = pd.read_csv(require("source_data/native_metric_changes.csv"))
    if len(native) != 13 or native["task"].nunique() != 13:
        raise AssertionError("native-metric source data must contain all 13 Matbench tasks")

    paired = pd.read_csv(require("source_data/low_label_calibration_paired_splits.csv"))
    paired_counts = paired.groupby("dataset")["split_id"].nunique().to_dict()
    if paired_counts != {"B2-18": 18, "HEA-95": 100} or len(paired) != 118:
        raise AssertionError("low-label paired-split inventory differs")
    distributions = pd.read_csv(require("source_data/low_label_calibration_distribution_summary.csv"))
    if len(distributions) != 18 or set(distributions["dataset"]) != {"HEA-95", "B2-18"}:
        raise AssertionError("low-label distribution summary differs")

    mechanism = pd.read_csv(require("source_data/mechanism_uncertainty.csv"))
    if len(mechanism) != 5:
        raise AssertionError("mechanism uncertainty inventory differs")
    elastic = pd.read_csv(require("source_data/elastic_expert_fold_cka.csv"))
    cka = elastic["bulk_shear_linear_cka"].astype(float)
    if sorted(elastic["outer_fold"].astype(int).tolist()) != list(range(5)):
        raise AssertionError("elastic-expert CKA must contain the five official outer folds")
    if not np.isclose(float(cka.mean()), 0.8518246410456765, rtol=0.0, atol=1e-12):
        raise AssertionError("elastic-expert mean CKA differs")
    if not np.isclose(float(cka.std(ddof=1)), 0.07259682024713017, rtol=0.0, atol=1e-12):
        raise AssertionError("elastic-expert CKA standard deviation differs")


def check_release_scope() -> None:
    require("CITATION.cff")

    variants = pd.read_csv(require("results/matbench/task_variant_mean_std.csv"))
    expected_variants = {"Stage1", "HP-SafeMoE", "No cross-expert", "No safety"}
    if set(variants["variant"]) != expected_variants or len(variants) != 13 * 4:
        raise AssertionError("Matbench summary contains an unreported experiment variant")
    fold_scores = pd.read_csv(require("analysis/routing_and_benefit/tables/outer_fold_scores.csv"))
    if set(fold_scores["variant"]) != expected_variants or len(fold_scores) != 5 * 13 * 4:
        raise AssertionError("Matbench fold table contains an unreported experiment variant")

    matched = pd.read_csv(require("results/matbench/task_matched_comparisons.csv"))
    expected_columns = [
        "task",
        "gain_stage1_specialist_only",
        "gain_no_cross_expert",
        "gain_full_method",
        "gain_no_safety",
        "cross_expert_increment_percent",
        "safety_increment_percent",
    ]
    if list(matched.columns) != expected_columns or len(matched) != 13:
        raise AssertionError("Primary matched-comparison table differs")

    blocked_fragments = (
        "audit_submission",
        "fig2_rounding_audit",
        "linear_probe_preprocessing_audit",
        "submission_numeric_audit",
        "pre_fold_scaling",
        "task_ablation_decomposition",
        "render_submission_figures",
    )
    blocked_paths = [
        path.relative_to(ROOT).as_posix()
        for path in publication_paths()
        if any(fragment in path.name.lower() for fragment in blocked_fragments)
    ]
    if blocked_paths:
        raise AssertionError(f"manuscript-production or historical files are present: {blocked_paths[:3]}")


def publication_paths():
    """Yield public working-tree paths while ignoring local Git metadata."""
    for path in ROOT.rglob("*"):
        if ".git" not in path.relative_to(ROOT).parts:
            yield path


def check_hygiene() -> None:
    blocked_names = {".DS_Store", ".pytest_cache", ".ruff_cache", "__pycache__"}
    encountered = [path for path in publication_paths() if path.name in blocked_names]
    if encountered:
        raise AssertionError(f"generated cache files are present: {encountered[:3]}")
    empty_directories = [path for path in publication_paths() if path.is_dir() and not any(path.iterdir())]
    if empty_directories:
        raise AssertionError(f"empty directories are present: {empty_directories[:3]}")
    blocked_text = (
        "/Users/", "/home/", "/mnt/", "/data/models", "results_0917",
        "/" + "absolute/path",
        "PLACE" + "HOLDER",
        "place" + "holder",
        "T" + "BD",
        "\u5f85\u8865",
        "\u5f85\u5b9a",
        "historical exploratory",
        "final__balanced",
        "Add the final article DOI",
        "ablation_data_only",
        "ablation_relation_only",
        "ablation_no_learned_relation",
        "ablation_shared_output_basis",
        "ablation_task_only_safety",
        "ablation_no_task_certificate",
        "ablation_shuffled_relation_alignment",
    )
    text_suffixes = {".md", ".py", ".yaml", ".yml", ".toml", ".json", ".csv", ".sh", ".txt"}
    for path in publication_paths():
        if path == Path(__file__).resolve() or not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in blocked_text:
            if token in content:
                raise AssertionError(f"private path token {token!r} found in {path.relative_to(ROOT)}")
        if re.search(r"\bTODO\b", content):
            raise AssertionError(f"TODO marker found in {path.relative_to(ROOT)}")
        if re.search(r"[\u4e00-\u9fff]", content):
            raise AssertionError(f"Chinese text found in {path.relative_to(ROOT)}")
    numbered_variant_tag = re.compile(r"(?:^|[_-])v\d+(?:[._-]\d+)*(?:[._-]|$)", re.IGNORECASE)
    tagged_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in publication_paths()
        if path.is_file() and numbered_variant_tag.search(path.name)
    )
    if tagged_paths:
        raise AssertionError(f"release filename contains an internal numbered variant tag: {tagged_paths[:3]}")
    legacy_names = {
        "build_table5.py",
        "table5_baselines.csv",
        "table5_baselines.md",
        "Fig2.pdf", "Fig2.png", "Fig2.svg",
        "Fig3.pdf", "Fig3.png", "Fig3.svg",
        "Fig4.pdf", "Fig4.png", "Fig4.svg",
        "Fig5.pdf", "Fig5.png", "Fig5.svg",
        "Fig6.pdf", "Fig6.png", "Fig6.svg",
        "Fig7.pdf", "Fig7.png", "Fig7.svg",
        "FigS2_pca.pdf", "FigS2_pca.png", "FigS2_pca.svg",
    }
    legacy_paths = sorted(
        path.relative_to(ROOT).as_posix()
        for path in publication_paths()
        if path.name in legacy_names or "results/table5" in path.relative_to(ROOT).as_posix()
    )
    if legacy_paths:
        raise AssertionError(f"legacy manuscript-numbered release paths are present: {legacy_paths[:3]}")


def check_document_links() -> None:
    for relative in (
        "README.md",
        "docs/DATASETS.md",
        "docs/ANALYSIS.md",
        "docs/REPRODUCIBILITY.md",
        "docs/TERMINOLOGY.md",
        "docs/WEIGHTS_AND_DEPENDENCIES.md",
    ):
        path = require(relative)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists() or ROOT not in (resolved, *resolved.parents):
                raise AssertionError(f"Broken local documentation link in {relative}: {target}")


def main() -> None:
    check_datasets()
    check_external_results()
    check_matbench_predictions()
    check_full_benchmark_comparison()
    check_method_contract()
    check_analysis_assets()
    check_manuscript_source_data()
    check_release_scope()
    check_hygiene()
    check_document_links()
    print(
        "Public release verified: structure, datasets, results, protocol contracts, "
        "analysis assets, manuscript source data, and path hygiene"
    )


if __name__ == "__main__":
    main()

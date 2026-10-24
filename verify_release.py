#!/usr/bin/env python3
"""Validate the public release structure, results, terminology, and path hygiene."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
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
    module_path = require("results/JARVIS/recompute_metrics.py")
    spec = importlib.util.spec_from_file_location("jarvis_release_metrics", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert result["tasks"] == 34 and result["test_rows"] == 95477
    paired_path = require("scripts/recompute_external_paired_intervals.py")
    paired_spec = importlib.util.spec_from_file_location("external_paired_intervals", paired_path)
    paired_module = importlib.util.module_from_spec(paired_spec)
    paired_spec.loader.exec_module(paired_module)
    paired_module.verify()


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
    if runtime["experts"]["mattervial_modnet"].get("stage2_native_latent") != "ordered_member_penultimate_concatenation":
        raise AssertionError("MatterVial Stage-2 representation description differs")
    if "cross_task_leakage" in protocol or "production_latent_dim" in protocol.get("artifacts", {}):
        raise AssertionError("Unimplemented filtering or uniform hidden-width metadata is present")
    registry = yaml.safe_load(require("configs/sota_registry.yaml").read_text(encoding="utf-8"))
    if any({"current_reference", "notable_candidates"}.intersection(task) for task in registry["tasks"].values()):
        raise AssertionError("Exploratory reference-selection records are present")
    jarvis = json.loads(require("integrations/jarvis/configs/protocol.json").read_text())
    assert jarvis["calibration_data_roles"] == ["val"]
    assert jarvis["prediction_barrier_before_test_scoring"] is True
    assert jarvis["task_count"] == 34
    assert jarvis["stage2_seeds"] == [20261071, 20261072]
    assert jarvis["stage2_steps"] == 1400 and jarvis["source_dropout"] == 0.05
    tasks = json.loads(require("integrations/jarvis/configs/registry.json").read_text())["tasks"]
    assert len(tasks) == 34 and sum(r["metric"] == "ACC" for r in tasks) == 7
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
        "source_data/external_correction_acceptance.csv",
        "results/matbench/task_matched_comparisons.csv",
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
        "source_data/JARVIS_task_metrics.csv",
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
    fold_scores = pd.read_csv(require("analysis/routing_and_benefit/tables/outer_fold_scores.csv"))
    paired = fold_scores.pivot(index=["task", "outer_fold"], columns="variant", values="normalized_gain_percent")
    expected_effects = {
        "Full HP-SafeMoE vs Stage 1": float(paired["HP-SafeMoE"].mean()),
        "Full HP-SafeMoE vs No-cross-expert": float((paired["HP-SafeMoE"] - paired["No cross-expert"]).mean()),
    }
    if set(aggregate["comparison"]) != set(expected_effects):
        raise AssertionError("Matbench aggregate comparison inventory differs")
    for comparison, expected in expected_effects.items():
        row = aggregate.loc[aggregate["comparison"] == comparison].iloc[0]
        if not np.isclose(float(row["macro_effect_percentage_points"]), expected, rtol=0.0, atol=1e-12):
            raise AssertionError(f"Matbench aggregate effect differs for {comparison}")
        if int(row["n_tasks"]) != 13 or int(row["n_folds_per_task"]) != 5:
            raise AssertionError(f"Matbench aggregate analysis scope differs for {comparison}")

    jarvis_scores = pd.read_csv(require("source_data/JARVIS_task_metrics.csv"))
    assert len(jarvis_scores) == 34 and jarvis_scores["task"].is_unique
    shares = pd.read_csv(require("source_data/JARVIS_acceptance_and_source_weights.csv"))
    assert len(shares) == 34 * 34
    assert shares["source_weight_percent"].notna().sum() == 858

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
        "core_analysis_summary",
        "task_mechanism_diagnosis",
        "full_benchmark_comparison.md",
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
        if not {".git", "run_outputs"}.intersection(path.relative_to(ROOT).parts):
            yield path


def check_hygiene() -> None:
    blocked_names = {".DS_Store", ".pytest_cache", ".ruff_cache", "__pycache__"}
    encountered = [path for path in publication_paths() if path.name in blocked_names]
    if encountered:
        raise AssertionError(f"generated cache files are present: {encountered[:3]}")
    history_names = {"history.json", "progress.json", "resume.pt"}
    history_suffixes = ("_history.json", "_progress.json", "_resume.pt", ".log")
    process_records = [
        path.relative_to(ROOT).as_posix()
        for path in publication_paths()
        if path.is_file()
        and (path.name.lower() in history_names or path.name.lower().endswith(history_suffixes))
    ]
    if process_records:
        raise AssertionError(f"training histories or process logs are present: {process_records[:3]}")
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
    text_suffixes = {".md", ".py", ".yaml", ".yml", ".toml", ".json", ".csv", ".sh", ".txt", ".cff"}
    unfinished_text = re.compile(
        r"\b(?:TODO|FIXME|TBD|TBC|XXX|WIP|placeholder)\b|/path/to|example\.com|"
        r"local publication candidate|local candidate|preparation version|"
        r"remote_snapshot|paper_bundle_\d{8}|pre-fix bridge|incomplete earlier three-fold",
        re.IGNORECASE,
    )
    for path in publication_paths():
        if path == Path(__file__).resolve() or not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in blocked_text:
            if token in content:
                raise AssertionError(f"private path token {token!r} found in {path.relative_to(ROOT)}")
        if unfinished_text.search(content):
            raise AssertionError(f"Unfinished or internal-process text found in {path.relative_to(ROOT)}")
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
    for path in publication_paths():
        if not path.is_file() or path.suffix.lower() != ".md":
            continue
        relative = path.relative_to(ROOT)
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists() or ROOT not in (resolved, *resolved.parents):
                raise AssertionError(f"Broken local documentation link in {relative}: {target}")



def check_inference_records() -> None:
    """Verify calibration/provenance records and the final checkpoint inventory."""
    require("source_data/JARVIS_protocol_metadata.json")
    require("source_data/external_correction_acceptance.csv")
    records = json.loads(require("results/matbench/experiment/full/fold_records.json").read_text())
    assert [row["outer_fold"] for row in records] == list(range(5))
    for row in records:
        assert row["proposal_model_count"] == 10 and len(row["checkpoint_paths"]) == 10
        assert len(row["calibration_files"]) == 13
        for filename, digest in row["calibration_files"].items():
            path = require(f"results/matbench/experiment/full/calibration/outer_{row['outer_fold']}/{filename}")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assets = json.loads(require("results/JARVIS/experiment/stage1_assets.json").read_text())
    assert assets["artifact_count"] == len(assets["artifacts"]) == 204
    assert sum(row["fold"] == -1 for row in assets["artifacts"]) == 34
    cost = json.loads(require("results/JARVIS/experiment/training_cost.json").read_text())
    assert len(cost["members"]) == cost["single_gpu_members"] == 4
    assert cost["maximum_concurrent_workers"] == 2
    assert all(row["step"] == row["steps"] == 1400 for row in cost["members"])
    assert np.isclose(sum(row["seconds"] for row in cost["members"]) / 3600,
                      cost["training_gpu_hours"], rtol=0, atol=1e-12)
    spec = importlib.util.spec_from_file_location("external_diagnostics", require("scripts/verify_external_diagnostics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()
    spec = importlib.util.spec_from_file_location("checkpoint_verification", require("scripts/verify_checkpoints.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()
    media = {".pdf", ".png", ".jpg", ".jpeg", ".svg", ".pptx", ".docx"}
    assert not [p for p in publication_paths() if p.is_file() and p.suffix.lower() in media]


def main() -> None:
    check_datasets()
    check_external_results()
    check_matbench_predictions()
    spec = importlib.util.spec_from_file_location("matbench_metric_verification", require("results/matbench/recompute_metrics.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(ROOT)
    check_full_benchmark_comparison()
    check_method_contract()
    check_analysis_assets()
    check_manuscript_source_data()
    check_release_scope()
    check_inference_records()
    check_hygiene()
    check_document_links()
    print(
        "Code, experiment records and final checkpoints verified: "
        "datasets, predictions, metrics, method contracts, calibration, analyses, "
        "source data and path hygiene"
    )


if __name__ == "__main__":
    main()

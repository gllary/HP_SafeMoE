#!/usr/bin/env python3
"""Recompute and verify the released descriptor, representation, PCA, and benefit-gate analyses."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "analysis/representation/input"
TASKS = (
    "steels", "expt_gap", "glass", "expt_is_metal", "jdft2d", "dielectric",
    "log_kvrh", "log_gvrh", "perovskites", "phonons", "mp_gap",
    "mp_is_metal", "mp_e_form",
)
CLASSIFICATION = {"glass", "expt_is_metal", "mp_is_metal"}
RAW_EXCLUDE = {
    "task", "sample_id", "global_position", "outer_fold", "y_true",
    "input_type", "formula", "crystal_system",
}
DESCRIPTOR_GROUPS = {
    "composition_stoichiometry": (
        "n_elements", "composition_entropy", "max_atomic_fraction",
        "min_atomic_fraction", "metal_fraction", "transition_metal_fraction",
        "alkali_fraction", "alkaline_fraction", "halogen_fraction",
        "chalcogen_fraction", "rare_earth_fraction",
    ),
    "electronic_chemistry": (
        "atomic_number_mean", "atomic_number_std", "atomic_number_min",
        "atomic_number_max", "atomic_number_range", "electronegativity_mean",
        "electronegativity_std", "electronegativity_min",
        "electronegativity_max", "electronegativity_range", "period_mean",
        "period_std", "period_min", "period_max", "period_range",
        "group_mean", "group_std", "group_min", "group_max", "group_range",
    ),
    "size_mass": (
        "atomic_mass_mean", "atomic_mass_std", "atomic_mass_min",
        "atomic_mass_max", "atomic_mass_range", "atomic_radius_mean",
        "atomic_radius_std", "atomic_radius_min", "atomic_radius_max",
        "atomic_radius_range",
    ),
    "structure_geometry_density": (
        "n_sites", "volume", "volume_per_atom", "density", "lattice_a",
        "lattice_b", "lattice_c", "lattice_length_anisotropy",
        "lattice_alpha", "lattice_beta", "lattice_gamma",
    ),
    "symmetry": ("spacegroup_number",),
}
TASK_DISPLAY = {
    "steels": "Steel yield strength",
    "expt_gap": "Experimental band gap",
    "jdft2d": "2D exfoliation energy",
    "dielectric": "Refractive index",
    "log_kvrh": "Bulk modulus (log K)",
    "log_gvrh": "Shear modulus (log G)",
    "perovskites": "Perovskite formation energy",
    "phonons": "Optical phonon frequency",
    "mp_gap": "MP band gap",
    "mp_e_form": "MP formation energy",
}
FEATURE_DISPLAY = {
    "anchor_prediction": "Stage-1 anchor prediction",
    "raw_proposal_prediction": "Pre-safety Stage-2 proposal",
    "proposal_residual": "Signed proposal correction",
    "abs_proposal_residual": "Proposal correction magnitude",
    "proposal_ensemble_std": "Proposal ensemble standard deviation",
    "directional_agreement": "Proposal directional agreement",
    "route_consensus": "Route consensus",
    "data_minus_relation_prediction": "Sample-conditioned minus Relation-augmented prediction",
    "abs_data_minus_relation_prediction": "Absolute branch-prediction difference",
    "route_anchor_probability": "Anchor-route probability",
    "route_data_probability": "Sample-conditioned route probability",
    "route_relation_probability": "Relation-augmented route probability",
    "route_max_probability": "Maximum route probability",
    "route_entropy": "Route entropy",
    "data_attention_max": "Maximum Sample-conditioned attention weight",
    "relation_attention_max": "Maximum Relation-augmented attention weight",
    "data_attention_entropy": "Sample-conditioned attention entropy",
    "relation_attention_entropy": "Relation-augmented attention entropy",
    "anchor_standardized": "Standardized anchor position",
    "abs_anchor_standardized": "Anchor distance from training center",
    "available_expert_fraction": "Available-expert fraction",
    "expert_prediction_disagreement": "Expert prediction disagreement",
    "anchor_vs_expert_mean_abs": "Anchor-to-expert-mean absolute difference",
    "mean_expert_uncertainty": "Mean expert uncertainty",
    "max_expert_uncertainty": "Maximum expert uncertainty",
    "max_abs_expert_prediction": "Maximum absolute expert prediction",
}
FEATURE_GROUP = {
    "anchor_prediction": "Prediction and correction",
    "raw_proposal_prediction": "Prediction and correction",
    "proposal_residual": "Prediction and correction",
    "abs_proposal_residual": "Prediction and correction",
    "proposal_ensemble_std": "Prediction and correction",
    "directional_agreement": "Prediction and correction",
    "max_abs_expert_prediction": "Prediction and correction",
    "route_consensus": "Branch disagreement",
    "data_minus_relation_prediction": "Branch disagreement",
    "abs_data_minus_relation_prediction": "Branch disagreement",
    "route_anchor_probability": "Route state",
    "route_data_probability": "Route state",
    "route_relation_probability": "Route state",
    "route_max_probability": "Route state",
    "route_entropy": "Route state",
    "data_attention_max": "Attention structure",
    "relation_attention_max": "Attention structure",
    "data_attention_entropy": "Attention structure",
    "relation_attention_entropy": "Attention structure",
    "anchor_standardized": "Sample position",
    "abs_anchor_standardized": "Sample position",
    "available_expert_fraction": "Sample position",
    "expert_prediction_disagreement": "Expert disagreement and uncertainty",
    "anchor_vs_expert_mean_abs": "Expert disagreement and uncertainty",
    "mean_expert_uncertainty": "Expert disagreement and uncertainty",
    "max_expert_uncertainty": "Expert disagreement and uncertainty",
}


def load_arrays(task: str) -> dict[str, np.ndarray]:
    with np.load(INPUT / "representations" / f"{task}.npz", allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) <= 1e-14 or np.nanstd(b) <= 1e-14:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def local_gain(task: str, y: np.ndarray, anchor: np.ndarray, final: np.ndarray) -> np.ndarray:
    if task not in CLASSIFICATION:
        return np.abs(y - anchor) - np.abs(y - final)
    eps = 1e-8
    anchor = np.clip(anchor, eps, 1 - eps)
    final = np.clip(final, eps, 1 - eps)
    anchor_loss = -(y * np.log(anchor) + (1 - y) * np.log1p(-anchor))
    final_loss = -(y * np.log(final) + (1 - y) * np.log1p(-final))
    return anchor_loss - final_loss


def raw_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    numeric_names = [name for name in frame.columns if name not in RAW_EXCLUDE]
    numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    categories = pd.get_dummies(
        frame["crystal_system"].fillna("unknown"),
        prefix="crystal_system",
        prefix_sep="=",
    )
    names = [*numeric_names, *categories.columns.tolist()]
    bases = [*numeric_names, *(["crystal_system"] * categories.shape[1])]
    return np.column_stack((numeric, categories.to_numpy(dtype=np.float64))), names, bases


def fit_transform(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.zeros(train.shape[1], dtype=np.float64)
    observed = np.isfinite(train).any(axis=0)
    median[observed] = np.nanmedian(train[:, observed], axis=0)
    train_i = np.where(np.isfinite(train), train, median)
    test_i = np.where(np.isfinite(test), test, median)
    mean = train_i.mean(axis=0)
    scale = np.where(train_i.std(axis=0) > 1e-12, train_i.std(axis=0), 1.0)
    return (train_i - mean) / scale, (test_i - mean) / scale


def endpoint_values(task: str, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = arrays["y_true"].astype(np.float64)
    anchor = arrays["stage1_anchor_prediction"].astype(np.float64)
    final = arrays["hpsafemoe_prediction"].astype(np.float64)
    return {
        "stage1_anchor_prediction": anchor,
        "hpsafemoe_prediction": final,
        "stage2_correction": final - anchor,
        "benefit_gate_probability": arrays["benefit_probability"].astype(np.float64),
        "realized_local_gain": local_gain(task, y, anchor, final),
    }


def recompute_surrogates(descriptors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fidelity_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []
    alphas = np.logspace(-2, 4, 13)
    for task in TASKS:
        frame = descriptors.loc[descriptors.task == task].copy()
        arrays = load_arrays(task)
        if not np.array_equal(frame.sample_id.astype(str).to_numpy(), arrays["sample_ids"].astype(str)):
            raise ValueError(f"Descriptor/representation sample-ID mismatch: {task}")
        x, feature_names, base_names = raw_design(frame)
        groups = arrays["outer_fold"].astype(int)
        for endpoint, target in endpoint_values(task, arrays).items():
            predictions = np.full(len(target), np.nan, dtype=np.float64)
            selected_alphas: list[float] = []
            if np.nanstd(target) > 1e-14:
                for fold in sorted(np.unique(groups)):
                    train, test = groups != fold, groups == fold
                    x_train, x_test = fit_transform(x[train], x[test])
                    model = RidgeCV(alphas=alphas).fit(x_train, target[train])
                    predictions[test] = model.predict(x_test)
                    selected_alphas.append(float(model.alpha_))
                cv_r2 = float(r2_score(target, predictions))
                cv_mae = float(mean_absolute_error(target, predictions))
                cv_spearman = safe_spearman(target, predictions)
                x_all, _ = fit_transform(x, x)
                model = RidgeCV(alphas=alphas).fit(x_all, target)
                shap = (x_all - x_all.mean(axis=0)) * model.coef_[None, :]
                coefficient = model.coef_
                selected_alpha = float(model.alpha_)
                baseline = float(model.intercept_ + np.dot(model.coef_, x_all.mean(axis=0)))
            else:
                cv_r2 = cv_spearman = float("nan")
                cv_mae = 0.0
                shap = np.zeros_like(x)
                coefficient = np.zeros(x.shape[1])
                selected_alpha = float("nan")
                baseline = float(target[0])
            fidelity_class = (
                "high" if np.isfinite(cv_r2) and cv_r2 >= 0.75
                else "moderate" if np.isfinite(cv_r2) and cv_r2 >= 0.5
                else "low"
            )
            fidelity_rows.append({
                "task": task,
                "endpoint": endpoint,
                "n_samples": len(target),
                "n_raw_design_columns": x.shape[1],
                "official_fold_cv_r2": cv_r2,
                "official_fold_cv_spearman": cv_spearman,
                "official_fold_cv_mae": cv_mae,
                "median_inner_selected_alpha": float(np.median(selected_alphas)) if selected_alphas else float("nan"),
                "full_sample_selected_alpha": selected_alpha,
                "surrogate_baseline": baseline,
                "fidelity_class": fidelity_class,
                "interpretation": "linear_shap_for_raw_descriptor_surrogate",
            })
            grouped: dict[str, dict[str, Any]] = {}
            for _column, base, values, coef in zip(feature_names, base_names, shap.T, coefficient, strict=True):
                item = grouped.setdefault(
                    base,
                    {"abs": np.zeros(len(target)), "signed": np.zeros(len(target)), "coefs": []},
                )
                item["abs"] += np.abs(values)
                item["signed"] += values
                item["coefs"].append(float(coef))
            total = sum(float(np.mean(item["abs"])) for item in grouped.values())
            for feature, item in grouped.items():
                importance = float(np.mean(item["abs"]))
                shap_rows.append({
                    "task": task,
                    "endpoint": endpoint,
                    "raw_feature": feature,
                    "mean_abs_linear_shap": importance,
                    "normalized_mean_abs_linear_shap": importance / max(total, 1e-15),
                    "mean_signed_linear_shap": float(np.mean(item["signed"])),
                    "standardized_coefficient_sum": float(np.sum(item["coefs"])),
                    "surrogate_official_fold_cv_r2": cv_r2,
                    "surrogate_fidelity_class": fidelity_class,
                    "attribution_scope": "raw_composition_structure_to_endpoint_linear_surrogate",
                })
    return pd.DataFrame(fidelity_rows), pd.DataFrame(shap_rows)


def standardize_complete(values: np.ndarray) -> np.ndarray:
    median = np.zeros(values.shape[1], dtype=np.float64)
    observed = np.isfinite(values).any(axis=0)
    median[observed] = np.nanmedian(values[:, observed], axis=0)
    result = np.where(np.isfinite(values), values, median)
    keep = result.std(axis=0) > 1e-12
    if not np.any(keep):
        return np.empty((len(values), 0))
    return StandardScaler().fit_transform(result[:, keep])


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    numerator = float(np.sum((a.T @ b) ** 2))
    denominator = math.sqrt(float(np.sum((a.T @ a) ** 2)) * float(np.sum((b.T @ b) ** 2)))
    return numerator / denominator if denominator > 0 else float("nan")


def reduce_hidden(hidden: np.ndarray) -> np.ndarray:
    values = standardize_complete(hidden)
    if values.shape[1] == 0:
        return values
    components = min(32, values.shape[1], values.shape[0] - 1)
    return PCA(n_components=components, random_state=20260908).fit_transform(values)


def probe_predictions(task: str, hidden: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, str, int]:
    """Return strictly out-of-probe-fold predictions.

    Missing-value imputation and feature standardization are fitted only on the
    probe-training partition and then applied to its validation partition.  This
    keeps every preprocessing statistic inside the four-fold probe CV loop.
    """
    prediction = np.full(len(y), np.nan)
    if task in CLASSIFICATION:
        counts = np.bincount(y.astype(int), minlength=2)
        splits = max(2, min(4, int(counts.min())))
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        for train, test in cv.split(hidden, y):
            hidden_train, hidden_test = fit_transform(hidden[train], hidden[test])
            model = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000).fit(
                hidden_train, y[train]
            )
            prediction[test] = model.predict_proba(hidden_test)[:, 1]
        return prediction, "roc_auc", splits
    splits = min(4, len(y))
    cv = KFold(n_splits=splits, shuffle=True, random_state=seed)
    for train, test in cv.split(hidden):
        hidden_train, hidden_test = fit_transform(hidden[train], hidden[test])
        model = Ridge(alpha=100.0).fit(hidden_train, y[train])
        prediction[test] = model.predict(hidden_test)
    return prediction, "r2", splits


def recompute_representations(
    descriptors: pd.DataFrame,
    metadata: dict[str, Any],
    permutations: int,
    *,
    include_cka_and_pca: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probe_rows: list[dict[str, Any]] = []
    probe_fold_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    pca_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260908)
    for expert_index, task in enumerate(TASKS):
        frame = descriptors.loc[descriptors.task == task].copy()
        arrays = load_arrays(task)
        hidden_all = arrays[f"expert_{expert_index:02d}_hidden"].astype(np.float64)
        available_all = arrays[f"expert_{expert_index:02d}_available"].astype(bool)
        y_all = arrays["y_true"].astype(np.float64)
        fold_all = arrays["outer_fold"].astype(int)
        prediction_all = arrays[f"expert_{expert_index:02d}_prediction"].astype(np.float64)
        if not np.all(available_all):
            raise ValueError(f"Own expert unexpectedly unavailable: {task}")
        fold_scores: list[float] = []
        fold_secondary: list[float] = []
        anchor_scores: list[float] = []
        for fold in sorted(np.unique(fold_all)):
            use = fold_all == fold
            hidden, y = hidden_all[use], y_all[use]
            seed = 20260908 + fold
            prediction, metric_name, probe_folds = probe_predictions(task, hidden, y, seed)
            if task in CLASSIFICATION:
                fold_score = float(roc_auc_score(y, prediction))
                fold_spearman = float("nan")
                anchor_scores.append(float(roc_auc_score(y, prediction_all[use])))
            else:
                fold_score = float(r2_score(y, prediction))
                fold_spearman = safe_spearman(y, prediction)
                anchor_scores.append(float(r2_score(y, prediction_all[use])))
            fold_scores.append(fold_score)
            fold_secondary.append(fold_spearman)
            probe_fold_rows.append({
                "expert_task": task,
                "outer_fold": int(fold),
                "probe_metric": metric_name,
                "probe_score": fold_score,
                "probe_spearman": fold_spearman,
                "n_samples": int(len(y)),
                "probe_cv_folds": int(probe_folds),
                "probe_seed": int(seed),
                "preprocessing": "imputation_and_standardization_fit_on_probe_training_fold",
            })
            if not include_cka_and_pca:
                continue
            reduced = reduce_hidden(hidden)
            group_values: dict[str, np.ndarray] = {}
            for group, columns in DESCRIPTOR_GROUPS.items():
                usable = [name for name in columns if name in frame.columns]
                prepared = standardize_complete(frame.loc[use, usable].to_numpy(dtype=np.float64))
                if prepared.shape[1] > 0:
                    group_values[group] = prepared
            crystal = standardize_complete(pd.get_dummies(frame.loc[use, "crystal_system"], dtype=float).to_numpy())
            if crystal.shape[1] > 0:
                group_values["symmetry"] = np.column_stack(
                    (group_values.get("symmetry", np.empty((len(y), 0))), crystal)
                )
            group_values["target_property"] = standardize_complete(y[:, None])
            for group, descriptor in group_values.items():
                observed = linear_cka(reduced, descriptor)
                null = np.asarray([
                    linear_cka(reduced, descriptor[rng.permutation(len(descriptor))])
                    for _ in range(permutations)
                ])
                cka_rows.append({
                    "expert_task": task,
                    "outer_fold": int(fold),
                    "descriptor_group": group,
                    "linear_cka": observed,
                    "permutation_null_mean": float(np.mean(null)),
                    "permutation_z": (observed - float(np.mean(null))) / max(float(np.std(null)), 1e-15),
                    "one_sided_permutation_p": float((1 + np.sum(null >= observed)) / (permutations + 1)),
                    "n_samples": len(y),
                    "hidden_dim": hidden.shape[1],
                    "reduced_dim_for_cka": reduced.shape[1],
                })
        expert = metadata["tasks"][task]["experts"][expert_index]
        probe_rows.append({
            "expert_task": task,
            "expert_id": expert["expert_id"],
            "provider": expert["provider"],
            "input_modality": expert["input_modality"],
            "hidden_dim": hidden_all.shape[1],
            "n_samples": len(y_all),
            "probe_metric": metric_name,
            "within_outer_fold_probe_mean": float(np.nanmean(fold_scores)),
            "within_outer_fold_probe_min": float(np.nanmin(fold_scores)),
            "within_outer_fold_probe_max": float(np.nanmax(fold_scores)),
            "within_outer_fold_probe_spearman_mean": float(np.nanmean(fold_secondary)) if np.any(np.isfinite(fold_secondary)) else float("nan"),
            "native_expert_prediction_metric_mean": float(np.nanmean(anchor_scores)),
            "coordinate_handling": "probe_preprocessing_fit_within_each_probe_training_fold;cka_fit_within_outer_fold",
        })
        if not include_cka_and_pca:
            continue
        chosen = int(pd.Series(fold_all).value_counts().index[0])
        use = fold_all == chosen
        xy = PCA(n_components=2, random_state=20260908).fit_transform(standardize_complete(hidden_all[use]))
        for sample_id, x, y_coord, target, accepted in zip(
            arrays["sample_ids"][use].astype(str),
            xy[:, 0],
            xy[:, 1],
            y_all[use],
            arrays["correction_accepted"][use].astype(bool),
            strict=True,
        ):
            pca_rows.append({
                "task": task,
                "outer_fold": chosen,
                "sample_id": sample_id,
                "pc1": float(x),
                "pc2": float(y_coord),
                "y_true": float(target),
                "correction_accepted": bool(accepted),
            })
    return (
        pd.DataFrame(probe_rows),
        pd.DataFrame(probe_fold_rows),
        pd.DataFrame(cka_rows),
        pd.DataFrame(pca_rows),
    )


def recompute_benefit_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = ROOT / "analysis/routing_and_benefit/tables/benefit_gate_linear_shap.csv"
    source = pd.read_csv(path)
    task_feature = (
        source.groupby(["task", "feature"], sort=False, as_index=False)
        .agg(
            mean_abs_shap=("p99_trimmed_mean_abs_linear_shap_log_odds", "mean"),
            mean_signed_shap=("mean_signed_linear_shap_log_odds", "mean"),
        )
    )
    totals = task_feature.groupby("task")["mean_abs_shap"].transform("sum")
    task_feature["normalized_importance_percent"] = 100.0 * task_feature["mean_abs_shap"] / totals
    task_feature["task_display"] = task_feature["task"].map(TASK_DISPLAY)
    task_feature["feature_display"] = task_feature["feature"].map(FEATURE_DISPLAY)
    if task_feature[["task_display", "feature_display"]].isna().any().any():
        raise ValueError("Benefit-gate display mapping is incomplete")
    grouped = task_feature.assign(feature_group=task_feature["feature"].map(FEATURE_GROUP))
    task_group = (
        grouped.groupby(["task", "feature_group"], as_index=False)["normalized_importance_percent"]
        .sum()
    )
    return task_feature, task_group


def compare_frame(name: str, observed: pd.DataFrame, path: Path, keys: list[str]) -> None:
    expected = pd.read_csv(path)
    if list(expected.columns) != list(observed.columns):
        raise AssertionError(f"{name}: column mismatch")
    expected = expected.sort_values(keys).reset_index(drop=True)
    observed = observed.sort_values(keys).reset_index(drop=True)
    if expected.shape != observed.shape:
        raise AssertionError(f"{name}: shape mismatch {expected.shape} != {observed.shape}")
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(
                expected[column].to_numpy(dtype=float),
                observed[column].to_numpy(dtype=float),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            ):
                delta = np.nanmax(np.abs(expected[column].to_numpy(dtype=float) - observed[column].to_numpy(dtype=float)))
                raise AssertionError(f"{name}: numeric drift in {column}; max absolute delta={delta}")
        else:
            if not expected[column].fillna("").astype(str).equals(observed[column].fillna("").astype(str)):
                raise AssertionError(f"{name}: text drift in {column}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate released tables after verification")
    parser.add_argument(
        "--probes-only",
        action="store_true",
        help="Recompute only the lightweight linear probes; leave CKA, PCA and other analyses untouched",
    )
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    if args.permutations != 200:
        raise ValueError("The release protocol fixes the CKA permutation count at 200")
    descriptors = pd.read_csv(INPUT / "physical_descriptors.csv")
    metadata = json.loads((INPUT / "METADATA.json").read_text(encoding="utf-8"))
    probes, probe_folds, cka, pca = recompute_representations(
        descriptors,
        metadata,
        args.permutations,
        include_cka_and_pca=not args.probes_only,
    )
    probe_path = ROOT / "analysis/representation/tables/representation_target_probes.csv"
    probe_fold_path = ROOT / "analysis/representation/tables/representation_target_probe_folds.csv"
    if args.probes_only:
        if args.write:
            probes.to_csv(probe_path, index=False, lineterminator="\n")
            probe_folds.to_csv(probe_fold_path, index=False, lineterminator="\n")
        else:
            compare_frame("representation target probes", probes, probe_path, ["expert_task"])
            compare_frame(
                "representation target probe folds",
                probe_folds,
                probe_fold_path,
                ["expert_task", "outer_fold"],
            )
        print(json.dumps({
            "status": "pass",
            "mode": "probes_only",
            "tasks": len(TASKS),
            "representation_probe_rows": len(probes),
            "representation_probe_fold_rows": len(probe_folds),
        }, indent=2, sort_keys=True))
        return
    fidelity, raw_shap = recompute_surrogates(descriptors)
    benefit_feature, benefit_group = recompute_benefit_tables()
    outputs = [
        ("raw surrogate fidelity", fidelity, ROOT / "analysis/descriptors/tables/raw_surrogate_fidelity.csv", ["task", "endpoint"]),
        ("raw descriptor SHAP", raw_shap, ROOT / "analysis/descriptors/tables/raw_descriptor_linear_shap.csv", ["task", "endpoint", "raw_feature"]),
        ("representation target probes", probes, probe_path, ["expert_task"]),
        ("representation target probe folds", probe_folds, probe_fold_path, ["expert_task", "outer_fold"]),
        ("representation descriptor CKA", cka, ROOT / "analysis/representation/tables/representation_descriptor_cka.csv", ["expert_task", "outer_fold", "descriptor_group"]),
        ("benefit-gate task-feature", benefit_feature, ROOT / "analysis/routing_and_benefit/tables/benefit_gate_shap_task_feature.csv", ["task", "feature"]),
        ("benefit-gate task-group", benefit_group, ROOT / "analysis/routing_and_benefit/tables/benefit_gate_shap_task_group.csv", ["task", "feature_group"]),
    ]
    for name, frame, path, keys in outputs:
        compare_frame(name, frame, path, keys)
    pca_path = ROOT / "analysis/representation/tables/pca_coordinates.csv"
    if args.write:
        for _name, frame, path, _keys in outputs:
            frame.to_csv(path, index=False, lineterminator="\n")
        pca.to_csv(pca_path, index=False, lineterminator="\n")
    elif pca_path.exists():
        compare_frame("PCA coordinates", pca, pca_path, ["task", "sample_id"])
    else:
        raise AssertionError("PCA coordinates are missing; run once with --write")
    print(json.dumps({
        "status": "pass",
        "tasks": len(TASKS),
        "descriptor_rows": len(descriptors),
        "surrogate_fidelity_rows": len(fidelity),
        "raw_descriptor_shap_rows": len(raw_shap),
        "representation_probe_rows": len(probes),
        "representation_cka_rows": len(cka),
        "pca_rows": len(pca),
        "benefit_feature_rows": len(benefit_feature),
        "benefit_group_rows": len(benefit_group),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

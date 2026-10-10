# Manuscript source data

This directory contains the numerical source data for the statistical results and supplementary analyses reported in the article. The files are derived from the released held-out predictions, matched Matbench summaries, routing diagnostics, or paired low-label calibration outputs; they do not represent additional model training.

## File-to-manuscript map

- `mechanism_uncertainty.csv` and `mechanism_uncertainty_method.json`: source-use, weight--benefit association, and matched Full-minus-No-cross uncertainty in Supplementary Table S5.
- `elastic_expert_fold_cka.csv`: five-fold bulk-modulus--shear-modulus representation CKA summarized in Supplementary Table S6.
- `matbench_aggregate_uncertainty.csv`: two-level bootstrap intervals and exact task-level sign-flip tests in Supplementary Table S1.
- `jarvis_paired_bootstrap.csv`: paired JARVIS test-record bootstrap results in Supplementary Table S7.
- `gain_normalization_sensitivity.csv`, `gain_normalization_sensitivity_method.json`, and `native_metric_changes.csv`: normalization and summary sensitivity analyses in Supplementary Table S3.
- `low_label_calibration_paired_splits.csv` and `low_label_calibration_distribution_summary.csv`: HEA-95 and B2-18 split-wise calibration results summarized in Supplementary Fig. S7 and Supplementary Table S12.
- `hea95_shared_label_group_sensitivity.csv` and `hea95_shared_label_group_sensitivity_summary.csv`: HEA-95 sensitivity analysis that retains each original five-record calibration set and excludes an evaluation record when its same-composition paired structure supplied a calibration label; values summarize all 100 fixed splits, 36 of which are affected.

The underlying Matbench, JARVIS, and external-case-study outputs are retained under `results/`. Representation and routing inputs are retained under `analysis/`.

# Manuscript source data

Numerical files for the article figures and tables, with dataset and method metadata.

## File-to-manuscript map

- `mechanism_uncertainty.csv` and `mechanism_uncertainty_method.json`: source-use, weight--benefit association, and matched Full-minus-No-cross uncertainty in Supplementary Table S5.
- `external_correction_acceptance.csv`: model-specific sample counts and correction-acceptance proportions for DS2-248 and Core-23. Reported percentages average the five model-specific sample proportions.
- `external_source_weights.csv`: non-target source weights pooled across accepted samples of the five frozen outer-fold ensembles. Recompute this table and verify the acceptance counts with `scripts/verify_external_diagnostics.py`; the aligned source arrays and their provenance are under each dataset's `results/.../experiment/` directory.
- `elastic_expert_fold_cka.csv`: five-fold bulk-modulus--shear-modulus representation CKA summarized in Supplementary Table S6.
- `matbench_aggregate_uncertainty.csv`: two-level bootstrap intervals and exact task-level sign-flip tests in Supplementary Table S1.
- `JARVIS_protocol_metadata.json`: frozen split assignment, source-role and normalization provenance cited in the Supplementary Methods.
- `JARVIS_task_metrics.csv`: all 34 JARVIS task scores and matched gains.
- `JARVIS_acceptance_and_source_weights.csv`: numerical source data for main Fig. 4 and Supplementary Fig. S7.
- `JARVIS_acceptance_and_source_weights_method.json`: diagnostic aggregation and plotting definitions.
- `gain_normalization_sensitivity.csv`, `gain_normalization_sensitivity_method.json`, and `native_metric_changes.csv`: normalization and summary sensitivity analyses in Supplementary Table S3.
- `low_label_calibration_paired_splits.csv` and `low_label_calibration_distribution_summary.csv`: HEA-95 and B2-18 split-wise calibration results summarized in Supplementary Fig. S8 and Supplementary Table S12.
- `hea95_shared_label_group_sensitivity.csv` and `hea95_shared_label_group_sensitivity_summary.csv`: paired-record sensitivity results for the 100 HEA-95 calibration splits.

The underlying Matbench, JARVIS, and external-case-study outputs are retained under `results/`. Representation and routing inputs are retained under `analysis/`.

## Use

Run `python scripts/recompute_matbench_comparisons.py --check` to check the No-cross-expert fields against the released fold scores, or use `--write` to regenerate them. Method JSON files provide the source paths, resampling settings and column definitions.

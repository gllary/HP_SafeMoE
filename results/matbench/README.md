# Matbench results

This directory contains the released Matbench summaries and the complete held-out prediction records for the three models used in the primary matched comparison.

## Complete held-out predictions

`per_sample_predictions/` contains 195 compressed NumPy archives: 5 official outer folds × 13 tasks × 3 matched variants.

- `stage1_specialist_only`: frozen task-matched Stage-1 specialist.
- `ablation_no_cross_expert__oof_locked`: matched Stage 2 control without separate non-target expert tokens or the learned target--source relation. Coarse pool-level reliability summaries remain available.
- `final__oof_locked`: Full HP-SafeMoE with the locked out-of-fold safeguard.

Within each task and outer fold, the three variants contain the same ordered `sample_ids`. Concatenating the five `final__oof_locked` files produces one held-out prediction for every benchmark record: 312 steels, 4,604 experimental-band-gap, 5,680 glass, 4,921 experimental-metallicity, 636 2D-exfoliation-energy, 4,764 dielectric, 10,987 bulk-modulus, 10,987 shear-modulus, 18,928 perovskite, 1,265 phonon, 106,113 Materials-Project-band-gap, 106,113 Materials-Project-metallicity, and 132,752 Materials-Project-formation-energy records.

Each archive contains:

- `sample_ids`: benchmark sample identifiers in official outer-test order;
- `anchor_prediction`: frozen Stage-1 prediction;
- `raw_prediction`: Stage-2 proposal before the locked safeguard;
- `prediction`: released prediction for the selected variant;
- `accepted`, `benefit_probability`, `task_certificate_accepted`, and `residual_scale`: safeguard decisions and parameters;
- `route_index` and `route_probabilities`: Anchor, Sample-conditioned, and Relation-augmented routing outputs, in that order;
- `directional_agreement` and `route_consensus`: proposal-ensemble diagnostics;
- `data_attention` and `learned_relation_attention`: source-attention arrays for the Sample-conditioned and Relation-augmented branches. The field names preserve the exported array schema.

Targets and official fold definitions are obtained from Matbench v0.1 as described in [`../../docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md). File sizes and SHA-256 digests are included in the repository-level `RELEASE_MANIFEST.csv`.

## Summary tables

- `task_variant_mean_std.csv`: five-fold means and standard deviations for Stage 1, Full HP-SafeMoE, No-cross-expert, and No-safety.
- `task_gain_statistics.csv`: task-level matched gains and evidence categories.
- `task_matched_comparisons.csv`: the Stage 1, No-cross-expert, Full, and No-safety gains used by the reported comparisons.
- `core_analysis_summary.json`: compact machine-readable summary of the primary Matbench analysis.

The separate `analysis/representation/input/representations/*.npz` files are stratified diagnostic samples of at most 1,500 records per task and additionally contain expert hidden representations. They should not be confused with the complete prediction archives released here.

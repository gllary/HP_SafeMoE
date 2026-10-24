# Matbench results

This directory contains the released Matbench summaries and the complete held-out prediction records for the three models used in the primary matched comparison.

## Complete held-out predictions

`per_sample_predictions/` contains 195 compressed NumPy archives: 5 official outer folds × 13 tasks × 3 matched variants.

- `stage1_specialist_only`: frozen task-matched Stage-1 specialist.
- `ablation_no_cross_expert__oof_locked`: No-cross-expert Stage 2 predictions.
- `final__oof_locked`: Full HP-SafeMoE predictions.

Within each task and outer fold, the three variants share the same ordered `sample_ids`. The five outer folds cover all official benchmark records for each of the 13 tasks.

Each archive contains:

- `sample_ids`: benchmark sample identifiers in official outer-test order;
- `anchor_prediction`: frozen Stage-1 prediction;
- `raw_prediction`: Stage-2 proposal before the locked safeguard;
- `prediction`: released prediction for the selected variant;
- `accepted`, `benefit_probability`, `task_certificate_accepted`, and `residual_scale`: safeguard decisions and parameters;
- `route_index` and `route_probabilities`: Anchor, Sample-conditioned, and Relation-augmented routing outputs, in that order;
- `directional_agreement` and `route_consensus`: proposal-ensemble diagnostics;
- `data_attention` and `learned_relation_attention`: source-attention arrays for the Sample-conditioned and Relation-augmented branches. The field names preserve the exported array schema.

`evaluation_targets/` contains separate, sample-aligned official targets and outer-fold membership for all 13 tasks. These files are used by `recompute_metrics.py`. Their source-manifest hashes are recorded in `experiment/evaluation_targets.json`. Official Matbench data preparation is described in [`../../docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md). File sizes and SHA-256 digests are included in the repository-level `RELEASE_MANIFEST.csv`.

## Summary tables

- `task_variant_mean_std.csv`: five-fold means and standard deviations for Stage 1, Full HP-SafeMoE, No-cross-expert, and No-safety.
- `task_gain_statistics.csv`: task-level matched gains and evidence categories.
- `task_matched_comparisons.csv`: the Stage 1, No-cross-expert, Full, and No-safety gains used by the reported comparisons.

`../full_benchmark_comparison/full_benchmark_comparison.csv` lists reference and baseline scores. Its `status`, `provider` and `evidence` columns identify the result type and source. `reproduced_baseline_fold_scores.csv` contains individual fold scores, and `baseline_fold_provenance.json` defines their units and aggregation.

The separate `analysis/representation/input/representations/*.npz` files are stratified diagnostic samples of at most 1,500 records per task and additionally contain expert hidden representations.

## Use

```bash
python results/matbench/recompute_metrics.py --check
python scripts/recompute_matbench_comparisons.py --check
```

The first command recomputes the 195 task-fold-variant scores from prediction arrays and targets. The second checks the matched comparison and uncertainty tables. [Experiment files](experiment/README.md) lists calibration, model-selection and provenance records. Final model paths are in [checkpoints/README.md](../../checkpoints/README.md).

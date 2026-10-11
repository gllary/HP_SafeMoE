# Mechanism-analysis assets

## Representation probes, PCA, and descriptor CKA

`analysis/representation/` contains:

- `tables/representation_target_probes.csv`: five-outer-fold summaries of the target probes for every Stage-1 representation.
- `tables/representation_target_probe_folds.csv`: the 65 outer-fold probe scores used in the main-text representation-probe figure.
- `tables/representation_descriptor_cka.csv`: fold-resolved linear CKA against descriptor groups, permutation nulls, z scores, and one-sided permutation p values.
- `tables/pca_coordinates.csv`: two-dimensional PCA coordinates for the first official outer fold of each task, together with target values and retained correction-acceptance metadata. The PCA panel is coloured by class or target value, not correction acceptance. The stored `outer_fold=0` corresponds to outer fold 1 in the manuscript.
- `input/physical_descriptors.csv`: descriptor matrix used by the probe, PCA, and CKA analyses.
- `input/representations/*.npz`: stratified diagnostic samples of the 13 expert-representation matrices, with at most 1,500 records per task.
- `figures/representation_target_probes.*`, `representation_pca.*`, `descriptor_group_cka.*`, and `descriptor_group_cka_permutation_z.*`: the current probe, PCA, CKA, and permutation-standardized CKA visualizations used in the article and Supplementary Information.

The diagnostic exports are drawn from the union of official outer-test partitions, retain the corresponding outer-fold identifier, and contain at most 1,500 records per task. Classification tasks use class-stratified sampling and regression tasks target-decile-stratified sampling; tasks with fewer than 1,500 records retain all samples (seed 20260908). Probe and CKA models are fitted separately within each fold-specific diagnostic subset. For the target probes, shuffled four-fold cross-validation produces one out-of-probe-fold prediction per record (stratified for classification; seed `20260908 + outer_fold`). Missing-value imputation and standardization are fitted on each probe-training partition and then applied to its probe-validation partition. Classification uses class-balanced logistic regression (`C=0.05`) and regression uses ridge regression (`alpha=100`); ROC–AUC or R² is scored across all diagnostic records in that outer fold before the five fold scores are averaged. CKA quantifies representational association with descriptor groups and retains its separate outer-fold-local preprocessing. Routing uses the learned task–source relation described in the method configuration.

The complete Matbench held-out prediction records are released separately under `results/matbench/per_sample_predictions/`. They cover every official outer-test record for Stage 1, the matched No-cross-expert control, and Full HP-SafeMoE. See [`results/matbench/README.md`](../results/matbench/README.md) for the array schema and record counts.

## Descriptor analysis

`analysis/descriptors/` contains the raw-feature inventory, linear-surrogate fidelity, and exact linear-SHAP values for the fitted descriptor surrogates. `raw_surrogate_fidelity.csv` reports how closely each surrogate represents its corresponding model signal.

## Source use, matched controls, and benefit gate

`analysis/routing_and_benefit/` contains:

- fold-resolved source exposure and realized gain;
- matched Stage 1, No-cross-expert, and Full decompositions;
- task-level gain statistics and mechanism diagnoses;
- linear benefit-gate SHAP outputs;
- task-feature and task-group benefit-gate summaries;
- `tables/task_correction_and_source_summary.csv`: task-level anchor-retention and correction-application rates, together with conditional non-target source-weight shares;
- `figures/safety_decisions_and_source_weights.*` and `figures/electronic_transfer_evidence.*`: the current safety/source-use and electronic-transfer figures used in the main text;
- two benefit-gate SHAP panels.

The displayed source weights are soft-routing diagnostics: branch attention is multiplied by the corresponding route probability even though deployed inference uses a hard route. They are normalized across non-target experts only for samples that receive a correction; they are not percentages of the final prediction. The target expert supplies the numerical anchor for every sample. Predictive benefit is evaluated separately through the matched Full versus No-cross-expert comparison and realized held-out gain.

The released comparison tables are limited to Stage 1, the independently fitted No-cross-expert control, Full HP-SafeMoE, and the No-safety control reported in the article.

The benefit-gate panels use the release task and feature terminology. Positive standardized log-odds weights for proposal-residual magnitude on experimental band gap, MP formation energy, perovskite formation energy, and optical phonons, and the negative weight on steel yield strength, are recorded directly in `tables/benefit_gate_linear_shap.csv`.

Run the complete numerical reconstruction and table comparison with:

```bash
python scripts/recompute_analysis_outputs.py
```

The CKA check fixes the random seed at `20260908` and uses 200 permutations, matching the released one-sided p-value resolution.

## Manuscript statistical source data

`source_data/` contains the numerical tables underlying the uncertainty and sensitivity analyses. These include aggregate matched Matbench comparisons, paired JARVIS bootstrap intervals, task-normalization sensitivity checks, native-metric changes, split-wise HEA-95 and B2-18 calibration results, the HEA-95 shared-label group sensitivity analysis, mechanism-analysis uncertainty, and fold-resolved bulk-modulus--shear-modulus CKA. The accompanying [`source_data/README.md`](../source_data/README.md) maps each file to its supplementary figure or table.

These files were calculated from existing held-out predictions, matched task summaries, routing diagnostics, and fixed calibration splits. No model fitting is performed when they are inspected or used to reproduce the reported summaries.

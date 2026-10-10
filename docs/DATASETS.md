# Datasets

## Redistributed datasets

| Dataset | Samples | Target | Public protocol | Files |
|---|---:|---|---|---|
| HEA-95 | 95 | `log10(E/GPa)` | 100 paired splits; 5 calibration and 90 test samples | `data/HEA-95/` |
| B2-18 | 18 | `log10(E/GPa)` | all 18 single-sample calibration choices; 17 test samples each | `data/B2-18/` |
| Core-23 | 23 | refractive index | target-label zero-shot | `data/Core-23/` |
| DS2-248 | 248 | experimental band gap in eV | target-label zero-shot | `data/DS2-248/` |

Each dataset directory contains `materials.csv` and `labels.csv` joined by `sample_id`. The B2-18 `youngs_modulus` column and the HEA-95 `target` column both store `log10(E/GPa)`.

The HEA-95 calibration partitions are fixed in `results/HEA-95/split_definitions.csv`. The release contains 95 distinct serialized structures and 90 unique compositions. Five pairs share a displayed target and literature source but correspond to different released structures. Their serialized identities and structure-matching checks are recorded in `results/HEA-95/shared_label_structure_validation.csv`. The validation script reproduces the group-separated sensitivity analysis in `source_data/hea95_shared_label_group_sensitivity*.csv`: the original five calibration records and fitted bias are retained, and an evaluation record is excluded only when its paired structure supplied a calibration label. Results cover all 100 fixed splits; 36 are affected.

B2-18 uses the row index of every sample exactly once as the calibration choice. Base predictions and paired per-split metrics are stored with each result set. For both low-label datasets, the reported target is `log10(E/GPa)` and the scalar output shift is the median calibration residual (a single residual for B2-18).

For Core-23 and DS2-248, zero-shot evaluation means that Core-23 sample records are absent from `matbench_dielectric`, DS2-248 compositions are absent from `matbench_expt_gap`, the models remain frozen, and no external labels enter fitting, output calibration, safety-rule selection, target-domain adaptation, or checkpoint selection. The reproducible cleaning and matching rules are summarized in [`EXTERNAL_DATA_PROVENANCE.md`](EXTERNAL_DATA_PROVENANCE.md).

## Standard benchmark sources

Matbench and JARVIS benchmark inputs are obtained from the official [Matbench](https://matbench.materialsproject.org/) and [JARVIS-DFT](https://pages.nist.gov/jarvis/databases/) resources. Scored JARVIS test outputs are provided under `results/JARVIS/`.

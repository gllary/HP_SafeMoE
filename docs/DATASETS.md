# Dataset files

| Dataset | Records | Target | Evaluation files |
| --- | ---: | --- | --- |
| HEA-95 | 95 | `log10(E/GPa)` | `data/HEA-95/`, `results/HEA-95/` |
| B2-18 | 18 | `log10(E/GPa)` | `data/B2-18/`, `results/B2-18/` |
| Core-23 | 23 | Refractive index at 589.3 nm | `data/Core-23/`, `results/Core-23/` |
| DS2-248 | 248 | Experimental band gap in eV | `data/DS2-248/`, `results/DS2-248/` |

## File layout

- Each dataset has `materials.csv` and `labels.csv`, joined by `sample_id`. The B2-18 `youngs_modulus` column and HEA-95 `target` column contain `log10(E/GPa)`.
- `results/HEA-95/split_definitions.csv` lists 100 calibration/evaluation splits, each with 5 calibration records and 90 evaluation records.
- B2-18 uses each of its 18 rows once as the calibration record, with the other 17 rows used for evaluation.
- `data/Core-23/screening_decisions.csv` contains candidate identifiers, inclusion flags and reason codes for the 23-record dataset.
- `results/HEA-95/shared_label_structure_validation.csv` contains paired-record identifiers and structure-comparison results.
- Prediction and metric files are stored in the corresponding dataset directory under `results/`.

## Commands

```bash
python scripts/recompute_external_case_studies.py
python scripts/recompute_external_paired_intervals.py --check
python scripts/verify_external_diagnostics.py
python scripts/validate_core23_screening.py
python scripts/validate_hea95_shared_labels.py
```

Official benchmark inputs are available from [Matbench](https://matbench.materialsproject.org/) and [JARVIS-DFT](https://pages.nist.gov/jarvis/databases/). Matbench filenames, download URLs and checksums are listed in `configs/reference/data_sources.yaml`. Input preparation is described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

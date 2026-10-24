# Figure and analysis data

## Files

| Path | Contents |
| --- | --- |
| `analysis/representation/input/physical_descriptors.csv` | Descriptor columns and sample identifiers |
| `analysis/representation/input/representations/` | Expert representations for up to 1,500 records per task |
| `analysis/representation/input/METADATA.json` | Array schema, expert order, dimensions and sampling settings |
| `analysis/representation/tables/representation_target_probes.csv` | Task-level probe summaries |
| `analysis/representation/tables/representation_target_probe_folds.csv` | 65 task-fold probe scores |
| `analysis/representation/tables/representation_descriptor_cka.csv` | Fold-level CKA, permutation statistics and descriptor groups |
| `analysis/representation/tables/pca_coordinates.csv` | PCA coordinates, targets and outer-fold identifiers |
| `analysis/descriptors/tables/` | Feature inventory, surrogate fidelity and linear-SHAP values |
| `analysis/routing_and_benefit/tables/` | Fold scores, source weights, correction rates and benefit-gate summaries |
| `source_data/` | Numerical tables mapped to manuscript figures and tables in [source_data/README.md](../source_data/README.md) |

`outer_fold` is zero-based. Complete Matbench prediction arrays are documented in [results/matbench/README.md](../results/matbench/README.md).

## Reconstruction

```bash
python scripts/recompute_analysis_outputs.py
python scripts/recompute_matbench_comparisons.py --check
python scripts/verify_external_diagnostics.py
```

Use `python scripts/recompute_matbench_comparisons.py --write` to regenerate control-comparison summaries and uncertainty tables from the released outer-fold scores.

`analysis/render_jarvis_supplementary_figures.py` writes figures and rendering metadata to `run_outputs/figures/`.

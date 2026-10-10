# JARVIS five-task results

This directory contains row-level outputs and independently recomputable metrics for five official JARVIS regression tasks.

| Task | Test samples | Unit | coGN Stage-1 MAE | HP-SafeMoE MAE | Relative gain |
|---|---:|---|---:|---:|---:|
| Formation energy | 5,572 | eV/atom | 0.0273 | 0.0271 | 0.602% |
| OptB88vdW band gap | 5,572 | eV | 0.1168 | 0.1120 | 4.109% |
| OptB88vdW total energy | 5,572 | eV/atom | 0.0263 | 0.0263 | 0.000% |
| MBJ band gap | 1,815 | eV | 0.2512 | 0.1963 | 21.854% |
| Energy above hull | 5,537 | eV/atom | 0.0409 | 0.0409 | 0.000% |

The 24,068 rows under `per_sample_predictions/` provide the official test target, Stage-1 anchor, raw Stage-2 proposal, HP-SafeMoE prediction, safety variables, routing diagnostics, and source-utilization weights. `field_dictionary.csv` defines every field.

For every sample, the HP-SafeMoE prediction satisfies

```text
hpsafemoe_prediction = stage1_anchor_prediction
                       + task_safety_accepted
                       × correction_accepted
                       × residual_scale
                       × (raw_stage2_prediction - stage1_anchor_prediction)
```

The task policy is selected from training OOF and official-validation evidence and locked before test prediction. The prediction barrier records the five prediction files before scoring. `protocol_certificate.json` records the protocol and task-level parameters.

`task_metrics.csv` is the full-precision result summary. `benchmark_comparison.csv` reproduces the comparison values, while marking ReciNet-MT and MoCE values as literature-reported references rather than matched-pipeline results.

Run the independent row-level check with:

```bash
python results/JARVIS/recompute_metrics.py --check
```

# Matbench experiment files

| Path | Contents |
| --- | --- |
| `full/calibration/` | 65 Full task-fold calibration records |
| `full/fold_records.json` | Full ensemble membership and calibration-file digests |
| `calibration/` | 65 No-cross-expert task-fold calibration records |
| `profiles/` | Selected safety profiles and their OOF score tables |
| `result_summary.json` | No-cross-expert fold scores and aggregates |
| `run_record.json` | Task count, folds, seeds, training steps and source hashes |
| `prediction_commit.json` | Prediction-file digests, array-key mapping and timestamps |
| `evaluation_targets.json` | Evaluation-target and source-manifest digests |
| `fold_records.json` | Final fold-level training, prediction and scoring records |
| `checkpoint_hashes.json` | Source identifiers and hashes for the 50 No-cross-expert members |
| `preflight.json` | Input-validation results |
| `implementation_verification.json` | Model-output, gradient and optimizer-update checks |

Final weights are under `checkpoints/matbench/`; `checkpoints/required_checkpoints.json` lists their files and calibration dependencies. Source-artifact hashes identify the source files, while `RELEASE_MANIFEST.csv` lists public-file hashes.

## Use

```bash
python results/matbench/recompute_metrics.py --check
python scripts/recompute_matbench_comparisons.py --check
```

See [REPRODUCIBILITY.md](../../../docs/REPRODUCIBILITY.md) for training commands and [Matbench result files](../README.md) for prediction-array definitions.

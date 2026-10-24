# JARVIS result files

| File or directory | Contents |
| --- | --- |
| `task_metrics.csv`, `experiment_results.json` | Final scores for 34 tasks: 27 regression tasks using MAE and seven classification tasks using accuracy |
| `per_sample_predictions/` | 95,477 official test records with labels, Stage 1 anchors, Full/No-cross-expert predictions and correction parameters |
| `field_dictionary.csv` | Prediction-column definitions |
| `experiment/normalization.json` | Fitted prediction-token normalization |
| `experiment/full/calibration.json`, `experiment/no_cross_expert/calibration.json` | Configuration-specific gates, scales and acceptance settings |
| `experiment/training_cost.json` | Training-cost and selected-checkpoint records for four Stage 2 members |
| `experiment/stage1_assets.json` | Specialist identifiers and hashes for 34 full-fit and 170 OOF roles |
| `experiment/` | Final checkpoint-selection, prediction and scoring records |
| `prediction_provenance.json` | Source-prediction, target and public-file digests |
| `implementation_provenance.json` | Source-module digests and implementation checks |
| `protocol_certificate.json` | Experiment settings, selected profiles and model steps |

Regression gain columns use relative MAE reduction; classification gain columns use accuracy differences in percentage points. `full_prediction` and `no_cross_prediction` contain the final outputs. Their `raw_prediction` counterparts contain proposals before calibration.

## Use

```bash
python results/JARVIS/recompute_metrics.py --check
```

Final models are under `checkpoints/jarvis/`. [Checkpoint instructions](../../checkpoints/README.md) give loading and calibration paths. Training commands are in the [JARVIS integration README](../../integrations/jarvis/README.md). Figure-source tables are listed in [source_data/README.md](../../source_data/README.md).

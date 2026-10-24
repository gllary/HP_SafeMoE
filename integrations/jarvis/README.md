# HP-SafeMoE on 34 JARVIS tasks

The integration uses 34 frozen property experts, 27 regression targets evaluated by MAE, and seven binary targets evaluated by accuracy. `configs/registry.json` fixes task order, official benchmark files and reference predictions from the specified leaderboard snapshot. `configs/protocol.json` gives training and evaluation settings.

## Required expert assets

Run `python -B scripts/download_checkpoints.py --benchmark jarvis` from the repository root to obtain the final Stage 1 weights for 34 experts and four Stage 2 members (two Full and two No-cross-expert) under `checkpoints/jarvis/`. The 170 Stage 1 OOF models used for training, full-universe hidden arrays, raw crystal structures and training labels are separate reproduction inputs. This Stage 2 workflow requires the following export layout, validated by `baseline_inputs.py`:

- `inputs/jarvis/data/universe_ids.npy`: the common ordered material identifiers.
- `inputs/jarvis/data/tasks/`: one directory for each task ID in `configs/registry.json`, containing `manifest.json`, `train.json`, `val.json` and `test.json`.
- `inputs/jarvis/artifacts/`: one directory per task, with subdirectories `full`, `fold0`, `fold1`, `fold2`, `fold3` and `fold4`. Each contains `prediction.npy`, `hidden.npy`, `config.json`, `complete.json` and its native model file (`model.pt`, `model.weights.h5` or `model.joblib`).
- `inputs/jarvis/public_full/`: one directory per task listed in `public_alignn_full_tasks`, containing `prediction.npy`, `hidden.npy` and `complete.json`.

`manifest.json` contains `role_ids`, `official_counts` and `within_task_role_overlap`. Predictions and hidden arrays have the common universe order. `config.json` records the fold, trained material identifiers and `test_labels_read: false`. `complete.json` records row counts, native hidden dimension and checkpoint hashes. Public ALIGNN full exports are used for the four tasks in `public_alignn_full_tasks`, with archive and array hashes in their completion records. Their OOF models are separately fitted. Experts are frozen throughout Stage 2.

Provider implementations and scientific references are listed in [REFERENCES.md](REFERENCES.md).

## Running Stage 2

From the repository root, place the expert exports in `inputs/jarvis/` using the layout above. The commands below train models and write run products to `run_outputs/jarvis/`.

```bash
python -m pip install -e integrations/jarvis
export HPSAFE_JARVIS_INPUT=inputs/jarvis
export HPSAFE_JARVIS_OUTPUT=run_outputs/jarvis
export HPSAFE_JARVIS_PUBLIC_FULL=inputs/jarvis/public_full
python -m jarvis_hpsafe.preflight
python -m jarvis_hpsafe.train
```

Released prediction files and metric commands are documented in [results/JARVIS](../../results/JARVIS/). Run `python -m pytest -q -p no:cacheprovider integrations/jarvis/tests` from the repository root to check the integration.

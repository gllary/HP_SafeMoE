# Reproduction workflow

Download the final model files into `checkpoints/` with `python -B scripts/download_checkpoints.py`. Use the commands below to prepare inputs, train models or recompute the released results. Provider environments are listed in [WEIGHTS_AND_DEPENDENCIES.md](WEIGHTS_AND_DEPENDENCIES.md).

## 1. Prepare official Matbench inputs

Obtain the official Matbench v0.1 files listed in `configs/reference/data_sources.yaml`, retain their listed filenames, and place them under `data/raw/`. Obtain the official `matbench_v0.1_validation.json` file recorded in `configs/reference/data_protocol.yaml`. The manifest builder verifies every published SHA-256 digest before writing the split artifacts.

```bash
python scripts/prepare_matbench_manifests.py \
  --official-validation data/raw/matbench_v0.1_validation.json
```

This produces the immutable five-fold manifests in `data/manifests/`. The same manifests are consumed by every Stage-1 provider and by Stage 2.

## 2. Produce the frozen Stage-1 exports

The provider assignment is declared once in `configs/reference/stage1_provider_pool.yaml`. Provider-specific environments and configurations are:

- TPOT-Mat, steels: `envs/tpot_mat.yml` and `configs/experiments/stage1_tpot_mat_steels.yaml`;
- AnchorBoost, glass: `envs/anchorboost.yml` and `configs/experiments/stage1_anchorboost_glass.yaml`;
- MatterVial/MODNet, six tasks: `envs/mattervial.yml` and the release recipes in `configs/expert_runtime.yaml`;
- JMP-L, five tasks: `envs/jmp_l.yml`, `configs/experiments/jmp_l_stage1.yaml`, and `configs/experiments/jmp_l_stage1_mp_tasks.yaml`.

Provider launchers are the `render_stage1_*`, `render_jmp_l_*`, `run_*`, `export_*` and `validate_*` scripts. Export files contain predictions, hidden representations, uncertainty, availability, sample IDs and checkpoint hashes.

MatterVial Stage 2 exports use `src/hpsafe_sota/stage2_common/mattervial_replay.py`. Required hidden dimensions are listed in `configs/reference/stage1_provider_pool.yaml`.

Place or link each expert export in a subdirectory of `stage1_exports/` named with its exact `expert_id` from the 13 entries in the provider-pool configuration. Then run the common export validation:

```bash
python scripts/validate_stage1_provider_exports.py --verify-hashes
```

The validator writes `stage1_exports/input_validation.json`, which is required by Stage 2.

## 3. Run Matbench Stage 2

The following commands use the released provider pool, validated exports, method configuration, and declared GPU resource profile:

```bash
python scripts/preflight_stage2.py \
  --feature-root . \
  --output-root run_outputs/stage2 \
  --config configs/experiments/stage2.yaml \
  --resource-config configs/server_resources_h20_8gpu_stage2.yaml

python scripts/render_stage2.py \
  --feature-root . \
  --output-root run_outputs/stage2 \
  --config configs/experiments/stage2.yaml \
  --resource-config configs/server_resources_h20_8gpu_stage2.yaml

python scripts/run_job_graph.py \
  --graph run_outputs/stage2/orchestration/experiment_job_graph.json
```

The generated graph runs training, calibration, prediction and scoring.

### Train No-cross-expert with the shared OOF safety profiles

No-cross-expert comprises 50 models and uses its own inner-OOF predictions for calibration. Full and No-cross-expert share the safety profiles selected from Full OOF evidence. With steps 1 and 2 completed, run:

```bash
python scripts/run_no_cross.py prepare --output-root run_outputs/no_cross --feature-root .
for outer in 0 1 2 3 4; do
  python scripts/run_no_cross.py train --output-root run_outputs/no_cross --outer-fold "$outer" --device cuda
done
for outer in 0 1 2 3 4; do
  python scripts/run_no_cross.py predict --output-root run_outputs/no_cross --outer-fold "$outer" --device cuda
done
python scripts/run_no_cross.py commit --output-root run_outputs/no_cross
python scripts/run_no_cross.py score --output-root run_outputs/no_cross
```

Use a fresh output directory for each fit. The commands prepare inputs, train 50 No-cross-expert members, generate predictions, commit their hashes and compute scores.

The general graph trains Full and No-cross-expert (100 members); the control-only commands train the 50 No-cross-expert members.

## 4. Verify released results

The shared environment supports release verification without installing the provider-specific training stacks:

```bash
python -m pip install -r requirements.txt
python scripts/build_full_benchmark_comparison.py --check
python results/matbench/recompute_metrics.py --check
python scripts/recompute_matbench_comparisons.py --check
python scripts/recompute_analysis_outputs.py
python scripts/recompute_external_case_studies.py
python scripts/recompute_external_paired_intervals.py --check
python scripts/verify_external_diagnostics.py
python scripts/validate_core23_screening.py
python scripts/validate_hea95_shared_labels.py
python results/JARVIS/recompute_metrics.py --check
python scripts/build_manifest.py --check
python verify_release.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests integrations/jarvis/tests
```

These commands check the released comparison tables, analysis outputs, external evaluations, JARVIS scores and file inventory.

`configs/experiments/stage2.yaml` defines Full and No-cross-expert. No-safety outputs use the unscaled Full proposal.

The 195 complete Matbench prediction archives and aligned evaluation targets are documented in [results/matbench/README.md](../results/matbench/README.md).
# Reproduction workflow

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

The corresponding launchers are the `render_stage1_*`, `render_jmp_l_*`, `run_*`, `export_*`, and `validate_*` commands under `scripts/`. Each export contains prediction, task-trained hidden representation, uncertainty, availability, sample identifiers, split provenance, and the checkpoint SHA-256.

Place or link each expert export at `stage1_exports/<expert_id>`, following the 13 exact expert identifiers in the provider-pool configuration. Then run the common export validation:

```bash
python scripts/validate_stage1_provider_exports.py --verify-hashes
```

The command writes `stage1_exports/input_validation.json`. Stage 2 starts after validation confirms all 13 target anchors for every official outer fold and validates the export schemas, hashes, split roles, and prediction/representation fields.

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

The graph first freezes the validated Stage-1 source policy, then trains and calibrates HP-SafeMoE and its matched No-cross-expert control using inner-OOF evidence. It commits all outer-test predictions and then runs the scoring phase.

## 4. Verify released results

The shared environment supports release verification without installing the provider-specific training stacks:

```bash
python -m pip install -r requirements.txt
python scripts/build_full_benchmark_comparison.py --check
python scripts/recompute_analysis_outputs.py
python scripts/recompute_external_case_studies.py
python scripts/validate_core23_screening.py
python scripts/validate_hea95_shared_labels.py
python results/JARVIS/recompute_metrics.py --check
python scripts/build_manifest.py --check
python verify_release.py
python -m pytest -q
python -m pytest -q integrations/jarvis/tests
```

These commands independently rebuild the full-benchmark comparison, derived analyses, the four external case studies, JARVIS metrics, the release manifest, and the automated test suites.

The released `configs/experiments/stage2.yaml` contains only the Full HP-SafeMoE proposal and the independently fitted No-cross-expert control used by the article. The No-safety control is derived from the Full proposal by applying the unscaled raw residual without the task certificate or sample-level benefit gate.

The complete held-out Matbench predictions used by the primary matched comparison are available under `results/matbench/per_sample_predictions/`. The release verifier checks all 195 archives, confirms identical sample ordering across Stage 1, No-cross-expert and Full within each fold, and verifies that the five held-out folds cover every benchmark record exactly once. Targets and fold definitions remain sourced from the official Matbench v0.1 files prepared in step 1.

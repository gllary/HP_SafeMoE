# HP-SafeMoE

This repository is the publication release of Heterogeneous-Provider Safe Mixture-of-Experts (HP-SafeMoE), a materials-property learning framework that combines independently trained task experts through a shared, safety-controlled residual model.

Public repository: <https://github.com/gllary/HP_SafeMoE>

HP-SafeMoE treats each frozen Stage 1 expert as a heterogeneous provider. Stage 2 maps provider-specific representations into a common token space. Its Sample-conditioned and Relation-augmented branches construct candidate residuals relative to the frozen target-task anchor. Cross-fitted training predictions determine residual shrinkage, task certification and sample-level benefit gating; unsupported corrections leave the anchor unchanged.

The repository contains the Matbench Stage 2 implementation, Stage 1 provider interfaces and configurations, the JARVIS validation workflow, four external-evaluation datasets and outputs, analysis and metric-recomputation scripts, environment specifications, automated tests, and a SHA-256 manifest. Full Stage 1 retraining additionally requires the upstream data and model assets listed in the documentation.

## Quick verification

Use Python 3.10 or later in a clean environment:

```bash
python -m pip install -r requirements.txt
python verify_release.py
python scripts/build_manifest.py --check
python -m pytest -q
```

Provider-specific training environments are documented separately because the four Stage-1 provider families have incompatible dependency stacks.

## Repository structure

### Source code

- `src/hpsafe_sota/`: Stage-1 provider adapters, shared artifact contracts, orchestration utilities, and the Matbench HP-SafeMoE implementation.
- `integrations/jarvis/`: five-task JARVIS implementation, protocol configuration, data interfaces, and tests.
- `scripts/`: data preparation, provider export, experiment orchestration, analysis reconstruction, and release-verification commands.
- `tests/`: unit and protocol tests for the shared implementation.

### Configuration and environments

- `configs/reference/`: benchmark data sources, split protocol, task definitions, and the Stage-1 provider pool.
- `configs/experiments/`: Matbench Stage-1 and Stage-2 experiment configurations.
- `configs/server_resources*.yaml`: execution profiles for the supported compute environments.
- `envs/`: provider-specific environment definitions.
- `requirements.txt` and `pyproject.toml`: dependencies and package metadata for the shared repository tools.

### Data, outputs, and analyses

- `data/`: the redistributed HEA-95, B2-18, Core-23, and DS2-248 datasets, together with public task metadata.
- `results/full_benchmark_comparison/`: the complete full-benchmark Matbench comparison reported in the article.
- `results/matbench/`: Matbench task summaries, primary matched comparisons, and complete row-level held-out predictions for Stage 1, No-cross-expert, and Full HP-SafeMoE.
- `results/HEA-95/`, `results/B2-18/`, `results/Core-23/`, and `results/DS2-248/`: external case-study inputs and outputs.
- `results/JARVIS/`: row-level predictions, task summaries, a field dictionary, and metric-recomputation utilities.
- `source_data/`: numerical source data for manuscript statistics, uncertainty analyses, and selected supplementary tables and figures.
- `analysis/representation/`: representation probes, PCA, descriptor CKA, and associated figures.
- `analysis/descriptors/`: descriptor-surrogate fidelity and attribution assets.
- `analysis/routing_and_benefit/`: routing, source-weight diagnostics, primary matched comparisons, task-safety, and benefit-gate analyses.

### Reproducibility and release integrity

- `docs/REPRODUCIBILITY.md`: end-to-end data preparation, Stage-1 export, Stage-2 execution, and verification workflow.
- `docs/WEIGHTS_AND_DEPENDENCIES.md`: upstream provider assets, environments, and checkpoint requirements.
- `verify_release.py`: structural, methodological, result, release-scope, terminology, and path-hygiene checks.
- `RELEASE_MANIFEST.csv`: relative paths, file sizes, and SHA-256 digests for the release.

## Documentation

- [`docs/DATASETS.md`](docs/DATASETS.md): redistributed datasets and official benchmark sources.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): complete reproduction and verification workflow.
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md): representation, descriptor, routing, ablation, and benefit-gate analyses.
- [`results/matbench/README.md`](results/matbench/README.md): schema and coverage of the complete Matbench held-out prediction release.
- [`source_data/README.md`](source_data/README.md): file-level map from manuscript statistics to released numerical source data.
- [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md): canonical method and public-field terminology.
- [`docs/WEIGHTS_AND_DEPENDENCIES.md`](docs/WEIGHTS_AND_DEPENDENCIES.md): provider environments and upstream assets.

## Citation and release scope

Citation metadata for this release are provided in [`CITATION.cff`](CITATION.cff).

Repository code is released under the MIT License in [`LICENSE`](LICENSE). The MIT License does not relicense third-party software, datasets, or pretrained assets; those remain subject to their source terms and citations described in the documentation.

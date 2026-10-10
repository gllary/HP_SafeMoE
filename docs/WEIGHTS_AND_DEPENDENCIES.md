# Model assets and third-party dependencies

The repository provides source code, configurations, evaluation outputs, analysis inputs, and result-verification paths. Full training from raw Matbench records uses the following upstream provider environments and assets:

- MatterVial/MODNet provider dependencies;
- JMP-L provider dependencies and base weights;
- TPOT-Mat and AnchorBoost runtime dependencies;
- the official Matbench v0.1 data files listed with SHA-256 digests in `configs/reference/data_sources.yaml`.

The JMP-L launcher uses the current Python interpreter by default. Set `JMP_ENV_PYTHON` to an explicit interpreter when a dedicated JMP-L environment is used.

Official Matbench data are prepared with `scripts/prepare_matbench_manifests.py`. Provider exports are validated with `scripts/validate_stage1_provider_exports.py` against `configs/reference/stage1_provider_pool.yaml`. The exact command sequence is documented in `docs/REPRODUCIBILITY.md`.

Third-party assets retain their upstream licenses and citations.

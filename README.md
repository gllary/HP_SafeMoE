# HP-SafeMoE

Code, configurations, final experiment records and model weights for HP-SafeMoE.

## Contents

| Directory | Contents |
| --- | --- |
| `src/hpsafe_sota/` | Provider interfaces, data preparation and Matbench Stage 2 |
| `integrations/jarvis/` | JARVIS code, configurations and tests |
| `configs/`, `envs/` | Model settings and dependency environments |
| `scripts/`, `tests/` | Preparation, training, evaluation and verification commands |
| `data/` | External evaluation datasets and benchmark metadata |
| `results/matbench/` | Held-out predictions, targets, scores and calibration records |
| `results/JARVIS/` | Test predictions, scores, normalization and calibration records |
| `results/full_benchmark_comparison/` | Reference and baseline score tables |
| `results/HEA-95/`, `results/B2-18/`, `results/Core-23/`, `results/DS2-248/` | External evaluation records |
| `analysis/`, `source_data/` | Numerical data and reconstruction code for article figures and tables |
| `checkpoints/` | 99 Stage 1 model packages and 104 Stage 2 members: 52 Full and 52 No-cross-expert |

## Use

Install the shared dependencies, download the weights from the
[final checkpoints release](https://github.com/gllary/HP_SafeMoE/releases/tag/final-checkpoints),
and verify the released files:

```bash
python -m pip install -r requirements.txt
python -B scripts/download_checkpoints.py
PYTHONDONTWRITEBYTECODE=1 python verify_release.py
python scripts/build_manifest.py --check
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests integrations/jarvis/tests
```

- [Training and evaluation commands](docs/REPRODUCIBILITY.md)
- [Checkpoint files and loading](checkpoints/README.md)
- [Provider dependencies](docs/WEIGHTS_AND_DEPENDENCIES.md)
- [Dataset files](docs/DATASETS.md)
- [Figure and analysis data](docs/ANALYSIS.md)
- [Manuscript source-data map](source_data/README.md)

`RELEASE_MANIFEST.csv` lists source and downloaded-weight file sizes and SHA-256 digests. Generated outputs use `run_outputs/`. Citation metadata are in [`CITATION.cff`](CITATION.cff). Code is licensed under [MIT](LICENSE); third-party assets retain their upstream terms and citations.

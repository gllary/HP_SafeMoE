# Final checkpoints

Final Stage 1 experts and Full/No-cross-expert Stage 2 ensembles are distributed
through the [final checkpoints release](https://github.com/gllary/HP_SafeMoE/releases/tag/final-checkpoints).
Download all 208 weight and preprocessor files (17.56 GiB) into this directory:

```bash
python -B scripts/download_checkpoints.py
```

The downloader restores the paths below, verifies SHA-256 digests and skips
verified existing files. Use `--benchmark matbench` or `--benchmark jarvis` to
select a benchmark, and `--stage 1` or `--stage 2` to select a stage. `--list`
prints the selected files. Release asset names join the path components below
`checkpoints/` with `__`.

| Benchmark | Stage 1 model packages | Full Stage 2 members | No-cross-expert Stage 2 members |
| --- | ---: | ---: | ---: |
| Matbench | 65 (13 experts × 5 outer folds) | 50 (5 outer × 5 inner folds × 2 seeds) | 50 |
| JARVIS | 34 final experts | 2 seeds | 2 seeds |
| Total | 99 | 52 | 52 |

The directory contains 203 model files and five fitted Glass preprocessors. Native Stage 1 ensembles can contain multiple members in one model file. [`required_checkpoints.json`](required_checkpoints.json) records every file's SHA-256, source digest, model configuration and necessary preprocessing parameters.

## Layout and loading

- `matbench/stage1/`: one directory per task, each containing `outer_0` through `outer_4`. These hold native expert models and, for Glass, fitted preprocessors.
- `matbench/stage2/full/` and `matbench/stage2/no_cross_expert/`: each has `outer_0` through `outer_4`, then `inner_0` through `inner_4`, and files `seed_20261071.pt` and `seed_20261072.pt`. Each file contains `state_dict`, `model_config` and fitted normalization, using the public Relation field names.
- `jarvis/stage1/`: one directory per task, with native final expert weights in its `full/` subdirectory. Official ALIGNN files contain a `model` state dictionary. Native model configuration is recorded in the inventory.
- `jarvis/stage2/full/` and `jarvis/stage2/no_cross_expert/`: files `seed20261071.pt` and `seed20261072.pt` contain `model`, `config` and the fitted training-target normalization.

No-cross-expert sets `use_cross_expert_input=False` and reuses the matching frozen Stage 1 experts. Apply each variant's own fitted calibration: Matbench Full uses `results/matbench/experiment/full/calibration/`, and No-cross-expert uses `results/matbench/experiment/calibration/`. JARVIS Full uses `results/JARVIS/experiment/full/calibration.json`, and No-cross-expert uses `results/JARVIS/experiment/no_cross_expert/calibration.json`. The Matbench safety profiles are shared across variants.

JMP files contain `state_dict` and architecture fields under `model_config`. The upstream model class and graph preprocessing are required to instantiate these models. Torch files can be read with `torch.load(path, map_location="cpu", weights_only=True)`. Native MODNet and tree models require their matching provider environments. Benefit gates, residual scales, thresholds, task certificates and token normalization are retained under the result paths listed in `shared_artifacts` in the inventory.

## JMP-L assets

The 25 bundled JMP-L files are the task- and outer-fold-specific fine-tuned weights used by this study: five folds each for `jdft2d`, `perovskites`, `phonons`, `mp_gap` and `mp_e_form`. Together they occupy approximately 15 GiB. Each file contains the fitted backbone and task head.

For retraining, obtain the base weights from the [official JMP repository](https://github.com/facebookresearch/JMP#pre-trained-checkpoints), which provides the [JMP-L pretrained checkpoint](https://jmp-iclr-datasets.s3.amazonaws.com/jmp-l.pt). This project pins upstream commit `937b14874381d9b80809582e323ef82f2d4e1291`; use `envs/jmp_l.yml` and `scripts/setup_jmp_l_runtime.py`. The [reproduction workflow](../docs/REPRODUCIBILITY.md) describes the study-specific provider recipes.

The compact JMP files use an inference parameter format. Read `state_dict` and `model_config` explicitly; instantiate the pinned upstream model and apply its graph/target preprocessing. The saved `backbone.scale_file` is a basename to resolve within the upstream source checkout.

No-safety uses the Full weights. Repeating JARVIS training requires 170 Stage 1 OOF fits in addition to the 34 final experts included here.

## Verification

```bash
python -B scripts/verify_checkpoints.py --load-stage2
```

This checks model-file hashes and loads Stage 2 weights. Provider environments and feature extractors are listed in [dependencies](../docs/WEIGHTS_AND_DEPENDENCIES.md).

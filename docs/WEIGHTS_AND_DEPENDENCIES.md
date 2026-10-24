# Model assets and dependencies

## Included assets

Run `python -B scripts/download_checkpoints.py` to populate `checkpoints/` with 99 Stage 1 model packages and 104 Stage 2 members. [`required_checkpoints.json`](../checkpoints/required_checkpoints.json) lists model files, architecture settings, source and public-file digests, preprocessing parameters and calibration paths. See [checkpoint loading](../checkpoints/README.md) for the directory layout.

## Dependencies

| Component | Environment or reference |
| --- | --- |
| Shared verification tools | `requirements.txt` |
| MatterVial/MODNet | `envs/mattervial.yml` |
| JMP-L | `envs/jmp_l.yml`, `scripts/setup_jmp_l_runtime.py` |
| TPOT-Mat | `envs/tpot_mat.yml` |
| AnchorBoost | `envs/anchorboost.yml` |
| JARVIS specialists | [JARVIS references](../integrations/jarvis/REFERENCES.md) |
| Official Matbench inputs | `configs/reference/data_sources.yaml` |

Native models require their corresponding provider implementations, feature extractors and graph/descriptor preprocessing. The JMP launcher uses the current Python interpreter; set `JMP_ENV_PYTHON` to use a dedicated interpreter. JARVIS training requires 170 Stage 1 OOF fits in addition to the 34 final experts included here.

Third-party software, datasets and pretrained assets retain their upstream licenses and citations.

## Verification

```bash
python -B scripts/verify_checkpoints.py --load-stage2
python -B verify_release.py
```

The first command verifies model-file hashes and loads Stage 2 weights. The second checks released predictions, metrics, calibration inventories and file contracts. Training commands are listed in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

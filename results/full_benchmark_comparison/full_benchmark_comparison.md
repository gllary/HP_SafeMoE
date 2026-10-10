# Full Matbench comparison

Values are five-fold mean ± SD unless the entry is a literature-reported point estimate. A dash marks an unavailable task-level result.

| Task and metric | Frozen Matbench v0.1 reference | MatterVial | JMP-L | coGN | coNGN | HP-SafeMoE |
|---|---:|---:|---:|---:|---:|---:|
| Experimental metallicity, ROC-AUC ↑ | 0.9598 ± 0.0041 | 0.9763 | — | — | — | 0.9805 ± 0.0038 |
| Glass-forming ability, ROC-AUC ↑ | 0.9603 ± 0.0075 | 0.9370 | — | — | — | 0.9559 ± 0.0078 |
| MP metallicity, ROC-AUC ↑ | 0.9520 ± 0.0074 | 0.9780 | — | 0.9519 ± 0.0021 | 0.9554 ± 0.0020 | 0.9871 ± 0.0011 |
| Refractive index, MAE ↓ | 0.2711 ± 0.0714 | 0.2337 | 0.2530 ± 0.0940 | 0.3017 ± 0.1009 | 0.3238 ± 0.1032 | 0.2434 ± 0.0938 |
| Experimental band gap, MAE (eV) ↓ | 0.2865 ± 0.0083 | 0.2900 | — | — | — | 0.2461 ± 0.0033 |
| 2D exfoliation energy, MAE (meV atom^-1) ↓ | 33.1918 ± 7.3428 | 28.8650 | 29.6696 ± 10.2381 | 37.4127 ± 13.0932 | 39.7885 ± 13.5337 | 28.9400 ± 11.2200 |
| Log shear modulus, MAE ↓ | 0.0670 ± 0.0006 | 0.0325 | 0.0587 ± 0.0010 | 0.0691 ± 0.0015 | 0.0676 ± 0.0015 | 0.0346 ± 0.0012 |
| Log bulk modulus, MAE ↓ | 0.0491 ± 0.0026 | 0.0270 | 0.0453 ± 0.0030 | 0.0528 ± 0.0029 | 0.0496 ± 0.0028 | 0.0266 ± 0.0018 |
| MP formation energy, MAE (eV atom^-1) ↓ | 0.0170 ± 0.0003 | 0.0138 | 0.0145 ± 0.0005 | 0.0169 ± 0.0003 | 0.0178 ± 0.0006 | 0.0112 ± 0.0003 |
| MP band gap, MAE (eV) ↓ | 0.1559 ± 0.0017 | 0.1368 | 0.0992 ± 0.0022 | 0.1558 ± 0.0024 | 0.1720 ± 0.0019 | 0.0976 ± 0.0022 |
| Perovskite formation energy, MAE (eV unit cell^-1) ↓ | 0.0269 ± 0.0008 | 0.0386 | 0.0257 ± 0.0011 | 0.0271 ± 0.0006 | 0.0287 ± 0.0016 | 0.0265 ± 0.0013 |
| Optical phonon peak, MAE (cm^-1) ↓ | 28.7606 ± 2.5767 | 30.0800 | 21.1923 ± 2.1026 | 30.6247 ± 1.7338 | 28.4937 ± 2.2786 | 20.1000 ± 1.5600 |
| Steel yield strength, MAE (MPa) ↓ | 79.9468 ± 13.5883 | 85.1200 | — | — | — | 78.5100 ± 14.3200 |

The frozen reference column is task-wise and names its provider in `full_benchmark_comparison.csv`. MatterVial values are literature-reported. JMP-L, coGN, coNGN, and HP-SafeMoE values use the official Matbench v0.1 outer folds.

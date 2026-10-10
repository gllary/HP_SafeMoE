# External-data provenance and preprocessing

This document records the preprocessing facts that can be verified from the retained source packages and released files for the corresponding target-task evaluations.

## DS2-248

The source archive contained 507 structure-level records representing 480 normalized compositions. Composition keys were invariant to element order and formula scaling. Removing 236 records from 232 compositions overlapping `matbench_expt_gap`, followed by median aggregation of multiple measurements for a retained composition, produced 248 composition-disjoint records. The released `sample_id`, normalized composition, structure and experimental band gap are joined across `data/DS2-248/materials.csv` and `data/DS2-248/labels.csv`. In the paired bootstrap, each of these 248 material records is one resampling unit and the same sampled indices are used for Stage 1 and HP-SafeMoE.

## Core-23

The retained RefractiveIndex.INFO screening inventory covered 4,182 YAML files, including 3,480 linear-optical records. Prediction-independent screening produced 101 composition candidates; 83 had both a scalar refractive index at 0.5893 micrometres and a same-composition structure, and the primary set retained 23 transparent crystalline records with suitable phase and sample-form correspondence.

Formula records were evaluated at 0.5893 micrometres. Tabulated records were linearly interpolated only within their reported wavelength range; no extrapolation was used. The 23 scalar targets comprise nine original scalar or direction-unspecified values, seven complete uniaxial groups combined as `sqrt((2*n_o^2+n_e^2)/3)`, and seven complete biaxial groups combined as `sqrt((n_alpha^2+n_beta^2+n_gamma^2)/3)`. All reported extinction coefficients are at most `1e-3`; 21 of 23 records lack an extinction coefficient, and missing values were retained as missing rather than treated as zero.

Structures were selected from same-composition entries in `matbench_mp_gap`; its band-gap labels were not used as Core-23 targets. Ambient phases were preferred where identifiable, and phase-uncertain or sample-form proxy records were excluded from the primary 23. The matched Matbench target is based on Materials Project DFPT electronic dielectric response, whereas Core-23 contains finite-wavelength experimental refractive indices. The evaluation is therefore cross-source and cross-definition.

`data/Core-23/screening_decisions.csv` records all 101 composition candidates, the 83 candidates with both a matched structure and a scalar value at 589.3 nm, the final 23-record core cohort, and the reason codes for every non-core record. Run `python scripts/validate_core23_screening.py` to verify the counts and final sample identifiers. In the paired bootstrap, each of the 23 retained material records is one resampling unit and the same sampled indices are used for Stage 1 and HP-SafeMoE.

The screening reason codes identify absorption at 589.3 nm, metallic or conducting behaviour, absence of an exact-composition structure in the `matbench_mp_gap` structure bank, absence of a scalar refractive index at 589.3 nm, structure or phase proxies, sample-form mismatch with an ideal bulk crystal, ordered proxies for site disorder, limited optical-tensor/phase alignment, and thin-film-to-bulk structure transfer. Codes are non-exclusive because a candidate may fail more than one core-cohort criterion.

## HEA-95

The released cohort contains 95 serialized structures and 90 unique compositions. Five pairs share the same displayed composition, target and literature source in Supplementary Table S10. Their serialized structures have different SHA-256 identities and do not match under the released validation settings (`ltol=0.2`, `stol=0.3`, `angle_tol=5`, no lattice scaling, supercell attempts enabled). They are therefore retained as different released structures rather than collapsed to HEA-90. The exact pair mapping and identity checks are provided in `results/HEA-95/shared_label_structure_validation.csv`.

## B2-18

All 18 labels were obtained in the present study under one preparation and dynamic-mechanical-analysis workflow. The release evaluates every alloy once as the sole calibration record and the remaining 17 as evaluation records. For both Stage 1 and HP-SafeMoE, each outer-fold model predicts `log10(K_VRH/GPa)` and `log10(G_VRH/GPa)`. These values are exponentiated to `K_VRH` and `G_VRH` in GPa, combined as `E = 9*K*G/(3*K+G)` to give an isotropic-equivalent estimate, and transformed to `log10(E/GPa)`. The five fold-specific log estimates are averaged before the scalar calibration residual is fitted and applied. HEA-95 uses the same prediction and averaging sequence.

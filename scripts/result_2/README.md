# Figure Result 2

This folder contains scripts used for the second paper result/figure block, focused on individual parcellation evaluation.

## Scripts

- `evaluation_HBP_indiv.py`: evaluates individual parcellations produced by the HBP pipeline.
- `evaluation_MSHBM_indiv.py`: evaluates individual parcellations produced by the MSHBM baseline.

## Expected inputs

- Pretrained group and individual parcellation outputs.
- External datasets referenced through `global_config.py` and hard-coded lab paths inside the scripts.
- Dependencies from `HierarchBayesParcel`, `Functional_Fusion`, and `FusionModel`.

## Outputs

- Evaluation tables and derived figure-ready summaries for comparing HBP and MSHBM individual maps.

## Release note

These scripts are paper-reproduction utilities, not stable public CLI entry points. They currently assume access to internal filesystem locations and may need path cleanup before fully public execution.

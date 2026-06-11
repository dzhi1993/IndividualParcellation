# Figure Result 6

This folder contains scripts used for the sixth paper result/figure block, focused on Dice-overlap analyses.

## Scripts

- `dice_overlap_hcp.py`: computes Dice-overlap results for HCP individual parcellations.
- `dice_overlap_randy.py`: computes Dice-overlap results for RANDY individual parcellations.

## Expected inputs

- Individual parcellation outputs for the target datasets.
- Subject-list support files under `../../replication/subject_list`.
- Atlas and model configuration available through this repository and its external dependencies.

## Outputs

- Dice-overlap metrics and related summaries used in the manuscript figure.

## Release note

These scripts are currently best understood as paper-reproduction code. They should be parameterized and path-cleaned before being presented as official public entry points.

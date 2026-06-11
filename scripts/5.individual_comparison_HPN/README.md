# Figure Result 5

This folder contains scripts used for the fifth paper result/figure block, focused on cosine-error analyses and thresholding comparisons.

## Scripts

- `indiv_coserr_hcp.py`: computes cosine-error analyses for HCP individual parcellations.
- `indiv_coserr_randy.py`: computes cosine-error analyses for RANDY individual parcellations.
- `naive_thresholding.py`: evaluates or visualizes thresholding-based comparison baselines.

## Expected inputs

- HCP and RANDY evaluation data.
- HCP subject-list support files under `../../replication/subject_list`.
- Trained model outputs and atlas resources.
- External dependencies from the Diedrichsen lab software stack.

## Outputs

- Cosine-error summaries, comparison tables, and figure inputs for this result block.

## Release note

The scripts in this folder are manuscript analysis scripts. They are useful to preserve for reproducibility, but they are not yet generalized for public execution across environments.

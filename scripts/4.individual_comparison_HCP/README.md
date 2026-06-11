# Figure Result 4

This folder contains scripts used for the fourth paper result/figure block, focused on HPN dataset individual-map homogeneity analyses.

## Scripts

- HPN homogeneity analysis script: computes or summarizes homogeneity metrics for individual parcellations on the HPN dataset.
- HPN subject-level analysis script: supports subject-level HPN analyses used in the same result block.

## Expected inputs

- HPN dataset files available at the locations expected by the scripts.
- HCP subject-list support files under `../../replication/subject_list` where required.
- Model outputs and auxiliary resources from the project dependencies.

## Outputs

- Subject-level and aggregate evaluation results used to assemble the corresponding manuscript figure.

## Release note

These scripts currently rely on machine-specific storage paths and should be considered paper-specific analysis code until they are parameterized.

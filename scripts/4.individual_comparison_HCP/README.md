# Result 4: HCP Individual Comparison

This folder contains the HCP individual-parcellation comparison analyses used in the paper. This section compares individual parcellations derived from resting-state and fusion group priors using task inhomogeneity, DCBC, resting-state homogeneity, task-domain-wise inhomogeneity, within-/between-individual parcellation similarity, and run-wise prediction accuracy.

## Files

- `indiv_eval_hcp.py`: in-house script used to generate HCP individual parcellations and evaluate them under task conditions, including task inhomogeneity and DCBC.
- `homo_indiv_hcp.py`: in-house script used for resting-state homogeneity evaluation.
- `individual_comparison_HCP.ipynb`: release-facing notebook that loads the precomputed result files and recreates the paper result plots for this section.

## Result Files

`individual_comparison_HCP.ipynb` reads the following repository-local files from `../../results/4.hcp`:

- `eval_indiv-rest_vs_fusion-HCP200_1-2run_K-17_test_on_HCPtask-contrast.tsv`: Figures 4e-f, task inhomogeneity and DCBC.
- `eval_indiv-rest_vs_fusion-HCP200_1-2run_K-17_test_on_HCPrest-Tseries.tsv`: Figure 4g, resting-state homogeneity.
- `eval_indiv-rest_vs_fusion-HCP200_1-2run_K-17_test_on_HCPtask-contrast_domain-wise.tsv`: Figure 4h, task-domain-wise inhomogeneity.
- `dice_within_vs_between_HCP_rest1run_K-17.tsv`: Figure 4i, within-/between-individual parcellation similarity.
- `dice_pair-wise_REST+HCPrest-1run1_vs_1run2.npy`: run-wise pairwise Dice matrix for rest-prior individual parcellations.
- `dice_pair-wise_FUSION+HCPrest-1run1_vs_1run2.npy`: run-wise pairwise Dice matrix for fusion-prior individual parcellations.

The notebook also reports subject-level statistical tests and run-wise prediction accuracy for rest-prior versus fusion-prior comparisons.

## Replication Note

The Python scripts in this folder were written for the lab analysis environment. They depend on lab-specific storage paths, HCP derivatives, precomputed model outputs, subject lists, distance matrices, and atlas resources configured through the project environment. They are retained to document how the paper analyses were produced, but they are not intended as minimal portable examples.

For a general train/evaluate workflow in a new environment, refer to the root-level example scripts:

- `train_individual.py`: minimal individual parcellation training example.
- `evaluation.py`: minimal individual parcellation evaluation example.

For reproducing the paper figures from released result files, use `individual_comparison_HCP.ipynb`.

## Outputs

- Interactive plots for the HCP individual-comparison result panels.
- Paired t-test result tables printed in the notebook.
- Run-wise prediction accuracy table printed in the notebook.
- Optional figure files under `../../results/4.hcp/figures` if `SAVE_FIGURES = True` in the notebook.

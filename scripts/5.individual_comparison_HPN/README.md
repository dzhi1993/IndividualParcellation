# Result 5: HPN Individual Comparison

This folder contains the HPN individual-parcellation comparison analyses used in the paper. This section compares individual parcellations derived from resting-state and fusion group priors using task inhomogeneity, DCBC, resting-state homogeneity, within-/between-individual parcellation similarity, and selected task-contrast z-value examples.

## Files

- `indiv_eval_hpn.py`: in-house script used to generate HPN individual parcellations and evaluate them under task conditions, including task inhomogeneity and DCBC.
- `homo_indiv_hpn.py`: in-house script used for resting-state homogeneity evaluation.
- `indiv_hpn_subjects.py`: support script for the HPN subject set used in the analysis.
- `individual_comparison_HPN.ipynb`: release-facing notebook that loads the precomputed result files and recreates the paper result plots for this section.

## Figure Panels

`individual_comparison_HPN.ipynb` reads the repository-local HPN result files from `../../results/5.hpn`:

- Figure 5d: task inhomogeneity from `eval_rest_vs_fusion_RANDYrest-1-allrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv`.
- Figure 5e: task DCBC from `eval_rest_vs_fusion_RANDYrest-1-allrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv`.
- Figure 5f: resting-state homogeneity from `eval_rest_vs_fusion_RANDYrest-1-allrun_K-15_indiv-mRBM_test_on_RANDYrest-Tseries_sm0_11subjects.tsv`.
- Figure 6b: within-/between-individual Dice similarity from `dice_within_vs_between_RANDY_rest1run_K-15.tsv`.
- Figure 6c: edited language zoom-in panel from `zoom-in_effect_LANG.pdf`.
- Figure 6d: language-contrast z values from `Z_value_LANG.tsv`, restricted to DN-B, LANG, and DN-A.
- Figure 6e: edited somatomotor zoom-in panel from `zoom-in_effect_SMB.pdf`.
- Figure 6f: motor-contrast z values from `Z_value_SMB.tsv`, restricted to SMOT-B and SMOT-A.

The notebook also reports subject-level paired statistical tests for rest-prior versus fusion-prior comparisons.

## Replication Note

The Python scripts in this folder were written for the lab analysis environment. They depend on lab-specific storage paths, HPN derivatives, precomputed model outputs, subject lists, distance matrices, and atlas resources configured through the project environment. They are retained to document how the paper analyses were produced, but they are not intended as minimal portable examples.

For a general train/evaluate workflow in a new environment, refer to the root-level example scripts:

- `train_individual.py`: minimal individual parcellation training example.
- `evaluation.py`: minimal individual parcellation evaluation example.

For reproducing the paper figures from released result files, use `individual_comparison_HPN.ipynb`.

## Outputs

- Interactive plots for Figures 5d-f and 6b-f.
- Paired t-test result tables printed in the notebook.
- Within-/between-individual Dice similarity statistical table.
- Language-contrast and somatomotor z-value statistical tables.
- Optional figure files under `../../results/5.hpn/figures` if `SAVE_FIGURES = True` in the notebook.

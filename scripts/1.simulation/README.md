# Figure Result 1

This folder contains the simulation workflow used to illustrate the manuscript's Supplementary Figures 1 and 2.

There are two user-facing files:

- `simulation_panels.ipynb`: the recommended interactive walkthrough, organized panel by panel.
- `simulation.py`: the source script that contains the simulation, plotting, and helper functions used by the notebook.

## Recommended workflow

For most users, start with:

- `simulation_panels.ipynb`

The notebook follows the current logic in `simulation.py`, but splits it into smaller sections so you can:

- run Supplementary Figure 1 and Supplementary Figure 2 separately,
- inspect intermediate variables and plots,
- rerun only one panel block instead of the full script,
- reuse cached outputs when they already exist.

Use `simulation.py` when you want the all-in-one scripted execution or when you need to modify the implementation itself.

## What this workflow covers

Based on the in-code comments in `simulation.py`, the workflow is organized into two main sections.

### Supplementary Figure 1

This section generates example priors and example individual maps under different settings.

- Panel A: group-level priors for the 5 parcels across multiple `theta_mu` settings.
- Panel C: example individual parcellations across different `theta_mu` and `theta_w` settings.
- Panels B and D: a burn-in/example subject simulation with `theta_mu=240` and `theta_w=1.2`, followed by example individual maps.

In the script version, these plots are produced in the `if __name__ == '__main__':` block. In the notebook version, they are split into separate cells.

### Supplementary Figure 2

This section runs the cmpRBM comparison simulation and then visualizes model behavior and evaluation summaries.

- Panel A: individual posterior/probability visualization via `plot_individual_Uhat(...)`.
- Panels B/C/D: evaluation summaries from a cached or freshly generated TSV file.

The current evaluation plotting workflow uses:

- `results/1.simulation/eval_cpmRBM_fit.tsv`

for the Supplementary Figure 2 summary plots.

## How to run it

### Notebook

Open and run:

- `scripts/1.simulation/simulation_panels.ipynb`

This is the easiest way to reproduce the panels one section at a time.

### Script

Run from the repository root:

```bash
python scripts/1.simulation/simulation.py
```

If your environment is configured correctly, the script will open matplotlib figures for the supplementary panels described above.

## Expected environment

This workflow depends on the same scientific Python and lab packages used elsewhere in the repository, especially:

- `torch`
- `torchvision`
- `matplotlib`
- `pandas`
- `seaborn`
- `HierarchBayesParcel`
- `FusionModel`

This folder should be treated as research/paper-reproduction code rather than as a polished public CLI package. In practice, users should make sure:

- the required sibling packages are importable,
- PyTorch/CUDA is configured the same way as in the working manuscript-reproduction environment,
- interactive plotting is available.

## Outputs and caching

Both the notebook and the script mainly show figures interactively. Some save/export lines are present in the code but are commented out.

The most important cached result path is:

- `results/1.simulation/eval_cpmRBM_fit.tsv`

This file is used for the Supplementary Figure 2 evaluation plots. If it already exists, the notebook or the script can reuse it. If you want to regenerate it from scratch, inspect the commented `DD.to_csv(...)` line and the simulation settings in the Supplementary Figure 2 section before running a longer simulation.

## Practical usage notes

- Start with the notebook unless you specifically want the all-in-one script execution.
- Read the notebook cells or the `__main__` comments first; they are the clearest mapping from code to figure panels.
- If you only need one figure block, run only the corresponding notebook section or comment out the other block in `simulation.py`.
- The simulation section can be expensive. Reusing the cached TSV is the easier path when you only need the final Supplementary Figure 2 summary plots.
- This folder is best treated as paper-reproduction code, not as a stable reusable API.

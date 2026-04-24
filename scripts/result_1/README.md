# Figure Result 1

This folder contains the simulation script used for the manuscript's Supplementary Figures 1 and 2.

The main entry point is:

- `simulation.py`

## What this script does

From a user perspective, `simulation.py` is a figure-generation and simulation workflow for the cmpRBM supplementary analyses. It is not a general-purpose command-line tool. The script mixes:

- data simulation for cmpRBM-style spatial priors,
- model fitting and evaluation,
- figure-panel plotting for the supplement,
- optional reading/writing of cached simulation results.

## Which supplementary figures it covers

Based on the in-code comments in `simulation.py`, the script is organized into two sections.

### Supplementary Figure 1

This section generates example priors and example individual maps under different settings.

- Panel A: group-level priors for the 5 parcels across multiple `theta_mu` settings.
- Panel C: example individual parcellations across different `theta_mu` and `theta_w` settings.
- Panels B and D: a burn-in/example subject simulation with `theta_mu=240` and `theta_w=1.2`, followed by example individual maps.

These plots are generated directly in the `if __name__ == '__main__':` block.

### Supplementary Figure 2

This section runs the cmpRBM comparison simulation and then visualizes model behavior and evaluation summaries.

- Panel A: individual posterior/probability visualization via `plot_individual_Uhat(...)`.
- Panels B/C/D: evaluation summaries from a cached or freshly generated TSV file.

The code currently points to:

- `results/result_1/eval_cpmRBM_fit.tsv`

for the summary plotting step.

## How to run it

Run from the repository root:

```bash
python scripts/result_1/simulation.py
```

If your environment is configured correctly, this will open matplotlib figures for the supplementary panels described above.

## Expected environment

This script depends on the same scientific Python and lab packages used elsewhere in the repository, especially:

- `torch`
- `torchvision`
- `matplotlib`
- `pandas`
- `seaborn`
- `HierarchBayesParcel`
- `FusionModel`

The script is currently meant for the lab/research environment rather than for a clean public installation. In practice, users should make sure:

- the required sibling packages are importable,
- PyTorch/CUDA is configured the same way as in the working environment used for the manuscript reproduction,
- interactive plotting is available.

## Outputs and caching

The script mainly shows figures interactively. Some save/export lines are present in the code but are commented out.

The most important cached result path is:

- `results/result_1/eval_cpmRBM_fit.tsv`

This file is used for Supplementary Figure 2 evaluation plots. If it already exists, the script can reuse it. If you want to regenerate it from scratch, inspect the commented `DD.to_csv(...)` line in the main block and the simulation settings in that section before running a longer simulation.

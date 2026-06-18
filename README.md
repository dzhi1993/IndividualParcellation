# IndividualParcellation

This repository contains code for generating and evaluating individual cortical parcellations using a multinomial restricted
Boltzmann Machine (mRBM) built on top of [HierarchBayesParcel](https://github.com/DiedrichsenLab/HierarchBayesParcel). It combines group atlas information with subject-level 
localizer data to produce reliable individual cortical parclelations. The repository also host all code scripts for replication
the analyses in the manuscript.

Details are described in the paper:

- Zhi, D., Diedrichsen, J., Ge, T. (2026). "Precision Functional Parcellation of the Human Cortex via Rest-Task fMRI Fusion."

## What this repository contains

- End-to-end scripts for fitting and applying individual parcellation models.
- Evaluation utilities for HCP, HPN, and related comparisons.
- Reproduction code for manuscript figure/result blocks.

## Repository layout

- [scripts](./scripts): paper-flow analysis scripts and manuscript reproduction workflows. Each numbered subfolder corresponds to a major analysis block and contains local documentation when more detail is needed.
- [example_data](./example_data): minimal example input data used by the root training and evaluation examples.
- [replication](./replication): static supporting files required to reproduce the analyses, including group parcellations, subject lists, and MSHBM/Kong2019 metadata.
- [results](./results): generated outputs, precomputed summary tables, and figure-related artifacts used by the reproduction workflows.
- [docs](./docs): project documentation assets such as pipeline diagrams and method notes.
- [deprecated](./deprecated): older scripts retained for provenance but no longer maintained as active release entry points.

## Root example files

The root examples are intended to run from the project folder without editing lab-local paths:

```bash
python train_individual.py
python evaluation.py
```

[train_individual.py](./train_individual.py) requires:

- `example_data/example_rest_space-fs32k_Ico642Run_desc-sm4fwhm_binarized.dscalar.nii`
- `example_data/example_rest_Ico642Run.tsv`
- `replication/group_parcellations/17Networks/HBP17_FUSION_networks_prob.dscalar.nii`
- `replication/MSHBM_17networks/17network_labels.mat`
- `replication/MSHBM_17networks/group.mat`
- `replication/MSHBM_17networks/Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii`

[evaluation.py](./evaluation.py) evaluates the individual map produced by `train_individual.py` and requires:

- `results/train_individual/example_subject_HBP17_FUSION_indiv_prob.npy`
- `example_data/example_task_contrasts_s4_MSMAll.dscalar.nii`
- `example_data/example_task_contrasts.tsv`
- `example_data/distGOD_fs32k.pt`

`distGOD_fs32k.pt` is the precomputed fs32k surface-distance tensor used for DCBC evaluation, matching the manuscript scripts. It is a large local support file and is intentionally ignored by Git.

The manuscript-replication scripts also expect support files under [replication](./replication):

- `replication/group_parcellations`: HBP group parcellation priors and label maps.
- `replication/subject_list`: subject-list TSV files used by HCP benchmark/reproducibility scripts.
- `replication/MSHBM_17networks`: MSHBM/Kong2019 group prior, network names, and color definitions.

## Dependencies

This project depends on standard scientific Python packages listed in [requirements.txt](./requirements.txt), including `numpy`, `pandas`, `matplotlib`, `nibabel`, `nilearn`, and `torch`.

It also depends on the following packages:

- [HierarchBayesParcel](https://github.com/DiedrichsenLab/HierarchBayesParcel)
- [Functional_Fusion](https://github.com/DiedrichsenLab/Functional_Fusion)
- [FusionModel](https://github.com/DiedrichsenLab/FusionModel)
- [SUITPy](https://suitpy.readthedocs.io/en/latest/index.html)
- [nitools](https://nitools.readthedocs.io/en/latest/)

Install the general Python dependencies with:

```bash
pip install -r requirements.txt
```

For the lab packages above, clone the repositories and add their parent directory to `PYTHONPATH`.

On macOS/Linux:

```bash
export PYTHONPATH=<path_to_repo_parent>:$PYTHONPATH
```

On Windows, add the same parent directory to the system `Path` or Python environment configuration.

## Notes on portability

- Many scripts assume local lab storage layouts or dataset mounts configured in [global_config.py](./global_config.py).
- Some manuscript reproduction scripts still contain machine-specific assumptions and should be treated as research code rather than stable public CLI tools.
- The numbered `scripts/*` paper-flow folders are best understood as result-reproduction modules tied to the paper figures.



## License

This project is released under the MIT license. See [LICENSE](./LICENSE).

## Contact

For questions or bug reports, contact Da Zhi at `dzhi@mgh.harvard.edu`.

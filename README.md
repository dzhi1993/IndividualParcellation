# IndividualParcellation

This repository contains code for generating and evaluating individual cerebellar parcellations with a hierarchical Bayesian framework built on top of [HierarchBayesParcel](https://github.com/DiedrichsenLab/HierarchBayesParcel). It combines group atlas information with subject-level localizer data to produce individual maps and the manuscript analyses derived from those maps.

Mathematical details are described in the paper:

- Zhi, D., Shahshahani, L., Nettekoven, C., Pinho, A. L., Bzdok, D., Diedrichsen, J. (2023). "A hierarchical Bayesian brain parcellation framework for fusion of functional imaging datasets." [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.24.542121v1)

## What this repository contains

- End-to-end scripts for fitting and applying individual parcellation models.
- Evaluation utilities for HCP, RANDY, MSC, leave-one-out, and related comparisons.
- Reproduction code for manuscript figure/result blocks.
- Notebooks used for exploration, figure assembly, and method checks.

## Repository layout

- [scripts](./scripts): core pipeline scripts for model fitting, individual parcellation, and evaluation.
- [scripts/result_2](./scripts/result_2): manuscript result block 2, focused on individual parcellation evaluation.
- [scripts/result_3](./scripts/result_3): manuscript result block 3, focused on task coverage maps.
- [scripts/result_4](./scripts/result_4): manuscript result block 4, focused on RANDY homogeneity analyses.
- [scripts/result_5](./scripts/result_5): manuscript result block 5, focused on cosine-error and thresholding comparisons.
- [scripts/result_6](./scripts/result_6): manuscript result block 6, focused on Dice-overlap analyses.
- [notebooks](./notebooks): interactive notebooks for prototyping, evaluation, and figure generation.
- [results](./results): output location for generated summaries, figures, or derived artifacts.
- [docs](./docs): project documentation assets such as the pipeline calling-structure diagram.
- [global_config.py](./global_config.py): machine-specific path and runtime configuration used by many scripts.

## Result section folders

Each result folder has its own local README with script-level detail. The top-level summary is:

- [scripts/result_2](./scripts/result_2): evaluates individual parcellations, including comparisons between the HBP pipeline and the MSHBM baseline.
- [scripts/result_3](./scripts/result_3): generates task coverage maps used in the manuscript.
- [scripts/result_4](./scripts/result_4): runs subject-level and aggregate homogeneity analyses on the RANDY dataset.
- [scripts/result_5](./scripts/result_5): computes cosine-error analyses and thresholding-based comparison baselines.
- [scripts/result_6](./scripts/result_6): computes Dice-overlap analyses for individual parcellations across datasets.

## Core workflow

The typical workflow in this repository is:

1. Configure dataset and model paths in [global_config.py](./global_config.py).
2. Load atlas and pretrained group model resources.
3. Load subject-level localizer or evaluation data.
4. Run the individual parcellation pipeline from scripts such as [scripts/individual_parcellation.py](./scripts/individual_parcellation.py).
5. Evaluate outputs with the relevant scripts in [scripts](./scripts) or the manuscript-specific `scripts/result_*` folders.

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
- The `scripts/result_*` folders are best understood as result-reproduction modules tied to the paper figures.



## License

This project is released under the MIT license. See [LICENSE](./LICENSE).

## Contact

For questions or bug reports, contact Da Zhi at `dzhi@mgh.harvard.edu`.

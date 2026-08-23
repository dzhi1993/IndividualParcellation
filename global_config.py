#!/usr/bin/env python3
"""Shared runtime configuration for local and HPC environments.

Paths are selected automatically from the two lab layouts.  Every setting can
also be overridden with an environment variable, which keeps source files
identical across synchronized checkouts.
"""
import os
from pathlib import Path

import torch as pt


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
EXAMPLE_DIR = REPO_ROOT / 'example_data'


def _select_path(environment_variable, candidates):
    """Use an explicit override, otherwise the first existing candidate."""
    override = os.environ.get(environment_variable)
    if override:
        return Path(override).expanduser().resolve()
    candidates = [Path(path).expanduser() for path in candidates]
    return next((path for path in candidates if path.exists()), candidates[0])


# Local: ~/eris_mount; HPC: /data/tge.
DATA_ROOT_PATH = _select_path(
    'INDIVPAR_DATA_ROOT',
    (Path('/data/tge'), Path.home() / 'eris_mount'),
)
MODEL_DIR_PATH = _select_path(
    'INDIVPAR_MODEL_DIR',
    (
        DATA_ROOT_PATH / 'dzhi' / 'Indiv_par' / 'Models',
        Path('/srv/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Models'),
        Path('/cifs/diedrichsen/data/Cerebellum/ProbabilisticParcellationModel/Models'),
        Path('/Volumes/diedrichsen_data$/data/Cerebellum/ProbabilisticParcellationModel/Models'),
        Path.home() / 'diedrichsen_data/data/Cerebellum/ProbabilisticParcellationModel/Models',
    ),
)
BASE_DIR_PATH = _select_path(
    'FUNCTIONAL_FUSION_DIR',
    (
        DATA_ROOT_PATH / 'Tian' / 'UKBB_full' / 'imaging',
        Path('/srv/diedrichsen/data/FunctionalFusion'),
        Path('/cifs/diedrichsen/data/FunctionalFusion'),
        Path('/Volumes/diedrichsen_data$/data/FunctionalFusion'),
        Path.home() / 'diedrichsen_data/data/FunctionalFusion',
    ),
)

ATLAS_DIR_PATH = BASE_DIR_PATH / 'Atlases'
HCP_DIR_PATH = DATA_ROOT_PATH / 'Tian' / 'HCP_img'
RANDY_DIR_PATH = DATA_ROOT_PATH / 'Tian' / 'RANDY15'
MSC_DIR_PATH = DATA_ROOT_PATH / 'Tian' / 'MSC'
GROUP_DIR_PATH = HCP_DIR_PATH / 'derivatives' / 'group'
TASK_FUSION_DIR_PATH = MODEL_DIR_PATH / 'Models_03' / 'task_fusion'
FS32K_SURFACE_DIR_PATH = ATLAS_DIR_PATH / 'tpl-fs32k'
SUBJECT_LIST_DIR = REPLICATION_DIR / 'subject_list'
FIGURE_DIR = Path(
    os.environ.get('INDIVPAR_FIGURE_DIR', RESULTS_DIR / 'figures')
).expanduser()

# String aliases preserve compatibility with the existing analysis scripts.
ERIS_DIR = str(DATA_ROOT_PATH)
MODEL_DIR = str(MODEL_DIR_PATH)
BASE_DIR = str(BASE_DIR_PATH)
ATLAS_DIR = str(ATLAS_DIR_PATH)
HCP_DIR = str(HCP_DIR_PATH)
RANDY_DIR = str(RANDY_DIR_PATH)
MSC_DIR = str(MSC_DIR_PATH)
GROUP_DIR = str(GROUP_DIR_PATH)
TASK_FUSION_DIR = str(TASK_FUSION_DIR_PATH)
FS32K_SURFACE_DIR = str(FS32K_SURFACE_DIR_PATH)
WORKBENCH_COMMAND = os.environ.get('WORKBENCH_COMMAND', 'wb_command')

HCP_PARTICIPANTS_FILE = HCP_DIR_PATH / 'participants.tsv'
HCP_TRAINING_SUBJECT_LIST_FILE = (
    HCP_DIR_PATH / 'subj_list' / 'HCP40_training_KONG2019.tsv'
)

# Set to 'cpu', 'cuda', or 'auto'. INDIVPAR_DEVICE can override this value.
DEVICE_MODE = 'cpu'


def _select_device():
    requested = os.environ.get('INDIVPAR_DEVICE', DEVICE_MODE).lower()
    if requested == 'auto':
        return 'cuda' if pt.cuda.is_available() else 'cpu'
    if requested not in {'cpu', 'cuda'}:
        raise ValueError('INDIVPAR_DEVICE must be auto, cpu, or cuda.')
    if requested == 'cuda' and not pt.cuda.is_available():
        raise RuntimeError('INDIVPAR_DEVICE=cuda, but CUDA is not available.')
    return requested


DEVICE = _select_device()
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)


if __name__ == '__main__':
    print(f'DEVICE={DEVICE}')
    print(f'REPO_ROOT={REPO_ROOT}')
    print(f'ERIS_DIR={ERIS_DIR}')
    print(f'MODEL_DIR={MODEL_DIR}')
    print(f'BASE_DIR={BASE_DIR}')
    print(f'ATLAS_DIR={ATLAS_DIR}')

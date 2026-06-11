#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: dzhi
"""
import time, os, warnings, scipy
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as hev
import HierarchBayesParcel.util as hut
import FusionModel.util as futil
import FusionModel.evaluate as ev
import IndividualParcellation.scripts.group_eval as ge

import scipy.io as spio
from pathlib import Path
from train_group import build_data_list

import IndividualParcellation.utils as ut
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
from scripts.group_parcellation import ERIS_DIR

# from scripts.dual_regression import model_name

hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

HCP_DIR = '/home/dzhi/eris_mount/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    HCP_DIR = '/data/tge/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
SUBJECT_LIST_DIR = REPLICATION_DIR / 'subject_list'
RES_DIR = RESULTS_DIR / Path(__file__).resolve().parent.name
RES_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = str(RES_DIR)

ERIS_DIR = '/home/dzhi/eris_mount'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)


def plot_zvalues(input, contrast_idx, t_info, parcel_name=None):
    # Validate input
    assert input.ndim == 3, "Input tensor must be 3D: (subjects, parcels, contrasts)"
    num_subjects, num_parcels, num_contrasts = input.shape
    assert 0 <= contrast_idx < num_contrasts, "Invalid contrast index"

    # Extract the data for the given contrast
    data = input[:, :, contrast_idx]  # shape: (num_subjects, number_parcels)

    # Calculate mean and standard error across subjects
    means = np.mean(data, axis=0)
    std_errs = np.std(data, axis=0, ddof=1) / np.sqrt(num_subjects)

    # Create parcel labels if not provided
    if parcel_name is None:
        parcel_name = [f'Parcel {i}' for i in range(num_parcels)]
    else:
        assert len(parcel_name) == num_parcels, 'Invalid parcel names'

    # Create DataFrame for plotting
    df = pd.DataFrame()
    for i in range(num_parcels):
        this_df = pd.DataFrame({'subject': np.arange(num_subjects),
            'parcel': parcel_name[i],
            'z_value': data[:,i],
            'domain': t_info.iloc[contrast_idx].task_name
        })
        df = pd.concat([df, this_df], ignore_index=True)

    df['contrast_name'] = t_info.iloc[contrast_idx].contrast_name

    return df


if __name__ == "__main__":
    atlas, am_info = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    test_ses = 'ses-rest1'
    K=17

    hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    ## Making distance metric
    # dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    ######## Step 2. Generate group / indiv parcellations
    ## laod Kong 2019 17net - HCP40
    align, net_name, colors = ut.get_kong2019_group_parcellation()
    align = pt.tensor(align, dtype=pt.get_default_dtype(), device=DEVICE)
    Pgroup = pt.argmax(align, dim=0) + 1
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess'  # fusion 17 (3 datasets)
    # model_name = f'/Models_03/task_fusion/asym_MdNiIb_space-fs32k_K-17_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1'  # task 17 (3 datasets)
    U, _ = hut.load_group_parcellation(MODEL_DIR + model_name, index=None, device=DEVICE)
    U = align
    Pgroup = pt.argmax(U, dim=0) + 1


    ######## Step 2. Load HCP test data for indiv parcellation
    print(f'Start loading data ...')
    tic = time.perf_counter()
    ## 3 datasets task data (CondHalf)
    t_data, cond_vec, part_vec, subj_ind = build_data_list(['IBC'], atlas=atlas.name,
                                                sess=["all"], cond_ind=['cond_num'],
                                                type=['CondHalf'], part_ind=['half'],
                                                subj=[None], part_num=[None],
                                                join_sess=[True],
                                                join_sess_part=False, smooth=[
                                                                              '5_zstat_masked-hi0.1lo0.1'])

    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    hut.report_cuda_memory()
    n_subj = t_data[0].shape[0]

    ## Calculate vertex-wise task converage map
    t_data = [np.where(d!=0, 1, 0) for d in t_data]
    t_data = [d.reshape(-1, d.shape[-1]) for d in t_data]
    coverage_map = np.vstack(t_data).mean(axis=0)

    ## Save indiv parcellation in cifti
    img = atlas.data_to_cifti(coverage_map.reshape(1,-1), ["MdNiIb_coverage_all"])
    nb.save(img, ERIS_DIR + f'/dzhi/Indiv_par/Results/section_3' +
            f'/asym_MdNiIb_space-fs32k_zstat_masked-hi0.1lo0.1_converage-map.dscalar.nii')

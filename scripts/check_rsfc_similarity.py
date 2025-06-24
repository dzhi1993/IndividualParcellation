#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 5/23/2025 at 12:40 PM
Author: dzhi
"""
import time, os, warnings
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

import group_parcellation as gp
from indiv_eval_hcp import load_hcp_timeseries
import group_eval as ge
import scipy.io as spio
from pathlib import Path

import IndividualParcellation.utils as ut
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
# from scripts.dual_regression import model_name

hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

HCP_DIR = '/home/dzhi/eris_mount/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    HCP_DIR = '/data/tge/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

RES_DIR = '/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations'
if not Path(RES_DIR).exists():
    RES_DIR = '/data/tge/dzhi/Indiv_par/Evaluations'
if not Path(RES_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

ERIS_DIR = '/home/dzhi/eris_mount'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : True
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

def cal_rsfc_similarity(data1, data2):
    """ Calculate the rsfc similarity between two models"""
    # rsfc1 = np.corrcoef(data1.cpu().numpy().flatten())
    rsfc1 = np.corrcoef(data1.T)
    rsfc2 = np.corrcoef(data2.T)

    similarity = np.corrcoef(rsfc1.flatten(), rsfc2.flatten())
    return similarity

if __name__ == '__main__':
    # get HCP subject list
    A = pd.read_csv('/home/dzhi/eris_mount/Tian/HCP_img/participants.tsv', delimiter='\t')
    B = pd.read_csv(f'/home/dzhi/eris_mount/Tian/HCP_img/subj_list/test.tsv', delimiter='\t')
    hcp_subj_ind = np.array(A[A['participant_id'].isin(B['participant_id'])].index)

    # Load HCP raw time series
    # Calculate the rsFC similarity
    results = pd.DataFrame()
    for type in ['ROI1483Run', 'Ico642Run']:
        for i, ses in enumerate(['ses-rest1', 'ses-rest2']):
            data, info, _ = ds.get_dataset(BASE_DIR, 'HCP', atlas='fs32k', sess=ses,
                                            type=type, subj=hcp_subj_ind, smooth='4fwhm_binarized')
            for j, run in enumerate([1,2]):
                global_run = i * 2 + j
                t_data = load_hcp_timeseries(HCP_DIR, "subj_list/test.tsv",
                                             space='fs32k', run_list=[global_run],
                                             type='Tseries', hemis=None, smooth='4fwhm')

                this_dat = data[:,info.run==run,:]
                n_subj = this_dat.shape[0]
                similarity = [cal_rsfc_similarity(this_dat[n], t_data[0][n]) for n in range(n_subj)]
                df = pd.DataFrame({'subj_num': np.arange(n_subj),
                                   'type': [type] * n_subj,
                                   'ses': [ses] * n_subj,
                                   'run': [run] * n_subj,
                                   'global_run': [global_run] * n_subj,
                                   'similarity': similarity})
                results = pd.concat([results, df], axis=0)

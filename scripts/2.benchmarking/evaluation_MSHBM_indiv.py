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

import scipy.io as spio
from pathlib import Path

import utils as ut
from global_config import (ATLAS_DIR, BASE_DIR, DEVICE, ERIS_DIR, HCP_DIR,
                           MODEL_DIR, REPLICATION_DIR, RESULTS_DIR,
                           SUBJECT_LIST_DIR)

# from scripts.dual_regression import model_name

hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

RES_DIR = RESULTS_DIR / Path(__file__).resolve().parent.name
RES_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = str(RES_DIR)

def make_eval_info(K, atlas='fs32k', train_info=['UKB'], train_sess='ses-2',
                   tdata='MDTB', test_sess='ses-1', model_type='Models_03',
                   group_map_name='Buckner7', test_kappa=None):
    """ Collects all the information from the model and the
        training and test data sets into a single dictionary

    Args:
        M (fm.Model): model object
        train_info (dict): training data information
        test_info (dict): test data information

    Returns:
        minfo (dict): model information
    """
    minfo = pd.Series()
    minfo['atlas'] = atlas
    minfo['K'] = K
    minfo['datasets'] = train_info
    minfo['train_sess'] = train_sess
    minfo['test_data'] = tdata
    minfo['test_sess'] = test_sess
    minfo['model_type'] = model_type
    minfo['group_map_name'] = group_map_name
    # minfo['indiv_train_subj_kappa'] = M.emissions[0].subject_specific_kappa
    # minfo['indiv_train_par_kappa'] = M.emissions[0].parcel_specific_kappa
    minfo['indiv_test_kappa'] = test_kappa
    return minfo


def plot_zvalues_inhomo(input, contrast_idx, t_info, type='z_value', parcel_name=None):
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
            type: data[:,i],
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

    group_strength_list = [100]
    spatial_list = [30]
    hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    ## Making distance metric
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    # ####### Step 2. Generate group / indiv parcellations
    # # laod Kong 2019 17net - HCP40
    align, net_name, colors = ut.get_kong2019_group_parcellation()
    align = pt.tensor(align, dtype=pt.get_default_dtype(), device=DEVICE)
    Pgroup = pt.argmax(align, dim=0) + 1

    results = pd.DataFrame()
    for global_counter in [1,2,3,4]:
        subj_list_file = f"HCP200_test_{global_counter}.tsv"
        subj_list_path = SUBJECT_LIST_DIR / subj_list_file
        if not subj_list_path.exists():
            subj_list_path = Path(HCP_DIR) / 'subj_list' / subj_list_file
        T = pd.read_csv(subj_list_path, sep='\t')

        ####################################################################################################################
        ## Evaluation
        ######## Step 2. Load HCP test data for indiv parcellation
        print(f'Start loading data: HCP - {test_ses} - task contrasts ...')
        tic = time.perf_counter()
        ## 1. HCP task contrasts
        # t_data, t_info = ut.load_hcp_contrasts(HCP_DIR, f"/subj_list/{subj_list_file}", space='fs32k',
        #                                        return_positive=False, hemis=None, smooth='4_MSMAll')
        ## 2. HCP resting state time series
        t_data = ut.load_hcp_timeseries(HCP_DIR, f"subj_list/{subj_list_file}",
                                    space=atlas.name, run_list=[2,3],
                                    type='Tseries', hemis=None, smooth=None)
        t_info = pd.DataFrame({"task_name": ['REST'] * t_data[0].shape[1]})

        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        hut.report_cuda_memory()
        n_subj = t_data[0].shape[0]

        for train_runs in [2]:
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    # Load indiv parcellation
                    # Pindiv = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP203_test_set' +
                    #         f'/asym_KONG2019+HCPrest-2run-indiv_space-fs32k_K-17_CondHalf_groupstrengh-{p}_spatial-{w}_1.dlabel.nii').get_fdata()[:]
                    # Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
                    # Load Kong2019 individual parcellations
                    Pindiv = ut.get_kong2019_indiv_parcellations(ERIS_DIR + '/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
                                        str(subj_list_path),
                                        w=p, c=w, num_sess=train_runs)
                    # Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)
                    Pindiv = Pindiv.nan_to_num()

                    # Making evaluation information
                    minfo = make_eval_info(K, train_info=['HCP'], train_sess=f'run-{train_runs}',
                                                tdata='HCP', test_sess='contrasts',
                                                model_type='Models_03', group_map_name='MSHBM17',
                                                test_kappa=None)

                    t_info['task_name']=[s.rstrip('2') for s in t_info.task_name]
                    this_res = pd.DataFrame()
                    # Evaluate on each run's time series
                    for r, td in enumerate(t_data):
                        if type(td) is np.ndarray:
                            td = pt.tensor(td, dtype=pt.get_default_dtype())

                        tasks_list = ['all']
                        for task in tasks_list:
                            if task == 'all':
                                idx = [True] * len(t_info)
                                # Individual evaluation
                                # homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:,idx,:])
                                # zvalue_indiv = ev.calc_test_zvalue(Pindiv, td[:,idx,:], return_single=False)
                                # np.save(ERIS_DIR + f'/dzhi/Indiv_par/Kong_2019/indiv_par/zvalues' +
                                #         f'/zvalue_indiv_MSHBM-{train_runs}run-indiv_K-{K}_strengh-{p}_spatial-{w}_contrasts-sm4.npy',
                                #         zvalue_indiv.cpu().numpy())
                                # inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :], return_single=False)
                                # inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                                # np.save(ERIS_DIR + f'/dzhi/Indiv_par/Kong_2019/indiv_par/inhomogeneity' +
                                #         f'/inhomo_nets_asym_MSHBM-{train_runs}run-indiv_K-{K}_strengh-{p}_spatial-{w}_contrasts-sm4.npy',
                                #         inhomo_nets.cpu().numpy())
                            else:
                                idx = t_info['task_name'] == task

                            res = pd.DataFrame({'atlas': [minfo.atlas] * n_subj,
                                    'K': [minfo.K] * n_subj,
                                    'train_data': [minfo.datasets] * n_subj,
                                    'train_sess': [minfo.train_sess] * n_subj,
                                    'test_data': [minfo.test_data] * n_subj,
                                    'test_sess': [minfo.test_sess] * n_subj,
                                    'model_type': [minfo.model_type] * n_subj,
                                    'group_map_name': [minfo.group_map_name] * n_subj,
                                    'subj_num': [f'{i}' for i in T.participant_id],
                                    'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj})

                            ## Factorize DCBC calcuation
                            # hut.report_cuda_memory()
                            # pt.cuda.empty_cache()
                            # dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:,idx,:], dist, trim_nan=True)
                            # pt.cuda.empty_cache()
                            # hut.report_cuda_memory()

                            hut.report_cuda_memory()
                            pt.cuda.empty_cache()
                            homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:, idx, :])
                            # inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :], return_single=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            # res['dcbc_indiv'] = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
                            res['homo_indiv'] = homo_indiv.cpu()
                            # res['inhomo_indiv'] = inhomo_indiv.cpu()
                            res['task_name'] = task
                            res['test_run'] = r + 1
                            res['train_smooth'] = "6fwhm"
                            res['test_smooth'] = 'sm0'
                            res['test_type'] = 'Tseries'
                            this_res = pd.concat([this_res, res], ignore_index=True)

                    # QC
                    # dice = [hev.dice_coefficient(Pgroup, Pindiv[i], label_matching=True).item()
                    #                 for i in range(Pindiv.shape[0])]
                    # # ari = [hev.ARI(pt.argmax(U, dim=0), pt.argmax(U_indiv, dim=1)[i]).item()
                    # #     for i in range(U_indiv.shape[0])]
                    # # nmi = [1- hev.nmi(pt.argmax(U, dim=0).cpu(), pt.argmax(U_indiv, dim=1)[i].cpu())
                    # #     for i in range(U_indiv.shape[0])]
                    #
                    # this_res['dice_group'] = dice * len(tasks_list) * len(t_data)
                    # res['ari_group'] = ari
                    # res['nmi_group'] = nmi
                    this_res['strength'] = p
                    this_res['spatial_w'] = w

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/eval_indiv-mRBM_MSHBM-HCP203_1-2run_K-17_test_on_HCPtask-contrast_sm4.tsv',
        index=False, sep='\t')
    print('Done')

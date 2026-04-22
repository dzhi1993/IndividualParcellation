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
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)


def make_eval_info(M, atlas='MNIAsymC2', train_info=['UKB'], train_sess='ses-2',
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
    minfo['K'] = M.K
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

def eval_parcel_DCBC(U_group, U_indiv, t_data, dist, minfo, out_file=None):
    # convert tdata to tensor
    if type(t_data) is np.ndarray:
        t_data = pt.tensor(t_data, dtype=pt.get_default_dtype())
    # convert U_group and U_indiv to tensor
    if type(U_group) is np.ndarray:
        U_group = pt.tensor(U_group, dtype=pt.get_default_dtype())
    if type(U_indiv) is np.ndarray:
        U_indiv = pt.tensor(U_indiv, dtype=pt.get_default_dtype())

    num_subj = t_data.shape[0]
    # Now run the DCBC evaluation fo the group
    # Pgroup = pt.argmax(U_group, dim=0) + 1
    # Pindiv = pt.argmax(U_indiv, dim=1) + 1
    homo_group = ev.calc_test_homogeneity(U_group, t_data)
    homo_indiv = ev.calc_test_homogeneity(U_indiv, t_data)
    dcbc_group = ev.calc_test_dcbc(U_group, t_data, dist)
    dcbc_indiv = ev.calc_test_dcbc(U_indiv, t_data, dist)

    # ------------------------------------------
    # Collect the information from the evaluation
    # in a data frame
    train_datasets = minfo.datasets
    ev_df = pd.DataFrame({'atlas': [minfo.atlas] * num_subj,
                          'K': [minfo.K] * num_subj,
                          'train_data': [train_datasets] * num_subj,
                          'train_sess': [minfo.train_sess] * num_subj,
                          'test_data': [minfo.test_data] * num_subj,
                          'test_sess': [minfo.test_sess] * num_subj,
                          'model_type': [minfo.model_type] * num_subj,
                          'group_map_name': [minfo.group_map_name] * num_subj,
                          'subj_num': np.arange(num_subj),
                          'indiv_test_kappa': [minfo.indiv_test_kappa] * num_subj})
    # Add all the evaluations to the data frame
    ev_df['dcbc_group'] = dcbc_group.cpu()
    ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
    ev_df['homo_group'] = homo_group.cpu()
    ev_df['homo_indiv'] = homo_indiv.cpu()
    # ev_df.to_csv(out_file, index=False, sep='\t')
    return ev_df

def plot_multi_flat(data, atlas, grid, cmap='tab20b', dtype='label',
                    cscale=None, titles=None, colorbar=False,
                    save_fig=False):
    """ Plot multiple flatmaps in a grid

    Args:
        data: the input parcellations, shape(N, K, P) where N indicates
              the number of parcellations, K indicates the number of
              parcels, and P is the number of vertices.
        atlas: the atlas name used to plot the flatmap
        grid: the grid shape of the subplots
        cmap: the colormap used to plot the flatmap
        dtype: the data type of the input data, 'label' or 'prob'
        cscale: the color scale used to plot the flatmap
        titles: the titles of the subplots
        colorbar: whether to plot the colorbar
        save_fig: whether to save the figure, default format is png

    Returns:
        The plt figure plot
    """

    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        futil.plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=None,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(titles[i])
        plt.tight_layout()

    if save_fig:
        plt.savefig('/indiv_parcellations.png')


def eval_task_inhomo_MSHBM_vs_HBP_indiv():
    atlas, _ = am.get_atlas('fs32k')
    contrast_file = HCP_DIR + '/rfMRI/fix_32k/group/' + \
                    'HCP_S1200_997_tfMRI_ALLTASKS_level2_cohensd_hp200_s2_MSMAll.dscalar.nii'
    dist = futil.load_fs32k_dist(file_type='distAvrg_sp', hemis='half',
                                device=DEVICE if pt.cuda.is_available() else 'cpu')

    task_name = np.array(nb.load(contrast_file).header.get_axis(0).name, dtype=object)
    t_data = atlas.cifti_to_data(contrast_file)[:,0:29759]
    t_data = pt.tensor(t_data, dtype=pt.get_default_dtype())

    # Load Kong2019 individual parcellations
    U_kong = get_kong2019_indiv_parcellations(ERIS_DIR + '/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
                                HCP_DIR + "/subj_list/HCP203_test_set.tsv",
                                w=100, c=30, num_sess=1)[:,0:29759]
    U_kong = pt.tensor(U_kong, dtype=pt.get_default_dtype())
    U_kong = pt.where(U_kong == 0, pt.nan, U_kong)

    # Load HBP individual parcellations
    U_hbp = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/test_set' +
                    f'/asym_HCP923test_space-fs32k_K-17_Ico642Run_indiv_groupstrength-60_cRBM-w90.dlabel.nii').get_fdata()[:,0:29759]
    U_hbp = pt.tensor(U_hbp, dtype=pt.get_default_dtype())
    U_hbp = pt.where(U_hbp == 0, pt.nan, U_hbp)

    gmap_names = ['MSHBM_HCP40', 'HBP_HCP40']
    results = pd.DataFrame()
    for i, U_indiv in enumerate([U_kong, U_hbp]):
        num_subj = U_indiv.shape[0]
        # homo_indiv_kong = ev.calc_test_homogeneity(U_kong, t_data.unsqueeze(0).repeat(num_subj_kong, 1, 1))
        inhomo_indiv = ev.calc_test_task_inhomogeneity(U_indiv, t_data.unsqueeze(0).repeat(num_subj, 1, 1),
                                                       return_single=False)
        # dcbc_indiv = ev.calc_test_dcbc(U_indiv, t_data.unsqueeze(0).repeat(num_subj, 1, 1), dist)

        ev_df = pd.DataFrame({'K': [17] * num_subj,
                            'group_map_name': [gmap_names[i]] * num_subj,
                            'subj_num': np.arange(num_subj)})
        # ev_df['dcbc_indiv'] = dcbc_indiv.cpu()

        if inhomo_indiv.ndim == 1:
            assert inhomo_indiv.shape[0] == num_subj, \
                "task inhomo should match subject number!"
            ev_df['inhomo_indiv'] = inhomo_indiv.cpu()
        elif inhomo_indiv.ndim == 2:
            assert inhomo_indiv.shape[1] == len(task_name), \
                "task inhomo should match subject number or task contrast number!"
            for i, tnam in enumerate(task_name):
                ev_df[f'inhomo_indiv_{tnam}'] = inhomo_indiv[:,i].cpu()

        results = pd.concat([results, ev_df], ignore_index=True)

    results.to_csv(RES_DIR + f'/eval_MSHBM_vs_HBP_indiv_teston-Task_separate_test.tsv', index=False, sep='\t')


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

    group_strength_list = [10]
    spatial_list = [5]
    hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    ## Making distance metric
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

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

    results = pd.DataFrame()
    for global_counter in [1]:
        subj_list_file = f"HCP200_test_new-added.tsv"
        T = pd.read_csv(HCP_DIR + f"/subj_list/{subj_list_file}", sep='\t')
        ####################################################################################################################
        ## Evaluation
        ######## Step 2. Load HCP test data for indiv parcellation
        print(f'Start loading data: HCP resting - {test_ses} - Tseries ...')
        tic = time.perf_counter()
        ## 1. HCP task contrasts
        # t_data, t_info = ut.load_hcp_contrasts(HCP_DIR, f"/subj_list/{subj_list_file}", space='fs32k',
        #                                        return_positive=False, hemis=None, smooth='4_MSMAll')
        ## 2. HCP resting state time series
        t_data = ut.load_hcp_timeseries(HCP_DIR, f"subj_list/{subj_list_file}",
                                    space=atlas.name, run_list=[2,3],
                                    type='Tseries', hemis=None, smooth=None)
        t_info = pd.DataFrame({"task_name": ['REST'] * t_data[0].shape[1]})

        ## 3. HCP task betas
        # t_data, _, _, _, t_info = ut.build_hcp_datasets(HCP_DIR, f"/subj_list/{subj_list_file}",
        #                                                 atlas, ses_list=['ses-task'], join_sess=False, join_sess_part=False,
        #                                      part_ind=['half'], part_num=None, cond_ind=['reg_id'],
        #                                      type=['CondHalf'], hemis=None, smooth='6fwhm')
        # t_data = [t_data[2-global_counter]]
        # t_info = np.array_split(t_info, 2)[0]

        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        hut.report_cuda_memory()
        n_subj = t_data[0].shape[0]

        for train_runs in [1]:
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    print(f"Evalulating on group strength {p}, spatial {w} ...")
                    # Load indiv parcellation
                    Pindiv = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                            f'/asym_MdNiIbHc+HCPrest-{train_runs}run1-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-{p}_spatial-{w}.dlabel.nii').get_fdata()[:]
                    Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
                    # Load Kong2019 individual parcellations
                    # Pindiv = ut.get_kong2019_indiv_parcellations(ERIS_DIR + '/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
                    #                     HCP_DIR + f"/subj_list/{subj_list_file}",
                    #                     w=p, c=w, num_sess=train_runs)
                    Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

                    HCP200_list = pd.read_csv(HCP_DIR + f"/subj_list/HCP200_test.tsv", sep='\t')
                    idx = HCP200_list.index[HCP200_list['participant_id'].isin(T['participant_id'])]
                    Pindiv = Pindiv[idx]
                    # Making evaluation information
                    minfo = ge.make_eval_info(K, train_info=['HCP'], train_sess=f'run-{train_runs}',
                                                tdata='HCP', test_sess='Tseries',
                                                model_type='Models_03', group_map_name='MdNiIbHc',
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
                                # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set/zvalues' +
                                #         f'/zvalue_indiv_asym_KONG2019+HCPrest-{train_runs}run-indiv_K-{K}_strengh-{p}_spatial-{w}_allcontrast_sm4_{global_counter}.npy',
                                #         zvalue_indiv.cpu().numpy())
                                # inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :], return_single=False)
                                # inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                                # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set/inhomogeneity' +
                                #         f'/inhomo_nets_asym_KONG2019+HCPrest-{train_runs}run-indiv_K-{K}_strengh-{p}_spatial-{w}_allcontrast_sm4_{global_counter}.npy',
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
                            hut.report_cuda_memory()
                            dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:,idx,:], dist, trim_nan=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            hut.report_cuda_memory()
                            homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:, idx, :])
                            # inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:,idx,:], return_single=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            res['dcbc_indiv'] = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
                            res['homo_indiv'] = homo_indiv.cpu()
                            # res['inhomo_indiv'] = inhomo_indiv.cpu()
                            res['task_name'] = task
                            res['test_run'] = 1
                            res['train_smooth'] = "6fwhm"
                            res['test_smooth'] = None
                            res['test_type'] = 'Tseries'
                            this_res = pd.concat([this_res, res], ignore_index=True)

                    # QC
                    dice = [hev.dice_coefficient(Pgroup, Pindiv[i], label_matching=True).item()
                                    for i in range(Pindiv.shape[0])]
                    # ari = [hev.ARI(pt.argmax(U, dim=0), pt.argmax(U_indiv, dim=1)[i]).item()
                    #     for i in range(U_indiv.shape[0])]
                    # nmi = [1- hev.nmi(pt.argmax(U, dim=0).cpu(), pt.argmax(U_indiv, dim=1)[i].cpu())
                    #     for i in range(U_indiv.shape[0])]

                    this_res['dice_group'] = dice * len(tasks_list) * len(t_data)
                    # res['ari_group'] = ari
                    # res['nmi_group'] = nmi
                    this_res['strength'] = p
                    this_res['spatial_w'] = w

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/eval_indiv-mRBM_HBP-HCP200_1run_K-17_test_on_HCPrest-Tseries_sm0.tsv',
        index=False, sep='\t')
    print('Done')


    ## Per train run
    df_hbp_1run_indiv = pd.read_csv(RES_DIR+ "/eval_HBP-1run_K-17_indiv-mRBM_test_on_HCPtask-allcontrasts_sm4.tsv", sep='\t')
    df_hbp_1run_indiv['type'] = "hbp_1run"
    df_hbp_2run_indiv = pd.read_csv(RES_DIR + "/eval_HBP-2run_K-17_indiv-mRBM_test_on_HCPtask-allcontrasts_sm4.tsv", sep='\t')
    df_hbp_2run_indiv['type'] = "hbp_2run"
    df_fusion_1run_indiv = pd.read_csv(RES_DIR + "/eval_MdNiIbHc-1run_K-17_indiv-mRBM_test_on_HCPtask-allcontrasts_sm4.tsv", sep='\t')
    df_fusion_1run_indiv['type'] = "fusion_1run"
    df_fusion_2run_indiv = pd.read_csv(RES_DIR + "/eval_MdNiIbHc-2run_K-17_indiv-mRBM_test_on_HCPtask-allcontrasts_sm4.tsv", sep='\t')
    df_fusion_2run_indiv['type'] = "fusion_2run"
    df_hbp = pd.concat([df_hbp_1run_indiv, df_hbp_2run_indiv], ignore_index=True)
    df_fusion = pd.concat([df_fusion_1run_indiv, df_fusion_2run_indiv], ignore_index=True)
    df = pd.concat([df_hbp, df_fusion], ignore_index=True)
    df_fusion['dcbc'] = df_fusion['dcbc_indiv']
    df_fusion['inhomo'] = df_fusion['inhomo_indiv']

    # load group
    df_group = pd.DataFrame()
    for i in [1, 2, 3, 4]:
        res = pd.read_csv(
            f'/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations/eval_group_rest_vs_task_vs_fusion_K-17_on-HCPtest-task-allcontrasts_sm4_{i}.tsv',
            sep='\t')
        df_group = pd.concat([df_group, res], ignore_index=True)

    df_group['type'] = "group"
    df_group['dcbc'] = df_group['dcbc_group']
    df_group['inhomo'] = df_group['inhomo_group']
    df_group = df_group.loc[df_group["group_map_name"] == "HBP17_rest"]
    # df_group['group_map_name'] = df_group['group_map_name'].replace({'YEO2011': 'X', 'C': 'Z'})
    df = pd.concat([df_group, df_fusion], ignore_index=True)
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    sb.barplot(df, x='task_name', y='dcbc', hue='type')
    # plt.title(f'mshbm, {t_run} runs')
    # plt.ylim(0.68, 0.76)
    # plt.axhline(df_group.loc[df_group["group_map_name"] == "Fusion(Nfeature 3+1)"].dcbc_group.mean(), color='orange',
    #             linestyle=':')
    # plt.axhline(df_group.loc[df_group["group_map_name"] == "HBP17_rest"].dcbc_group.mean(), color='blue',
    #             linestyle=':')

    plt.subplot(1, 2, 2)
    sb.barplot(df, x='task_name', y='inhomo', hue='type')
    # plt.title(f'rest-only, {t_run} runs')
    # plt.axhline(df_group.loc[df_group["group_map_name"] == "Fusion(Nfeature 3+1)"].inhomo_group.mean(), color='orange',
    #             linestyle=':')
    # plt.axhline(df_group.loc[df_group["group_map_name"] == "HBP17_rest"].inhomo_group.mean(), color='blue',
    #             linestyle=':')
    plt.ylim(0.8, 1)

    plt.suptitle("Test on HCP task contrasts sm=4mm")
    plt.tight_layout()
    plt.show()

    ## Mean z-value
    num_run = 1
    d_type = ['rest-only', 'fusion', 'task-only']
    color_list = [tuple(color) for color in colors[1:, :]]

    # res1 = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                f'/zvalue_DU-indiv_K-15.npy')

    # res2 = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/leave_one_out/zvalues' +
    #                f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest5run+task5run-woVODDK-indiv_K-15_strengh-1_spatial-1_sm2_equalweights.npy')

    res_rest = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                   f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest-{num_run}run-indiv_K-15_strengh-1_spatial-1_sm2.npy')

    res_fusion = []
    for do in ['EPROJ','MOTOR','NBACK','VODDK','LANG', 'TOM', 'VISME']:
        res = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/leave_one_out/zvalues' +
                      f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest5run+task5run-wo{do}-indiv_K-15_strengh-2_spatial-2_sm2_task0.5rest0.5.npy')
        res_fusion.append(res)
        print(f'Writing domain {do}, number of contrasts {res.shape[1]}')
    res_fusion = np.concatenate(res_fusion, axis=2)

    res_task = []
    for do in ['EPROJ', 'MOTOR', 'NBACK', 'VODDK', 'LANG', 'TOM', 'VISME']:
        res = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/leave_one_out/zvalues' +
                      f'/zvalue_indiv_asym_MdNiIbHc+RANDYtask5run-wo{do}-indiv_K-15_strengh-1_spatial-1_sm2_4fwhm-masked.npy')
        res_task.append(res)
        print(f'Writing domain {do}, number of contrasts {res.shape[1]}')
    res_task = np.concatenate(res_task, axis=2)

    df_all = pd.DataFrame()
    _, t_info = ut.load_randy_contrasts(subj=None, hemis='L', smooth=2)
    t_info['task_name'] = t_info['domain']
    for j, res in enumerate([res_rest, res_fusion, res_task]):
        for i in [0,1,5,10,11,12]:
            df = plot_zvalues(res, i, t_info, parcel_name=net_name[1:])
            df['dtype'] = d_type[j]
            df_all = pd.concat([df_all, df], ignore_index=True)

    con_nam = df_all['contrast_name'].unique()
    plt.figure(figsize=(10, 5 * len(con_nam)))

    for n, c_name in enumerate(con_nam):
        this_df_all = df_all.loc[df_all['contrast_name'] == c_name]
        domain = this_df_all['domain'].unique()

        plt.subplot(len(con_nam), 1, n + 1)
        sb.barplot(data=this_df_all, x='parcel', y='z_value', hue='dtype', errorbar='se', width=0.7)
        plt.xticks(rotation=45)
        plt.title(f'{domain}: {c_name}')

    plt.tight_layout()
    plt.suptitle(f'{num_run} runs, group priors: MSHBM, HBP_rest, HBP_fusion', y=1.05)
    # plt.savefig(f'{num_run}runs.pdf', format='pdf')
    plt.show()
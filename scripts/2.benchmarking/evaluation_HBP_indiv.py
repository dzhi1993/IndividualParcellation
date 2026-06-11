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

import IndividualParcellation.utils as ut
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
from scripts.group_parcellation import ERIS_DIR

# from scripts.dual_regression import model_name

hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']

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


def make_eval_info(K, atlas='MNIAsymC2', train_info=['UKB'], train_sess='ses-2',
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

if __name__ == "__main__":
    atlas, am_info = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    K=17

    ## Making distance metric
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    ######## Main evaluation loop: indiv parcellations
    results = pd.DataFrame()
    for global_counter in [1]:
        subj_list_file = f"HCP200_test_{global_counter}.tsv"
        subj_list_path = SUBJECT_LIST_DIR / subj_list_file
        if not subj_list_path.exists():
            subj_list_path = Path(HCP_DIR) / 'subj_list' / subj_list_file
        T = pd.read_csv(subj_list_path, sep='\t')
        ####################################################################################################################
        ## Evaluation
        ######## Step 2. Load HCP test data for indiv parcellation
        print(f'Start loading HCP test data ...')
        tic = time.perf_counter()
        ## 1. HCP task contrasts
        t_data, t_info = ut.load_hcp_contrasts(HCP_DIR, f"/subj_list/{subj_list_file}", space='fs32k',
                                               return_positive=False, hemis=None, smooth='4_MSMAll')
        ## 2. HCP resting state time series
        # t_data = ut.load_hcp_timeseries(HCP_DIR, f"subj_list/{subj_list_file}",
        #                             space=atlas.name, run_list=[2,3],
        #                             type='Tseries', hemis=None, smooth=None)
        # t_info = pd.DataFrame({"task_name": ['REST'] * t_data[0].shape[1]})

        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        hut.report_cuda_memory()
        n_subj = t_data[0].shape[0]

        for train_runs in [2]:
            print(f"Evalulating ...")
            # Load indiv parcellation
            Pindiv = nb.load(ERIS_DIR + f'/dzhi/Indiv_par/Results/section_4/HCP' +
                    f'/RestPrior+HCPrest-2run-indiv_space-fs32k_K-17_Ico642Run.dlabel.nii').get_fdata()[:]
            Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
            Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

            HCP200_list = pd.read_csv(SUBJECT_LIST_DIR / 'HCP200_test.tsv', sep='\t')
            idx = HCP200_list.index[HCP200_list['participant_id'].isin(T['participant_id'])]
            Pindiv = Pindiv[idx]
            # Making evaluation information
            minfo = make_eval_info(K, train_info=['HCP'], train_sess=f'run-{train_runs}',
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


            this_res['dice_group'] = dice * len(tasks_list) * len(t_data)
            # res['ari_group'] = ari
            # res['nmi_group'] = nmi

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
            RES_DIR + f'/eval_group_rest_vs_task_vs_fusion_K-17_on-HCPtest-task-allcontrasts_sm4_{i}.tsv',
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

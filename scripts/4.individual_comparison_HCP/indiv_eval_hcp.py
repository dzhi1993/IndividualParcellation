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

import group_parcellation as gp
import group_eval as ge
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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
SUBJECT_LIST_DIR = REPLICATION_DIR / 'subject_list'
RES_DIR = RESULTS_DIR / Path(__file__).resolve().parent.name
RES_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = str(RES_DIR)

def get_subject_list_path(file_name):
    subj_list_path = SUBJECT_LIST_DIR / file_name
    if not subj_list_path.exists():
        subj_list_path = Path(HCP_DIR) / 'subj_list' / file_name
    return subj_list_path

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
                                str(get_subject_list_path("HCP203_test_set.tsv")),
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

def train_indiv_par(atlas, p_list, w_list, U, Wc, data, cond_vec, part_vec, subj_ind):
    for counter_p, p in enumerate(p_list):
        for counter_w, w in enumerate(w_list):
            print(f'prior strength is {p}; the MRF strength is {w} ...')
            # m-RBM
            ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
                                                  sym_type='asym', model_type='cRBM_Wc',
                                                  Wc=Wc, theta=w, epos_iter=20, num_chain=n_subj)
            ar_model.bu = ar_model.bu * p

            # Independent
            # ar_model = ar.build_arrangement_model(U*p, prior_type='logpi', atlas=atlas,
            #                                       sym_type='asym', model_type='independent')

            U_indiv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
                                                    cond_vec, part_vec, subj_ind, Vs=None,
                                                    sym_type='asym', n_iter=10,
                                                    em_params={'num_subj': n_subj,
                                                                'uniform_kappa': True,
                                                                'subjects_equal_weight':True,
                                                                'subject_specific_kappa': False,
                                                                'parcel_specific_kappa': False})
            Pindiv = pt.argmax(U_indiv, dim=1) + 1

            del M
            pt.cuda.empty_cache()
            hut.report_cuda_memory()

            # Load Kong2019 individual parcellations
            # U_indiv = get_kong2019_indiv_parcellations(ERIS_DIR + '/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
            #                     HCP_DIR + "/subj_list/HCP203_test_set_filtered_1.tsv",
            #                     w=100, c=30, num_sess=1)
            # Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

            ## Save indiv parcellation in cifti
            # colors = np.concatenate([np.array([[0,0,0,0]]),plt.cm.get_cmap('tab20', 17).colors], axis=0)
            img = nt.make_label_cifti(Pindiv.T.cpu().numpy(), atlas.get_brain_model_axis(),
                                    column_names=[f'subj_{i}' for i in range(Pindiv.shape[0])],
                                    label_names=net_name, label_RGBA=colors)
            nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP203_test_set' +
                    f'/asym_KONG2019+HCPrest+task-1run-indiv_space-fs32k_K-{K}_{fc_type}_groupstrengh-{p}_spatial-{w}.dlabel.nii')




if __name__ == "__main__":
    atlas, am_info = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    # DEVICE = 'cpu'
    training_ses = 'all'
    test_ses = 'ses-rest1'
    fc_type = 'Ico642Run'
    ext = '_binarized'
    K=17
    group_strength_list = [2]
    spatial_list = [2]
    hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']

    ######## Step 2. Generate group / indiv parcellations
    ## laod Kong 2019 17net - HCP40
    align, net_name, colors = ut.get_kong2019_group_parcellation()
    align = pt.tensor(align, dtype=pt.get_default_dtype(), device=DEVICE)

    ## Load the group prior from a pre-trained model
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800' # resting 17 800subj
    # model_name = f'/Models_03/task_fusion/asym_Hc_space-fs32k_K-17_HCP40-Kong_ROI1483Run_sm6fwhm_binarized' # resting 17 kong40subject(idx=32)
    # model_name = f'/Models_03/task_fusion/asym_MdNiIbWmDeSoHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess_equalweights'  # fusion 17 (6 datasets)
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess' # fusion 17 (3 datasets)
    # model_name = f'/Models_03/task_fusion/asym_MdNiIb_space-fs32k_K-17_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1'  # task 17 (3 datasets)
    U, _ = hut.load_group_parcellation(MODEL_DIR + model_name, index=None, device=DEVICE)
    # Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE) # list of N*K matrix
    # U = atlas.cifti_to_data(ERIS_DIR + '/dzhi/workspace/res/priors/MSHBM_group_prior_HCP40training_k-15.dscalar.nii') # MSHBM 15
    # U = pt.tensor(U, dtype=pt.get_default_dtype(), device=DEVICE)
    # Align with prior
    indx = hev.matching_greedy(align, pt.softmax(U, dim=0))
    U = U[indx, :]
    U = pt.softmax(U, dim=0)
    # Vs = [v[:,indx] for v in Vs]
    # U = align
    Pgroup = pt.argmax(U, dim=0) + 1

    ## Determine the connectivity profile for mRBM
    Wc = pt.load(ERIS_DIR + '/Tian/UKBB_full/imaging/Atlases/tpl-fs32k/fs32k_neighbours.pt', weights_only=True)
    hut.report_cuda_memory()

    for global_counter in [1,2,3,4]:
        subj_list_file = f"HCP200_test_{global_counter}.tsv"
        ######## Step 1. Load subjects individual training data
        print(f'Start loading data {global_counter}: HCP resting - {training_ses}, {fc_type} {ext} ...')
        tic = time.perf_counter()
        ## HCP task data
        # data1, cond_vec1, part_vec1, subj_ind1, t_info = gp.build_hcp_datasets(HCP_DIR, f'subj_list/{subj_list_file}',
        #                                                            atlas, ses_list=['ses-task'],
        #                                      join_sess=False, join_sess_part=False,
        #                                      part_ind=['half'], part_num=None, cond_ind=['reg_id'],
        #                                      type=['CondHalf'], hemis=None, smooth='6fwhm_zstat_masked-hi0.1lo0.1')
        ## HCP resting data
        data_all, cond_vec_all, part_vec_all, subj_ind_all, rs_info = gp.build_hcp_datasets(HCP_DIR, f"subj_list/{subj_list_file}",
                                                    atlas, ses_list=['all'],
                                                    join_sess=False, join_sess_part=False,
                                                    part_ind='run', part_num=[1,2], cond_ind=['net_id'],
                                                    type=['Ico642Run'], hemis=None, smooth='4fwhm_binarized')

        for half in [1,2]:
            # data = [data1[half-1]] + [data[1]]
            data = [(d - np.mean(d, axis=1, keepdims=True)).astype(np.float32) for d in [data_all[half-1]]]
            cond_vec = [cond_vec_all[half-1]]
            part_vec = [part_vec_all[half-1]]
            subj_ind = [subj_ind_all[half-1]]
            n_subj = np.unique(np.concatenate(subj_ind, axis=0)).size

            toc = time.perf_counter()
            print(f'Done loading. Used {toc - tic:0.4f} seconds!')
            hut.report_cuda_memory()

            # p = 1,30,60,90,120; w = 0,30,60,90,120
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    print(f'prior strength is {p}; the MRF strength is {w} ...')
                    # m-RBM
                    ar_model = ar.build_arrangement_model(U.clone(), prior_type='prob', atlas=atlas,
                                                          sym_type='asym', model_type='cRBM_Wc',
                                                          Wc=Wc, theta=w, epos_iter=20, num_chain=n_subj)
                    ar_model.bu = ar_model.bu * p

                    # Independent
                    # ar_model = ar.build_arrangement_model(U.clone()*p, prior_type='logpi', atlas=atlas,
                    #                                       sym_type='asym', model_type='independent')

                    U_indiv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
                                                            cond_vec, part_vec, subj_ind, Vs=None,
                                                            sym_type='asym', n_iter=10 if ar_model.name == 'cRBM_Wc' else 200,
                                                            em_params={'num_subj': n_subj,
                                                                        'uniform_kappa': True,
                                                                        'subjects_equal_weight':True,
                                                                        'subject_specific_kappa': False,
                                                                        'parcel_specific_kappa': False})
                    Pindiv = pt.argmax(U_indiv, dim=1) + 1
                    np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set/prob' +
                            f'/asym_MdNiIbHc+HCPrest-1run{half}-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-{p}_spatial-{w}_{global_counter}.npy',
                            U_indiv.cpu().numpy())

                    del M
                    pt.cuda.empty_cache()
                    hut.report_cuda_memory()

                    ## Save indiv parcellation in cifti
                    T = pd.read_csv(get_subject_list_path(subj_list_file), sep='\t')
                    img = nt.make_label_cifti(Pindiv.T.cpu().numpy(), atlas.get_brain_model_axis(),
                                            column_names=[f'{i}' for i in T.participant_id],
                                            label_names=net_name, label_RGBA=colors)
                    nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                            f'/asym_MdNiIbHc+HCPrest-1run{half}-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-{p}_spatial-{w}_{global_counter}.dlabel.nii')

    # # Combine all test subjects parcellation
    # indiv_par = []
    # # colors[[1, 6]] = colors[[6, 1]]
    # for i in [1, 2, 3, 4]:
    #     par = nb.load('/home/dzhi/eris_mount/dzhi/Indiv_par/Models/Models_03/indiv_parcellation/HCP203_test_set' +
    #                 f'/asym_KONG2019+HCPrest-1run-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-2_spatial-2_{i}.dlabel.nii').get_fdata()[:]
    #     indiv_par.append(par)
    # indiv_par = np.vstack(indiv_par)
    # T = pd.read_csv(get_subject_list_path("HCP200_test.tsv"), sep='\t')
    # img = nt.make_label_cifti(indiv_par.T, atlas.get_brain_model_axis(),
    #                           column_names=[f'{i}' for i in T.participant_id],
    #                           label_names=net_name, label_RGBA=colors)
    # nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
    #         f'/asym_MdNiIbHc+HCPrest-2261s2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1.dlabel.nii')


    ####################################################################################################################
    ## Evaluation
    ######## Step 2. Load HCP test data for indiv parcellation
    ## Making distance metric
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    results = pd.DataFrame()
    for global_counter in [1, 2, 3, 4]:
        subj_list_file = f"HCP200_test_{global_counter}.tsv"
        T = pd.read_csv(get_subject_list_path(subj_list_file), sep='\t')

        print(f'Start loading data: HCP {global_counter} test data ...')
        tic = time.perf_counter()
        ## 1. HCP task contrasts
        # t_data, t_info = ut.load_hcp_contrasts(HCP_DIR, "/subj_list/HCP200_test_1.tsv", space='fs32k',
        #                                        return_positive=True, hemis=None, smooth='4_MSMAll')
        ## 2. HCP resting state time series
        # t_data = ut.load_hcp_timeseries(HCP_DIR, "subj_list/HCP203_test_set_filtered_1.tsv",
        #                             space=atlas.name, run_list=[2,3],
        #                             type='Tseries', hemis=None, smooth='4fwhm')

        ## 3. HCP task betas
        t_data, _, _, _, t_info = ut.build_hcp_datasets(HCP_DIR, f"/subj_list/{subj_list_file}",
                                                        atlas, ses_list=['ses-task'], join_sess=False, join_sess_part=False,
                                                        part_ind=['half'], part_num=None, cond_ind=['reg_id'],
                                                        type=['ZstatHalf'], hemis=None, smooth='4fwhm')
        # t_data = [t_data[1]]
        t_info = np.array_split(t_info, 2)[0]
        contrast_idx = t_info['positive'] == 1
        t_info = t_info[contrast_idx]
        t_data = [d[:,contrast_idx,:] for d in t_data]

        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        hut.report_cuda_memory()
        n_subj = t_data[0].shape[0]

        for half in [1,2]:
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    # Load indiv parcellation
                    Pindiv = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                            f'/asym_MdNiIbHc+task-half{half}-sm6zstat_masked-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-{p}_spatial-{w}.dlabel.nii').get_fdata()[:]
                    Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
                    # Load Kong2019 individual parcellations
                    # Pindiv = ut.get_kong2019_indiv_parcellations(ERIS_DIR + '/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
                    #                     HCP_DIR + "/subj_list/HCP203_test_set_filtered_1.tsv",
                    #                     w=100, c=30, num_sess=2)
                    Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

                    HCP200_list = pd.read_csv(get_subject_list_path("HCP200_test.tsv"), sep='\t')
                    idx = HCP200_list.index[HCP200_list['participant_id'].isin(T['participant_id'])]
                    Pindiv = Pindiv[idx]

                    # Making evaluation information
                    minfo = ge.make_eval_info(K, train_info=['HCP'], train_sess=f'run-{half}',
                                                tdata='HCP', test_sess='contrasts',
                                                model_type='Models_03', group_map_name='MdNiIbHc',
                                                test_kappa=None)

                    t_info['task_name']=[s.rstrip('2') for s in t_info.task_name]
                    this_res = pd.DataFrame()
                    # Evaluate on each run's time series
                    for r, td in enumerate([t_data[2-half]]):
                        if type(td) is np.ndarray:
                            td = pt.tensor(td, dtype=pt.get_default_dtype())

                        tasks_list = hcp_tasks + ['all']
                        for task in tasks_list:
                            if task == 'all':
                                idx = [True] * len(t_info)

                                # Individual evaluation
                                # homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:,idx,:])
                                zvalue_indiv = ev.calc_test_zvalue(Pindiv, td[:, idx, :], return_single=False)
                                np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set/zvalues' +
                                        f'/zvalue_indiv_asym_MdNiIbHc+HCPtask-half{half}-indiv_K-{K}_strengh-{p}_spatial-{w}_zstat-sm4_{global_counter}.npy',
                                        zvalue_indiv.cpu().numpy())
                                inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :], return_single=False)
                                inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                                np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set/inhomogeneity' +
                                        f'/inhomo_nets_asym_MdNiIbHc+HCPtask-half{half}-indiv_K-{K}_strengh-{p}_spatial-{w}_zstat-sm4_{global_counter}.npy',
                                        inhomo_nets.cpu().numpy())

                            else:
                                idx = t_info['task_name'] == task
                                idx = list(idx)

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

                            hut.report_cuda_memory()
                            inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:,idx,:], return_single=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            ## Factorize DCBC calcuation
                            hut.report_cuda_memory()
                            dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:,idx,:], dist, trim_nan=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            res['dcbc_indiv'] = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
                            # res['homo_indiv'] = homo_indiv.cpu()
                            res['inhomo_indiv'] = inhomo_indiv.cpu()
                            res['task_name'] = task
                            res['test_run'] = 3-half
                            res['train_smooth'] = "6fwhm"
                            res['test_smooth'] = None
                            res['test_type'] = 'contrasts'
                            this_res = pd.concat([this_res, res], ignore_index=True)

                    # QC
                    dice = [hev.dice_coefficient(Pgroup, Pindiv[i], label_matching=True).item()
                                    for i in range(Pindiv.shape[0])]
                    # ari = [hev.ARI(pt.argmax(U, dim=0), pt.argmax(U_indiv, dim=1)[i]).item()
                    #     for i in range(U_indiv.shape[0])]
                    # nmi = [1- hev.nmi(pt.argmax(U, dim=0).cpu(), pt.argmax(U_indiv, dim=1)[i].cpu())
                    #     for i in range(U_indiv.shape[0])]

                    this_res['dice_group'] = dice * len(tasks_list) * len([t_data[2-half]])
                    # res['ari_group'] = ari
                    # res['nmi_group'] = nmi
                    this_res['strength'] = p
                    this_res['spatial_w'] = w

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/eval_MdNiIbHc+RANDYrest-allrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2.tsv',
        index=False, sep='\t')
    print('Done')
    #     plt.figure(figsize=(20, 8))
    #     plot_multi_flat(U_indiv[0:10].cpu().numpy(), 'MNIAsymC2', grid=(2, 5),
    #                     cmap=colors, dtype='prob',
    #                     titles=["subj_{}".format(i+1) for i in range(10)])
    #     plt.show()




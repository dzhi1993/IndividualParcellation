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
import Functional_Fusion.util as fut
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.evaluation as hev
import HierarchBayesParcel.util as hut
import FusionModel.util as futil
import FusionModel.evaluate as ev

import IndividualParcellation.scripts.group_parcellation as gp
import IndividualParcellation.scripts.group_eval as ge
import scipy.io as spio
from pathlib import Path
from copy import deepcopy

import IndividualParcellation.utils as ut
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
from scripts.group_parcellation import ERIS_DIR

# from scripts.dual_regression import model_name
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

RANDY_DIR = '/home/dzhi/eris_mount/Tian/RANDY15'
if not Path(RANDY_DIR).exists():
    RANDY_DIR = '/data/tge/Tian/RANDY15'
if not Path(RANDY_DIR).exists():
    raise (NameError('Could not find RANDY_DIR'))

RES_DIR = '/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations'
if not Path(RES_DIR).exists():
    RES_DIR = '/data/tge/dzhi/Indiv_par/Evaluations'
if not Path(RES_DIR).exists():
    raise (NameError('Could not find RANDY_DIR'))

ERIS_DIR = '/home/dzhi/eris_mount'

if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    raise (NameError('Could not find RANDY_DIR'))

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : True
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

    df['contrast_name'] = t_info.iloc[contrast_idx].cond_name

    return df

def plot():
    df1 = pd.read_csv(
        RES_DIR + '/eval_MSHBM40+RANDYrest-1-10run_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv',
        sep='\t')
    df2 = pd.read_csv(
        RES_DIR + '/eval_MdNiIbHc+RANDYrest-1-10run_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv',
        sep='\t')
    df = pd.concat([df1, df2], ignore_index=False)

    for i in range(1,8):
        this_run = df.loc[df.train_runs == i]
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        sb.barplot(this_run.loc[this_run.group_map_name == 'MSHBM40'], x='strength', y='inhomo_indiv', hue='spatial_w')
        plt.ylim(0.68, 0.75)
        plt.title('MSHBM40')

        plt.subplot(2, 2, 2)
        sb.barplot(this_run.loc[this_run.group_map_name == 'MdNiIbHc'], x='strength', y='inhomo_indiv', hue='spatial_w')
        plt.ylim(0.68, 0.75)
        plt.title('MdNiIbHc')

        plt.subplot(2, 2, 3)
        sb.barplot(this_run.loc[this_run.group_map_name == 'MSHBM40'], x='strength', y='dcbc_indiv', hue='spatial_w')
        plt.title('MSHBM40')

        plt.subplot(2, 2, 4)
        sb.barplot(this_run.loc[this_run.group_map_name == 'MdNiIbHc'], x='strength', y='dcbc_indiv', hue='spatial_w')
        plt.title('MdNiIbHc')

        plt.suptitle(f'run_{i}')
        plt.tight_layout()
        plt.show()

def assemble_result(domains, tinfo, type_sets=[('','N-feature'),
                                               ('_equalweights','equal'),
                                               ('_task0.5rest0.5','task1rest1')],
                    prior_name='MSHBM40', strength=2, spatial=2):
    results = pd.DataFrame()
    contrast_names = tinfo.domain_abbr + '_' + tinfo.contrast_name
    for type, type_name in type_sets:
        # inhomo_indiv = []
        # for do in domains:
        inhomo_indiv = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/inhomogeneity' +
                       f'/inhomo_nets_asym_{prior_name}+RANDYtask-5run-indiv_K-15_strengh-{strength}_spatial-{spatial}_sm2.npy')
        #     inhomo_indiv.append(res)
        #
        # inhomo_indiv = np.hstack(inhomo_indiv)
        num_subj = inhomo_indiv.shape[0]
        for i in range(inhomo_indiv.shape[1]):
            df = pd.DataFrame({'atlas': ['fs32k'] * num_subj,
                               'domain': [tinfo.domain_abbr[i]] * num_subj,
                               'contrast': [tinfo.contrast_name[i]] * num_subj,
                               'subj_num': np.arange(num_subj),
                               'type': type_name,
                               'inhomo_indiv': inhomo_indiv[:,i]})
            results = pd.concat([results, df], ignore_index=True)

        df_all = pd.DataFrame({'atlas': ['fs32k'] * num_subj,
                           'domain': ['all'] * num_subj,
                           'contrast': ['all'] * num_subj,
                           'subj_num': np.arange(num_subj),
                           'type': type_name,
                           'inhomo_indiv': inhomo_indiv.mean(axis=1)})
        results = pd.concat([results, df_all], ignore_index=True)
        results['K'] = 15
        results['strength'] = strength
        results['spatial'] = spatial
        results['rest_runs'] = 5
        results['task_runs'] = 5

    return results

def integrate_parcels(group_strength_list, spatial_list):
    T = pd.read_csv(ERIS_DIR + f"/Tian/RANDY15/participants.tsv", sep='\t')
    for counter_p, p in enumerate(group_strength_list):
        for counter_w, w in enumerate(spatial_list):
            print(f'prior strength is {p}; the MRF strength is {w} ...')
            par = []
            for index, row in T.iterrows():
                ######## Step 1. Load subjects individual training data
                print(f'This is subject {row["participant_id"]}')
                this_par = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set' +
                        f'/asym_MdNiIbHc+RANDYrest-allrun-indiv_space-fs32k_K-{K}_{fc_type}_groupstrengh-{p}_spatial-{w}_{row["participant_id"]}.dlabel.nii')
                par.append(this_par.get_fdata())

            par = np.vstack(par)
            img = nt.make_label_cifti(par.T, atlas.get_brain_model_axis(),
                                    column_names=T.participant_id.tolist(),
                                    label_names=net_name, label_RGBA=colors)
            nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set' +
                    f'/asym_MdNiIbHc+RANDYrest-allrun-indiv_space-fs32k_K-{K}_{fc_type}_groupstrengh-{p}_spatial-{w}.dlabel.nii')


if __name__ == "__main__":
    # plot()
    atlas, am_info = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    # DEVICE = 'cpu'
    training_ses = 'all'
    test_ses = 'ses-rest1'
    fc_type = 'Ico642Run'
    ext = '_binarized'
    K=15
    group_strength_list = [1]
    spatial_list = [0]
    global_run_list = [1]
    randy_tasks = ['EPROJ', 'LANG', 'MOTOR', 'NBACK', 'TOM', 'VISME', 'VODDK']
    randy_good_subjlist = [1,2,3,5,6,7,8,11,12,13,14]
    randy_bad_subjlist = [0,4,9,10]

    ######## Step 2. Generate group / indiv parcellations
    ## laod Kong 2019 17net - HCP40
    # align, net_name, colors = ut.get_kong2019_group_parcellation()
    # align = pt.tensor(align, dtype=pt.get_default_dtype(), device=DEVICE)

    # Load DU15 networks
    DU, net_name, colors = gp.get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    DU15 = ar.expand_mn_1d(DU, K=16)
    align = DU15[1:,:]
    nanidx = pt.where(align.sum(dim=0)==0)[0]

    ## Load the group prior from a pre-trained model
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_DU15-inits' # resting 15
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_DU15-inits' # task 15
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800' # resting 17
    # model_name = f'/Models_03/task_fusion/asym_MdNiIbWmDeSoHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess_equalweights'  # task 17
    U, _ = hut.load_group_parcellation(MODEL_DIR + model_name, device=DEVICE)
    # Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE) # list of N*K matrix
    # U = atlas.cifti_to_data(ERIS_DIR + '/dzhi/workspace/res/priors/MSHBM_group_prior_HCP40training_k-15.dscalar.nii') # MSHBM 15
    # U = pt.tensor(np.nan_to_num(U), dtype=pt.get_default_dtype(), device=DEVICE)
    # Align with prior
    indx = hev.matching_greedy(align, pt.softmax(U, dim=0).to(pt.get_default_dtype()))
    U = U[indx, :]
    U = pt.softmax(U, dim=0)
    # Vs = [v[:,indx] for v in Vs]
    Pgroup = pt.argmax(U, dim=0) + 1
    # U[:, nanidx] = 0

    ## Determine the connectivity profile for mRBM
    hut.report_cuda_memory()
    T = pd.read_csv(ERIS_DIR + f"/Tian/RANDY15/participants.tsv", sep='\t')




    ####################################################################################################################
    ## Evaluation
    ######## Step 2. Load HCP test data for indiv parcellation
    print(f'Start loading test data: RANDY - {test_ses} - {fc_type} ...')
    tic = time.perf_counter()
    ## 3. RANDY task contrasts
    # t_data, t_info = ut.load_randy_contrasts(space='fs32k', subj=randy_good_subjlist, hemis=None, smooth=2)
    t_data, t_info = ut.load_randy_contrasts_runwise(run_idx=[0,1], space='fs32k',
                                                     subj=randy_good_subjlist, hemis=None, smooth=2)

    ## 4. RANDY resting timeseries
    # t_data = ut.build_resting_data('RANDY15', space='fs32k',
    #                               ses_list=[f'ses-rest{sn}' for sn in range(6, 8)],
    #                               type='Tseries', subj=randy_bad_subjlist, hemis=None, smooth=None)
    # t_info = pd.DataFrame({'task_name': ['REST'] * t_data[0].shape[1]})

    ## 5. RANDY task betas
    # t_data, _, _, _, t_info = ut.build_dataset('RANDY15', atlas=atlas.name,
    #                                             sess=randy_tasks, cond_ind='reg_id',
    #                                             type='CondRun', part_ind='sn', subj=randy_good_subjlist,
    #                                             part_num=np.arange(1,2), join_sess=False,
    #                                             join_sess_part=False, smooth=None)
    # t_info = t_info[0]

    toc = time.perf_counter()
    print(f"Done loading. Used {time.strftime('%M:%S', time.gmtime(toc - tic))} (MM:SS)!")
    hut.report_cuda_memory()
    # n_subj = t_data[0].shape[0]
    # df_MSHBM40 = assemble_result(randy_tasks, t_info)
    # df_MSHBM40 = df_MSHBM40[df_MSHBM40['subj_num'].isin(randy_good_subjlist)]
    # df_HBPHc = assemble_result(randy_tasks, t_info, type_sets=[('','task')],
    #                 prior_name='MdNiIbHc', strength=1, spatial=1)
    # df_HBPHc = df_HBPHc[df_HBPHc['subj_num'].isin(randy_good_subjlist)]
    #
    # ## Making distance metric
    # dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    # results = pd.DataFrame()
    # for run_indx in [1,2]:
    #     n_subj = t_data[0].shape[0]
    #     for counter_p, p in enumerate(group_strength_list):
    #         for counter_w, w in enumerate(spatial_list):
    #             print(f'Evaluating {run_indx} runs data, prior strength {p}, the spatial w {w} ...')
    #             # Load indiv parcellation
    #             Pindiv = np.stack([fut.mask_data_by_percent(dat, high_percent=0.2, low_percent=0.0, binarized=True)
    #                       for dat in t_data[run_indx-1]])
    #
    #             Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
    #             Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)
    #
    #             # Making evaluation information
    #             minfo = ge.make_eval_info(K, train_info=['RANDY'], train_sess=f'run-{run_indx}',
    #                                         tdata='RANDY', test_sess='contrasts',
    #                                         model_type='Models_03', group_map_name='MdNiIbHc',
    #                                         test_kappa=None)
    #
    #             t_info['task_name'] = t_info['task_name'] if 'task_name' in t_info \
    #                                         else t_info.get('domain_abbr', None)
    #             t_info['task_name']=[s.rstrip('2') for s in t_info.task_name]
    #             t_info['cond_name'] = t_info['cond_name'] if 'cond_name' in t_info \
    #                 else t_info.get('contrast_name', None)
    #             t_info = t_info.reset_index(drop=True)
    #             this_res = pd.DataFrame()
    #             # Evaluate on each run's time series
    #             for r, td in enumerate(t_data[:run_indx-1] + t_data[run_indx:]):
    #                 if type(td) is np.ndarray:
    #                     td = pt.tensor(td, dtype=pt.get_default_dtype())
    #
    #                 for i, cond in enumerate(t_info['cond_name']):
    #                     idx = t_info['cond_name'] == cond
    #
    #                     ## Z-value / task inhomogeneity (per network)
    #                     zvalue_indiv = ev.calc_test_zvalue(Pindiv[:,i,:], td[:,idx,:], return_single=True)
    #                     # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                     #         f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest3387s-indiv_K-{K}_strengh-{p}_spatial-{w}_contrasts-loo{global_counter}_sm2_11sub.npy',
    #                     #         zvalue_indiv.cpu().numpy())
    #
    #                     res = pd.DataFrame({'atlas': [minfo.atlas] * n_subj,
    #                             'K': [minfo.K] * n_subj,
    #                             'train_data': [minfo.datasets] * n_subj,
    #                             'train_sess': [minfo.train_sess] * n_subj,
    #                             'test_data': [minfo.test_data] * n_subj,
    #                             'test_type': [minfo.test_sess] * n_subj,
    #                             'model_type': [minfo.model_type] * n_subj,
    #                             'group_map_name': [minfo.group_map_name] * n_subj,
    #                             'subj_num': np.arange(n_subj)})
    #
    #                     res['cond_name'] = cond
    #                     res['task_name'] = t_info.iloc[i]['task_name']
    #                     res['test_run'] = r + 1
    #                     res['train_smooth'] = "2fwhm"
    #                     res['test_smooth'] = None
    #                     res['z_value'] = zvalue_indiv.cpu()
    #                     this_res = pd.concat([this_res, res], ignore_index=True)
    #
    #             this_res['strength'] = p
    #             this_res['spatial_w'] = w
    #             this_res['train_runs'] = run_indx
    #
    #             results = pd.concat([results, this_res], ignore_index=True)
    #
    # results.to_csv(
    #     RES_DIR + f'/zvalue_RANDYtask-contrast_masked_hi-0.2_indiv_test_on_RANDYtask-contrast_sm2_11subjects.tsv',
    #     index=False, sep='\t')
    # print('Done')

    df_thres = pd.read_csv('/home/dzhi/eris_mount/dzhi/Indiv_par/Results/section_5/RANDY'
                           + f'/zvalue_RANDYtask-5contrast_masked_hi-0.2_indiv_test_on_RANDYtask-5contrast_sm2_11subjects.tsv', sep='\t')
    df_thres = df_thres.groupby(["subj_num","task_name","cond_name"])["z_value"].mean().reset_index()
    df_thres = df_thres.rename(columns={"subj_num": "subject",
                                  "task_name": "domain",
                                  "cond_name": "contrast_name"})
    df_thres["parcel"] = "thres_0.2"
    df_thres["dtype"] = "naive_threshold"

    ## Mean z-value
    res1 = []
    train_run = [1, 2, 3]
    for p in [1]:
        for r in train_run:
            test_run = train_run.copy()
            test_run.remove(r)
            for tr in test_run:
                res1.append(
                    np.load(f'{ERIS_DIR}/dzhi/Indiv_par/Models/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                            f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest2run+task1run{r}-indiv_K-15_strengh-{p}_spatial-0_betas-run{tr}_sm0_11sub.npy'))

    res1 = np.mean(np.stack(res1), axis=0)

    d_type = ['F+fusion']
    # color_list = [tuple(color) for color in colors[1:, :]]
    #
    # res1 = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                f'/zvalue_DU-indiv_K-15.npy')[randy_good_subjlist]
    #
    # res2 = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                f'/zvalue_indiv_asym_HBPHc+RANDYrest-{num_run}run-indiv_K-15_strengh-2_spatial-0_sm2.npy')[randy_good_subjlist]
    #
    # res3 = np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest-{num_run}run-indiv_K-15_strengh-2_spatial-0_sm2.npy')[randy_good_subjlist]

    df_all = pd.DataFrame()
    # _, t_info = ut.load_randy_contrasts(subj=None, hemis='L', smooth=2)
    # t_info['task_name'] = t_info['domain']
    for j, res in enumerate([res1]):
        for i in range(t_info.shape[0]):
            df = plot_zvalues(res, i, t_info, parcel_name=net_name[1:])
            df['dtype'] = d_type[j]
            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all = pd.concat([df_all, df_thres], ignore_index=True)
    con_nam = df_all['contrast_name'].unique()
    plt.figure(figsize=(10, 5 * len(con_nam)))
    for n, c_name in enumerate(con_nam):
        this_df_all = df_all.loc[df_all['contrast_name'] == c_name]
        domain = this_df_all['domain'].unique()

        plt.subplot(len(con_nam), 1, n + 1)
        sb.barplot(data=this_df_all, x='parcel', y='z_value', hue='dtype', errorbar='se', width=0.7)
        sb.stripplot(
            data=this_df_all,
            x='parcel', y='z_value', hue='dtype',
            dodge=True, alpha=0.6, jitter=False
        )

        # for parcel in this_df_all['parcel'].unique():
        #     sub_df = this_df_all[this_df_all['parcel'] == parcel]
        #     for subj in sub_df['subject'].unique():
        #         subj_df = sub_df[sub_df['subject'] == subj].sort_values('dtype')
        #         plt.plot(
        #             [parcel] * len(subj_df),
        #             subj_df['z_value'].values,
        #             color='gray', alpha=0.3, linewidth=1,
        #             zorder=0
        #         )

        plt.xticks(rotation=45)
        plt.title(f'{domain}: {c_name}')

    plt.tight_layout()
    plt.suptitle(f'{num_run} runs, group priors: MSHBM, HBP_rest, HBP_fusion', y=1.05)
    plt.savefig(RES_DIR + f'/zvalues_du_vs_fusion.pdf', format='pdf')
    plt.show()


    # ## Per train run
    # for t_run in [1,5,10,15,'all']:
    #     plt.figure(figsize=(15, 10))
    #     plt.subplot(2, 3, 1)
    #     sb.barplot(df_mshbm.loc[df_mshbm["train_runs"] == str(t_run)], x='strength', y='inhomo_indiv', hue='spatial_w')
    #     plt.title(f'mshbm, {t_run} runs')
    #     plt.ylim(0.68, 0.76)
    #
    #     plt.subplot(2, 3, 2)
    #     sb.barplot(df_rest.loc[df_rest["train_runs"]==t_run], x='strength', y='inhomo_indiv', hue='spatial_w')
    #     plt.title(f'rest-only, {t_run} runs')
    #     plt.ylim(0.68, 0.76)
    #
    #     plt.subplot(2, 3, 3)
    #     sb.barplot(df_fusion.loc[df_fusion["train_runs"]==t_run], x='strength', y='inhomo_indiv', hue='spatial_w')
    #     plt.title(f'fusion, {t_run} runs')
    #     plt.ylim(0.68, 0.76)
    #
    #     plt.subplot(2, 3, 4)
    #     sb.barplot(df_mshbm.loc[df_mshbm["train_runs"] == str(t_run)], x='strength', y='dcbc_indiv', hue='spatial_w')
    #     plt.title(f'mshbm, {t_run} runs')
    #
    #     plt.subplot(2, 3, 5)
    #     sb.barplot(df_rest.loc[df_rest["train_runs"]==t_run], x='strength', y='dcbc_indiv', hue='spatial_w')
    #     plt.title(f'rest-only, {t_run} runs')
    #
    #     plt.subplot(2, 3, 6)
    #     sb.barplot(df_fusion.loc[df_fusion["train_runs"]==t_run], x='strength', y='dcbc_indiv', hue='spatial_w')
    #     plt.title(f'fusion, {t_run} runs')
    #
    #     plt.tight_layout()
    #     plt.show()

    #     plt.figure(figsize=(20, 8))
    #     plot_multi_flat(U_indiv[0:10].cpu().numpy(), 'MNIAsymC2', grid=(2, 5),
    #                     cmap=colors, dtype='prob',
    #                     titles=["subj_{}".format(i+1) for i in range(10)])
    #     plt.show()



    # ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
    #                                       sym_type='asym')
    # U_indv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
    #                                          cond_vec, part_vec, subj_ind,
    #                                          sym_type='asym',
    #                                          em_params={'num_subj': data[0].shape[0],
    #                                                     'uniform_kappa': True,
    #                                                     'subjects_equal_weight':True,
    #                                                     'subject_specific_kappa': False,
    #                                                     'parcel_specific_kappa': False})

    # em_params={'subjects_equal_weight':True,
    #             'uniform_kappa': None,
    #             'subject_specific_kappa': False,
    #             'parcel_specific_kappa': True}

    # del data
    # pt.cuda.empty_cache()
    # fm.report_cuda_memory()

    # ######## Step 3: Evaluate individual maps using DCBC
    # # Step 3.1: compute the distance matrix
    # dist = ev.compute_dist(atlas.world.T, resolution=1)
    # # Step 3.2: Gatering all necessary information for evaluation
    # eval_info = make_eval_info(M, train_info=['UKB'], train_sess='ses-rest1',
    #                         tdata='UKB', test_sess='ses-rest2',
    #                         model_type='Models_04', group_map_name='Buckner7',
    #                         test_kappa=None)
    # # Step 3.3: Do DCBC evaluation on the second half data
    # res = eval_parcel_DCBC(U, U_indv, t_data[0], dist, eval_info,
    #                         out_file='eval_dcbc_indiv_Buckner7_k-7_model-04_test.tsv')
    # dice = [hev.dice_coefficient(pt.tensor(U_hard), pt.argmax(U_indv, dim=1)[i])
    #         for i in range(U_indv.shape[0])]
    # # res.to_csv(f'eval_dcbc_indiv_Buckner7_k-7_model-04_test2_prior.tsv', index=False, sep='\t')

    # ######## Step 4: Visualization
    # # Step 4.1 (optional): plot the DCBC results
    # ev_df = pd.read_csv('eval_dcbc_indiv_parcellations.tsv', sep='\t')
    # plt.figure(figsize=(5, 5))
    # df = pd.melt(ev_df, var_name='group', value_name='value')
    # df = df.loc[(df['group'] == 'dcbc_group') | (df['group'] == 'dcbc_indiv')]
    # sb.barplot(x='group', y='value', errorbar="se", width=0.7, data=df)
    # plt.show()

    # # Step 4.2: plot group parcellation
    # plt.figure(figsize=(10, 10))
    # plot_multi_flat(U.unsqueeze(0).cpu().numpy(), 'MNIAsymC2', grid=(1, 1),
    #                 cmap='tab20', dtype='prob', titles=['group prior'])
    # plt.show()

    # # Step 4.3: plot individual parcellation
    # plt.figure(figsize=(40,20))
    # plot_multi_flat(U_indv.cpu().numpy(), 'MNIAsymC2', grid=(2, 5),
    #                 cmap='tab20', dtype='prob',
    #                 titles=["subj_{}".format(i+1) for i in range(U_indv.shape[0])])
    # plt.show()
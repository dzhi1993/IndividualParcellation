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
    # group_strength_list = np.linspace(0, 2, 21).round(1).tolist() + np.linspace(2, 10, 17).tolist()
    # group_strength_list = [x for x in group_strength_list if x not in [1,2,5,10]]
    group_strength_list = [0,0.2,0.4,0.6,0.8,1,2,3,4,5,6,7,8,9,10]
    spatial_list = [0]
    global_run_list = [1,2,3,4,5]
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
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_DU15-inits' # fusion 15
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
    Wc = pt.load(ERIS_DIR + '/Tian/UKBB_full/imaging/Atlases/tpl-fs32k/fs32k_neighbours.pt', weights_only=True)
    hut.report_cuda_memory()
    T = pd.read_csv(ERIS_DIR + f"/Tian/RANDY15/participants.tsv", sep='\t')

    ## Find optimal task battery
    maps = nb.load(
        '/home/dzhi/eris_mount/Tian/RANDY15/derivatives/group/data/group_space-fs32k_ses-task_CondAll_masked-hi0.1lo0.1_binarized.dscalar.nii').get_fdata()
    # maps = nb.load(
    #     '/home/dzhi/eris_mount/Tian/RANDY15/derivatives/group/data/group_space-fs32k_ses-task_CondAll.dscalar.nii').get_fdata()

    # for num_tasks in range(9, 10):
    #     # best_score = -np.inf
    #     # task_idx = None
    #     # for i in range(maps.shape[0]):
    #     #     best, cover, overlap = ut.task_beta_selection(maps, num_tasks, lamda=0, selected=[i])
    #     #     score = cover - 0 * overlap
    #     #
    #     #     # best, score = ut.greedy_select_max_trace_inverse_cov(maps, num_tasks, selected=[i],
    #     #     #                                                      z_transfer=False)
    #     #     if score > best_score:
    #     #         best_score = score
    #     #         task_idx = best
    #
    #     task_idx, best_score = ut.greedy_task_beta_selection(maps, num_tasks, lamda=0)
    #     print(f"Best subsets are {task_idx}, with coverage {best_score}")

    for s in T.participant_id[randy_good_subjlist]:
        print(s)
        for global_counter in global_run_list:
            # task_names = randy_tasks.copy()
            # task_names.remove(global_counter)
            ######## Step 1. Load subjects individual training data
            print(f'Start loading data {global_counter} run: RANDY resting - {training_ses}, {fc_type} {ext} ...')
            tic = time.perf_counter()
            ## RANDY15 resting data
            data = ut.build_resting_data('RANDY15', space='fs32k',
                                          ses_list=[f'ses-rest{i}' for i in range(1, global_counter+1)],
                                          type='Ico642Run', subj=s, hemis=None, smooth='binarized-0.1')
            data = [(d - np.mean(d, axis=1, keepdims=True)).astype(np.float32) for d in data]
            # cond_vec1 = [np.tile(np.arange(1,1211), len(data))]
            # part_vec1 = [np.repeat(np.arange(1, len(data)+1), 1210)]
            cond_vec = [np.arange(1, 1211)] * len(data)
            part_vec = [None] * len(data)
            # data = [np.concatenate(data, axis=1)]
            subj_ind = [np.arange(dat.shape[0]) for dat in data]

            ## RANDY15 task data
            # data, cond_vec, part_vec, subj_ind, infos = ut.build_dataset('RANDY15', atlas=atlas.name,
            #                                                         sess=randy_tasks, cond_ind='reg_id',
            #                                                         type='CondRun', part_ind='sn', subj=randy_good_subjlist,
            #                                                         part_num=np.arange(2, 2+1), join_sess=True,
            #                                                         join_sess_part=False, smooth='4fwhm_zstat_masked-hi0.1lo0.1')
            # indices = [np.where(np.isin(cond, np.array(task_idx) + 1))[0] for cond in cond_vec]
            # for i, idx in enumerate(indices):
            #     data[i] = data[i][:,idx,:]
            #     cond_vec[i] = cond_vec[i][idx]
            #     part_vec[i] = part_vec[i][idx]

            # data = data1 + data
            # cond_vec = cond_vec1 + cond_vec
            # part_vec = part_vec1 + part_vec
            # subj_ind = subj_ind1 + subj_ind
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
                    # ar_model = ar.build_arrangement_model(U.clone(), prior_type='logpi', atlas=atlas,
                    #                                       sym_type='asym', model_type='independent')
                    # ar_model.logpi = ar_model.logpi * p

                    # ll_all = []
                    # max_ll = float("-inf")
                    # best_Pindiv = None
                    # best_U_indiv = None
                    # for mini_iter in range(20):
                    U_indiv, ll, M = fm.get_indiv_parcellation(deepcopy(ar_model), atlas, data,
                                                            cond_vec, part_vec, subj_ind, Vs=None,
                                                            sym_type='asym', n_iter=200 if ar_model.name == 'cRBM_Wc' else 200,
                                                            em_params={'num_subj': n_subj,
                                                                        'uniform_kappa': True,
                                                                        'subjects_equal_weight':True,
                                                                        'subject_specific_kappa': False,
                                                                        'parcel_specific_kappa': False})

                    Pindiv = pt.argmax(U_indiv, dim=1) + 1
                        # this_ll = pt.stack([hev.dice_coefficient(pt.tensor(DU, dtype=pt.int),
                        #                                             Pindiv[i], label_matching=True)
                        #               for i in range(Pindiv.shape[0])]).mean()
                        # this_ll = ll[-1]
                        # if float(this_ll) > max_ll:
                        #     print(f'Current best - iteration {mini_iter} ..')
                        #     max_ll = float(this_ll)
                        #     best_Pindiv = Pindiv
                        #     best_U_indiv = U_indiv

                    # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                    #         f'/asym_HBPHc+RANDYrest{global_counter}run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}.npy',
                    #         U_indiv.cpu().numpy())
                    del M
                    # ll_all.append(ll)
                    pt.cuda.empty_cache()
                    hut.report_cuda_memory()

                    ## Save indiv parcellation in cifti
                    img = nt.make_label_cifti(Pindiv.T.cpu().numpy(), atlas.get_brain_model_axis(),
                                            column_names=[s],
                                            label_names=net_name, label_RGBA=colors)
                    nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/subjects' +
                            f'/asym_HBPHc+RANDYrest-{global_counter}run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_{s}.dlabel.nii')


    #### Generating border files
    # for prior in ["MSHBM40"]:
    #     for subj in range(1,16):
    #         for hem in ['L', 'R']:
    #             ut.make_border(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set' +
    #                            f'/asym_{prior}+RANDYrest-1run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-2_spatial-0.dlabel.nii',
    #                            f'sub-{subj:02d}', hem=hem,
    #                            outfile=ERIS_DIR + f'/dzhi/Indiv_par/Results/section_4/RANDY/example_overlap_1runs/sub-{subj:02d}_{prior}_1run_border_{hem}.border')


    ####################################################################################################################
    ## Evaluation
    ######## Step 2. Load HCP test data for indiv parcellation
    ## Making distance metric
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)

    results = pd.DataFrame()
    for s in T.participant_id[randy_good_subjlist]:
        print(s)
        print(f'Start loading test data: RANDY - {test_ses} - {fc_type} ...')
        tic = time.perf_counter()
        ## 3. RANDY task contrasts
        t_data, t_info = ut.load_randy_contrasts(space='fs32k', subj=s, hemis=None, smooth=2)
        # t_data, t_info = ut.load_randy_contrasts_wo_run(run_exclude=1, space='fs32k', subj=randy_good_subjlist,
        #                                                 hemis=None, smooth=2)

        ## 4. RANDY resting timeseries
        # t_data = ut.build_resting_data('RANDY15', space='fs32k',
        #                               ses_list=[f'ses-rest{sn}' for sn in range(6, 8)],
        #                               type='Tseries', subj=randy_good_subjlist, hemis=None, smooth=None)
        # t_info = pd.DataFrame({'task_name': ['REST'] * t_data[0].shape[1]})

        ## 5. RANDY task betas
        # t_data, _, _, _, t_info = ut.build_dataset('RANDY15', atlas=atlas.name,
        #                                             sess=randy_tasks, cond_ind='reg_id',
        #                                             type='CondRun', part_ind='sn', subj=randy_good_subjlist,
        #                                             part_num=np.arange(1,4), join_sess=False,
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


        for global_counter in global_run_list:
            # t_data, t_info = ut.load_randy_contrasts_wo_run(run_exclude=global_counter, space='fs32k', subj=randy_good_subjlist,
            #                                                 hemis=None, smooth=2)
            n_subj = t_data[0].shape[0]
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    print(f'Evaluating {global_counter} runs data, prior strength {p}, the spatial w {w} ...')
                    # Load indiv parcellation
                    Pindiv = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/subjects' +
                            f'/asym_MdNiIbHc+RANDYrest-{global_counter}run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_{s}.dlabel.nii').get_fdata()
                    # Pindiv = Pindiv[randy_good_subjlist] if Pindiv.shape[0] == 15 else Pindiv

                    # Load DU15 individual parcellations
                    # Pindiv_du = nb.load(ERIS_DIR + '/dzhi/workspace/DU15NET/Harvard_Precision_Neuroimaging_Group_gz_zK166' +
                    #                  '/DU15_fs32k_individual_parcellation_wb.dlabel.nii').get_fdata()[randy_good_subjlist]
                    # Pindiv_du = pt.tensor(Pindiv_du, dtype=pt.get_default_dtype(), device=DEVICE)

                    Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)
                    Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

                    # Making evaluation information
                    minfo = ge.make_eval_info(K, train_info=['RANDY'], train_sess=f'run-{global_counter}',
                                                tdata='RANDY', test_sess='contrasts',
                                                model_type='Models_03', group_map_name='MdNiIbHc',
                                                test_kappa=None)

                    t_info['task_name'] = t_info['task_name'] if 'task_name' in t_info \
                                                else t_info.get('domain_abbr', None)
                    t_info['task_name']=[s.rstrip('2') for s in t_info.task_name]
                    t_info = t_info.reset_index(drop=True)
                    this_res = pd.DataFrame()
                    # Evaluate on each run's time series
                    for r, td in enumerate(t_data):
                        # if r == global_counter-1:
                        #     continue

                        if type(td) is np.ndarray:
                            td = pt.tensor(td, dtype=pt.get_default_dtype())

                        tasks_list = ['all']
                        for task in tasks_list:
                            if task == 'all':
                                idx = [True] * len(t_info)
                                # ## Soft Dice overalp
                                # soft_dice_indiv = ev.calc_test_soft_dice(Pindiv, td[:,idx,:], return_single=False)
                                # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/soft_dice' +
                                #         f'/soft_dice_indiv_asym_MdNiIbHc+RANDYrest-{global_counter}run-indiv_K-{K}_strengh-{p}_spatial-{w}_Tseires_sm2_11sub.npy',
                                #         soft_dice_indiv.cpu().numpy())
                                #
                                # ## Z-value / task inhomogeneity (per network)
                                # zvalue_indiv = ev.calc_test_zvalue(Pindiv, td[:,idx,:], return_single=False)
                                # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                                #         f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest-{global_counter}run-indiv_K-{K}_strengh-{p}_spatial-{w}_sm2_11sub.npy',
                                #         zvalue_indiv.cpu().numpy())
                                # inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :], z_transfer=False,
                                #                                               return_single=False)
                                # inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                                # np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/inhomogeneity' +
                                #         f'/inhomo_nets_asym_MdNiIbHc+RANDYrest-{global_counter}run-indiv_K-{K}_strengh-{p}_spatial-{w}_sm2_11sub.npy',
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
                                    'subj_num': s,
                                    'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj})

                            ## Factorize DCBC calcuation
                            hut.report_cuda_memory()
                            dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:,idx,:], dist, trim_nan=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            ### Rest homogeneity / task inhomogeneity (global)
                            hut.report_cuda_memory()
                            # homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:,idx,:])
                            inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:,idx,:], return_single=True)
                            pt.cuda.empty_cache()
                            hut.report_cuda_memory()

                            res['dcbc_indiv'] = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
                            # res['homo_indiv'] = homo_indiv.cpu()
                            res['inhomo_indiv'] = inhomo_indiv.cpu()
                            res['task_name'] = task
                            res['test_run'] = r + 1
                            res['train_smooth'] = "2fwhm"
                            res['test_smooth'] = None
                            res['test_type'] = 'contrasts'
                            this_res = pd.concat([this_res, res], ignore_index=True)

                    # QC
                    dice_group = [hev.dice_coefficient(Pgroup, Pindiv[i], label_matching=True).item()
                                    for i in range(Pindiv.shape[0])]
                    # dice_indiv = [hev.dice_coefficient(Pindiv_du[i], Pindiv[i], label_matching=True).item()
                    #               for i in range(Pindiv.shape[0])]
                    # ari = [hev.ARI(pt.argmax(U, dim=0), pt.argmax(U_indiv, dim=1)[i]).item()
                    #     for i in range(U_indiv.shape[0])]
                    # nmi = [1- hev.nmi(pt.argmax(U, dim=0).cpu(), pt.argmax(U_indiv, dim=1)[i].cpu())
                    #     for i in range(U_indiv.shape[0])]

                    this_res['dice_group'] = dice_group * (len(t_data)) * len(tasks_list)
                    # this_res['dice_indiv'] = dice_indiv * (len(t_data)) * len(tasks_list)
                    # res['ari_group'] = ari
                    # res['nmi_group'] = nmi
                    this_res['strength'] = p
                    this_res['spatial_w'] = w
                    this_res['train_runs'] = global_counter

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/eval_MSHBM40+RANDYrest-1to15run_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv',
        index=False, sep='\t')
    print('Done')

    ## Plot group-prior comparison by different runs
    # df_du = pd.read_csv(RES_DIR + '/eval_DU15_indiv_test_on_RANDYtask-contrast_sm2_11subjects.tsv', sep='\t')
    # df_rest = pd.read_csv(RES_DIR + '/eval_HBPHc+RANDYrest-1to15run_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv', sep='\t')
    # df_fusion = pd.read_csv(RES_DIR + '/eval_MdNiIbHc+RANDYrest-1to15run_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv',sep='\t')
    # df_rest_all = pd.read_csv(RES_DIR + '/eval_HBPHc+RANDYrest-allrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv', sep='\t')
    # df_fusion_all = pd.read_csv(RES_DIR + '/eval_MdNiIbHc+RANDYrest-allrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv', sep='\t')
    # df_rest = pd.concat([df_rest, df_rest_all], ignore_index=True)
    # df_fusion = pd.concat([df_fusion, df_fusion_all], ignore_index=True)
    # df_mshbm = pd.read_csv(RES_DIR + '/eval_MSHBM40+RANDYrest-1toallrun_K-15_indiv-mRBM_test_on_RANDYtask-contrast_sm2_11subjects.tsv', sep='\t')
    # #
    # df_rest = df_rest.loc[((df_rest["strength"] == 1) | (df_rest["strength"] == 2) |
    #                        (df_rest["strength"] == 5)) & (df_rest["spatial_w"] == 0)]
    # df_fusion = df_fusion.loc[((df_fusion["strength"] == 1) | (df_fusion["strength"] == 2) |
    #                            (df_fusion["strength"] == 5)) & (df_fusion["spatial_w"] == 0)]
    # df_mshbm = df_mshbm.loc[((df_mshbm["strength"] == 1) | (df_mshbm["strength"] == 2) |
    #                            (df_mshbm["strength"] == 5)) & (df_mshbm["spatial_w"] == 0)]
    # df_rest['group_map_name'] = 'rest'
    # df_fusion['group_map_name'] = 'fusion'
    # from scipy import stats
    # a = df_rest.groupby(['subj_num', 'train_runs']).mean(numeric_only=True).reset_index()
    # b = df_fusion.groupby(['subj_num', 'train_runs']).mean(numeric_only=True).reset_index()
    #
    # print(stats.ttest_rel(df_du.dcbc_indiv, b.loc[b["train_runs"]=='all'].dcbc_indiv))
    # print(stats.ttest_rel(df_du.inhomo_indiv, b.loc[b["train_runs"]=='all'].inhomo_indiv))
    #
    # df = pd.concat([df_mshbm, df_rest, df_fusion], ignore_index=False)
    # sb.barplot(df, x='train_runs', y='inhomo_indiv', hue='group_map_name')
    # plt.ylim(0.68, 0.75)
    # plt.show()
    # sb.barplot(df, x='train_runs', y='dcbc_indiv', hue='group_map_name')
    # plt.show()

    ## Mean z-value
    t_data, t_info = ut.load_randy_contrasts_wo_run(run_exclude=1, space='fs32k', subj=randy_good_subjlist,
                                                    hemis=None, smooth=2)
    dataset = ds.DataSetRANDY15(RANDY_DIR)
    for i, s in enumerate(T.iloc[randy_good_subjlist].participant_id):
        C = atlas.data_to_cifti(t_data[0][i], t_info.contrast_name)
        nb.save(C, dataset.contrast_dir.format(s) + f'/{s}_5Contrasts_fs32k_sm2_meanZmap_exclude-run1.dscalar.nii')

    res1, res2, res3 = [],[],[]
    for i in [1]:
        res1.append(np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                            f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest-2run-indiv_K-15_strengh-1_spatial-0_contrasts-loo{i}_sm2_11sub.npy'))
        res2.append(np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                            f'/zvalue_indiv_asym_MdNiIbHc+RANDYrest2run+task1run{i}-indiv_K-15_strengh-1_spatial-0_contrasts-loo_sm2_11sub.npy'))
        res3.append(np.load(MODEL_DIR + '/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                            f'/zvalue_du-indiv_K-15_strengh-1_spatial-0_contrasts-loo{i}_sm2_11sub.npy'))

    res1 = np.mean(np.stack(res1), axis=0)
    res2 = np.mean(np.stack(res2), axis=0)
    res3 = np.mean(np.stack(res3), axis=0)

    num_run = 2
    d_type = ['F+rest', 'F+fusion', 'DU15']
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
    for j, res in enumerate([res1, res2, res3]):
        for i in range(5):
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


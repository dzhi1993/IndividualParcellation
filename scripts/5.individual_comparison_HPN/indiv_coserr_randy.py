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
from copy import deepcopy
import IndividualParcellation.scripts.group_eval as ge
import IndividualParcellation.scripts.group_parcellation as gp

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
pt.cuda.is_available = lambda : True
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

def trim_sparse_tensor(sparse_tensor, rows_to_remove, cols_to_remove):
    """
    Trims the given sparse tensor by removing specified rows and columns.

    Parameters:
    sparse_tensor (torch.sparse_coo_tensor or torch.sparse_csr_tensor): The input sparse tensor.
    rows_to_remove (torch.Tensor): A 1D tensor containing the indices of rows to remove.
    cols_to_remove (torch.Tensor): A 1D tensor containing the indices of columns to remove.

    Returns:
    torch.sparse.Tensor: The trimmed sparse tensor.
    """
    # Check the format of the sparse tensor and convert to COO if necessary
    if sparse_tensor.layout == pt.sparse_coo:
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
    elif sparse_tensor.layout == pt.sparse_csr:
        sparse_tensor = sparse_tensor.to_sparse_coo()
        indices = sparse_tensor._indices()
        values = sparse_tensor._values()
    else:
        raise ValueError("The sparse tensor must be in COO or CSR format.")

    # Extract row and column indices from the COO format
    row_indices = indices[0]
    col_indices = indices[1]

    # Create masks to keep only the rows and columns that are NOT in the rows_to_remove and cols_to_remove
    row_mask = ~pt.isin(row_indices, rows_to_remove)
    col_mask = ~pt.isin(col_indices, cols_to_remove)

    # Combine the masks to filter out the unwanted rows and columns
    combined_mask = row_mask & col_mask

    # Filter the indices and values using the combined mask
    filtered_indices = indices[:, combined_mask]
    filtered_values = values[combined_mask]

    # Compute the new size of the tensor after removing rows and columns
    new_size = (sparse_tensor.size(0) - len(rows_to_remove),
                sparse_tensor.size(1) - len(cols_to_remove))

    # Create the new sparse tensor
    trimmed_sparse_tensor = pt.sparse_coo_tensor(filtered_indices,
                                                 filtered_values,
                                                 size=new_size)

    return trimmed_sparse_tensor

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
    K = 15
    group_strength_list = [1]
    spatial_list = [0]
    global_run_list = [1,2,3]
    randy_tasks = ['EPROJ', 'LANG', 'MOTOR', 'NBACK', 'TOM', 'VISME', 'VODDK']
    randy_good_subjlist = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14]
    randy_bad_subjlist = [0, 4, 9, 10]

    ######## Step 2. Generate group / indiv parcellations
    # Load DU15 networks
    DU, net_name, colors = gp.get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    DU15 = ar.expand_mn_1d(DU, K=16)
    align = DU15[1:, :]
    nanidx = pt.where(align.sum(dim=0) == 0)[0]

    ## Load the group prior from a pre-trained model
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_DU15-inits'  # resting 15
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_DU15-inits' # task 15
    U, _ = hut.load_group_parcellation(MODEL_DIR + model_name, device=DEVICE)
    # U = atlas.cifti_to_data(ERIS_DIR + '/dzhi/workspace/res/priors/MSHBM_group_prior_HCP40training_k-15.dscalar.nii') # MSHBM 15
    # U = pt.tensor(np.nan_to_num(U), dtype=pt.get_default_dtype(), device=DEVICE)
    # Align with prior
    indx = hev.matching_greedy(align, pt.softmax(U, dim=0).to(pt.get_default_dtype()))
    U = U[indx, :]
    U = pt.softmax(U, dim=0)
    # Vs = [v[:,indx] for v in Vs]
    Pgroup = pt.argmax(U, dim=0) + 1

    ## Determine the connectivity profile for mRBM
    Wc = pt.load(ERIS_DIR + '/Tian/UKBB_full/imaging/Atlases/tpl-fs32k/fs32k_neighbours.pt', weights_only=True)
    hut.report_cuda_memory()
    T = pd.read_csv(ERIS_DIR + f"/Tian/RANDY15/participants.tsv", sep='\t')

    ####################################################################################################################
    ## Evaluation
    print(f'Start loading data: RANDY resting - {test_ses} - {fc_type} ...')
    tic = time.perf_counter()
    ## 3. RANDY task contrasts
    # t_data, t_info = ut.load_randy_contrasts(space='fs32k', subj=randy_good_subjlist, hemis=None, smooth=2)

    ## 4. RANDY resting timeseries
    # t_data = ut.build_resting_data('RANDY15', space='fs32k',
    #                               ses_list=[f'ses-rest{sn}' for sn in range(6, 8)],
    #                               type='Tseries', subj=randy_bad_subjlist, hemis=None, smooth=None)
    # t_info = pd.DataFrame({'task_name': ['REST'] * t_data[0].shape[1]})

    ## 5. RANDY task betas
    t_data, cond_vec, part_vec, subj_ind, t_info = ut.build_dataset('RANDY15', atlas=atlas.name,
                                               sess=randy_tasks, cond_ind='reg_id',
                                               type='CondRun', part_ind='sn', subj=randy_good_subjlist,
                                               part_num=np.arange(1, 4), join_sess=False,
                                               join_sess_part=False, smooth=None)
    t_info = t_info[0]

    toc = time.perf_counter()
    print(f"Done loading. Used {time.strftime('%M:%S', time.gmtime(toc - tic))} (MM:SS)!")
    hut.report_cuda_memory()
    n_subj = t_data[0].shape[0]

    results = pd.DataFrame()

    for num_test in range(10):
        for global_counter in global_run_list:
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    print(f'Evaluating {global_counter} runs data, prior strength {p}, the spatial w {w} ...')
                    U_indiv_rest = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYrest2run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_rest = pt.tensor(U_indiv_rest, dtype=pt.get_default_dtype())

                    U_indiv_rest_3387s = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                           f'/asym_MdNiIbHc+RANDYrest3387s-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-5_spatial-{w}_11sub.npy')
                    U_indiv_rest_3387s = pt.tensor(U_indiv_rest_3387s, dtype=pt.get_default_dtype())

                    U_indiv_task = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYtask1run{global_counter}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_task = pt.tensor(U_indiv_task, dtype=pt.get_default_dtype())

                    U_indiv_fusion = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYrest2run+task1run{global_counter}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_fusion = pt.tensor(U_indiv_fusion, dtype=pt.get_default_dtype())

                    # m-RBM
                    ar_model = ar.build_arrangement_model(U.clone(), prior_type='prob', atlas=atlas,
                                                          sym_type='asym', model_type='cRBM_Wc',
                                                          Wc=Wc, theta=w, epos_iter=20, num_chain=n_subj)
                    ar_model.bu = ar_model.bu * p

                    # Making evaluation information
                    this_glist = global_run_list.copy()
                    this_glist.remove(global_counter)
                    minfo = ge.make_eval_info(K, train_info=['RANDY'], train_sess=global_counter,
                                              tdata='RANDY', test_sess=this_glist,
                                              model_type='Models_03', group_map_name='MdNiIbHc',
                                              test_kappa=None)
                    this_res = pd.DataFrame()
                    # Evaluate on each run's time series
                    for r, td in enumerate(t_data):
                        if r == global_counter - 1:
                            continue

                        res = pd.DataFrame({'atlas': [minfo.atlas] * n_subj,
                                            'K': [minfo.K] * n_subj,
                                            'train_data': [minfo.datasets] * n_subj,
                                            'train_sess': [minfo.train_sess] * n_subj,
                                            'test_data': [minfo.test_data] * n_subj,
                                            'test_sess': [minfo.test_sess] * n_subj,
                                            'model_type': [minfo.model_type] * n_subj,
                                            'group_map_name': [minfo.group_map_name] * n_subj,
                                            'subj_num': [f'{i}' for i in T.participant_id[randy_good_subjlist]],
                                            'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj})

                        ## prediction error
                        hut.report_cuda_memory()
                        em_params = {'num_subj': n_subj, 'uniform_kappa': True, 'subjects_equal_weight': True,
                                     'subject_specific_kappa': False, 'parcel_specific_kappa': False}

                        em_model = em.build_emission_model(K, atlas, 'VMF', hut.indicator(cond_vec[r]),
                                                           part_vec[r], V=None, em_params=em_params)
                        M_test = fm.FullMultiModel(deepcopy(ar_model), [em_model])
                        coserr_indiv = ev.calc_test_error(M_test, td,['group', 'floor',
                                                                      U_indiv_rest, U_indiv_rest_3387s, U_indiv_task, U_indiv_fusion])
                        del M_test
                        pt.cuda.empty_cache()
                        hut.report_cuda_memory()

                        res['group_coserr'] = coserr_indiv[0]
                        res['floor_coserr'] = coserr_indiv[1]
                        res['indiv-rest_coserr'] = coserr_indiv[2]
                        res['indiv-rest3387s_coserr'] = coserr_indiv[3]
                        res['indiv-task_coserr'] = coserr_indiv[4]
                        res['indiv-fusion_coserr'] = coserr_indiv[5]
                        res['test_run'] = r + 1
                        # res['ari_group'] = ari
                        # res['nmi_group'] = nmi
                        res['strength'] = p
                        res['spatial_w'] = w
                        res["num_test"] = num_test

                        this_res = pd.concat([this_res, res], ignore_index=True)

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/coserr_indiv-mRBM_MdNiIbHc-RANDY_rest2run_vs_task1run_vs_fusion_K-15_test_on_RANDYtask-beta_sm0_run123_11subjects.tsv',
        index=False, sep='\t')
    print('Done')


    ## Combine all test subjects parcellation
    indiv_par = []
    for i in [1, 2, 3, 4]:
        par = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                    f'/asym_MdNiIbHc+HCPrest-run1+task-half2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1_{i}.dlabel.nii').get_fdata()[:]
        indiv_par.append(par)
    indiv_par = np.vstack(indiv_par)
    T = pd.read_csv(get_subject_list_path("HCP200_test.tsv"), sep='\t')
    img = nt.make_label_cifti(indiv_par.T, atlas.get_brain_model_axis(),
                              column_names=[f'{i}' for i in T.participant_id],
                              label_names=net_name, label_RGBA=colors)
    nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
            f'/asym_MdNiIbHc+HCPrest-run1+task-half2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1.dlabel.nii')

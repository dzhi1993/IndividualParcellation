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
        subj_list_file = f"test.tsv"
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

    # Combine all test subjects parcellation
    indiv_par = []
    # colors[[1, 6]] = colors[[6, 1]]
    for i in [1, 2, 3, 4]:
        par = nb.load('/home/dzhi/eris_mount/dzhi/Indiv_par/Models/Models_03/indiv_parcellation/HCP203_test_set' +
                    f'/asym_KONG2019+HCPrest-1run-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-2_spatial-2_{i}.dlabel.nii').get_fdata()[:]
        indiv_par.append(par)
    indiv_par = np.vstack(indiv_par)
    T = pd.read_csv(get_subject_list_path("HCP200_test.tsv"), sep='\t')
    img = nt.make_label_cifti(indiv_par.T, atlas.get_brain_model_axis(),
                              column_names=[f'{i}' for i in T.participant_id],
                              label_names=net_name, label_RGBA=colors)
    nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
            f'/asym_MdNiIbHc+HCPrest-2261s2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1.dlabel.nii')


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

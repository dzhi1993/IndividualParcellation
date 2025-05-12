#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
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

def get_kong2019_17_parcellation():
    network_names = spio.loadmat(ERIS_DIR + '/dzhi/workspace/CBIG/stable_projects/'
                                 'brain_parcellation/Kong2019_MSHBM/lib/'
                                 'group_priors/HCP_40/17network_labels.mat')['network_name']
    network_names = ['???'] + [network_names[0][i][0] for i in range(17)]

    colors = spio.loadmat(ERIS_DIR + '/dzhi/workspace/CBIG/stable_projects/'
                     'brain_parcellation/Kong2019_MSHBM/lib/'
                     'group_priors/HCP_40/group.mat')['colors']/255
    colors = colors[1:,:]
    colors = np.hstack((colors, np.ones((17, 1))))
    colors = np.vstack((np.zeros(4), colors))
    KONG2019 = nb.load(ERIS_DIR + '/dzhi/Indiv_par/Kong_2019/group_prior' \
                       '/HCP_40/Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii').get_fdata()[:]
    
    return KONG2019, network_names, colors

def get_kong2019_indiv_parcellations(dir, subj_list, w=80, c=40, num_sess=1):
    atlas, _ = am.get_atlas('fs32k')
    
    parcellations = []
    sub_name = []
    T = pd.read_csv(subj_list, delimiter='\t')
    for i, s in enumerate(T.participant_id):
        if num_sess == 1:
            mat_file = dir + f'/Ind_parcellation_MSHBM_sub{i+1}_w{w}_MRF{c}.mat'
        else:
            mat_file = dir + f'/Ind_parcellation_MSHBM_sub{i+1}_w{w}_MRF{c}_num-sess{num_sess}.mat'

        left = spio.loadmat(mat_file)['lh_labels'].reshape(-1)
        right = spio.loadmat(mat_file)['rh_labels'].reshape(-1)
        left_labels = left[atlas.mask[0]]
        right_labels = right[atlas.mask[1]]
        parcel = np.concatenate([left_labels, right_labels])

        parcellations.append(parcel)
        sub_name.append(f'sub_{s}')
    
    _, nets, colors = get_kong2019_17_parcellation()
    img = nt.make_label_cifti(np.stack(parcellations).T, atlas.get_brain_model_axis(),
                            column_names=sub_name, label_names=nets, label_RGBA=colors)
    outdir = '/data/tge/dzhi/Indiv_par/Kong_2019/indiv_par'
    nb.save(img, outdir + f'/{os.path.basename(dir)}_indiv_par_w{w}_MRF{c}_n-sess{num_sess}.dlabel.nii')
    return np.stack(parcellations)
        
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


def load_hcp_timeseries(dataset_dir, subj_list, space='MNIAsymC2', run_list=[0,1,2,3],
                       type='Tseries', hemis=None, smooth=None, ext=None):

    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    
    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    
    data = []
    for i, run_id in enumerate(run_list):
        ses_data=[]
        for s in T.participant_id:
            # Assemble file name            
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_run{run_id}'
            else:
                file_name = f'/{s}_run{run_id}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dtseries.nii'

            # Load data / remove medial wall
            dat = nb.load(data_dir.format(s) + file_name)
            dat = dat.get_fdata().astype(np.float32)
            dat = dat[:, np.concatenate(this_at.vertex_mask)]

            if hemis is not None: # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:,this_at.indx_full[stru_idx]]
            
            ses_data.append(dat)
        
        data.append(np.stack(ses_data))

    return data


def load_hcp_contrasts(dataset_dir, subj_list, space='MNIAsymC2', sess='all',
                        type='Tseries', hemis=None, smooth=None):
    hcp_ds = ds.DataSetHcpTask(dataset_dir)
    T = hcp_ds.get_participants(subj_list)
    T = T.iloc[0:20]
    if sess == 'all':
        sess = hcp_ds.task_domain

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    data, info, domains = [],[],[]
    for s in T.participant_id:
        ses_data, ses_info, ses_domains = [],[],[]
        for ses_id in sess:
            # Assemble file name
            if smooth is None:
                file_name = f'/ses-{ses_id}/{s}_tfMRI_{ses_id}_level2_hp200_s2.dscalar.nii'
            else:
                file_name = f'/ses-{ses_id}/{s}_tfMRI_{ses_id}_level2_hp200_s{smooth}.dscalar.nii'

            # Load data / info
            dat = nb.load(hcp_ds.func_dir.format(s) + file_name)
            this_info = dat.header.get_axis(0).name.tolist()
            prefix = os.path.commonprefix(this_info)
            this_info = [s[len(prefix):] for s in this_info]
            dat = dat.get_fdata().astype(np.float32)

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]
            else:
                dat = dat[:, np.concatenate(this_at.indx_full)]

            # Remove the betas (contrast) from this session
            reg_info = pd.read_csv(hcp_ds.estimates_dir.format(s) +
                                   f'/ses-task/{s}_ses-task_reginfo.tsv', sep='\t')
            beta_names = reg_info.loc[reg_info.task_name == ses_id].cond_name.unique()
            beta_names = [s + '_' for s in beta_names]
            contrast_idx = [i for i, s in enumerate(this_info) if not s.startswith(tuple(beta_names))]

            ses_data.append(dat[contrast_idx,:])
            ses_info.append([this_info[i] for i in contrast_idx])
            ses_domains.append([ses_id] * len(contrast_idx))

        data.append(np.vstack(ses_data))
        info.append(np.concatenate(ses_info))
        domains.append(np.concatenate(ses_domains))

    # Check if all arrays are identical
    assert all(np.array_equal(info[0], arr) for arr in info)
    assert all(np.array_equal(domains[0], arr) for arr in domains)

    info_com = pd.DataFrame({'contrast_name': info[0],
                             'task_name': domains[0]})

    data = np.stack(data)
    return [data], info_com


def load_msc_contrasts(ds_name, space='fs32k', sess='all', subj=None, smooth=None):
    atlas, _ = am.get_atlas(space)

    if sess == 'all':
        sess = ['motor','memory','mixed']
    elif isinstance(sess, str):
        sess = [sess]
    assert isinstance(sess, list)

    dataset = ds.DataSetMSC('/home/dzhi/eris_mount/Tian/MSC')
    T = dataset.get_participants()
    # Assemble the data
    Data = None
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a list of str or int'))

    # Loop again to assemble the data
    Data_list, info = [], []
    for i, s in enumerate(subj):
        subj_dat, subj_info = [], []
        for ses_id in sess:
            # Load the data
            if smooth is not None:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR_smooth{smooth}.dscalar.nii')
            else:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR.dscalar.nii')
            this_data = atlas.cifti_to_data(C)
            this_info = C.header.get_axis(0).name.tolist()

            # indices = [i for i, s in enumerate(this_info) if not re.search(r"[+-]", s)
            #            and not s.startswith("Alltasks")]
            indices = [i for i, s in enumerate(this_info) if not s.startswith("Alltasks")]

            subj_info.append([s for i, s in enumerate(this_info) if i in indices])
            subj_dat.append(this_data[indices])

        Data_list.append(np.vstack(subj_dat))
        info.append(np.concat(subj_info))

    # concatenate along the first dimension (subjects)
    Data = np.stack(Data_list)
    Data[np.isinf(Data)] = np.nan

    # Assemble info file
    # Check if all arrays are identical
    if all(np.array_equal(info[0], arr) for arr in info):
        # Convert to DataFrame with a single column
        info_com = pd.DataFrame(info[0], columns=['task_name'])
    else:
        # Convert to DataFrame with multiple columns
        info_com = pd.DataFrame(info).T
        info_com.columns = [f'subj_{i + 1}' for i in range(len(info))]

    return Data, info_com


def build_msc_resting_data(dataset_dir, subj_list, this_at, ses_list='all',
                        type='Tseries', hemis=None, smooth=None, ext=None):
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')

    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

    data = []
    for i, run_id in enumerate(ses_list):
        ses_data = []
        for s in T.participant_id:
            # Assemble file name
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_run{run_id}'
            else:
                file_name = f'/{s}_run{run_id}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dtseries.nii'

            # Load data / remove medial wall
            dat = nb.load(data_dir.format(s) + file_name)
            dat = dat.get_fdata().astype(np.float32)
            dat = dat[:, np.concatenate(this_at.vertex_mask)]

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]

            ses_data.append(dat)

        data.append(np.stack(ses_data))

    return data


def load_randy_contrasts(space='fs32k', ses_id='ses-s1', type=None,
                         subj=None, hemis=None, smooth=2, verbose=False):
    """Loads all the CIFTI files in the data directory of a certain space / type and returns they content as a Numpy array

    Args:
        space (str): Atlas space (Defaults to 'SUIT3').
        ses_id (str): Session ID (Defaults to 'ses-s1').
        type (str): Type of data (Defaults to 'CondHalf').
        subj (ndarray, str, or list):  Subject numbers /names to get [None = all]
    Returns:
        Data (ndarray): (n_subj, n_contrast, n_voxel) array of data
        info (DataFramw): Data frame with common descriptor
    """
    dataset = ds.DataSetRANDY15(ERIS_DIR + '/Tian/RANDY15')
    T = dataset.get_participants()
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, str):
        subj = [subj]
    elif isinstance(subj, (int, np.integer)):
        subj = [T.participant_id.iloc[subj]]
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a str, int, list or ndarray'))
    if type is None:
        type = dataset.default_type

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    max = 0
    # Loop over the different subjects to find the most complete info
    for s in subj:
        # Get an check the information
        info_raw = pd.read_csv(dataset.contrast_dir.format(s)
                               + f'/{s}_AllContrasts.tsv', sep='\t')
        # Keep the most complete info
        if info_raw.shape[0] > max:
            info_com = info_raw
            max = info_raw.shape[0]
    base = np.asarray(info_com['contrast_name'])

    # Loop again to assemble the data
    Data_list = []
    for i, s in enumerate(subj):
        # If you add verbose printout, make it so
        # that by default it is suppressed by a verbose=False option
        if verbose:
            print(f'- Getting data for {s} in {space}')
        # Load the data
        if smooth is not None:
            C = nb.load(dataset.contrast_dir.format(s)
                        + f'/{s}_AllContrasts_{space}_sm{smooth}_Zmap.dscalar.nii')
        else:
            C = nb.load(dataset.contrast_dir.format(s)
                        + f'/{s}_AllContrasts_{space}_Zmap.dscalar.nii')
        this_data = C.get_fdata()

        if hemis is not None:  # if cortical data
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            this_data = this_data[:, this_at.indx_full[stru_idx]]
        else:
            this_data = this_data[:, np.concatenate(this_at.indx_full)]

        # Check if this subject data in incomplete
        if this_data.shape[0] != info_com.shape[0]:
            this_info = pd.read_csv(dataset.contrast_dir.format(s)
                                    + f'/{s}_AllContrasts.tsv', sep='\t')
            incomplete = np.asarray(this_info['contrast_name'])
            contrast_to_row = {name: i for i, name in enumerate(incomplete)}
            aligned_data = np.full((len(base), this_data.shape[1]), np.nan)

            for j, name in enumerate(base):
                if name in contrast_to_row:
                    aligned_data[j] = this_data[contrast_to_row[name]]
                else:
                    warnings.warn(f'{s} - missing contrast {name}')
            this_data = aligned_data

        Data_list.append(this_data[np.newaxis, ...])
    # concatenate along the first dimension (subjects)
    Data = np.concatenate(Data_list, axis=0)
    # Ensure that infinite values (from div / 0) show up as NaNs
    Data[np.isinf(Data)] = np.nan
    return [Data], info_com


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
    U_kong = get_kong2019_indiv_parcellations('/data/tge/dzhi/workspace/res/ind_parcellation/test_set',
                                HCP_DIR + "/subj_list/test_split/HCP923_test_set_split_1.tsv", 
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
    # eval_task_inhomo_MSHBM_vs_HBP_indiv()
    # U_indiv = get_kong2019_indiv_parcellations('/data/tge/dzhi/workspace/res/ind_parcellation/HCP203_test_set',
    #                                     HCP_DIR + "/subj_list/HCP203_test_set.tsv", w=100, c=30, num_sess=2)

    # align, net_name, colors = get_kong2019_17_parcellation()
    # t_data, t_info = load_hcp_contrasts(HCP_DIR, "/subj_list/HCP203_test_set_filtered_1.tsv", space='fs32k',
    #                                     hemis='L', smooth='2_MSMAll')
    # zvalues = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP203_test_set' +
    #               '/zvalue_indiv_asym_HCP800+HCPtask+rest-half1-indiv_K-17_strengh-2_spatial-1.npy')
    #
    # df_all = pd.DataFrame()
    # for i in range(res.shape[2]):
    #     df = plot_zvalues(res, 0, t_info, parcel_name=net_name[1:])
    #     df_all = pd.concat([df_all, df], ignore_index=True)
    #
    # plt.figure(figsize=(15, 10))
    # # sb.boxplot(data=results, x='network', y='dice', hue='type', width=0.7)
    # sb.barplot(data=df, x='parcel', y='z_value', errorbar='se', width=0.7, palette=[tuple(c) for c in colors[1:]])
    # plt.tight_layout()
    # plt.show()

    atlas, _ = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    # DEVICE = 'cpu'
    training_ses = 'all'
    test_ses = 'ses-rest1'
    test_hemis = 'L'
    fc_type = 'CondHalf'
    ext = '_binarized'
    K=15

    # fname = MODEL_DIR + f'/Models_03/asym_Hc_space-fs32k_K-15_HCP40subjects_Ico642Run_desc-sm4fwhm_binarized'
    # U, _ = hut.load_group_parcellation(fname, device=DEVICE)
    # dist = futil.load_fs32k_dist(file_type=f'distGOD_mid_{test_hemis}', hemis='half',
    #                             device=DEVICE if pt.cuda.is_available() else 'cpu')
    # minfo = ge.make_eval_info(K, atlas='fs32k', train_info=['RANDY15'], train_sess='half-1', tdata='HCP',
    #                           test_sess='contrast', model_type='Models_03', group_map_name='HCP', test_kappa=None)
    # stru_idx = atlas.structure.index(hemis_dict[test_hemis])
    # Pgroup = pt.argmax(pt.softmax(U, dim=0), dim=0) + 1
    # Pgroup = Pgroup[atlas.indx_full[stru_idx]]
    #
    # # t_data = []
    # # for sn in [6,7,8,9,10]:
    # #     this_data, info, _ = ds.get_dataset(BASE_DIR, 'RANDY15', atlas=atlas.name, sess=f'ses-rest{sn}',
    # #                                    type='Tseries', subj=None, smooth=None)
    # #     t_data.append(this_data)
    #
    # t_data, t_info = load_randy_contrasts(ses_id='all', subj=None, hemis=None, smooth=2)
    # t_info["task_name"] = t_info["domain"]
    #
    # results = pd.DataFrame()
    # for i, td in enumerate(t_data):
    #     if type(td) is np.ndarray:
    #                 td = pt.tensor(td, dtype=pt.get_default_dtype())
    #
    #     for counter_p, p in enumerate([0.5]):
    #         for counter_w, w in enumerate([0]):
    #             print(f'prior strength is {p}; the MRF strength is {w} ...')
    #             indiv_par = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set' +
    #                     f'/asym_Ra-5run-indiv_space-fs32k_K-15_CondHalf_arrange-independent.dlabel.nii').get_fdata()
    #             # Pindiv = indiv_par[:,atlas.indx_full[stru_idx]]
    #             Pindiv = pt.tensor(indiv_par, dtype=pt.get_default_dtype())
    #
    #             # res = eval_parcel_DCBC(Pgroup, Pindiv, td[:,:,atlas.indx_full[stru_idx]], dist, minfo,
    #             #                 out_file='eval_dcbc_indiv_Buckner7_k-7_model-04_test.tsv')
    #
    #             zvalue_indiv = ev.calc_test_zvalue(Pindiv, td, return_single=False)
    #             np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
    #                     f'/zvalue_indiv_asym_Ra-5run-indiv_K-{K}.npy',
    #                     zvalue_indiv.cpu().numpy())
    #             inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td, return_single=False)
    #             inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
    #             np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/inhomogeneity' +
    #                     f'/inhomo_nets_asym_Ra-5run-indiv_K-{K}.npy',
    #                     inhomo_nets.cpu().numpy())
    #
    #             res['test_run'] = i+1
    #             # res['strength'] = p
    #             # res['spatial_w'] = w
    #             # results = pd.concat([results, res], ignore_index=True)
    #
    # results.to_csv(f'eval_HCP40+RANDYrest-5run_indiv_k-15_test_on_RANDYrest_Tseries.tsv', index=False, sep='\t')


    ######## Step 1. Load subjects individual training data
    print(f'Start loading data: HCP resting - {training_ses}, {fc_type} {ext} ...')
    tic = time.perf_counter()
    ## HCP task data
    # data1, cond_vec1, part_vec1, subj_ind1, t_info = gp.build_hcp_datasets(HCP_DIR, f'subj_list/HCP203_test_set_filtered_1.tsv',
    #                                                            atlas, ses_list=['ses-task'],
    #                                      join_sess=False, join_sess_part=False,
    #                                      part_ind=['half'], part_num=None, cond_ind=['reg_id'],
    #                                      type=['CondHalf'], hemis=None, smooth='6fwhm_zstat_masked-hi0.1lo0.1')
    # ## HCP resting data
    # data2, cond_vec2, part_vec2, subj_ind2, rs_info = gp.build_hcp_datasets(HCP_DIR, "subj_list/HCP203_test_set_filtered_1.tsv",
    #                                             atlas, ses_list=['ses-rest1'],
    #                                             join_sess=False, join_sess_part=False,
    #                                             part_ind='run', part_num=None, cond_ind=['net_id'],
    #                                             type=['Ico642Run'], hemis=None, smooth='4fwhm_binarized')
    #
    # data = data1 + data2
    # cond_vec = cond_vec1 + cond_vec2
    # part_vec = part_vec1 + part_vec2
    # subj_ind = subj_ind1 + subj_ind2

    # 2. MSC task
    # data, info, _ = ds.get_dataset(BASE_DIR, 'MSC', atlas=atlas.name, sess='ses-task',
    #                                 type='CondHalf', subj=None, smooth=None)
    # t_data = [t_data[:,t_info.half == 2,0:29759]]
    # t_info = t_info.loc[t_info.half == 2].reset_index(drop=True)
    #
    # data, info, _ = ds.get_dataset(BASE_DIR, 'MSC', atlas=atlas.name, sess='ses-task',
    #                                    type='CondRun', subj=None, smooth=None)
    #
    # data = [data[:,info.sn == i,:] for i in [1,3,5,7,9]]
    # cond_vec = [np.arange(1,13)] * 5
    # part_vec = [np.repeat(np.array([1]), 12)] * 5
    # subj_ind = [np.arange(10)] *5

    # 3. RANDY15
    ## Randy 15 resting-state
    # data = []
    # for sn in [1,2,3,4,5]:
    #     this_data, info, _ = ds.get_dataset(BASE_DIR, 'RANDY15', atlas=atlas.name, sess=f'ses-rest{sn}',
    #                                    type='Ico642Run', subj=None, smooth=None)
    #     data.append(this_data)
    data = ut.build_resting_data('RANDY15', space='fs32k', ses_list=[f'ses-rest{sn}' for sn in range(1,25)],
                              type='Ico642Run', hemis=None, smooth=None)


    # data, info, _ = ds.get_dataset(BASE_DIR, 'MSC', atlas=atlas.name, sess='ses-task',
    #                                    type='CondRun', subj=None, smooth=None)

    cond_vec = [np.arange(1,1211)] * 5
    part_vec = [np.repeat(np.array([1]), 1210)] * 5
    subj_ind = [np.arange(15)] * 5

    # 3. MDTB task
    # data, cond_vec, part_vec, subj_ind = gp.build_data_list(['MDTB'], atlas='fs32k', sess=['all'],
    #                                                         cond_ind=['cond_num_uni'], type=[fc_type],
    #                                                         part_ind=['half'], part_num=None, subj=None,
    #                                                         join_sess=False, join_sess_part=False,
    #                                                         smooth=['10fwhm_zstat_masked-hi0.1lo0.1'], hemis=None)
    # t_data = [data[1][:, :, 0:29759]]
    # data = [data[0]]
    # cond_vec = [cond_vec[0]]
    # part_vec = [part_vec[0]]
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    hut.report_cuda_memory()
    n_subj = np.unique(np.concatenate(subj_ind, axis=0)).size

    ## Load HCP 40 validation subjects data for indiv parcellation
    print(f'Start loading data: HCP resting - {test_ses} - Tseries ...')
    tic = time.perf_counter()    
    # t_data, _, _, _ = gp.build_hcp_datasets(HCP_DIR, "subj_list/HCP40_validation_set.tsv",
    #                                         space=atlas.name, ses_list=[test_ses],
    #                                         join_sess=False, join_sess_part=False, 
    #                                         part_ind='run', part_num=None, cond_ind=['time_id'],
    #                                         type=['Tseries'], hemis=test_hemis, smooth='4fwhm')

    # t_data, t_info = load_hcp_contrasts(HCP_DIR, "/subj_list/HCP203_test_set_filtered_1.tsv", space='fs32k',
    #                                     hemis='L', smooth='2_MSMAll')

    # t_data, t_info = load_msc_contrasts('MSC', sess='all', subj=None, smooth=2.55)
    t_data, t_info = load_randy_contrasts(ses_id='all', subj=None, hemis='L', smooth=2)
    t_info["task_name"] = t_info["domain"]

    # t_data = load_hcp_timeseries(HCP_DIR, "subj_list/HCP203_test_set_filtered.tsv",
    #                             space=atlas.name, run_list=[2,3],
    #                             type='Tseries', hemis=test_hemis, smooth='4fwhm')
    # t_data, t_info, _ = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, sess='ses-s2',
    #                                 type='CondHalf', subj=None, smooth=None)
    # t_data = [data[:,:,0:29759]]
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    hut.report_cuda_memory()

    ######## Step 2. Generate group / indiv parcellations
    # Option 1: calculate indiv parcellations directly from fitted model
    # U, U_indv, M = get_indiv_parcellation_from_model(MODEL_DIR + 
    #                   f'/Models_07/asym_Uk_space-MNIAsymC2_K-7_ses-rest1', data)

    ######## Option 2: calculate indiv parcellations from existing group map
    ## laod Kong 2019 - HCP40
    # atlas_dir = '/data/tge/dzhi/Indiv_par/Kong_2019/group_prior/HCP_40'
    # model_name = f'/Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii'
    # U = nb.load(atlas_dir + model_name).get_fdata()
    # U = U.T
    # align, net_name, colors = get_kong2019_17_parcellation()
    # align = pt.tensor(align, dtype=pt.get_default_dtype(), device=DEVICE)

    # Load K15 colors
    DU15, net_name, colors = gp.get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    DU15 = ar.expand_mn_1d(DU15, K=16)
    align = DU15[1:,:]

    ## Load the group prior from a pre-trained model
    model_name = f'/Models_03/asym_Hc_space-fs32k_K-15_HCP40subjects_Ico642Run_desc-sm4fwhm_binarized' # resting 15
    # model_name = f'/Models_03/task_fusion/asym_MdPoNiIbWmDeSo_space-fs32k_K-15_sm6fwhm_zstat_masked-hi0.1lo0.1' # task 15
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800' # resting 17
    # model_name = f'/Models_03/task_fusion/asym_MdPoNiIbWmDeSo_space-fs32k_K-17_sm6fwhm_zstat_masked-hi0.1lo0.1' # task 17

    fname = MODEL_DIR + model_name
    U, _ = hut.load_group_parcellation(fname, device=DEVICE)
    # Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE) # list of N*K matrix

    # U = nb.load('/home/dzhi/eris_mount/dzhi/Indiv_par/Kong_2019/group_prior/HCP_40/Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii').get_fdata()
    # U = pt.tensor(U, dtype=pt.get_default_dtype(), device=DEVICE)

    Wc = pt.load('/home/dzhi/eris_mount/Tian/UKBB_full/imaging/Atlases/tpl-fs32k/fs32k_neighbours.pt', weights_only=True)
    # Wc = futil.get_fs32k_weights(file_type='distGOD_sp', hemis='full', remove_mw=True,
    #                   max_dist=10, kernel='gaussian', sigma=4, device=DEVICE)

    # Align with prior
    indx = hev.matching_greedy(align, pt.softmax(U, dim=0))
    U = U[indx,:]
    # Vs = [v[:,indx] for v in Vs]
    ## Load distance metric - distAvrg is dijstra; distGOD is godesic
    dist = futil.load_fs32k_dist(file_type='distGOD_mid_L', hemis='half',
                                device=DEVICE if pt.cuda.is_available() else 'cpu')
    
    group_evaluation = []
    results = pd.DataFrame()
    # p = 1,30,60,90,120; w = 0,30,60,90,120
    for counter_p, p in enumerate([0.01,0.1,1,10,100]):
        for counter_w, w in enumerate([0]):
            print(f'prior strength is {p}; the MRF strength is {w} ...')
            # m-RBM
            # ar_model = ar.build_arrangement_model(U*p, prior_type='logpi', atlas=atlas,
            #                                     sym_type='asym', model_type='cRBM_Wc',
            #                                     Wc=Wc, theta=w, epos_iter=20, num_chain=n_subj)

            # Independent
            ar_model = ar.build_arrangement_model(U*p, prior_type='logpi', atlas=atlas,
                                                  sym_type='asym', model_type='independent')
            
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
            # U_indiv = get_kong2019_indiv_parcellations('/data/tge/dzhi/workspace/res/ind_parcellation/test_set',
            #                             HCP_DIR + "/subj_list/test_split/HCP923_test_set_split_1.tsv", 
            #                             w=100, c=30, num_sess=2)
            # Pindiv = pt.tensor(U_indiv, dtype=pt.get_default_dtype())
            # Pindiv = pt.where(Pindiv == 0, pt.nan, Pindiv)

            ## Save indiv parcellation in cifti
            # colors = np.concatenate([np.array([[0,0,0,0]]),plt.cm.get_cmap('tab20', 17).colors], axis=0)
            img = nt.make_label_cifti(Pindiv.T.cpu().numpy(), atlas.get_brain_model_axis(),
                                    column_names=[f'subj_{i}' for i in range(Pindiv.shape[0])],
                                    label_names=net_name, label_RGBA=colors)
            nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set' +
                    f'/asym_HCP40+RANDYrest-5run-indiv_space-fs32k_K-{K}_{fc_type}_arr-independent_groupstrengh-{p}.dlabel.nii')

            # Take the test hemisphere
            stru_idx = atlas.structure.index(hemis_dict[test_hemis])
            Pindiv = Pindiv[:,atlas.indx_full[stru_idx]]
            Pgroup = pt.argmax(pt.softmax(U, dim=0), dim=0) + 1

            # Load Kong2019 group parcellations
            # g_map = nb.load('/data/tge/dzhi/workspace/res/priors'+
            #                 '/MSHBM_group_prior_HCP40training_k-17.dscalar.nii').get_fdata()[:]
            # Pgroup = pt.argmax(pt.tensor(g_map), dim=0) + 1
            Pgroup = Pgroup[atlas.indx_full[stru_idx]]

            # Making evaluation information
            minfo = ge.make_eval_info(K, train_info=['RANDY15'], train_sess='half-1',
                                        tdata='RANDY', test_sess='contrast',
                                        model_type='Models_03', group_map_name='HCP',
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
                            'subj_num': np.arange(n_subj),
                            'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj})
                
                    # Individual evaluation
                    # homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:,idx,:])
                    zvalue_indiv = ev.calc_test_zvalue(Pindiv, td[:,idx,:], return_single=False)
                    np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/zvalues' +
                            f'/zvalue_indiv_asym_HCP40+RANDYrest-5run-indiv_K-{K}_arr-independent_strengh-{p}.npy',
                            zvalue_indiv.cpu().numpy())
                    inhomo_nets = ev.calc_test_task_inhomogeneity(Pindiv, td[:,idx,:], return_single=False)
                    inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                    np.save(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/inhomogeneity' +
                            f'/inhomo_nets_asym_HCP40+RANDYrest-5run-indiv_K-{K}_arr-independent_strengh-{p}.npy',
                            inhomo_nets.cpu().numpy())

                    inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:,idx,:], return_single=True)
                    dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:,idx,:], dist, trim_nan=True)

                    res['dcbc_indiv'] = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
                    # res['homo_indiv'] = homo_indiv.cpu()
                    res['inhomo_indiv'] = inhomo_indiv.cpu()
                    res['task_name'] = task
                    res['test_run'] = r + 1
                    res['train_smooth'] = "2fwhm"
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
            
            this_res['dice_group'] = dice * len(tasks_list)
            # res['ari_group'] = ari
            # res['nmi_group'] = nmi
            this_res['strength'] = p
            this_res['spatial_w'] = w
            
            results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(RES_DIR + f'/eval_HCP40+RANDYrest-5run_K-{K}_indiv-independent_test_on_RANDYtask-contrast.tsv',
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
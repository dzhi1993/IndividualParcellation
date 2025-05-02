 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for learning fusion on datasets

Created on 11/17/2022 at 2:16 PM
Author: dzhi, jdiedrichsen
"""
from time import gmtime
from pathlib import Path
import pandas as pd
import numpy as np
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import *
import Functional_Fusion.matrix as matrix
import nibabel as nb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.evaluation as ev
import HierarchBayesParcel.util as hut
import torch as pt
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
import time
import FusionModel.util as ut

GROUP_DIR = '/data/tge/Tian/HCP_img/derivatives/group'

def build_data_list(datasets, atlas='MNISymC3', sess=None, cond_ind=None, type=None,
                    part_ind=None, part_num=None, subj=None, join_sess=True,
                    join_sess_part=False, smooth=None, hemis=None):
    """Builds list of datasets, cond_vec, part_vec, subj_ind
    from different data sets
    Args:
        datasets (list): Names of datasets to include
        atlas (str): Atlas indicator
        sess (list): list of 'all' or list of sessions
        design_ind (list, optional): _description_. Defaults to None.
        part_ind (list, optional): _description_. Defaults to None.
        subj (list, optional): _description_. Defaults to None.
        join_sess (bool, optional): Model the sessions with a single model.
            Defaults to True.
    Returns:
        data, cond_vec, part_vec, subj_ind
    """
    n_sets = len(datasets)
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(atlas, ut.atlas_dir)
    data, cond_vec, part_vec, subj_ind = [],[],[],[]

    # Set defaults for data sets:
    if sess is None:
        sess = ['all'] * n_sets
    if part_ind is None:
        part_ind = [None] * n_sets
    if cond_ind is None:
        cond_ind = [None] * n_sets
    if type is None:
        type = [None] * n_sets
    if subj is None:
        subj = [None] * n_sets

    sub = 0
    # Run over datasets get data + design
    for i in range(n_sets):
        dat, info, ds = get_dataset(ut.base_dir, datasets[i],
                                    atlas=atlas, sess=sess[i],
                                    type=type[i], subj=subj[i], smooth=smooth[i])
        if hemis is not None:
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            dat = dat[:,:,this_at.indx_full[stru_idx]]

        n_subj = dat.shape[0]

        # Find correct indices
        if cond_ind[i] is None:
            cond_ind[i] = ds.cond_ind
        if part_ind[i] is None:
            part_ind[i] = ds.part_ind
        # Make different sessions either the same or different
        if join_sess:
            if part_num is not None:
                indx = info[part_ind[i]] == part_num
            else:
                indx = np.full(info[part_ind[i]].shape, True)
            # Check if we want to set no partition after join sessions
            if join_sess_part:
                part_vec.append(np.ones(indx.shape))
            else:
                part_vec.append(info[part_ind[i]].values[indx].reshape(-1, ))

            data.append(dat[:, indx, :])
            cond_vec.append(info[cond_ind[i]].values[indx].reshape(-1, ))
            subj_ind.append(np.arange(sub, sub + n_subj))
        else:
            if sess[i] == 'all':
                sessions = ds.sessions
            else:
                sessions = sess[i]
            # Now build and split across the correct sessions:
            for s in sessions:
                if part_num is None:
                    indx = info.sess == s
                else:
                    indx = (info.sess == s) & (info[part_ind[i]] == part_num)
                
                data.append(dat[:, indx, :])
                # indx = pt.tensor(np.where(idx == True)[0])
                # data.append(pt.index_select(dat, 1, indx))
                cond_vec.append(info[cond_ind[i]].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind[i]].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(sub, sub + n_subj))
        sub += n_subj
    return data, cond_vec, part_vec, subj_ind


def batch_fit(datasets, sess,
              type=None, cond_ind=None, part_ind=None, subj=None,
              atlas=None,
              K=10,
              arrange='independent',
              sym_type='asym',
              emission='VMF',
              n_rep=3, n_inits=10, n_iter=80, first_iter=10,
              name=None,
              uniform_kappa=True,
              join_sess=True,
              join_sess_part=False,
              part_num=None,
              weighting=None,
              smooth=None,
              hemis=None,
              second_converge=True,
              Wc_theta=1,
              em_params={}):
    """ Executes a set of fits starting from random starting values
    selects the best one from a batch and saves them

    Args:
        datasets (list): List of dataset names to be used as training
        sess (list): List of list of sessions to be used for each
        type (list): List the data types
        cond_ind (list): Name of the info-field that indicates the condition
        part_ind (list): Name of the field indicating independent partitions of the data
        subj (list, optional): _description_. Defaults to None
        atlas (Atlas): Atlas to be used. Defaults to None.
        K (int): Number of parcels. Defaults to 10.
        arrange (str): Type of arangement model. Defaults to 'independent'.
        sym_type (str): {'sym','asym'} - defaults to asymmetric model
        emission (list / strs): Type of emission models. Defaults to 'VMF'.
        n_inits (int): Number of random starting values. default: 10
        n_iter (int): Maximal number of iterations per fit: default: 20
        save (bool): Save the resulting fits? Defaults to True.
        name (str): Name of model (for filename). Defaults to None.

    Returns:
        info (pd.DataFrame):
    """
    print(f'Start loading data: {datasets} - {sess} - {type} ...')
    tic = time.perf_counter()
    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=atlas.name,
                                                         sess=sess,
                                                         cond_ind=cond_ind,
                                                         type=type,
                                                         part_ind=part_ind,
                                                         part_num=part_num,
                                                         subj=subj,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part,
                                                         smooth=smooth,
                                                         hemis=hemis)
    
    # data = [pt.load(GROUP_DIR + '/processed_ses-rest1.pt'),
    #         pt.load(GROUP_DIR + '/processed_ses-rest2.pt')]
    # cond_vec = [np.tile(np.arange(1,1211), 2), np.tile(np.arange(1,1211), 2)]
    # part_vec = [np.repeat(np.array([1,2]), 1210), np.repeat(np.array([1,2]), 1210)]
    # subj_ind = [np.arange(800), np.arange(800)]

    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    # data = [pt.tensor(dat, dtype=pt.int8).to_sparse() for dat in data]



def fit_all(set_ind=[0, 1, 2, 3], subj_list=None, weighting=None,
             this_sess=None, space=None, smooth=None, part_num=None):
    # Get dataset info
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    datasets = T.name.to_numpy()
    sess = np.array(['all'] * len(T), dtype=object)
    if this_sess is not None:
        for i, idx in enumerate(set_ind):
            sess[idx] = this_sess[i]

    type = T.default_type.to_numpy()
    # type[0:7] = ['CondAll','TaskAll','CondAll','CondAll','CondAll','CondAll','CondAll']
    cond_ind = T.default_cond_ind.to_numpy()
    part_ind = np.array(['half'] * len(T), dtype=object)

    # Make the atlas object
    if space is None:
        space = 'MNISymC3'

    hemis = None
    this_space = space
    if space.startswith('fs32k_'):
        hemis = space.split('_')[1]
        space = space.split('_')[0]

    dataname = ''.join(T.two_letter_code[set_ind])
    atlas, _ = am.get_atlas(space, ut.atlas_dir)

    # Provide different setttings for the different model types
    join_sess_part = False
    join_sess = False
    
    # Generate a dataname from first two letters of each training data set
    dataname = ''.join(T.name[set_ind])
    ut.print_memory_usage()

    print(f'Start loading data: {datasets[set_ind]} - {sess[set_ind]} - {type[set_ind]} ...')
    tic = time.perf_counter()
    data, cond_vec, part_vec, subj_ind = build_data_list(datasets[set_ind],
                                                         atlas=atlas.name,
                                                         sess=sess[set_ind],
                                                         cond_ind=cond_ind[set_ind],
                                                         type=type[set_ind],
                                                         part_ind=part_ind[set_ind],
                                                         part_num=part_num,
                                                         subj=subj_list,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part,
                                                         smooth=smooth,
                                                         hemis=hemis)

    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Assembly data and task union
    contrast = union_task_contrasts(data)
    return contrast, dataname


def union_task_contrasts(datasets, subject_wise=False, weighting=False, scaling=True):
    
    contrast, subj_list, task_list = [],[],[]
    for dat in datasets:
        num_subj = dat.shape[0]
        num_tasks = dat.shape[1]

        subj_list.append(num_subj)
        task_list.append(num_tasks)
    
        # Set all negative values to zero
        # dat = np.abs(dat)
        dat[dat < 0] = 0

        sum_contrasts = np.nansum(dat, axis=1)
        if not subject_wise:
            sum_contrasts = np.nansum(sum_contrasts, axis=0)
        
        contrast.append(sum_contrasts)
    
    contrast = np.nansum(np.stack(contrast), axis=0, keepdims=True)
    print(subj_list)
    print(task_list)

    if scaling:
        contrast = (contrast - contrast.min()) / (contrast.max() - contrast.min())

    # make percentile masks
    mask_percentiles = []
    for i in range(100):
        this_contrast = contrast.copy()
        threshold = np.percentile(this_contrast, i)
        this_contrast[this_contrast <= threshold] = 0
        mask_percentiles.append(this_contrast)
    
    mask_percentiles = np.stack(mask_percentiles)
    return contrast


if __name__ == "__main__":
    train_smooth = 10
    smooth_list=[f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                    f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                    f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                        '7_zstat_masked-hi0.1lo0.1',
                    f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                    f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                    f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1']
    
    contrasts, names = [], []
    for i, s in enumerate(smooth_list):
        ds_contrast, dname = fit_all(set_ind=[i], this_sess=None, space='fs32k',
                              smooth=[s], subj_list=None)
        contrasts.append(ds_contrast)
        names.append(dname)
    
    atlas, _ = am.get_atlas('fs32k', ut.atlas_dir)
    img = atlas.data_to_cifti(np.vstack(contrasts), names)
    nb.save(img, '/data/tge/dzhi/Indiv_par/contrast_mask_per_7datasets_only-activation.dscalar.nii')
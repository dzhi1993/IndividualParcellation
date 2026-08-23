#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script for training group parcellation map given datasets

Created on 3/11/2024 at 2:55 PM
Author: dzhi
"""
from time import gmtime
from pathlib import Path

import mat73
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
import pickle, time
from copy import deepcopy
import scipy.io as spio
import FusionModel.util as ut
from utils import plot_multi_flat, load_batch_best, convert_hard_to_prob

import sys
sys.path.append('..')
from global_config import (ATLAS_DIR, BASE_DIR, DEVICE, ERIS_DIR, HCP_DIR,
                           MODEL_DIR, REPLICATION_DIR)

MSHBM_17NETWORK_DIR = REPLICATION_DIR / 'MSHBM_17networks'

def get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k'):
    atlas, _ = am.get_atlas(atlas_space)
    DU15_dir = ERIS_DIR + '/dzhi/workspace/DU15NET'
    file = nb.load(DU15_dir + f'/HCP/fsLR_32k/{file_name}_fsLR_32k.dlabel.nii')
    DU15 = atlas.cifti_to_data(file).reshape(-1)

    info = pd.read_csv(DU15_dir + '/DU15NET_ColorLUT.csv')
    network_names = list(info['Abbreviation'])
    colors = info[["R","G","B","A"]].to_numpy().astype(float)
    colors[:, :3] = colors[:, :3] / 255

    return DU15, network_names, colors

def get_kong2019_group_parcellation():
    network_names = spio.loadmat(MSHBM_17NETWORK_DIR / '17network_labels.mat')['network_name']
    network_names = ['???'] + [network_names[0][i][0] for i in range(17)]

    colors = spio.loadmat(MSHBM_17NETWORK_DIR / 'group.mat')['colors']/255
    colors = colors[1:,:]
    colors = np.hstack((colors, np.ones((17, 1))))
    colors = np.vstack((np.zeros(4), colors))
    KONG2019 = nb.load(MSHBM_17NETWORK_DIR / 'Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii').get_fdata()[:]

    return KONG2019, network_names, colors

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

    sub = 0
    # Run over datasets get data + design
    for i in range(n_sets):
        dat, info, ds = get_dataset(ut.base_dir, datasets[i],
                                    atlas=atlas, sess=sess[i],
                                    type=type[i], smooth=smooth[i])
        if hemis is not None:
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            dat = dat[:,:,this_at.indx_full[stru_idx]]
        # Sub-index the subjects:
        if subj is not None:
            dat = dat[subj[i], :, :]
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
                cond_vec.append(info[cond_ind[i]].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind[i]].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(sub, sub + n_subj))
        sub += n_subj
    return data, cond_vec, part_vec, subj_ind


def build_model(K, arrange, sym_type, emission, atlas, cond_vec, part_vec,
                em_params={}, weighting=None, epos_iter=5, eneg_iter=5,
                num_chain=20, Wc=None, theta=None, sess=None):
    """ Builds a Full model based on your specification"""
    if arrange == 'independent':
        if sym_type == 'sym':
            ar_model = ar.ArrangeIndependentSymmetric(K,
                                                      atlas.indx_full,
                                                      atlas.indx_reduced,
                                                      same_parcels=False,
                                                      spatial_specific=True,
                                                      remove_redundancy=False)
            ar_model.name = 'indp_sym'
        elif sym_type == 'asym':
            ar_model = ar.ArrangeIndependent(K, atlas.P,
                                             spatial_specific=True,
                                             remove_redundancy=False)
            ar_model.name = 'indp_asym'
    elif arrange == 'cRBM_W':
        # Boltzmann with a arbitrary fully connected model - P hiden nodes
        n_hidden = atlas.P
        ar_model = ar.cmpRBM(K, atlas.P, nh=n_hidden, eneg_iter=eneg_iter,
                             epos_iter=epos_iter, eneg_numchains=num_chain)
        ar_model.name=f'cRBM_{n_hidden}'
    elif arrange == 'cRBM_Wc':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        if Wc is None:
            raise ValueError('Wc must be provided')

        ar_model = ar.wcmDBM(K, atlas.P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                             epos_iter=epos_iter, eneg_numchains=num_chain)
        ar_model.name = 'cRBM_Wc'
        ar_model.momentum = False
        ar_model.fit_W = False
    else:
        raise (NameError(f'unknown arrangement model:{arrange}'))

    # Initialize emission models
    em_models = []
    for j, ds in enumerate(cond_vec):
        if emission == 'VMF':
            em_model = em.MixVMF(K=K, P=atlas.P, X=matrix.indicator(cond_vec[j]),
                                 part_vec=part_vec[j], **em_params)
            # trained_emi = f'Models_03/asym_Ib_space-fs32k_L_K-{K}_independent'
            # _, model = ut.load_batch_best(trained_emi, device=DEVICE)
            # em_model.V = model.emissions[j].V
            # em_model.kappa = model.emissions[j].kappa
        elif emission == 'GMM':
            em_model = em.MixGaussian(K=K, P=atlas.P,
                                      X=matrix.indicator(cond_vec[j]),
                                      std_V=False)
        elif emission == 'wVMF':
            em_model = em.wMixVMF(K=K, P=atlas.P,
                                  X=matrix.indicator(cond_vec[j]),
                                  part_vec=part_vec[j],
                                  uniform_kappa=True,
                                  weighting='length')
        else:
            raise ((NameError(f'unknown emission model:{emission}')))
        em_models.append(em_model)
    M = fm.FullMultiModel(ar_model, em_models)
    if weighting is not None:
        M.ds_weight = weighting  # Weighting for each dataset

    return M


def batch_fit(datasets, sess, type=None, subj=None, atlas=None,
              K=10, arrange='independent', sym_type='asym', emission='VMF',
              n_fits=3, n_inits=10, n_iter=80, first_iter=10, name=None,
              uniform_kappa=True, weighting=None, smooth=None, hemis=None,
              extension=None, second_converge=True, em_params={}):
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
    # data, cond_vec, part_vec, subj_ind = build_ukb_datasets(BASE_DIR,
    #                                         "participants_filtered_final.tsv",
    #                                         space=atlas.name,
    #                                         ses_list=sess,
    #                                         type=type, smooth=smooth)

    # data, cond_vec, part_vec, subj_ind = build_hcp_datasets(HCP_DIR,
    #                                     "subj_list/HCP40_training_set.tsv",
    #                                     space=atlas.name, ses_list=sess,
    #                                     join_sess=False, join_sess_part=False, 
    #                                     part_ind='run', part_num=[1,2],
    #                                     type=type, hemis=hemis, smooth=smooth,
    #                                     ext=extension)
    
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
    
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    
    # # Make run-wise data structure
    # data = np.split(data[0], 2, axis=1) + np.split(data[1], 2, axis=1)
    # cond_vec = np.split(cond_vec[0], 2) + np.split(cond_vec[1], 2)
    # part_vec = [np.ones(cv.size, dtype=int) for cv in cond_vec]
    # subj_ind = subj_ind * 2

    # Load all necessary data and designs
    n_sets = len(data)
    n_subj = np.unique(np.concatenate(subj_ind, axis=0)).size
    if hemis == 'L':
        atlas, _ = am.get_atlas('fs32k_L', ut.atlas_dir)
    elif hemis == 'R':
        atlas, _ = am.get_atlas('fs32k_R', ut.atlas_dir)

    print(f'Building fullMultiModel {arrange} + {emission} for fitting...')
    # Load connectiviy matrix if cpRBM with connectiviy is used
    if arrange == 'cRBM_Wc':
        # surf = ATLAS_DIR + f'/tpl-fs32k/tpl_fs32k_hemi-{hemis}_sphere.surf.gii'
        # Wc = ut.get_fs32k_neighbours(remove_mw=True)
        Wc = ut.get_fs32k_weights(file_type='distGOD_sp',
                                  hemis='half' if (hemis=='L') or (hemis=='R') else 'full',
                                  remove_mw=True, max_dist=10, kernel='gaussian', sigma=10,
                                  device=DEVICE)

        # Use sparse tensor if CUDA is enabled, otherwise dense tensor
        # Wc = Wc.to_sparse_csr() if pt.cuda.is_available() else Wc.to_dense()
        # Wc.values().fill_(1)
    else:
        Wc = None
    M = build_model(K, arrange, sym_type, emission, atlas, cond_vec, part_vec,
                    Wc=Wc, num_chain=n_subj, em_params={'num_subj': n_subj, **em_params})
    fm.report_cuda_memory()

    # Initialize data frame for results
    models, priors = [], []
    info = pd.DataFrame({'name': [name] * n_fits,
                         'atlas': [atlas.name] * n_fits,
                         'K': [K] * n_fits,
                         'datasets': [datasets] * n_fits,
                         'sess': [sess] * n_fits,
                         'type': [type] * n_fits,
                         'subj': [n_subj] * n_fits,
                         'arrange': [arrange] * n_fits,
                         'emission': [emission] * n_fits,
                         'loglik': [np.nan] * n_fits,
                         'weighting': [weighting] * n_fits})

    # Iterate over the number of fits
    ll = np.empty((n_fits, n_iter))
    tt = np.empty((n_fits, n_iter))
    prior = pt.zeros((n_fits, K, atlas.P)) if second_converge else None
    for i in range(n_fits):
        print(f'Start fit: repetition {i} - {name}')

        iter_tic = time.perf_counter()
        # Copy the object (without data)
        m = deepcopy(M)
        # Attach the data
        m.initialize(data, subj_ind=subj_ind)
        pt.cuda.empty_cache()
        fm.report_cuda_memory()

        # Swith the learning process between independent and RBMs
        if m.arrange.name.startswith('indp'):
            m, ll, _, _, _ = m.fit_em_ninits(
                iter=n_iter,
                tol=0.01,
                fit_arrangement=True,
                fit_emission=True,
                init_arrangement=True,
                init_emission=True,
                n_inits=n_inits,
                first_iter=first_iter, verbose=False)
        elif m.arrange.name.startswith('cRBM'):
            m.random_params(init_arrangement=True,
                            init_emission=False)
            m, ll, theta, _ = m.fit_sml(
                iter=n_iter,
                batch_size=6,
                stepsize=0.5,
                seperate_ll=False,
                fit_arrangement=True,
                fit_emission=False)
            tt[i] = theta.cpu().numpy()

        info.loglik.at[i] = ll[-1].cpu().numpy()  # Convert to numpy
        m.clear()
        if second_converge:
            # Align group priors
            if i == 0:
                indx = pt.arange(K)
            else:
                indx = ev.matching_greedy(prior[0, :, :], m.marginal_prob())
            prior[i, :, :] = m.marginal_prob()[indx, :]

            this_similarity = []
            for j in range(i):
                # Option1: K*K similarity matrix between two Us
                # this_crit = cal_corr(prior[i, :, :], prior[j, :, :])
                # this_similarity.append(1 - pt.diagonal(this_crit).mean())

                # Option2: L1 norm between two Us
                this_crit = pt.abs(prior[i, :, :] - prior[j, :, :]).mean()
                this_similarity.append(this_crit)

            num_rep = sum(sim < 0.02 for sim in this_similarity)
            print(num_rep)

            # Convergence: 1. must run enough repetitions (50);
            #              2. num_rep greater than threshold (10% of max_iter)
            if (i > 50) and (num_rep >= int(n_fits * 0.1)):
                m.move_to(device='cpu')
                models.append(m)
                break

        # Move to CPU device before storing
        m.move_to(device='cpu')
        models.append(m)
        del m
        pt.cuda.empty_cache()

        iter_toc = time.perf_counter()
        print(
            f'Done fit: repetition {i} - {name} - {iter_toc - iter_tic:0.4f} seconds!')

    models = np.array(models, dtype=object)

    return info, models


def fit_all(set_ind=[0, 1, 2, 3], K=10, repeats=100, model_type='01',
            sym_type='asym', arrange='independent', subj_list=None,
            weighting=None, this_sess=None, this_type=None, space=None, smooth=None,
            extension=None, sc=True, Wc_theta=1, part_num=None, em_params={}):
    # Get dataset info
    T = pd.read_csv(BASE_DIR + '/dataset_description.tsv', sep='\t')
    datasets = T.name.to_numpy()
    sess = np.array(['all'] * len(T), dtype=object)
    if this_sess is not None:
        for i, idx in enumerate(set_ind):
            sess[idx] = this_sess[i]

    type = T.default_type.to_numpy()
    if this_type is not None:
        for i, idx in enumerate(set_ind):
            type[idx] = this_type[i]

    # Make the atlas object
    if space is None:
        space = 'MNISymC3'

    hemis = None
    this_space = space
    if space.startswith('fs32k_'):
        hemis = space.split('_')[1]
        space = space.split('_')[0]

    atlas, _ = am.get_atlas(space)

    # Generate a dataname from first two letters of each training data set
    dataname = ''.join(T.two_letter_code[set_ind])

    tic = time.perf_counter()
    name = sym_type + '_' + ''.join(dataname)
    info, models = batch_fit(datasets[set_ind],
                                sess=sess[set_ind],
                                type=type[set_ind],
                                subj=subj_list,
                                atlas=atlas,
                                K=K,
                                arrange=arrange,
                                sym_type=sym_type,
                                name=name,
                                n_inits=50,
                                n_iter=200,
                                n_fits=repeats,
                                first_iter=10,
                                uniform_kappa=None,
                                weighting=weighting,
                                smooth=smooth,
                                hemis=hemis,
                                extension=extension,
                                second_converge=sc,
                                em_params=em_params)

    toc = time.perf_counter()
    print(f'Done Model fitting - {sym_type}. Used {toc - tic:0.4f} seconds!')

    # Save the fits and information
    wdir = MODEL_DIR + f'/Models_{model_type}'
    fname = f'/{name}_space-{atlas.name}_K-{K}'

    return wdir, fname, info, models


def build_ukb_datasets(dataset_dir, subj_list, space='MNIAsymC2', ses_list=['ses-rest1'],
                       type=['Tseries'], smooth=None):
    '''Build datasets for functional fusion framework, each dataset is
    supposed to follow BIDS filing structure. Where each subject's data
    is located in <root of your dataset folder>/derivatives/sub-XXX/data
    folder. A <participant.tsv> file is expected in the root directory

    Args:
        dataset_dir: the dataset root directory
        subj_list: a csv file contains all subjects id

    Returns:

    '''
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    data_dir = dataset_dir + '/derivatives/{0}/data'
    raw_rfmri = dataset_dir + '/rfMRI/{0}/20227_2_0'
    
    data = []
    for i, ses_id in enumerate(ses_list):
        ses_data=[]
        for s in T.participant_id:
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}.dscalar.nii'
            else:
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}_desc-sm{smooth}.dscalar.nii'
            try:
                this_data = nb.load(data_dir.format(s) + file_name)
                # this_data.append(atlas.read_data(data_dir.format(s) + file_name).T)
                ses_data.append(this_data.get_fdata())
            except FileNotFoundError as e:
                # print(e)
                if not os.path.exists(raw_rfmri.format(s) + '/fMRI/unusable'):
                    print(f'fake missing raw fMRI {s}')
        
        data.append(np.stack(ses_data))

    # Step 2: Assemble condition and partition vectors
    cond_v, part_v, sub_ind = [], [], []
    for j, dat in enumerate(data):
        cond_v.append(np.arange(dat.shape[1]))
        part_v.append(np.ones((dat.shape[1],), dtype=int))
        sub_ind.append(np.arange(0, dat.shape[0]))

    return data, cond_v, part_v, sub_ind


def build_hcp_datasets(dataset_dir, subj_list, this_at, ses_list=['ses-rest1'],
                       type=['Tseries'], join_sess=False, join_sess_part=False, 
                       part_ind='half', part_num=None, cond_ind='net_id', hemis=None, 
                       smooth=None, ext=None):
    '''Build datasets for functional fusion framework, each dataset is
    supposed to follow BIDS filing structure. Where each subject's data
    is located in <root of your dataset folder>/derivatives/sub-XXX/data
    folder. A <participant.tsv> file is expected in the root directory

    Args:
        dataset_dir: the dataset root directory
        subj_list: a csv file contains all subjects id

    Returns:

    '''
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    data_dir = dataset_dir + '/derivatives/{0}/data'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    space = this_at.name
    assert len(ses_list) == len(type), "session list and type list must equal length!"
    
    if ses_list[0] == 'all':
        ses_list = ['ses-rest1','ses-rest2']
        type = np.repeat(type, 2)
    
    data = []
    info_l = []
    for i, ses_id in enumerate(ses_list):
        ses_data=[]
        for s in T.participant_id:
            info_raw = pd.read_csv(data_dir.format(s)
                                   + f'/{s}_{ses_id}_{type[i]}.tsv', sep='\t')

            # Assemble file name            
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}'
            else:
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dscalar.nii'

            # Load data
            # dat = nb.load(data_dir.format(s) + file_name)
            # # this_data.append(atlas.read_data(data_dir.format(s) + file_name).T)
            # dat = dat.get_fdata().astype(np.float32)
            dat = this_at.cifti_to_data(data_dir.format(s) + file_name).astype(np.float16)
            if "binarized" in file_name:
                dat = dat.astype(np.int8)

            if hemis is not None: # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:,this_at.indx_full[stru_idx]]
            
            ses_data.append(dat)
        
        data.append(np.stack(ses_data))
        info_l.append(info_raw)

    dat = np.concatenate(data, axis=1)
    info = pd.concat(info_l, ignore_index=True, sort=False)
    n_subj = dat.shape[0]

    # Step 2: Assemble condition and partition vectors
    data, cond_vec, part_vec, subj_ind = [], [], [], []
    # Make different sessions either the same or different
    if join_sess:
        if part_num is not None:
            indx = info[part_ind] == part_num
        else:
            indx = np.full(info[part_ind].shape, True)
        # Check if we want to set no partition after join sessions
        if join_sess_part:
            part_vec.append(np.ones(indx.shape))
        else:
            part_vec.append(info[part_ind].values[indx].reshape(-1, ))

        data.append(dat[:, indx, :])
        cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
        subj_ind.append(np.arange(0, n_subj))
    else:
        splitter = 'sess' if info.get('sess') is not None else 'half'
        sessions = np.unique(info[splitter])
        # Now build and split across the correct sessions:
        for s in sessions:
            if part_num is None:
                indx_list = [info[splitter] == s]
            else:
                indx_list = [(info[splitter] == s) & (info[part_ind] == pn) for pn in part_num]

            for indx in indx_list:
                data.append(dat[:, indx, :])
                cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(0, 0 + n_subj))

    return data, cond_vec, part_vec, subj_ind, info


def load_hcp_timeseries(dataset_dir, subj_list, this_at, run_list=[0,1,2,3],
                       type='Tseries', hemis=None, smooth=None, ext=None):

    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    # B = pd.read_csv(Path(HCP_DIR) / 'subj_list/HCP40_training_KONG2019.tsv', delimiter='\t')
    # T = T[~T['participant_id'].isin(B['participant_id'])]

    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    
    data = []
    for i, run_id in enumerate(run_list):
        ses_data=[]
        for s in T.participant_id:
            # Assemble file name
            print(f"Loading subj {s}")
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

def load_hcp_task_contrast(dataset_dir, subj_list, atlas, ses_list=['all'],
                           hemis=None):
    print('Loading HCP task contrasts...')
    if ses_list == 'all':
        ses_list = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    
    data_dir = dataset_dir + '/derivatives/{0}/func'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at = atlas
    
    data = []
    for s in T.participant_id:
        subj_data = []
        for i, run_id in enumerate(ses_list):

            try:
                # Assemble file name            
                file_name = f'/ses-{run_id}/{s}_tfMRI_{run_id}_level2_hp200_s2_MSMAll.dscalar.nii'
                # Load data / remove medial wall
                dat = this_at.cifti_to_data(data_dir.format(s) + file_name)

                if hemis is not None: # if cortical data
                    stru_idx = this_at.structure.index(hemis_dict[hemis])
                    dat = dat[:,this_at.indx_full[stru_idx]]
                
                subj_data.append(dat)
            except:
                print(f'Subject {s} missing data {run_id}')
                subj_data = []
                break
        
        if not subj_data:
            pass
        else:
            data.append(np.vstack(subj_data))

    return np.stack(data)

def fit_UKBrest_indv_sess(indx=11, K=10, model_type='07', space='MNIAsymC2',
                          indv_sess='ses-rest1', smooth=None, em_params={}):
    ukb_dir = MODEL_DIR + f'/Models_{model_type}'
    nam = f'/asym_Uk_space-{space}_K-{K}_{indv_sess}' if smooth is None \
        else f'/asym_Uk_space-{space}_K-{K}_{indv_sess}_sm{smooth}'

    if not Path(ukb_dir + nam + '.tsv').exists():
        print(
            f'fitting model {model_type} with K={K} on UKB sessions {indv_sess} ...')
        wdir, fname, info, models = fit_all([indx], K=K,
                                            model_type=model_type,
                                            repeats=100,
                                            sym_type='asym',
                                            space=space,
                                            this_sess=[indv_sess],
                                            smooth=smooth,
                                            em_params=em_params)
        fname = fname + f'_{indv_sess}'
        if smooth is not None:
            fname = fname + f'_sm{smooth}'

        info.to_csv(wdir + fname + '.tsv', sep='\t')
        with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)
    else:
        print(f'Already fitted: /Models_{model_type}' + nam)


def fit_HCPrest_indv_sess(indx=7, K=10, model_type='07', space='MNIAsymC2',
                          indv_sess='ses-rest1', indv_type='Ico42Run', ar_model='independent',
                          smooth=None, em_params={}):
    ukb_dir = MODEL_DIR + f'/Models_{model_type}'
    nam = f'/asym_Hc_space-{space}_K-{K}_{indv_sess}_{indv_type}' if smooth is None \
        else f'/asym_Hc_space-{space}_K-{K}_{indv_sess}_{indv_type}_sm{smooth}'

    if not Path(ukb_dir + nam + '.tsv').exists():
        print(
            f'fitting model {model_type} with K={K} on HCP sessions {indv_sess} ...')
        wdir, fname, info, models = fit_all([indx], K=K,
                                            model_type=model_type,
                                            repeats=5,
                                            sym_type='asym',
                                            arrange=ar_model,
                                            space=space,
                                            this_sess=[indv_sess],
                                            this_type=[indv_type],
                                            smooth=smooth,
                                            em_params=em_params)
        fname = fname + f'_{indv_sess}'
        if smooth is not None:
            fname = fname + f'_sm{smooth}'

        info.to_csv(wdir + fname + '.tsv', sep='\t')
        with open(wdir + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)
    else:
        print(f'Already fitted: /Models_{model_type}' + nam)


def get_indiv_parcellation_from_model(model_file, data, device=DEVICE):
    # 1. load group map
    U, _ = ar.load_group_parcellation(model_file, marginal=True, device=DEVICE)

    # 2. load individual parcellation
    if not isinstance(data, list):
        data = [data]
    
    _, model = load_batch_best(model_file, device=device)
    model.initialize(data)
    # U_emlik = model.collect_evidence([e.Estep() for e in model.emissions])
    U_indiv = model.Estep()[0]

    return U, U_indiv, model


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python group_parcellation.py <i>")
    #     sys.exit(1)

    ###### Assemble all the independent test into one pickle file
    # models = []
    # infos = pd.DataFrame()
    # for i in range(6, 31):
    #     fname = MODEL_DIR + f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800_split-{i}'
    #     info = pd.read_csv(fname + '.tsv', sep='\t')
    #     with open(fname + '.pickle', 'rb') as file:
    #         model = pickle.load(file)

    #     models.extend(model.tolist())
    #     infos = pd.concat([infos, info], ignore_index=True)

    # models = np.array(models, dtype=object)
    # infos.to_csv(MODEL_DIR + f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800' + '.tsv', sep='\t')
    # with open(MODEL_DIR + f'/Models_03/asym_Hc_space-fs32k_K-17_HCPsubjects-800' + '.pickle', 'wb') as file:
    #     pickle.dump(models, file)

    ###### Fit UKB group/indiv purely on UKB resting-state data
    # for s in [3]:
    #     for k in [10]:
    #         fit_UKBrest_indv_sess(K=k, model_type='03', space='MNIAsymC2', 
    #                             indv_sess='ses-rest1', smooth=s, 
    #                             em_params={'uniform_kappa': True,
    #                                         'subjects_equal_weight': False,
    #                                         'subject_specific_kappa': False,
    #                                         'parcel_specific_kappa': False})

    ###### Fit HCP group/indiv purely on HCP session resting-state data
    # fit_HCPrest_indv_sess(K=17, model_type='03', space='fs32k_L', 
    #                       indv_sess=None,
    #                       ar_model='independent',
    #                       indv_type='Ico42Run', smooth=None, 
    #                       em_params={'uniform_kappa': True,
    #                                 'subjects_equal_weight': False,
    #                                 'subject_specific_kappa': False,
    #                                 'parcel_specific_kappa': False})
    
    ###### Fit HCP group on 80 training+validation subjects
    # wdir, fname, info, models = fit_all([0, 7], K=17,
    #                                     model_type='03',
    #                                     repeats=50, sc=False,
    #                                     sym_type='asym',
    #                                     arrange='independent',
    #                                     space='fs32k',
    #                                     this_sess=None,
    #                                     this_type=['CondHalf','ROI1483Run'],
    #                                     smooth='4fwhm',
    #                                     extension=None,
    #                                     em_params={'uniform_kappa': True,
    #                                     'subjects_equal_weight': False,
    #                                     'subject_specific_kappa': False,
    #                                     'parcel_specific_kappa': False})
    
    # fname = fname + f'_HCP40subjects_ICA50Run_desc-sm4fwhm_' + sys.argv[1]
    # info.to_csv(wdir + fname + '.tsv', sep='\t')
    # with open(wdir + fname + '.pickle', 'wb') as file:
    #     pickle.dump(models, file)
    

    ###### Convert result to label cifti
    # atlas, _ = am.get_atlas('fs32kAsym')
    # data = mat73.loadmat(Path(ERIS_DIR) / 'dzhi/projects/RANDY15/HCP_avg_40sub/avg_40sub_avg4runs_900sphere_cen_sm4_profile.mat')['profile_mat'].T
    # data = data[np.newaxis, :, np.concat(atlas.vertex_mask)]
    # cond_vec = np.arange(1,1484)
    # part_vec = np.repeat(np.array([1]), 1483)
    # # Build spatial arrangement model
    # ar_model = ar.ArrangeIndependent(15, atlas.P, spatial_specific=True,
    #                                  remove_redundancy=False)
    # # Build emission models for each dataset
    # em_params = {'uniform_kappa': True,
    #              'subjects_equal_weight': True,
    #              'subject_specific_kappa': False,
    #              'parcel_specific_kappa': True}
    # em_model = em.MixVMF(K=15, P=atlas.P, X=hut.indicator(cond_vec),
    #                           part_vec=part_vec, **em_params)
    # M = fm.FullMultiModel(ar_model, [em_model])
    #
    # # Iterate over the number of fits
    # n_fits = 100
    # n_iter = 1000
    # ll = np.empty((n_fits, n_iter))
    # tt = np.empty((n_fits, n_iter))
    # models=[]
    # info = pd.DataFrame({'name': ['asym_Hc'] * n_fits,
    #                      'atlas': ['fs32kAsym'] * n_fits,
    #                      'K': [15] * n_fits,
    #                      'datasets': ['HCP_avrg'] * n_fits,
    #                      'sess': ['avrg'] * n_fits,
    #                      'type': ['ROI1483'] * n_fits,
    #                      'smooth': [None] * n_fits,
    #                      'subj': [40] * n_fits,
    #                      'arrange': ['independent'] * n_fits,
    #                      'emission': ['VMF'] * n_fits,
    #                      'loglik': [np.nan] * n_fits,
    #                      'weighting': [None] * n_fits})
    # for i in range(n_fits):
    #     print(f'Start fit: repetition {i}')
    #
    #     iter_tic = time.perf_counter()
    #     # Copy the object (without data)
    #     m = deepcopy(M)
    #     # Attach the data
    #     m.initialize([data], subj_ind='separate')
    #     pt.cuda.empty_cache()
    #     hut.report_cuda_memory()
    #
    #     # Swith the learning process between independent and RBMs
    #     m, ll, _, _, _ = m.fit_em_ninits(
    #         iter=n_iter,
    #         tol=0.01,
    #         fit_arrangement=True,
    #         fit_emission=True,
    #         init_arrangement=True,
    #         init_emission=True,
    #         n_inits=50,
    #         first_iter=10, verbose=False)
    #
    #     info.loglik.at[i] = ll[-1].cpu().numpy()  # Convert to numpy
    #     m.clear()
    #
    #     # Move to CPU device before storing
    #     m.move_to(device='cpu')
    #     models.append(m)
    #     pt.cuda.empty_cache()
    #
    #     print(f'Done fit: repetition {i}!')
    #
    # models = np.array(models, dtype=object)
    #
    # wdir = MODEL_DIR + '/Models_04'
    # fname = f'/asym_Hc_space-fs32kAsym_K-15_arrange-independent_HCP40-avrg'
    # info.to_csv(wdir + fname + '.tsv', sep='\t')
    # with open(wdir + fname + '.pickle', 'wb') as file:
    #     pickle.dump(models, file)



    KONG2019, network_names, colors = get_kong2019_group_parcellation()
    fname = f'Models_03/indiv_parcellation/validation_set/asym_HCP40+HCPrest-2run-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-2_spatial-10.dlabel.nii'
    colors[[1, 6]] = colors[[6, 1]]
    ut.write_model_to_labelcifti([fname], align=KONG2019, col_names=[f'Hc40'],
                                 label_names=network_names, label_RGBA=colors,
                                 load='all', oname=fname, device=DEVICE)


    DU15, network_names, colors = get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    DU15 = ar.expand_mn_1d(DU15, K=16)
    fname = f'Models_03/task_fusion/asym_MdNiIb_space-fs32k_K-15_sm6fwhm_zstat_masked-hi0.1lo0.1_Ib-jointsess_DU15-inits_equalweights'
    ut.write_model_to_labelcifti([fname], align=DU15[1:,:].cpu().numpy(), col_names=None,
                                    label_names=network_names, label_RGBA=colors,
                                    load='best', oname=fname, device=DEVICE)

    for i in range(50,51):
        fname = f'Models_03/task_fusion/asym_MdPoNiIbWmDeSo_space-fs32k_K-{i}'
        # colors = np.concatenate([np.array([[0,0,0,0]]),plt.cm.get_cmap('tab20', 17).colors], axis=0)
        ut.write_model_to_labelcifti([fname], align=None, col_names=[f'taskfusion-K-{i}'],
                                    label_names=None, label_RGBA=None,
                                    load='best', oname=fname, device=DEVICE)
        
        
    ## Step 1. Load UKB 736 subjects data
    # print(f'Start loading data: UKBresting - ses-rest1 - ICA25All ...')
    # tic = time.perf_counter()
    # data, cond_vec, part_vec, subj_ind = build_ukb_datasets(BASE_DIR,
    #                                                         "participants_filtered_final.tsv",
    #                                                         ses_list=['ses-rest1'],
    #                                                         type=['ICA25All'])
    # toc = time.perf_counter()
    # print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    ## Step 2. Derive individual parcellation
    # 2.1 calculate indiv parcellations directly from fitted model
    # U, U_indv, _ = get_indiv_parcellation_from_model(MODEL_DIR + 
    #                 f'/Models_07/asym_Uk_space-MNIAsymC2_K-7_ses-rest1', data)
    
    # 2.2 calculate indiv parcellations using an existing prior
    # atlas, _ = am.get_atlas('MNIAsymC2')
    # atlas_dir = ATLAS_DIR + '/tpl-MNI152NLin2009cSymC'
    # model_name = f'/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii'
    # U = atlas.read_data(atlas_dir + model_name)
    # U = convert_hard_to_prob(U, strength=3)
    # ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
    #                                       sym_type='asym')
    # U_indv, _, _ = fm.get_indiv_parcellation(ar_model, atlas, data,
    #                                          cond_vec, part_vec, subj_ind, 
    #                                          sym_type='asym', 
    #                                          em_params={'uniform_kappa':False})

    ## Step 3. Plotting parcellations
    # colors = ut.get_colormap_from_lut(fname=BASE_DIR + '/Atlases/tpl-SUIT/atl-Buckner7.lut')
    
    # #-- Group
    # plt.figure(figsize=(10, 10))
    # plot_multi_flat(U.unsqueeze(0).cpu().numpy(), 'MNIAsymC2', grid=(1, 1),
    #                 cmap=colors, dtype='prob', titles=['group prior'])
    # plt.show()

    # #-- Individual
    # plt.figure(figsize=(40, 20))
    # plot_multi_flat(U_indv[0:4].cpu().numpy(), 'MNIAsymC2', grid=(1, 4),
    #                 cmap=colors, dtype='prob',
    #                 titles=["subj_{}".format(i+1) for i in range(U_indv.shape[0])])
    # plt.show()

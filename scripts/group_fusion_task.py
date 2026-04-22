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
import Functional_Fusion.reliability as rel
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
import utils as fut
from utils import get_kong2019_group_parcellation, get_DU15_parcellation

# pytorch cuda global flag
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

GROUP_DIR = '/data/tge/Tian/HCP_img/derivatives/group'
KONG2019, _, _ = get_kong2019_group_parcellation()
KONG2019 = pt.tensor(KONG2019, dtype=pt.get_default_dtype())
DU15, _, _ = get_DU15_parcellation(file_name='DU15NET_PriorProb', atlas_space='fs32k')
DU15 = pt.tensor(DU15, dtype=pt.get_default_dtype())

def build_data_list(datasets, atlas='MNISymC3', sess=None, cond_ind=None, type=None,
                    part_ind=None, part_num=None, subj=None, join_sess=True,
                    join_sess_part=False, smooth=None, hemis=None, num_task=None):
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
        dat = np.nan_to_num(dat)

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
        if join_sess[i]:
            if part_num[i] is not None:
                indx = info[part_ind[i]] == part_num[i]
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
            indices = []
            for s in sessions:
                if part_num[i] is None:
                    indx = info.sess == s
                    indices.append(indx)
                else:
                    for parts in part_num[i]:
                        indx = (info.sess == s) & (info[part_ind[i]] == parts)
                        indices.append(indx)

            for indx in indices:
                this_dat = dat[:, indx, :]
                this_cond_vec = info[cond_ind[i]].values[indx].reshape(-1, )
                this_part_vec = info[part_ind[i]].values[indx].reshape(-1, )

                # QC
                # rw = rel.within_subj(np.nan_to_num(this_dat), cond_vec=this_cond_vec, part_vec=this_part_vec,
                #                      separate='condition_wise', subtract_mean=True)
                # cond_mask = np.where(rw <= 0.5, False, True)
                # cond_mask = np.hstack((cond_mask, cond_mask.copy()))
                # this_dat[~cond_mask, :] = np.nan
                #
                # print(f'{datasets[i]} session {s} - data drop {np.sum(~cond_mask)/2} conditions from {n_subj} subjects at cut-off > 0.5' )
                # print(f'{np.sum(~cond_mask)}/{cond_mask.size} contrasts, drop rate {np.sum(~cond_mask)/cond_mask.size}')

                # Make the zero voxels to nan
                # zero_cols = np.all(this_dat == 0, axis=1, keepdims=True)
                # data.append(np.where(zero_cols, np.nan, this_dat))
                data.append(this_dat)
                # indx = pt.tensor(np.where(idx == True)[0])
                # data.append(pt.index_select(dat, 1, indx))
                cond_vec.append(this_cond_vec)
                part_vec.append(this_part_vec)
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
            # _, model = ut.load_batch_best(trained_emi, device='cuda')
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
              num_task=None,
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
                                                         hemis=hemis,
                                                         num_task=num_task)
    # if num_task is not None:
    #     this_ses = sess[0][0]
    #     data_dir = f'/home/dzhi/eris_mount/Tian/{datasets[0]}/derivatives/group/data'
    #     maps = nb.load(
    #         data_dir + f'/group_space-fs32k_{this_ses}_CondAll_masked-hi0.1lo0.1_binarized.dscalar.nii').get_fdata()
    #     task_idx, _ = fut.greedy_task_beta_selection(maps, num_task, lamda=0)
    #
    #     indices = [np.where(np.isin(cond, np.array(task_idx) + 1))[0] for cond in cond_vec]
    #     for i, idx in enumerate(indices):
    #         data[i] = data[i][:, idx, :]
    #         cond_vec[i] = cond_vec[i][idx]
    #         part_vec[i] = part_vec[i][idx]

    # data = [pt.load(GROUP_DIR + '/processed_ses-rest1.pt'),
    #         pt.load(GROUP_DIR + '/processed_ses-rest2.pt')]
    # cond_vec = [np.tile(np.arange(1,1211), 2), np.tile(np.arange(1,1211), 2)]
    # part_vec = [np.repeat(np.array([1,2]), 1210), np.repeat(np.array([1,2]), 1210)]
    # subj_ind = [np.arange(800), np.arange(800)]
    # data = [(d - np.mean(d, axis=1, keepdims=True)).astype(np.float32) for d in data]
    # data = [np.concatenate(data[0:4], axis=1)]
    # cond_vec = [np.concatenate(cond_vec[0:4])]
    # part_vec = [np.repeat(np.array([1,2,3,4]), 1483)]
    # subj_ind = [subj_ind[0]]

    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    # data = [pt.tensor(dat, dtype=pt.int8).to_sparse() for dat in data]

    # data = mat73.loadmat('/home/dzhi/eris_mount/dzhi/projects/RANDY15/HCP_avg_40sub/avg_40sub_avg4runs_900sphere_cen_sm4_profile.mat')['profile_mat'].T
    # data = np.split(data[0], 2, axis=1) + np.split(data[1], 2, axis=1)
    # cond_vec = [np.arange(1,1484)] * 4
    # part_vec = [np.repeat(np.array([1]), 1483)] * 4
    # subj_ind = [np.arange(20)] * 4

    # Load all necessary data and designs
    n_subj = np.unique(np.concatenate(subj_ind, axis=0)).size
    if hemis == 'L':
        atlas, _ = am.get_atlas('fs32k_L', ut.atlas_dir)
    elif hemis == 'R':
        atlas, _ = am.get_atlas('fs32k_R', ut.atlas_dir)

    print(f'Building fullMultiModel {arrange} + {emission} for fitting...')
    # Load connectiviy matrix if cpRBM with connectiviy is used
    if arrange == 'cRBM_Wc':
        surf = 'Y:/data/FunctionalFusion/Atlases/tpl-fs32k/tpl_fs32k_hemi-L_sphere.surf.gii'
        Wc = ut.get_fs32k_neighbours(surf, remove_mw=True)
        # Wc = ut.get_fs32k_weights(file_type='distGOD_sp',
        #                           hemis='half' if (hemis=='L') or (hemis=='R') else 'full',
        #                           remove_mw=True, max_dist=10, kernel='gaussian', sigma=10,
        #                           device='cuda' if pt.cuda.is_available() else 'cpu')

        # Use sparse tensor if CUDA is enabled, otherwise dense tensor
        # Wc = Wc.to_sparse_csr() if pt.cuda.is_available() else Wc.to_dense()
        # Wc.values().fill_(1)
    else:
        Wc = None
    M = build_model(K, arrange, sym_type, emission, atlas, cond_vec,
                    part_vec, weighting=weighting, Wc=Wc, theta=Wc_theta,
                    num_chain=n_subj, sess=sess, em_params={'num_subj': n_subj, **em_params})
    weights = pt.ones(len(data)) / len(data)
    # weights = pt.tensor([td.shape[1] for td in data])
    # weights = (1 / weights) / pt.sum(1 / weights)
    # weights[0:5] = weights[0:5] / (2 * weights[0:5].sum())
    # weights[5:] = weights[5:] / (2 * weights[5:].sum())
    del Wc
    pt.cuda.empty_cache()
    hut.report_cuda_memory()

    # Initialize data frame for results
    models, priors = [], []
    n_fits = n_rep
    info = pd.DataFrame({'name': [name] * n_fits,
                         'atlas': [atlas.name] * n_fits,
                         'K': [K] * n_fits,
                         'datasets': [datasets] * n_fits,
                         'sess': [sess] * n_fits,
                         'type': [type] * n_fits,
                         'smooth': [smooth] * n_fits,
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
        m.ds_weight = weights
        m.arrange.logpi = pt.log(DU15 + 1e-2)
        m.arrange.logpi = m.arrange.logpi - m.arrange.logpi.mean(dim=0)
        pt.cuda.empty_cache()
        hut.report_cuda_memory()

        # Swith the learning process between independent and RBMs
        if m.arrange.name.startswith('indp'):
            m, ll, _, _, _ = m.fit_em_ninits(
                iter=n_iter,
                tol=0.01,
                fit_arrangement=True,
                fit_emission=True,
                init_arrangement=False,
                init_emission=True,
                n_inits=n_inits,
                first_iter=first_iter, verbose=False)
        elif m.arrange.name.startswith('cRBM'):
            m.random_params(init_arrangement=True,
                            init_emission=True)
            m.arrange.theta = pt.tensor(Wc_theta)
            m, ll, theta, _ = m.fit_sml(
                iter=n_iter,
                batch_size=37,
                stepsize=0.5,
                seperate_ll=False,
                fit_arrangement=True,
                fit_emission=True)
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
        pt.cuda.empty_cache()

        iter_toc = time.perf_counter()
        print(
            f'Done fit: repetition {i} - {name} - {iter_toc - iter_tic:0.4f} seconds!')

    models = np.array(models, dtype=object)
    # plt.plot(tt.T)
    # plt.show()

    return info, models


def fit_all(set_ind=[0, 1, 2, 3], K=10, repeats=100, model_type='01',
            sym_type=['asym', 'sym'], arrange='independent', subj_list=None,
            weighting=None, this_sess=None, space=None, smooth=None,
            sc=True, Wc_theta=1, part_num=None, num_task=None, em_params={}):
    # Get dataset info
    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    datasets = T.name.to_numpy()
    sess = np.array(['all'] * len(T), dtype=object)
    if this_sess is not None:
        for i, idx in enumerate(set_ind):
            sess[idx] = this_sess[i]

    type = T.default_type.to_numpy()
    type[7] = 'ROI1483Run'
    type[13] = 'Ico642Run'
    cond_ind = T.default_cond_ind.to_numpy()
    cond_ind[2] = 'cond_num'
    cond_ind[13] = 'net_id'
    part_ind = np.array(['half'] * len(T), dtype=object)

    # Make the atlas object
    if space is None:
        space = 'MNISymC3'

    hemis = None
    this_space = space
    if space.startswith('fs32k_'):
        hemis = space.split('_')[1]
        space = space.split('_')[0]

    atlas, _ = am.get_atlas(space, ut.atlas_dir)

    # Provide different setttings for the different model types
    join_sess_part = False
    if model_type == '01':
        uniform_kappa = True
        join_sess = True
    elif model_type == '02':
        uniform_kappa = False
        join_sess = True
    elif model_type[:6] == '01-HCP':
        uniform_kappa = True
        weighting = np.repeat(1, len(set_ind) - 1).tolist()
        hcp_weight = model_type.split('HCP')[1]
        weighting.extend([float(f'{hcp_weight[0]}.{hcp_weight[1]}')])
        join_sess = True
    elif model_type == '03':
        uniform_kappa = True
        join_sess = [False, False, True, False]
    elif model_type == '04':
        uniform_kappa = False
        join_sess = False
    elif model_type == '05':
        uniform_kappa = False
        join_sess = True
        join_sess_part = True
    elif model_type == '06':
        uniform_kappa = True
        join_sess = True
        join_sess_part = True
    elif model_type == '07':
        uniform_kappa = False
        join_sess = False

    # Generate a dataname from first two letters of each training data set
    dataname = ''.join(T.two_letter_code[set_ind])
    ut.print_memory_usage()

    for mname in sym_type:
        tic = time.perf_counter()
        name = mname + '_' + ''.join(dataname)
        info, models = batch_fit(datasets[set_ind],
                                 sess=sess[set_ind],
                                 type=type[set_ind],
                                 cond_ind=cond_ind[set_ind],
                                 part_ind=part_ind[set_ind],
                                 subj=subj_list,
                                 atlas=atlas,
                                 K=K,
                                 arrange=arrange,
                                 sym_type=mname,
                                 name=name,
                                 n_inits=20,
                                 n_iter=500,
                                 n_rep=repeats,
                                 first_iter=10,
                                 join_sess=join_sess,
                                 join_sess_part=join_sess_part,
                                 part_num=part_num,
                                 uniform_kappa=uniform_kappa,
                                 weighting=weighting,
                                 smooth=smooth,
                                 hemis=hemis,
                                 second_converge=sc,
                                 Wc_theta=Wc_theta,
                                 num_task=num_task,
                                 em_params=em_params)

        # Save the fits and information
        wdir = ut.model_dir + f'/Models/Models_{model_type}'
        fname = f'/{name}_space-{this_space}_K-{K}'

        # if smooth is not None:
        #     return wdir, fname, info, models

        # if this_sess is not None:
        #     return wdir, fname, info, models

        # if part_num is not None:
        #     return wdir, fname, info, models

        # if subj_list is not None:
        #     wdir = ut.model_dir + f'/Models/Models_{model_type}/leaveNout'
        #     fname = f'/{name}_space-{this_space}_K-{K}'

        #     toc = time.perf_counter()
        #     print(f'Done Model fitting - {mname}. Used {toc - tic:0.4f} seconds!')
        #     return wdir, fname, info, models

        toc = time.perf_counter()
        print(f'Done Model fitting - {mname}. Used {toc - tic:0.4f} seconds!')

        return wdir, fname, info, models


def clear_models(K, model_type='04'):
    for t in ['sym', 'asym']:
        for k in K:
            for s in ['MdPoNiIbHc_00', 'MdPoNiIbHc_02',
                      'MdPoNiIbHc_10']:  # Md','Po','Ni','Hc','Ib','MdPoNiIb','MdPoNiIbHc','MdPoNiIbHc_00']:
                fname = f"Models_{model_type}/{t}_{s}_space-MNISymC3_K-{k}"
                try:
                    ut.clear_batch(fname)
                    print(f"cleared {fname}")
                except:
                    print(f"skipping {fname}")


def leave_one_out_fit(dataset=[0], model_type=['01'], K=10):
    # Define some constant
    nsubj = [24, 8, 6, 12, 100]
    ########## Leave-one-out fitting ##########
    for m in model_type:
        this_nsub = nsubj[dataset[0]]
        for i in range(this_nsub):
            print(
                f'fitting dataset:{dataset} - model:{m} - leaveNout: {i} ...')
            sub_list = np.delete(np.arange(this_nsub), i)
            wdir, fname, info, models = fit_all(dataset, K,
                                                model_type=m,
                                                sym_type=['asym'],
                                                subj_list=[sub_list])
            fname = fname + f'_leave-{i}'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)


def fit_indv_sess(indx=3, model_type='01', K=10):
    datasets = np.array(['MDTB', 'Pontine', 'Nishimoto',
                         'IBC', 'WMFS', 'Demand', 'Somatotopic'],
                        dtype=object)
    _, _, my_dataset = get_dataset(ut.base_dir, datasets[indx])
    sess = my_dataset.sessions
    for indv_sess in sess:
        ibc_dir = ut.model_dir + f'/Models/Models_{model_type}'
        nam = f'/asym_Ib_space-MNISymC3_K-{K}_{indv_sess}'

        if not Path(ibc_dir + nam + '.tsv').exists():
            print(
                f'fitting model {model_type} with K={K} on IBC sessions {indv_sess} ...')
            wdir, fname, info, models = fit_all([indx], K,
                                                model_type=model_type,
                                                repeats=100,
                                                sym_type=['asym'],
                                                this_sess=[[indv_sess]])
            fname = fname + f'_{indv_sess}'
            info.to_csv(wdir + fname + '.tsv', sep='\t')
            with open(wdir + fname + '.pickle', 'wb') as file:
                pickle.dump(models, file)


def fit_two_IBC_sessions(K=10, sess1='clips4', sess2='rsvplanguage', model_type='04'):
    ibc_dir = ut.model_dir + f'/Models/Models_{model_type}/IBC_sessFusion'
    nam = f'/asym_Ib_space-MNISymC3_K-{K}_ses-{sess1}+{sess2}'

    if not Path(ibc_dir + nam + '.tsv').exists():
        print(
            f'fitting model {model_type} with K={K} on IBC sessions {sess1} + {sess2} ...')
        wdir, fname, info, models = fit_all([3], K, model_type=model_type, repeats=50,
                                            sym_type=['asym'], this_sess=[['ses-' + sess1,
                                                                           'ses-' + sess2]])
        fname = fname + f'_ses-{sess1}+{sess2}'
        info.to_csv(wdir + '/IBC_sessFusion' + fname + '.tsv', sep='\t')
        with open(wdir + '/IBC_sessFusion' + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)


def fit_all_datasets(space='MNISymC2',
                     msym='sym',
                     K=[68],
                     datasets_list=[[0, 1, 2, 3, 4, 5, 6]]):
    # -- Model fitting --
    # datasets_list = [[0], [1], [2], [3], [4], [5], [6], [0, 1, 2, 3, 4, 5, 6, 7]]

    T = pd.read_csv(ut.base_dir + '/dataset_description.tsv', sep='\t')
    # for i in range(7):
    #     datasets = [0, 1, 2, 3, 4, 5, 6]
    #     datasets.remove(i)
    for datasets in datasets_list:
        for k in K:
            for t in ['03', '04']:
                datanames = ''.join(T.two_letter_code[datasets])
                wdir = ut.model_dir + f'/Models'
                fname = f'/Models_{t}/{msym}_{datanames}_space-{space}_K-{k}'

                # inf, m = load_batch_fit(fname)
                # if not m[0].ds_weight.is_cuda:
                #     print(f'Convert model {t} with K={k} {fname} to cuda...')
                #     # move_batch_to_device(fname, device='cuda')
                if not Path(wdir + fname + '.tsv').exists():
                    print(f'fitting model {t} with K={k} as {fname}...')
                    fit_all(datasets, k, model_type=t,
                            repeats=100, sym_type=[msym])
                else:
                    print(f'model {t} with K={k} already fitted as {fname}')


def refit_model(model, new_info):
    """Refits model.

    Args:
        model:      Model to be refitted
        new_info:       Information for new model

    Returns:
        model: Refitted model

    """

    if type(model.arrange) is ar.ArrangeIndependentSymmetric:
        M = fm.FullMultiModel(model.arrange, model.emissions)
    else:
        M = fm.FullMultiModel(model.arrange, model.emissions)

    model_settings = {'Models_01': [True, True, False],
                      'Models_02': [False, True, False],
                      'Models_03': [True, False, False],
                      'Models_04': [False, False, False],
                      'Models_05': [False, True, True]}

    # uniform_kappa = model_settings[new_info.model_type][0]
    join_sess = model_settings[new_info.model_type][1]
    join_sess_part = model_settings[new_info.model_type][2]

    datasets = new_info.datasets.strip("'[").strip("]'").split("' '")
    sessions = new_info.sess.strip("'[").strip("]'").split("' '")
    types = new_info.type.strip("'[").strip("]'").split("' '")

    data, cond_vec, part_vec, subj_ind = build_data_list(datasets,
                                                         atlas=new_info.atlas,
                                                         sess=sessions,
                                                         type=types,
                                                         join_sess=join_sess,
                                                         join_sess_part=join_sess_part)

    # Attach the data
    M.initialize(data, subj_ind=subj_ind)

    # Refit emission models
    print(f'Freezing arrangement model and fitting emission models...\n')

    M, ll, _, _ = M.fit_em(iter=500, tol=0.01,
                           fit_emission=True,
                           fit_arrangement=False,
                           first_evidence=True)

    # make info from a Series back to a dataframe
    new_info = pd.DataFrame(new_info.to_dict(), index=[0])
    new_info['loglik'] = ll[-1].item()
    # Plot ll
    #
    # pt.Tensor.ndim = property(lambda self: len(self.shape))
    # x = pt.linspace(0,ll.shape[0], ll.shape[0])
    # plt.figure()
    # plt.plot(x[1:], ll[1:])

    return M, new_info

def convert_sparse_list_to_sparse(file):
    sparse_tensors = pt.load(file)
    
    # Prepare to stack tensors by adjusting indices and values
    final_indices = pt.empty((3, 0), dtype=pt.long)
    final_values = pt.empty(0)

    for i, tensor in enumerate(sparse_tensors):
        # Add an extra dimension to the indices to reflect the P dimension
        indices = tensor.indices()  # shape: (2, num_nonzero_elements)
        values = tensor.values()  # shape: (num_nonzero_elements)
        
        # Add the "P" dimension index (i) as a new row in the indices tensor
        p_indices = pt.full((1, indices.size(1)), i, dtype=pt.long)  # shape: (1, num_nonzero_elements)
        expanded_indices = pt.cat([p_indices, indices], dim=0)  # shape: (3, num_nonzero_elements)
        
        # Collect expanded indices and values
        final_indices = pt.cat([final_indices, expanded_indices], dim=1)
        final_values = pt.cat([final_values, values])

    # Concatenate all indices and values
    res = pt.sparse_coo_tensor(final_indices, final_values, (800, 2420, 59518))

def make_processed_Y(file, part=1):
    # Y = pt.load(GROUP_DIR + file)
    Y = load_large_sparse(GROUP_DIR + file, shape=(800, 1210, 59518))
    cond_vec = np.tile(np.arange(1,1211), 2)
    part_vec = pt.tensor(np.repeat(np.array([1,2]), 1210), dtype=pt.int)
    subj_ind = np.arange(Y.shape[0])
    X = pt.tensor(matrix.indicator(cond_vec), dtype=pt.float32)

    # Split the design matrix X and data and calculate (X^T*X)-1*X^T in each partition
    parts = pt.unique(part_vec)

    # Create array of new normalized data
    proc_Y, W = [], []

    Y_part, W_part = [],[]
    x = X[part_vec==part, :]
    idx = pt.where(part_vec == part)[0]
    ut.print_memory_usage()

    Y = pt.index_select(Y, 1, idx)
    ut.print_memory_usage()

    for sub in subj_ind:
        if sub % 100 == 0:
            print(f"Done subject {sub}")

        this_Y = pt.index_select(Y, 0, pt.tensor([sub]))[0]
        this_Y = pt.index_select(this_Y, 0, idx).to_dense()
        this_W = pt.sqrt(pt.sum(this_Y ** 2, dim=0, keepdim=True))
        
        Y_part.append((pt.matmul(pt.linalg.pinv(x), this_Y.to(pt.float32)) / this_W).unsqueeze(0).to_sparse())
        W_part.append(this_W)

    proc_Y.append(pt.cat(Y_part, dim=0))
    W.append(pt.stack(W_part))

    # Keep track of how many available partions per voxels
    num_part = pt.sum(~pt.stack(W).isnan(),dim=0)
    proc_Y = pt.nan_to_num(pt.stack(proc_Y)).sum(dim=0)
    M = proc_Y.shape[1]

    return proc_Y, num_part, M, proc_Y.shape[0]


def load_large_sparse(file_path, shape=(800, 2420, 59518)):
    # Load the saved sparse tensor data
    loaded_sparse_data = pt.load(file_path)

    # Reconstruct the sparse tensor
    return pt.sparse_coo_tensor(
        indices=loaded_sparse_data["indices"],
        values=loaded_sparse_data["values"],
        size=shape, dtype=pt.int8, is_coalesced=True
    )

def datasets_reliability(dataset, space='fs32k', sess=['all'], cond_ind=['cond_num_uni'],
                         part_ind=['half'], type=['CondHalf'], smooth=[None], separation='none'):
    # Get dataset info
    print(f'Start loading data: {dataset} - {sess} - {type} ...')
    tic = time.perf_counter()
    # data, cond_vec, part_vec, subj_ind = build_data_list(dataset, atlas=space,
    #                                                      sess=sess, cond_ind=cond_ind,
    #                                                      type=type, part_ind=part_ind,
    #                                                      subj=[None], join_sess=False,
    #                                                      join_sess_part=False, smooth=smooth)

    data, info, ds = get_dataset(ut.base_dir, 'WMFS', atlas=space, sess='ses-01',
                                type='CondHalf', smooth='6fwhm_zstat_masked-hi0.1lo0.1')
    rw = rel.within_subj(np.nan_to_num(data), cond_vec=info.cond_num, part_vec=info.half,
                         separate=separation, subtract_mean=True)

    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    # Calculate reliability
    # rw = rel.within_subj_loo(data[0], cond_vec=cond_vec[0], part_vec=part_vec[0],
    #                          separate=separation, subtract_mean=True)

    data, info, ds = get_dataset(ut.base_dir, 'WMFS', atlas=space, sess='all',
                                 type='CondHalf', smooth='6fwhm_zstat_masked-hi0.1lo0.1')
    rw2 = rel.within_subj(np.nan_to_num(data), cond_vec=info.cond_num, part_vec=info.half,
                         separate=separation, subtract_mean=True)

    print(rw)
    return rw


if __name__ == "__main__":
    # Y, num_part, M, num_subj = make_processed_Y('/backup_ses-rest1.pt')
    # convert_sparse_list_to_sparse(GROUP_DIR + '/processed_ses-rest2_Ico642Run.pt')
    # if len(sys.argv) != 3:
    #     print("Usage: python group_fusion_task.py <K> <smooth>")
    #     sys.exit(1)

    # mdtb_rw = datasets_reliability(['MDTB'], space='fs32k', sess=['ses-s1'], cond_ind=['cond_num_uni'],
    #                           part_ind=['half'], type=['CondHalf'], smooth=[None], separation='condition_wise')
    #
    # data, info = dataset.get_data('MNISymC3', ses, 'CondRun')
    # rw = reliability.within_subj(data,
    #                              part_vec=info.run,
    #                              cond_vec=info.reg_id,
    #                              separate='None',
    #                              subtract_mean=True)

    A = pd.read_csv('/home/dzhi/eris_mount/Tian/HCP_img/participants.tsv', delimiter='\t')
    B = pd.read_csv(f'/home/dzhi/eris_mount/Tian/HCP_img/subj_list/HCP40_training_KONG2019.tsv', delimiter='\t')
    hcp_subj_ind = np.array(A[A['participant_id'].isin(B['participant_id'])].index)

    num_parcel = [15]
    smoothing_levels = [6]
    # num_tasks = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    # for ses in ['ses-01', 'ses-02']:
    #     for num_task in num_tasks:
    for k in num_parcel:
        for train_smooth in smoothing_levels:
            print(f'Training K={k}, and smoothing = {train_smooth} ...')
            wdir, fname, info, models = fit_all(set_ind=[0,2,3], K=k, repeats=10, model_type='03',
                                                this_sess=None, sym_type=['asym'], space='fs32k',
                                                smooth=[f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                                                        f'{train_smooth}fwhm_zstat_masked-hi0.1lo0.1',
                                                        f'5_zstat_masked-hi0.1lo0.1'],
                                                subj_list=[None,None,None],
                                                arrange='independent', Wc_theta=0.0,
                                                part_num=[None, None, None],
                                                em_params={'uniform_kappa': True,
                                                           'subjects_equal_weight': True,
                                                           'subject_specific_kappa': False,
                                                           'parcel_specific_kappa': False})


        fname = fname + f'_sm{train_smooth}fwhm_zstat_masked-hi0.1lo0.1_Ib-jointsess_equalweights'
        info.to_csv(wdir + '/task_fusion' + fname + '.tsv', sep='\t')
        with open(wdir + '/task_fusion' + fname + '.pickle', 'wb') as file:
            pickle.dump(models, file)
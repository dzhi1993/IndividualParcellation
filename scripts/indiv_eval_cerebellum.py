#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: dzhi
"""
import time
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
import FusionModel.util as futil
import FusionModel.evaluate as ev

# from group_parcellation import build_ukb_datasets, convert_hard_to_prob, get_indiv_parcellation_from_model
# from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR, DEVICE

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
    Pgroup = pt.argmax(U_group, dim=0) + 1
    Pindiv = pt.argmax(U_indiv, dim=1) + 1
    homo_group = ev.calc_test_homogeneity(U_group, t_data)
    homo_indiv = ev.calc_test_homogeneity(U_indiv, t_data)
    dcbc_group = ev.calc_test_dcbc(Pgroup, t_data, dist)
    dcbc_indiv = ev.calc_test_dcbc(Pindiv, t_data, dist)

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

if __name__ == "__main__":
    atlas, _ = am.get_atlas('MNIAsymC2')
    ######## Step 1. Load UKB 736 subjects training data
    print(f'Start loading data: UKBresting - ses-rest1 - ICA25All ...')
    tic = time.perf_counter()
    data, cond_vec, part_vec, subj_ind = build_ukb_datasets(BASE_DIR,
                                                            "test.tsv",
                                                            space=atlas.name,
                                                            ses_list=['ses-rest1'],
                                                            type=['ICA25All'])
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    ## Load UKB 736 subjects test data
    print(f'Start loading data: UKBresting - ses-rest2 - Tseries ...')
    tic = time.perf_counter()
    t_data, _, _, _ = build_ukb_datasets(BASE_DIR,
                                         "test.tsv",
                                         space=atlas.name,
                                         ses_list=['ses-rest2'],
                                         type=['Tseries'])
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    ######## Step 2. Generate group / indiv parcellations
    # Option 1: calculate indiv parcellations directly from fitted model
    # U, U_indv, M = get_indiv_parcellation_from_model(MODEL_DIR + 
    #                   f'/Models_07/asym_Uk_space-MNIAsymC2_K-7_ses-rest1', data)

    # Option 2: calculate indiv parcellations from existing group map
    atlas_dir = ATLAS_DIR + '/tpl-MNI152NLin2009cSymC'
    model_name = f'/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii'
    U_hard = atlas.read_data(atlas_dir + model_name)
    conf_dir = '/data/tge/dzhi/Indiv_par/Buckner_JNeurophysiol11_MNI152'
    conf_name = f'/Buckner2011_7NetworksConfidence_MNI152_FreeSurferConformed1mm_LooseMask.nii.gz'
    conf_map = atlas.read_data(conf_dir + conf_name)
    U = convert_hard_to_prob(U_hard, strength=1, confidence=conf_map)

    ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
                                          sym_type='asym')
    U_indv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
                                             cond_vec, part_vec, subj_ind,
                                             sym_type='asym',
                                             em_params={'num_subj': data[0].shape[0],
                                                        'uniform_kappa': None,
                                                        'subjects_equal_weight':True,
                                                        'subject_specific_kappa': True,
                                                        'parcel_specific_kappa': True})

    # em_params={'subjects_equal_weight':True,
    #             'uniform_kappa': None,
    #             'subject_specific_kappa': False,
    #             'parcel_specific_kappa': True}

    # del data
    # pt.cuda.empty_cache()
    # fm.report_cuda_memory()

    ######## Step 3: Evaluate individual maps using DCBC
    # Step 3.1: compute the distance matrix
    dist = ev.compute_dist(atlas.world.T, resolution=1)
    # Step 3.2: Gatering all necessary information for evaluation
    eval_info = make_eval_info(M, train_info=['UKB'], train_sess='ses-rest1',
                            tdata='UKB', test_sess='ses-rest2', 
                            model_type='Models_04', group_map_name='Buckner7',
                            test_kappa=None)
    # Step 3.3: Do DCBC evaluation on the second half data
    res = eval_parcel_DCBC(U, U_indv, t_data[0], dist, eval_info,
                            out_file='eval_dcbc_indiv_Buckner7_k-7_model-04_test.tsv')
    dice = [hev.dice_coefficient(pt.tensor(U_hard), pt.argmax(U_indv, dim=1)[i]) 
            for i in range(U_indv.shape[0])]
    # res.to_csv(f'eval_dcbc_indiv_Buckner7_k-7_model-04_test2_prior.tsv', index=False, sep='\t')

    ######## Step 4: Visualization
    # Step 4.1 (optional): plot the DCBC results
    ev_df = pd.read_csv('eval_dcbc_indiv_parcellations.tsv', sep='\t')
    plt.figure(figsize=(5, 5))
    df = pd.melt(ev_df, var_name='group', value_name='value')
    df = df.loc[(df['group'] == 'dcbc_group') | (df['group'] == 'dcbc_indiv')]
    sb.barplot(x='group', y='value', errorbar="se", width=0.7, data=df)
    plt.show()

    # Step 4.2: plot group parcellation
    plt.figure(figsize=(10, 10))
    plot_multi_flat(U.unsqueeze(0).cpu().numpy(), 'MNIAsymC2', grid=(1, 1),
                    cmap='tab20', dtype='prob', titles=['group prior'])
    plt.show()

    # Step 4.3: plot individual parcellation
    plt.figure(figsize=(40,20))
    plot_multi_flat(U_indv.cpu().numpy(), 'MNIAsymC2', grid=(2, 5),
                    cmap='tab20', dtype='prob',
                    titles=["subj_{}".format(i+1) for i in range(U_indv.shape[0])])
    plt.show()
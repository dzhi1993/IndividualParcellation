#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: dzhi
"""
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
import FusionModel.util as futil
import FusionModel.evaluate as ev

from global_config import MODEL_DIR, BASE_DIR, DEVICE

def make_eval_info(M, train_info=None, tdata='MDTB',
                   group_type='Models_03', indivtrain_ind='half',
                   indivtrain_values=1, indivtest_values=2,
                   test_kappa=None):
    """ Collects all the information from the model and the
        training and test data sets into a single dictionary

    Args:
        M (fm.Model): model object
        train_info (dict): training data information
        test_info (dict): test data information

    Returns:
        minfo (dict): model information
    """
    if train_info is None:
        train_info = pd.Series()
    minfo = train_info
    minfo['test_data'] = tdata
    minfo['group_type'] = group_type
    minfo['indivtrain_ind'] = indivtrain_ind
    minfo['indivtrain_val'] = indivtrain_values
    minfo['indivtest_val'] = indivtest_values
    # minfo['indiv_train_kappa'] = M.emissions[0].uniform_kappa
    # minfo['indiv_test_kappa'] = test_kappa
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
    dcbc_group = ev.calc_test_dcbc(Pgroup, t_data, dist)
    dcbc_indiv = ev.calc_test_dcbc(Pindiv, t_data, dist)

    # ------------------------------------------
    # Collect the information from the evaluation
    # in a data frame
    train_datasets = minfo.datasets
    if isinstance(minfo.datasets, pd.Series):
        train_datasets = minfo.datasets.tolist()
    ev_df = pd.DataFrame({'model_name': [minfo['name']] * num_subj,
                          'atlas': [minfo.atlas] * num_subj,
                          'K': [minfo.K] * num_subj,
                          'train_data': [train_datasets] * num_subj,
                          'train_loglik': [minfo.loglik] * num_subj,
                          'test_data': [minfo.test_data] * num_subj,
                          'group_type': [minfo.group_type] * num_subj,
                          'indivtrain_ind': [minfo.indivtrain_ind] * num_subj,
                          'indivtrain_val': [minfo.indivtrain_val] * num_subj,
                          'indivtest_val': [minfo.indivtest_val] * num_subj,
                          'subj_num': np.arange(num_subj),
                        #   'indiv_train_kappa': [minfo.indiv_train_kappa] * num_subj,
                        #   'indiv_test_kappa': [minfo.indiv_test_kappa] * num_subj
                          })
    # Add all the evaluations to the data frame
    ev_df['dcbc_group'] = dcbc_group.cpu()
    ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
    ev_df.to_csv(out_file, index=False, sep='\t')
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
    ## Step 1: Loading a pre-trained group model
    atlas, _ = am.get_atlas('MNISymC3')
    model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'
    U, minfo = ar.load_group_parcellation(MODEL_DIR + model_name, device=DEVICE)
    ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                          sym_type='asym')
    # Step 2: Get data set and train the individual maps
    data, info, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, subj=None)
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)
    # Here, we, for example, use the first half to train the individual maps
    indiv_par, _, M = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
                                                [cond_v[0]], [part_v[0]],
                                                [sub_ind[0]])

    ## Step 3: Evaluate individual maps using DCBC
    # Step 3.1: compute the distance matrix
    dist = ev.compute_dist(atlas.world.T, resolution=1)
    # Step 2.2: Gatering all necessary information for evaluation
    eval_info = make_eval_info(M, minfo, info, tdata='MDTB', group_type='Models_03',
                               indivtrain_ind='half', indivtrain_values=1,
                               indivtest_values=2, test_kappa=None)
    # Step 2.3: Do DCBC evaluation on the second half data
    eval_parcel_DCBC(U, indiv_par, tdata[1], dist, eval_info,
                     out_file='eval_dcbc_indiv_parcellations.tsv')
    # Step 2.4 (optional): plot the DCBC results
    ev_df = pd.read_csv('eval_dcbc_indiv_parcellations.tsv', sep='\t')
    plt.figure(figsize=(5, 5))
    df = pd.melt(ev_df, var_name='group', value_name='value')
    df = df.loc[(df['group'] == 'dcbc_group') | (df['group'] == 'dcbc_indiv')]
    sb.barplot(x='group', y='value', errorbar="se", width=0.7, data=df)
    plt.show()

    # Step 3: Visualization
    plt.figure(figsize=(20,20))
    plot_multi_flat(indiv_par.cpu().numpy(), 'MNISymC3', grid=(6, 4),
                    cmap='tab20', dtype='prob',
                    titles=["subj_{}".format(i+1) for i in range(indiv_par.shape[0])])
    plt.show()
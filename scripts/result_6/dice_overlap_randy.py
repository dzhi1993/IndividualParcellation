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
from itertools import combinations
import IndividualParcellation.scripts.group_eval as ge

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

def make_network_plot(space='MNISymC3',
                      model_name='Models_03/asym_Md_space-MNISymC3_K-17'):
    atlas, _ = am.get_atlas(space)
    _, M = ut.load_batch_best(model_name, device='cuda')
    U_group_hard = pt.argmax(M.arrange.marginal_prob(), dim=0).cpu().numpy() + 1
    colors = ut.get_cmap(model_name)
    colors[12] = np.array([249 / 255, 178 / 255, 247 / 255, 1.])

    U_parcels = [U_group_hard]
    for i in range(1,18):
        this_U = np.copy(U_group_hard)
        this_U[this_U != i] = 0
        U_parcels.append(this_U)

    U_parcels = np.stack(U_parcels)
    plt.figure(figsize=(50, 25))
    ut.plot_multi_flat(U_parcels, atlas.name, grid=(3, 6),
                    cmap=colors, dtype='label', titles=[f"network_{i}" for i in range(18)])
    plt.savefig('asym_Md_space-MNISymC3_K-17_Networks.png', format='png')
    plt.show()


if __name__ == "__main__":
    ## Step 1: Get data set and train the individual maps
    atlas, _ = am.get_atlas('fs32k')
    randy_good_subjlist = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14]
    group_strength_list = [4]
    spatial_list = [0]
    run_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    prior_name = 'HBPHc'

    ## Step 3: Run dice coefficient between indivpar1 and indivpar2
    # network_names = ['VentralAttentionB', 'ControlA', 'VisualA', 'TemporalParietal', 'DorsalAttentionA',
    #                  'ControlC', 'DefaultA', 'ControlB', 'VentralAttentionA', 'VisualB', 'SomatomotorA',
    #                  'DefaultB', 'DefaultC', 'SomatomotorB', 'DorsalAttentionB', 'Auditory', 'VisualC']
    # mapping = np.array([3, 2, 7, 8, 4, 2, 1, 2, 3, 7, 6, 1, 1, 6, 4, 5, 7])
    # 1-Default, 2-Control, 3-vATTN, 4-dATTN, 5-AUD, 6-SMOT, 7-Visual, 8-LANG

    network_names = ['VIS-P','CG-OP','DN-B','SMOT-B','AUD','PM-PPr','dATN-B','SMOT-A',
                     'LANG','FPN-B','FPN-A','dATN-A','VIS-C','SAL/PMN','DN-A']
    mapping = np.array([7, 3, 1, 6, 5, 9, 4, 6, 8, 2, 2, 4, 7, 3, 1])
    # 1-Default, 2-FPN, 3-CGOP/SAL, 4-dATTN, 5-AUD, 6-SMOT, 7-Visual, 8-LANG, 9-PM-PPr

    df_all = pd.DataFrame()
    pair_wise_dice = np.zeros((len(run_list), len(run_list), 11, 11))
    for counter_p, p in enumerate(group_strength_list):
        for r1, r2 in combinations(run_list, 2):
            print(f'prior strength is {p}; the MRF strength is 0; run {r1} vs. run {r2} ...')
            # indiv_par_1 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
            #                 f'/asym_MdNiIbHc+HCPrest-1run1-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-10_spatial-5.dlabel.nii').get_fdata()[:]
            indiv_par_1 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/runs' +
                                  f'/asym_{prior_name}+RANDYrest-1run{r1}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-0.dlabel.nii').get_fdata()[:]

            # indiv_par_1 = mapping[indiv_par_1.astype(int) - 1]
            indiv_par_1 = pt.tensor(indiv_par_1, dtype=pt.get_default_dtype(), device=DEVICE)[randy_good_subjlist]

            # indiv_par_2 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
            #                 f'/asym_MdNiIbHc+HCPrest-1run2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-10_spatial-5.dlabel.nii').get_fdata()[:]
            indiv_par_2 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/runs' +
                                  f'/asym_{prior_name}+RANDYrest-1run{r2}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-0.dlabel.nii').get_fdata()[:]
            # indiv_par_2 = mapping[indiv_par_2.astype(int) - 1]
            indiv_par_2 = pt.tensor(indiv_par_2, dtype=pt.get_default_dtype(), device=DEVICE)[randy_good_subjlist]

            # indiv_par_1 = pt.argmax(indiv_par_1, dim=1)+1
            # indiv_par_2 = pt.argmax(indiv_par_2, dim=1)+1

            K=15
            # ## 1. Within dice
            # within_dice = [hev.dice_coefficient(indiv_par_1[i],
            #                                     indiv_par_2[i],
            #                                     label_matching=False).item()
            #                 for i in range(indiv_par_1.shape[0])]
            #
            # num_subj = indiv_par_1.shape[0]
            # ev_df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
            #                       'K': [K] * num_subj,
            #                       'networks': ['all'] * num_subj,
            #                       'subj': np.arange(1,num_subj+1)})
            # ev_df['dice'] = within_dice
            # ev_df['type'] = prior_name
            #
            # # ev_df = pd.DataFrame()
            # for sub in range(num_subj):
            #     networks_dice = hev.dice_coefficient(indiv_par_1[sub], indiv_par_2[sub],
            #                                          label_matching=False, separate=True)
            #     df = pd.DataFrame({'atlas': [atlas.name] * K,
            #                        'K': [K] * K,
            #                        'networks': np.arange(1, K+1),
            #                        'subj': [sub+1] * K})
            #     df['dice'] = networks_dice.cpu().numpy()
            #     df['type'] = prior_name
            #     ev_df = pd.concat([ev_df, df], ignore_index=True)
            #
            # ev_df["strength"] = p
            # ev_df["spatial_w"] = 0
            # ev_df["first_run"] = r1
            # ev_df["second_run"] = r2
            # df_all = pd.concat([df_all, ev_df], ignore_index=True)

            ## 2. Between dice
            print("1-Visual, 2-Motor, 3-dorsal attention, 4-ventral attention, 5-limbic, 6-frontoparietal, 7-DMN")
            between_dice = np.zeros((indiv_par_1.shape[0],indiv_par_2.shape[0], K))
            for i, par1 in enumerate(indiv_par_1):
                if i % 10 == 0:
                    print(f"Done {i} subjects between dice..")
                for j, par2 in enumerate(indiv_par_2):
                    # dice1 = hev.dice_coefficient(indiv_par_1[i], indiv_par_1[j],
                    #                              label_matching=False, separate=False).cpu().numpy()
                    # dice2 = hev.dice_coefficient(indiv_par_2[i], indiv_par_2[j],
                    #                              label_matching=False, separate=False).cpu().numpy()
                    dice3 = hev.dice_coefficient(indiv_par_1[i], indiv_par_2[j],
                                                 label_matching=False, separate=False).cpu().numpy()
                    # dice4 = hev.dice_coefficient(indiv_par_1[j], indiv_par_2[i],
                    #                              label_matching=False, separate=False).cpu().numpy()
                    # between_dice[i,j,:] = (dice1 + dice2 + dice3 + dice4)/4
                    between_dice[i, j, :] = dice3

            pair_wise_dice[r1-1,r2-1,:,:] = between_dice[:,:,0]
            y_values = between_dice[np.triu_indices(between_dice.shape[0], k=1)]
            num_subj = y_values.shape[0]
            df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
                                  'K': [K] * num_subj,
                                  'networks': ['all'] * num_subj,
                                  'subj': np.arange(1, num_subj+1)})
            df['dice'] = y_values[:,0]
            df['type'] = prior_name
            df["strength"] = p
            df["spatial_w"] = 0
            df["first_run"] = r1
            df["second_run"] = r2
            df_all = pd.concat([df_all, df], ignore_index=True)

    df_all.to_csv(MODEL_DIR +
                  "/Models_03/indiv_parcellation/HCP200_test_set/dice/dice_within_networks_indiv-mRBM_MdNiIbHc-HCP200_rest1run_K-17_group-5_spatial_1.tsv",
        index=False, sep='\t')

    # Calculate mean and standard deviation of the y values
    mean_y = np.mean(y_values)
    std_y = np.std(y_values)
    plt.axhline(y=mean_y, color='r', linestyle='--', label='Mean')
    plt.fill_betweenx(y=[mean_y - std_y, mean_y + std_y],
                      x1=0, x2=17,
                      color='grey', alpha=0.3, label='Standard Deviation')
    plt.savefig('dice.pdf', format='pdf')
    plt.show()
    # Vectorize the upper triangular part
    vectorized_upper_triangle = upper_triangle.flatten()

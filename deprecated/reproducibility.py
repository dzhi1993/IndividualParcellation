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
import HierarchBayesParcel.evaluation as hev
import scipy.io as spio
import matplotlib.pyplot as plt

from itertools import combinations
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR, DEVICE
HCP_DIR = '/data/tge/Tian/HCP_img'
hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
RES_DIR = '/data/tge/dzhi/Indiv_par/Evaluations'


def intra_subjects_dice(parcels1, parcels2):
    """ The function to calculate the intra subject similarity
        using the Dice coefficient. It takes input of two group
        of individual parcellations, supposing they have the
        same number of subjects.

    Args:
        parcels1 (pt.tensor): individual parcellations with
            shape of (n_subj, n_voxels)
        parcels2 (pt.tensor): individual parcellations with
            shape of (n_subj, n_voxels)
    
    Returns:
        results:

    """
    num_subj1 = parcels1.shape[0]
    num_subj2 = parcels2.shape[0]
    assert num_subj1 == num_subj2, \
        "The input individual parcellations must have same " \
        "number of subjects!"
    between_dice = []
    for i, j in combinations(range(num_subj1), 2):
        tic = time.perf_counter()
        dice1 = hev.dice_coefficient(parcels1[i], parcels1[j], 
                                    label_matching=False).item()
        dice2 = hev.dice_coefficient(parcels2[i], parcels2[j], 
                                    label_matching=False).item()
        dice3 = hev.dice_coefficient(parcels1[i], parcels2[j], 
                                    label_matching=False).item()
        dice4 = hev.dice_coefficient(parcels1[j], parcels2[i], 
                                    label_matching=False).item()
        toc = time.perf_counter()
        print(f'Dice: subject {i} and {j}. Used {toc - tic:0.4f} seconds!')

        between_dice.append((dice1+dice2+dice3+dice4)/4)

    return between_dice

def make_eval_dataframe(dice, K=17, type='within', network='average', 
                        group_strength=60, spatial_strength=90):
    if isinstance(dice, list):
        num_row = len(dice)
    else:
        raise ValueError('Input dice must be a list of value!')
         
    res = pd.DataFrame({'atlas': ['fs32k'] * num_row,
                        'K': [K] * num_row,
                        'type': [type] * num_row,
                        'network': [network] * num_row,
                        'group_strength': [group_strength] * num_row,
                        'spatial_strength': [spatial_strength] * num_row,
                        'dice': dice})
    return res
     

if __name__ == "__main__":
    # 1. load individual parcellations of the two sessions
    par_dir = '/data/tge/dzhi/Indiv_par/Models/Models_03/indiv_parcellation'
    parcel_ses1 = nb.load(par_dir + '/test_set' + '/HBP_HCPtest-indiv_space-fs32k_K-17_' + \
                          'ROI1483Run_groupstrengh-60_spatial-90_ses-rest1.dlabel.nii').get_fdata()[:]
    parcel_ses2 = nb.load(par_dir + '/test_set' + '/HBP_HCPtest-indiv_space-fs32k_K-17_' + \
                          'ROI1483Run_groupstrengh-60_spatial-90_ses-rest2.dlabel.nii').get_fdata()[:]
    parcel_ses1 = pt.tensor(parcel_ses1, dtype=pt.get_default_dtype(), device=DEVICE)
    parcel_ses2 = pt.tensor(parcel_ses2, dtype=pt.get_default_dtype(), device=DEVICE)

    
    # 2. calculate average within / between reproducibility
    within_dice = [hev.dice_coefficient(parcel_ses1[i], parcel_ses2[i], label_matching=False).item() 
                                for i in range(parcel_ses1.shape[0])]
    res_within = make_eval_dataframe(within_dice, K=17, type='within', network='average')
    
    between_dice = intra_subjects_dice(parcel_ses1, parcel_ses2)
    res_between = make_eval_dataframe(between_dice, K=17, type='between', network='average')
    
    results = pd.concat([res_within, res_between], ignore_index=True)

    # 3. calculate networks within / between reproducibility
    network_names = spio.loadmat('/data/tge/dzhi/workspace/CBIG/stable_projects/'
                                 'brain_parcellation/Kong2019_MSHBM/lib/'
                                 'group_priors/HCP_40/17network_labels.mat')['network_name']
    network_names = [network_names[0][i][0] for i in range(17)]

    for i, net in enumerate(network_names):
        print(f'calculating network {net}')
        net_parcel1 = pt.where(parcel_ses1 ==i+1, 1, 0)
        net_parcel2 = pt.where(parcel_ses2 ==i+1, 1, 0)
        # within
        net_within = [hev.dice_coefficient(net_parcel1[i], net_parcel2[i], label_matching=False).item() 
                        for i in range(parcel_ses1.shape[0])]
        res_net_within = make_eval_dataframe(net_within, K=17, type='within', network=net)

        # between
        net_between = intra_subjects_dice(net_parcel1, net_parcel2)
        res_net_between = make_eval_dataframe(net_between, K=17, type='between', network=net)

        results = pd.concat([results, res_net_within, res_net_between], ignore_index=True)

    results.to_csv(RES_DIR + f'/reproducibility_MSHBM_vs_HBP_indiv_HCPtest.tsv',
                   index=False, sep='\t')
    
    # Plot results
    plt.figure(figsize=(15, 5))
    # sb.boxplot(data=results, x='network', y='dice', hue='type', width=0.7)
    sb.barplot(data=results, x='network', y='dice', hue='type', 
               errorbar='se', width=0.7)
    
    plt.xticks(rotation=45)
    plt.suptitle('Reproducibility - HCP test individual parcellations')
    plt.tight_layout()
    plt.show()
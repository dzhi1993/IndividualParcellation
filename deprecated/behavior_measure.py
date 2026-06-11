#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 9/25/2024 at 11:45 AM
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
behaviour_file = HCP_DIR + '/behaviour/unrestricted_hcp_freesurfer.csv'


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
    df = pd.read_csv(behaviour_file, sep=',')

    T = pd.read_csv(HCP_DIR + f'/subj_list/HCP923_test_set.tsv', sep='\t')
    subject_list = T.participant_id[0:50]

    # Step 3: Extract rows where Subject_ID is in the list
    filtered_df = df[df['Subject'].isin(subject_list)]

    filtered_df.to_csv(HCP_DIR + f'/behaviour/restricted_first50hcp-test.csv', index=False, sep='\t')
    print(filtered_df)
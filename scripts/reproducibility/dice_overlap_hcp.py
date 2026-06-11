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

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
SUBJECT_LIST_DIR = REPLICATION_DIR / 'subject_list'
RES_DIR = RESULTS_DIR / Path(__file__).resolve().parent.name
RES_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = str(RES_DIR)

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


def load_and_intersect(path1, path2):
    # Load the CIFTI objects
    img1 = nb.load(path1)
    img2 = nb.load(path2)

    # Extract Map Names (Subject IDs) from the headers
    # Axis 0 is usually the 'CiftiScalarAxis' or 'CiftiLabelAxis' containing names
    names1 = [img1.header.get_axis(0).name[i] for i in range(img1.shape[0])]
    names2 = [img2.header.get_axis(0).name[i] for i in range(img2.shape[0])]

    # Find the intersection
    common_subjects = sorted(list(set(names1) & set(names2)))
    print(f"File 1: {len(names1)} subjects")
    print(f"File 2: {len(names2)} subjects")
    print(f"Intersection: {len(common_subjects)} common subjects")

    # Get the indices for the common subjects in each file
    idx1 = [names1.index(s) for s in common_subjects]
    idx2 = [names2.index(s) for s in common_subjects]

    # Extract and filter data
    # We use vstack/take to ensure we get the rows in the exact order of common_subjects
    data1 = img1.get_fdata()[idx1, :]
    data2 = img2.get_fdata()[idx2, :]

    return data1, data2, common_subjects


if __name__ == "__main__":
    ## Step 1: Get data set and train the individual maps
    atlas, _ = am.get_atlas('fs32k')
    randy_good_subjlist = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14]
    group_strength_list = [1]
    spatial_list = [0]
    # data, info, tds = ds.get_dataset(base_dir, 'MDTB', atlas=atlas.name, subj=None)
    # tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
    #                                                   info['cond_num_uni'].values,
    #                                                   info['half'].values,
    #                                                   join_sess=False,
    #                                                   join_sess_part=False)
    #
    # ##### Option1: Use full model E_step
    # _, M = ut.load_batch_best('Models_03/asym_Md_space-MNISymC3_K-17', device='cuda')
    # U_group_hard = pt.argmax(M.arrange.marginal_prob(), dim=0).cpu().numpy() + 1
    # M.initialize(tdata)
    # # emloglik = M.collect_evidence([e.Estep() for e in M.emissions])
    # U_indiv = M.Estep()[0]
    # indiv_par_1 = U_indiv[M.subj_ind[0]]
    # indiv_par_2 = U_indiv[M.subj_ind[1]]

    ##### Option2: pipeline
    ## Step 1: Loading a pre-trained group model
    # model_name = f'/Models/Models_03/asym_Md_space-MNISymC3_K-17'
    # U, minfo = ar.load_group_parcellation(model_dir + model_name, device='cuda')
    # ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
    #                                       sym_type='asym')
    # Get indiv parcellation from first half
    # indiv_par_1, _, M1 = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
    #                                               [cond_v[0]], [part_v[0]],
    #                                               [sub_ind[0]], sym_type='asym',
    #                                               em_params={'uniform_kappa': True})
    # # Get indiv parcellation from second half
    # indiv_par_2, _, M2 = fm.get_indiv_parcellation(ar_model, atlas, [tdata[1]],
    #                                               [cond_v[1]], [part_v[1]],
    #                                               [sub_ind[1]], sym_type='asym',
    #                                               em_params={'uniform_kappa': True})
    # em_params = {'num_subj': tdata[0].shape[0],
    #              'uniform_kappa': True,
    #              'subjects_equal_weight': False,
    #              'subject_specific_kappa': False,
    #              'parcel_specific_kappa': False}

    ## Step 3: Run dice coefficient between indivpar1 and indivpar2
    network_names = ['VentralAttentionB', 'ControlA', 'VisualA', 'TemporalParietal', 'DorsalAttentionA',
                     'ControlC', 'DefaultA', 'ControlB', 'VentralAttentionA', 'VisualB', 'SomatomotorA',
                     'DefaultB', 'DefaultC', 'SomatomotorB', 'DorsalAttentionB', 'Auditory', 'VisualC']
    # mapping = np.array([3, 2, 7, 8, 4, 2, 1, 2, 3, 7, 6, 1, 1, 6, 4, 5, 7])
    # 1-Default, 2-Control, 3-vATTN, 4-dATTN, 5-AUD, 6-SMOT, 7-Visual, 8-LANG

    # network_names = ['VIS-P','CG-OP','DN-B','SMOT-B','AUD','PM-PPr','dATN-B','SMOT-A',
    #                  'LANG','FPN-B','FPN-A','dATN-A','VIS-C','SAL/PMN','DN-A']
    mapping = np.array([7, 3, 1, 6, 5, 9, 4, 6, 8, 2, 2, 4, 7, 3, 1])
    # 1-Default, 2-FPN, 3-CGOP/SAL, 4-dATTN, 5-AUD, 6-SMOT, 7-Visual, 8-LANG, 9-PM-PPr

    df_all = pd.DataFrame()
    for counter_p, p in enumerate(group_strength_list):
        for counter_w, w in enumerate(spatial_list):
            print(f'prior strength is {p}; the MRF strength is {w} ...')
            indiv_par_1 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                            f'/asym_MdNiIbHc+HCPrest-1run1-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-10_spatial-5.dlabel.nii').get_fdata()[:]
            # indiv_par_1 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/runs' +
            #                       f'/asym_HBPHc+RANDYrest-1run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}.dlabel.nii').get_fdata()[:]
            # indiv_par_1 = mapping[indiv_par_1.astype(int) - 1]
            indiv_par_1 = pt.tensor(indiv_par_1, dtype=pt.get_default_dtype(), device=DEVICE)

            indiv_par_2 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                            f'/asym_MdNiIbHc+HCPrest-1run2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-10_spatial-5.dlabel.nii').get_fdata()[:]
            # indiv_par_2 = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/runs' +
            #                       f'/asym_HBPHc+RANDYrest-1run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}.dlabel.nii').get_fdata()[:]
            # indiv_par_2 = mapping[indiv_par_2.astype(int) - 1]
            indiv_par_2 = pt.tensor(indiv_par_2, dtype=pt.get_default_dtype(), device=DEVICE)

            # indiv_par_1 = pt.argmax(indiv_par_1, dim=1)+1
            # indiv_par_2 = pt.argmax(indiv_par_2, dim=1)+1

            # colors = ut.get_cmap('Models_03/asym_Md_space-MNISymC3_K-17')
            # colors[12] = np.array([249 / 255, 178 / 255, 247 / 255, 1.])
            # plt.figure(figsize=(40, 15))
            # ut.plot_multi_flat(indiv_par_2.cpu().numpy(), atlas.name, grid=(3, 8),
            #                    cmap=colors, dtype='label', titles=[f"sub_{i+1}" for i in range(indiv_par_2.shape[0])])
            # plt.savefig('asym_Md_indiv_ses2.png', format='png')
            # plt.show()

            K=17
            ## 1. Within dice
            within_dice = [hev.dice_coefficient(indiv_par_1[i],
                                                indiv_par_2[i],
                                                label_matching=True).item()
                            for i in range(indiv_par_1.shape[0])]

            # within_nmi = [hev.nmi(indiv_par_1[i].cpu().numpy(),
            #                       indiv_par_2[i].cpu().numpy()).item()
            #                for i in range(indiv_par_1.shape[0])]
            #
            # within_ari = [hev.ARI(indiv_par_1[i],indiv_par_2[i]).item()
            #                for i in range(indiv_par_1.shape[0])]

            num_subj = indiv_par_1.shape[0]
            ev_df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
                                  'K': [K] * num_subj,
                                  'networks': ['all'] * num_subj,
                                  'subj': np.arange(1,num_subj+1)})
            ev_df['dice'] = within_dice
            # ev_df['nmi'] = within_nmi
            # ev_df['ari'] = within_ari
            ev_df['type'] = 'MdNiIbHc'

            # ev_df = pd.DataFrame()
            for sub in range(num_subj):
                networks_dice = hev.dice_coefficient(indiv_par_1[sub], indiv_par_2[sub],
                                                     label_matching=True, separate=True)
                df = pd.DataFrame({'atlas': [atlas.name] * K,
                                   'K': [K] * K,
                                   'networks': np.arange(1, K+1),
                                   'subj': [sub+1] * K})
                df['dice'] = networks_dice.cpu().numpy()
                df['type'] = 'MdNiIbHc'
                ev_df = pd.concat([ev_df, df], ignore_index=True)

            ev_df["strength"] = p
            ev_df["spatial_w"] = w
            df_all = pd.concat([df_all, ev_df], ignore_index=True)

            # Between dice
            # plt.figure(figsize=(16, 8))
            # sb.boxplot(data=ev_df, x="networks", y="dice", width=0.5)

            # ## 2. Between dice
            # print("1-Visual, 2-Motor, 3-dorsal attention, 4-ventral attention, 5-limbic, 6-frontoparietal, 7-DMN")
            # between_dice = np.zeros((indiv_par_1.shape[0],indiv_par_2.shape[0], K))
            # for i, par1 in enumerate(indiv_par_1):
            #     if i % 10 == 0:
            #         print(f"Done {i} subjects between dice..")
            #     for j, par2 in enumerate(indiv_par_2):
            #         # dice1 = hev.dice_coefficient(indiv_par_1[i], indiv_par_1[j],
            #         #                              label_matching=False, separate=False).cpu().numpy()
            #         # dice2 = hev.dice_coefficient(indiv_par_2[i], indiv_par_2[j],
            #         #                              label_matching=False, separate=False).cpu().numpy()
            #         dice3 = hev.dice_coefficient(indiv_par_1[i], indiv_par_2[j],
            #                                      label_matching=False, separate=False).cpu().numpy()
            #         # dice4 = hev.dice_coefficient(indiv_par_1[j], indiv_par_2[i],
            #         #                              label_matching=False, separate=False).cpu().numpy()
            #         # between_dice[i,j,:] = (dice1 + dice2 + dice3 + dice4)/4
            #         between_dice[i, j, :] = dice3
            #
            # y_values = between_dice[np.triu_indices(between_dice.shape[0], k=1)]
            # num_subj = y_values.shape[0]
            # df = pd.DataFrame({'atlas': [atlas.name] * num_subj,
            #                       'K': [K] * num_subj,
            #                       'networks': ['all'] * num_subj,
            #                       'subj': np.arange(1, num_subj+1)})
            # df['dice'] = y_values[:,0]
            #
            # # ONLY FOR NETWORK-WISE Add 17 network columns
            # # network_cols = [f"dice_network_{i}" for i in range(1, K+1)]
            # # network_df = pd.DataFrame(y_values, columns=network_cols)
            # # df = pd.concat([df, network_df], axis=1)
            #
            # df['type'] = 'MdNiIbHc'
            # df["strength"] = p
            # df["spatial_w"] = w
            # df_all = pd.concat([df_all, df], ignore_index=True)

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

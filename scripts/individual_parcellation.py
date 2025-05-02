#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Individual parcellation using localizing data

Created on 10/16/2023 at 2:02 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm

from utils import plot_multi_flat, convert_hard_to_prob
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR, DEVICE


if __name__ == "__main__":
    ## Step 1: Load the atlas
    atlas, _ = am.get_atlas('fs32k')
    # sym_type = 'sym'
    sym_type = 'asym'

    ## Step 2a: Load the probabilstic group atlas from a _probseg.nii file
    # atlas_dir = ATLAS_DIR + '/tpl-MNI152NLin2009cSymC'
    # model_name = f'/atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii'
    # U = atlas.read_data(atlas_dir + model_name)
    # U = U.T

    # Step 2b: If _probseg.nii file stores hard parcellation (e.g Buckner7),
    # we convert it to a soft (probabilistic) version
    # atlas_dir = ATLAS_DIR + '/tpl-MNI152NLin2009cSymC'
    # model_name = f'/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii'
    # U = atlas.read_data(atlas_dir + model_name)
    # U = convert_hard_to_prob(U, strength=7)

    ## Step 2c: Or Load the group prior from a pre-trained model
    # model_name = f'/Models_03/sym_Md_space-MNISymC3_K-34'
    model_name = f'/Models_03/asym_Hc_space-MNISymC3_K-17'
    fname = MODEL_DIR + model_name
    U, _ = ar.load_group_parcellation(fname, device=DEVICE)
    Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE)

    ## Step 3: Build the arrangement model
    # ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
    #                                       sym_type=sym_type)
    ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
                                          sym_type='asym')

    ## Step 4a: Load the individual localizing data / info from Fusion project
    # Step 4a.1: Load the data into 3d tensor
    data, info, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, subj=None)
    # Step 4a.2: Prepare the data into the right format
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)

    # ## Step 4b: Build custom individual localizing data / info
    # # Step 4b.1: Build the data into list of 3d tensor
    # data_dir = 'Y:/data/FunctionalFusion/MDTB/derivatives/{0}/data'
    # mdtb_dataset = ds.get_dataset_class('Y:/data/FunctionalFusion','MDTB')
    # subj = mdtb_dataset.get_participants().participant_id
    # data, info = [], []
    # for ses_id in mdtb_dataset.sessions:
    #     this_data = []
    #     this_info = []
    #     info.append(mdtb_dataset.get_info(ses_id=ses_id, type='CondHalf'))
    #     for i, s in enumerate(subj):
    #         file_name = f'/{s}_space-{atlas.name}_{ses_id}_CondHalf.dscalar.nii'
    #         this_data.append(atlas.read_data(data_dir.format(s) + file_name).T)
    #     data.append(np.stack(this_data))
    # # Step 4b.2: Assemble condition and partition vectors
    # cond_v, part_v, sub_ind = [], [], []
    # for j, inf in enumerate(info):
    #     cond_v.append(inf['cond_num_uni'].values.reshape(-1,))
    #     part_v.append(inf['half'].values.reshape(-1,))
    #     sub_ind.append(np.arange(0, len(subj)))

    # ## Step 5: Compute the individual parcellations
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, tdata,
    #                                             cond_v, part_v, sub_ind, Vs=Vs,
    #                                             sym_type=sym_type)

    # # All dataset
    # # V fixed
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, tdata,

    #                                             cond_v, part_v, sub_ind,
    #                                             Vs=Vs, sym_type=sym_type)
    # # V not fixed
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, tdata,
    #                                             cond_v, part_v, sub_ind,
    #                                             sym_type=sym_type)

    # # First session (SC1) of MDTB (info.sess)
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
    #                                             [cond_v[0]], [part_v[0]],
    #                                             [sub_ind[0]])

    # # Second session (SC2) MDTB (info.sess)
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, [tdata[1]],
    #                                             [cond_v[1]], [part_v[1]],
    #                                             [sub_ind[1]])

    # # First halfs of session 1 and 2 of MDTB (info.half)
    # indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, [tdata[0]],
    #                                             [cond_v[0]], [part_v[0]],
    #                                             [sub_ind[0]])

    # Second halfs of session 1 and 2 of MDTB (info.half)
    indiv_par, _, _ = fm.get_indiv_parcellation(ar_model, atlas, [tdata[1]],
                                                [cond_v[1]], [part_v[1]],
                                                [sub_ind[1]])

    # Step 3: Save the individual parcellations as a nifti/gifti file
    # Step 3.1: Convert the individual parcellations to gifti file
    # gii_file = nt.make_label_gifti(indiv_par.cpu().numpy().transpose(),
    #                                labels=["label_{}".format(i) for i in range(K)],
    #                                column_names=["subj_{}".format(i+1)
    #                                              for i in range(indiv_par.shape[0])])
    # nb.save(gii_file, '/Md_Asym_17.dlabel.gii')
    # TODO: here we need to write a function to convert the
    #       individual parcellations to nifti file

    # Step 4: Visualization
    # Step 4.1: plot the individual parcellations
    plt.figure(figsize=(20, 20))
    plot_multi_flat(indiv_par.cpu().numpy(), 'MNISymC3', grid=(6, 4),
                    cmap='tab20', dtype='prob',
                    titles=["subj_{}".format(i+1) for i in range(indiv_par.shape[0])])
    plt.show()

    # Step 4.2: plot the group parcellations for comparison
    plt.figure(figsize=(10, 10))
    plot_multi_flat(U.unsqueeze(0).cpu().numpy(), 'MNISymC3', grid=(1, 1),
                    cmap='tab20', dtype='prob', titles=['group prior'])
    plt.show()







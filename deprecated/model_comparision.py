#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares different model for individual parcellation using the MDTB dataset. 

"""
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as mm

import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm

from global_config import MODEL_DIR, BASE_DIR, DEVICE


if __name__ == "__main__":


    ## Step 1: Load the atlas
    atlas, _ = am.get_atlas('MNISymC3')

    ## Step 2a: Load the probabilstic group atlas from a _probseg.nii file
    # atlas_dir = BASE_DIR + '/Atlases'
    # model_name = f'/tpl-MNI152NLin2009cSymC/atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii'
    # U = atlas.read_data(atlas_dir + model_name)
    # U = U.T

    ## Step 2b: Or Load the group prior from a pre-trained model
    model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'
    fname = MODEL_DIR + model_name
    U, _ = ar.load_group_parcellation(fname, device=DEVICE)
    Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE)

    ## Step 3: Build the arrangement model
    # ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
    #                                       sym_type=sym_type)
    ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                          sym_type='asym')

    ## Step 4a: Load the individual localizing data / info from Fusion project
    # Step 4a.1: Load the data into 3d tensor
    data, info, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, subj=None, type='CondHalf')
    # Step 4a.2: Prepare the data into the right format
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)

    # Initialize emission models
    i=0
    em_model = em.MixVMF(K=ar_model.K, P=atlas.P, X=mm.indicator(cond_v[i]),part_vec=part_v[i])
    M = fm.FullMultiModel(ar_model, [em_model])
    M.initialize([tdata[i]], subj_ind=[sub_ind[i]])

    M, ll, _, U_indiv = M.fit_em(iter=100, tol=0.01,
                                     fit_arrangement=False,
                                     fit_emission=True,
                                     first_evidence=False)
    
    pass 






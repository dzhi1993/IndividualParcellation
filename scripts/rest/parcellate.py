#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: Caro Nettekoven
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
import FusionModel.evaluate as ev
from IndividualParcellation.scripts.indiv_evaluation import make_eval_info, eval_parcel_DCBC
from FusionModel.util import plot_multi_flat, plot_data_flat


from global_config import DEVICE
import IndividualParcellation.scripts.paths as paths
from IndividualParcellation.plot import plot_diagnostics, plot_evolution

data_dir = paths.set_fusion_dir()
atlas_dir = paths.set_atlas_dir()

def get_prior(atlas='NettekovenSym32', sym_type='sym', space='MNISymC2'):
    space, _ = am.get_atlas(space)
    if space.name == 'MNISymC2':
        space_folder = 'tpl-MNI152NLin2009cSymC'

    # Load group prior and construct arrangement model
    atlas_dir = paths.set_atlas_dir()
    model_name = f'/{space_folder}/atl-{atlas}_space-{space_folder.split("tpl-")[1]}_probseg.nii'
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/{space_folder}/atl-{atlas}.lut')
    U = space.read_data(atlas_dir + model_name)
    U = U.T
    ar_model = ar.build_arrangement_model(U, prior_type='prob', sym_type=sym_type, atlas=space)
    
    return ar_model, cmap, labels


def parcellate(ar_model, train_data, cond_vec, part_vec,
                            subj_ind, Vs=None, sym_type='asym', n_iter=200,
                            em_params={}, 
                            fit_arrangement=False,
                            fit_emission=True, 
                            subject_specific_kappa=False,
                            subjects_equal_weight=False,
                            device=None):
    """ Calculates the individual parcellations using the given individual
        training data and the given arrangement model with the pre-defined
        group prior.

    Args:
        ar_model (arrangement model object):
            The arrangement model object with pre-defined group prior U.
        train_data (np.ndarray or pt.Tensor):
            Individual localizing data
        cond_vec (list):
            The condition vectors for each emission model
        part_vec (list):
            The partition vectors for each emission model
        subj_ind (list):
            The subject indices for each emission model
        Vs (list):
            The mean response vectors for each emission model, if None, the
            mean response vectors will be calculated from random inits.
            If not None, the Vs should be a list of the V vectors for each
            emission model, and V will be fixed during the learning.
        sym_type (str):
            The symmetry type of the arrangement model
        n_iter (int):
            The number of iterations for the EM algorithm
        em_params (dictionary):
            Dictionary setting optina parameters for the emission model
        fit_arrangement (boolean):
            If True, the arrangement model will be fitted using the given
            individual training data. However, in this case, the arrangement
            model should be freezed during the learning process.
        fit_emission (boolean):
            If True, the emission models will be fitted. The emission model
            parameters are freely learned.
        subject_specific_kappa: If True, each subject has a different kappa
        parcel_specific_kappa: If True, each parcel has a different kappa
        device (str):
            The device name to load trained model

    Returns:
        U_indiv (pt.Tensor):
            The individual probabilistic parcellations
        ll (list):
            The log-likelihood of the individual parcellations
        M (object):
            The trained arrangement model
    """
    # convert tdata to tensor
    if type(train_data) is np.ndarray:
        train_data = pt.tensor(train_data, dtype=pt.get_default_dtype())
    if Vs is None:
        Vs = [None] * len(train_data)

    # Check if the lists have equal length using assert
    assert len(train_data) == len(cond_vec) == len(part_vec) == len(Vs),\
        "training data, condition vector, and partition vector " \
        "must have equal length."

    # Check if the input arrangement model is valid
    if not isinstance(ar_model, ar.ArrangementModel):
        raise ValueError("The input model must be a valid arrangement"
                         " model object")

    # Initialize emission models
    em_models = []
    for j, this_cv in enumerate(cond_vec):
        if sym_type=='sym':
            K=ar_model.K_full
        else:
            K=ar_model.K
        em_model = em.MixVMF(K=K,
                        P=train_data[j].shape[-1],
                        X=fm.indicator(this_cv),
                        part_vec=part_vec[j],
                        num_subj=train_data[j].shape[0],
                        subject_specific_kappa=subject_specific_kappa,
                        subjects_equal_weight=subjects_equal_weight,
                        **em_params)
        if Vs[j] is not None:
            em_model.V = Vs[j]
            new_param_list = em_model.param_list.copy()
            new_param_list.remove('V')
            em_model.set_param_list(new_param_list)

        em_models.append(em_model)

    M = fm.FullMultiModel(ar_model, em_models)
    M.initialize(train_data, subj_ind=subj_ind)

    # ---------------------------------------------------------
    # Real training starts here with a frozen arrangement model
    # ---------------------------------------------------------
    if M.arrange.name.startswith('indp'):
        M, ll, _, U_indiv = M.fit_em(iter=n_iter, tol=0.01,
                                     fit_arrangement=fit_arrangement,
                                     fit_emission=fit_emission,
                                     first_evidence=False)
    else:
        raise NameError("The arrangement model is not supported yet.")

    # Return the individual PROBABILISTIC parcellations
    return U_indiv, ll, M
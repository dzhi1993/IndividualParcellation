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
from IndividualParcellation.scripts.rest import parcellate as par
from FusionModel.util import plot_multi_flat, plot_data_flat
import pickle

from global_config import DEVICE
import IndividualParcellation.scripts.paths as paths
from IndividualParcellation.plot import plot_diagnostics, plot_evolution

base_dir = paths.set_base_dir()
results_dir = paths.set_results_dir(base_dir)
data_dir = paths.set_fusion_dir(base_dir)

def get_mdtb(space='MNISymC2', sessions=['ses-rest'], type='Net67Run'):
    # -- Get MDTB rest --

    # Get subjects
    T = pd.read_csv(f'{data_dir}/MDTB/participants.tsv', delimiter='\t')
    participants = T.participant_id
    subject_list = participants[T['ses-rest'] == 1].tolist()
    
    if 'ses-rest' in sessions:
        condition_column = 'net_id'
    else:
        type='CondHalf'
        condition_column = 'cond_num_uni'

    # Load MDTB rest data
    data, info, tds = ds.get_dataset(data_dir, 'MDTB', atlas=space, sess=sessions, type=type, subj=subject_list)
    tdata, cond, part, subj_ind = fm.prep_datasets(data, info.sess,
                                                    info[condition_column].values,
                                                    info['half'].values,
                                                    join_sess=False,
                                                    join_sess_part=False)
    return tdata, cond, part, subj_ind, subject_list

def get_hcp(space='MNISymC2'):
    """Get HCP data"""

    # -- Get connectivity fingerprints of regions --
    # Load HCP data
    hcp, hcp_info, hcp_tds = ds.get_dataset(data_dir, 'HCP', atlas=space, type='Net67Run')

    space, _ = am.get_atlas(space)

    tdata_hcp, cond_v_hcp, part_v_hcp, sub_ind = fm.prep_datasets(hcp, hcp_info.sess,
                                                    hcp_info['net_id'].values,
                                                    hcp_info['half'].values,
                                                    join_sess=False,
                                                    join_sess_part=False)
    return tdata_hcp, cond_v_hcp, part_v_hcp, sub_ind

def fit_models_directly(type='Net67Run'):
    """Script to fit the individual parcellations for the rest data of MDTB and HCP directly on the data."""

    model_types = {                    
                'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'general_eq':            {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'subject_specific_eq':   {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                }

    sym_type='sym'
    ar_model, cmap, labels = par.get_prior(sym_type='sym')

    # Import MDTB
    mdtb_data_rest, mdtb_cond_rest, mdtb_part_rest, mdtb_sub_ind_rest, subject_list = get_mdtb(sessions='ses-rest', type=type)

    # Import HCP
    # hcp_data, hcp_cond, hcp_part, hcp_sub_ind = get_hcp()

    for dataset in ['mdtb', 'hcp']:
        if dataset == 'mdtb':
            tdata = mdtb_data_rest
            cond = mdtb_cond_rest
            part = mdtb_part_rest
            sub_ind = mdtb_sub_ind_rest
        # else:
        #     tdata = hcp_data
        #     cond = hcp_cond
        #     part = hcp_part
        #     sub_ind = hcp_sub_ind
        for mtype, model_type in model_types.items():
            # --- Rest ---
            
            restP, _, model = par.parcellate(ar_model,
                                        tdata,
                                        cond,
                                        part,
                                        sub_ind,
                                        sym_type = sym_type, 
                                        subject_specific_kappa=model_type['subject_specific_kappa'],
                                        subjects_equal_weight=model_type['subjects_equal_weight'])
            # Pickle the fitted emission model
            with open(f'{results_dir}/fitted_emissions/fitted_rest-{dataset}_dtype-{type}_mtype-{mtype}.pkl', 'wb') as f:
                pickle.dump(model, f)

def fit_models_with_hcp_Vs(type='Net67Run'):

    sym_type='sym'
    ar_model, cmap, labels = par.get_prior(sym_type=sym_type)

    v_types = {                    
                'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'general_eq':            {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'subject_specific_eq':   {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                }

    model_types = {
                'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                }

    # Import MDTB
    tdata, cond, part, sub_ind, subject_list = get_mdtb(sessions='ses-rest', type=type)

    # Import HCP Vs
    hcp_vs = {vtype: pickle.load(open(f'{results_dir}/fitted_emissions/fitted_rest-hcp_dtype-{type}_mtype-{vtype}.pkl', 'rb')) for vtype in v_types.keys()}

    for vtype, hcp_model in hcp_vs.items():
        # Select HCP Vs
        Vs = hcp_model.emissions[0].V
        
        for mtype, model_type in model_types.items():
            # --- Rest ---
            print(f'Fitting rest model {mtype} with Vs {vtype}')
            restP, _, model = par.parcellate(ar_model,
                                        tdata,
                                        cond,
                                        part,
                                        sub_ind,
                                        Vs=[Vs],
                                        sym_type = sym_type, 
                                        subject_specific_kappa=model_type['subject_specific_kappa'],
                                        subjects_equal_weight=model_type['subjects_equal_weight'])
            # Pickle the fitted emission model
            with open(f'{results_dir}/fitted_emissions/fitted_rest-mdtb_dtype-{type}_mtype-{mtype}_vtype-{vtype}.pkl', 'wb') as f:
                pickle.dump(model, f)




if __name__ == "__main__":
    # fit_models_directly(type='Fus06Run')
    fit_models_with_hcp_Vs(type='Fus06Run')
    



    pass

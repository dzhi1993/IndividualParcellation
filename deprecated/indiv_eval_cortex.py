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

from group_parcellation import build_ukb_datasets, build_hcp_datasets, load_hcp_timeseries, load_hcp_task_contrast
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
HCP_DIR = '/data/tge/Tian/HCP_img'
BRAIN_WISE = ['whole brain'] * 50
TRAIN_SMOOTH = [0,2,4,6,8,10]

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda:1'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)

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

def eval_indiv_parcellation(parcels, names, space='fs32k', test_smooth=None,
                                K=7, subj_list="HCP80_training+validation_set.tsv", out_file=1):
    out_file = int(out_file)

    if not isinstance(parcels, list):
        parcels = [parcels]

    space_sp = space.split('_')
    if len(space_sp) == 1:
        hemis = 'full'
        atlas, _ = am.get_atlas(space)
        vert_indx = np.arange(0,atlas.P)
    elif len(space_sp) == 2:
        hemis = 'half'
        space = space_sp[0]
        hem = space_sp[1]
        hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
        atlas, _ = am.get_atlas(space)
        stru_idx = atlas.structure.index(hemis_dict[hem])
        vert_indx = atlas.indx_full[stru_idx]
    else:
        raise NameError('Unrecognized `space` for atlasing!')
    
    dist = futil.load_fs32k_dist(file_type='distGOD_sp', hemis=hemis,
                                  device=DEVICE if pt.cuda.is_available() else 'cpu')
    
    ## Load HCP subjects test data
    print(f'Start loading data: HCP {space} resting - Tseries ...')
    tic = time.perf_counter()
    # t_data, _, _, _ = build_hcp_datasets(HCP_DIR, subj_list,
    #                                     space=atlas.name, ses_list=[test_ses],
    #                                     join_sess=False, join_sess_part=False, 
    #                                     part_ind=['run'], part_num=None,cond_ind=['time_id'],
    #                                     type=['Tseries'], hemis=hem, smooth=test_smooth)
    
    t_data = load_hcp_timeseries(HCP_DIR, subj_list, space=atlas.name, run_list=[0,1,2,3],
                                type='Tseries', hemis=hem, smooth=test_smooth)
    # t_data = [load_hcp_task_contrast(HCP_DIR, subj_list, space=atlas.name, ses_list='all',
    #                                 hemis=hem)]
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    
    results = pd.DataFrame()
    for i, par in enumerate(parcels):
        ######## Step 1. Load UKB 736 subjects training data
        print(f'Start evaluating HCP subjects bulk {out_file}')
        
        Pgroup = np.where(par==0, np.nan, par)
        ######## Step 3: Evaluate individual maps using DCBC
        # Step 3.2: Gatering all necessary information for evaluation
        eval_info = make_eval_info(K, atlas=space, train_info=['group_train'], train_sess='all',
                                    tdata='HCP_test', test_sess=None, 
                                    model_type='Models_03', group_map_name=names[i],
                                    test_kappa=None)
        for r, td in enumerate(t_data):
            # Step 3.3: Do DCBC evaluation on the second half data
            res = eval_group_DCBC(Pgroup, td, dist, eval_info, subj_list=None)
            res['brain_wise'] = 'whole_brain'
            res['test_run'] = r
            # res['train_smooth'] = TRAIN_SMOOTH[i]
            res['test_smooth'] = test_smooth
            res['test_type'] = 'Tseries'
            results = pd.concat([results, res], ignore_index=True)
    
    results.to_csv(f'/data/tge/dzhi/Indiv_par/Evaluations/eval_group-taskfusion-old_on-HCPtest-task.tsv',
                    index=False, sep='\t')
    

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

    eval_HCP_group_parcellation(parcels, names, space='fs32k_L', test_smooth=None, K=K,
                                subj_list=f'subj_list/HCP203_test_set.tsv', 
                                out_file=1)
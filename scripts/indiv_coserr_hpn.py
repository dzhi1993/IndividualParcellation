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
import IndividualParcellation.scripts.group_parcellation as gp

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

def get_subject_list_path(file_name):
    subj_list_path = SUBJECT_LIST_DIR / file_name
    if not subj_list_path.exists():
        subj_list_path = Path(HCP_DIR) / 'subj_list' / file_name
    return subj_list_path

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




if __name__ == "__main__":
    atlas, am_info = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    # DEVICE = 'cpu'
    training_ses = 'all'
    test_ses = 'ses-rest1'
    fc_type = 'Ico642Run'
    ext = '_binarized'
    K = 15
    group_strength_list = [1]
    spatial_list = [0]
    global_run_list = [1,2,3]
    randy_tasks = ['EPROJ', 'LANG', 'MOTOR', 'NBACK', 'TOM', 'VISME', 'VODDK']
    randy_good_subjlist = [1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14]
    randy_bad_subjlist = [0, 4, 9, 10]

    ######## Step 2. Generate group / indiv parcellations
    # Load DU15 networks
    DU, net_name, colors = gp.get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    DU15 = ar.expand_mn_1d(DU, K=16)
    align = DU15[1:, :]
    nanidx = pt.where(align.sum(dim=0) == 0)[0]

    ## Load the group prior from a pre-trained model
    # model_name = f'/Models_03/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_DU15-inits'  # resting 15
    model_name = f'/Models_03/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_DU15-inits' # task 15
    U, _ = hut.load_group_parcellation(MODEL_DIR + model_name, device=DEVICE)
    # U = atlas.cifti_to_data(ERIS_DIR + '/dzhi/workspace/res/priors/MSHBM_group_prior_HCP40training_k-15.dscalar.nii') # MSHBM 15
    # U = pt.tensor(np.nan_to_num(U), dtype=pt.get_default_dtype(), device=DEVICE)
    # Align with prior
    indx = hev.matching_greedy(align, pt.softmax(U, dim=0).to(pt.get_default_dtype()))
    U = U[indx, :]
    U = pt.softmax(U, dim=0)
    # Vs = [v[:,indx] for v in Vs]
    Pgroup = pt.argmax(U, dim=0) + 1

    ## Determine the connectivity profile for mRBM
    Wc = pt.load(ERIS_DIR + '/Tian/UKBB_full/imaging/Atlases/tpl-fs32k/fs32k_neighbours.pt', weights_only=True)
    hut.report_cuda_memory()
    T = pd.read_csv(ERIS_DIR + f"/Tian/RANDY15/participants.tsv", sep='\t')

    ####################################################################################################################
    ## Evaluation
    print(f'Start loading data: RANDY resting - {test_ses} - {fc_type} ...')
    tic = time.perf_counter()
    ## 3. RANDY task contrasts
    # t_data, t_info = ut.load_randy_contrasts(space='fs32k', subj=randy_good_subjlist, hemis=None, smooth=2)

    ## 4. RANDY resting timeseries
    # t_data = ut.build_resting_data('RANDY15', space='fs32k',
    #                               ses_list=[f'ses-rest{sn}' for sn in range(6, 8)],
    #                               type='Tseries', subj=randy_bad_subjlist, hemis=None, smooth=None)
    # t_info = pd.DataFrame({'task_name': ['REST'] * t_data[0].shape[1]})

    ## 5. RANDY task betas
    t_data, cond_vec, part_vec, subj_ind, t_info = ut.build_dataset('RANDY15', atlas=atlas.name,
                                               sess=randy_tasks, cond_ind='reg_id',
                                               type='CondRun', part_ind='sn', subj=randy_good_subjlist,
                                               part_num=np.arange(1, 4), join_sess=False,
                                               join_sess_part=False, smooth=None)
    t_info = t_info[0]

    toc = time.perf_counter()
    print(f"Done loading. Used {time.strftime('%M:%S', time.gmtime(toc - tic))} (MM:SS)!")
    hut.report_cuda_memory()
    n_subj = t_data[0].shape[0]

    results = pd.DataFrame()

    for num_test in range(10):
        for global_counter in global_run_list:
            for counter_p, p in enumerate(group_strength_list):
                for counter_w, w in enumerate(spatial_list):
                    print(f'Evaluating {global_counter} runs data, prior strength {p}, the spatial w {w} ...')
                    U_indiv_rest = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYrest2run-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_rest = pt.tensor(U_indiv_rest, dtype=pt.get_default_dtype())

                    U_indiv_rest_3387s = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                           f'/asym_MdNiIbHc+RANDYrest3387s-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-5_spatial-{w}_11sub.npy')
                    U_indiv_rest_3387s = pt.tensor(U_indiv_rest_3387s, dtype=pt.get_default_dtype())

                    U_indiv_task = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYtask1run{global_counter}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_task = pt.tensor(U_indiv_task, dtype=pt.get_default_dtype())

                    U_indiv_fusion = np.load(MODEL_DIR + f'/Models_03/indiv_parcellation/RANDY15_test_set/prob' +
                                        f'/asym_MdNiIbHc+RANDYrest2run+task1run{global_counter}-indiv_space-fs32k_K-15_Ico642Run_groupstrengh-{p}_spatial-{w}_11sub.npy')
                    U_indiv_fusion = pt.tensor(U_indiv_fusion, dtype=pt.get_default_dtype())

                    # m-RBM
                    ar_model = ar.build_arrangement_model(U.clone(), prior_type='prob', atlas=atlas,
                                                          sym_type='asym', model_type='cRBM_Wc',
                                                          Wc=Wc, theta=w, epos_iter=20, num_chain=n_subj)
                    ar_model.bu = ar_model.bu * p

                    # Making evaluation information
                    this_glist = global_run_list.copy()
                    this_glist.remove(global_counter)
                    minfo = ge.make_eval_info(K, train_info=['RANDY'], train_sess=global_counter,
                                              tdata='RANDY', test_sess=this_glist,
                                              model_type='Models_03', group_map_name='MdNiIbHc',
                                              test_kappa=None)
                    this_res = pd.DataFrame()
                    # Evaluate on each run's time series
                    for r, td in enumerate(t_data):
                        if r == global_counter - 1:
                            continue

                        res = pd.DataFrame({'atlas': [minfo.atlas] * n_subj,
                                            'K': [minfo.K] * n_subj,
                                            'train_data': [minfo.datasets] * n_subj,
                                            'train_sess': [minfo.train_sess] * n_subj,
                                            'test_data': [minfo.test_data] * n_subj,
                                            'test_sess': [minfo.test_sess] * n_subj,
                                            'model_type': [minfo.model_type] * n_subj,
                                            'group_map_name': [minfo.group_map_name] * n_subj,
                                            'subj_num': [f'{i}' for i in T.participant_id[randy_good_subjlist]],
                                            'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj})

                        ## prediction error
                        hut.report_cuda_memory()
                        em_params = {'num_subj': n_subj, 'uniform_kappa': True, 'subjects_equal_weight': True,
                                     'subject_specific_kappa': False, 'parcel_specific_kappa': False}

                        em_model = em.build_emission_model(K, atlas, 'VMF', hut.indicator(cond_vec[r]),
                                                           part_vec[r], V=None, em_params=em_params)
                        M_test = fm.FullMultiModel(deepcopy(ar_model), [em_model])
                        coserr_indiv = ev.calc_test_error(M_test, td,['group', 'floor',
                                                                      U_indiv_rest, U_indiv_rest_3387s, U_indiv_task, U_indiv_fusion])
                        del M_test
                        pt.cuda.empty_cache()
                        hut.report_cuda_memory()

                        res['group_coserr'] = coserr_indiv[0]
                        res['floor_coserr'] = coserr_indiv[1]
                        res['indiv-rest_coserr'] = coserr_indiv[2]
                        res['indiv-rest3387s_coserr'] = coserr_indiv[3]
                        res['indiv-task_coserr'] = coserr_indiv[4]
                        res['indiv-fusion_coserr'] = coserr_indiv[5]
                        res['test_run'] = r + 1
                        # res['ari_group'] = ari
                        # res['nmi_group'] = nmi
                        res['strength'] = p
                        res['spatial_w'] = w
                        res["num_test"] = num_test

                        this_res = pd.concat([this_res, res], ignore_index=True)

                    results = pd.concat([results, this_res], ignore_index=True)

    results.to_csv(
        RES_DIR + f'/coserr_indiv-mRBM_MdNiIbHc-RANDY_rest2run_vs_task1run_vs_fusion_K-15_test_on_RANDYtask-beta_sm0_run123_11subjects.tsv',
        index=False, sep='\t')
    print('Done')


    ## Combine all test subjects parcellation
    indiv_par = []
    for i in [1, 2, 3, 4]:
        par = nb.load(MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
                    f'/asym_MdNiIbHc+HCPrest-run1+task-half2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1_{i}.dlabel.nii').get_fdata()[:]
        indiv_par.append(par)
    indiv_par = np.vstack(indiv_par)
    T = pd.read_csv(get_subject_list_path("HCP200_test.tsv"), sep='\t')
    img = nt.make_label_cifti(indiv_par.T, atlas.get_brain_model_axis(),
                              column_names=[f'{i}' for i in T.participant_id],
                              label_names=net_name, label_RGBA=colors)
    nb.save(img, MODEL_DIR + f'/Models_03/indiv_parcellation/HCP200_test_set' +
            f'/asym_MdNiIbHc+HCPrest-run1+task-half2-indiv_space-fs32k_K-17_Ico642Run_groupstrengh-1_spatial-1.dlabel.nii')

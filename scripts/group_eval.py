#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: dzhi
"""
import argparse
import time, sys, json
from pathlib import Path
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import seaborn as sb
import scipy.io as spio
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
import IndividualParcellation.scripts.group_parcellation as gp
import utils as ut

from IndividualParcellation.scripts.group_parcellation import build_ukb_datasets, build_hcp_datasets, load_hcp_timeseries, load_hcp_task_contrast
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
HCP_DIR = '/home/dzhi/eris_mount/Tian/HCP_img'
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
RESULT_DIR = str(REPLICATION_DIR / 'group_parcellations')
EVAL_DIR = RESULTS_DIR / 'group_eval'
EVAL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR = str(EVAL_DIR)
# BRAIN_WISE = ['whole brain','whole brain','left hemisphere','whole brain',
#                 'left hemisphere','left hemisphere','whole brain','whole brain',
#                 'left hemisphere','left hemisphere','whole brain','whole brain']
BRAIN_WISE = ['whole brain'] * 50
TRAIN_SMOOTH = [0,4,6,8,10]

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : True
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)


def load_existing_atlas():
    atlas, _ = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    data_dir = '/home/dzhi/eris_mount'
    ######## Evaluate MS-HBM group vs. HBP group parcellations
    YEO2011 = nb.load(data_dir + f'/dzhi/workspace/res/group/Yeo2011_17.dlabel.nii').get_fdata().reshape(-1)[0:29759]
    KONG2019 = np.argmax(ut.get_kong2019_group_parcellation()[0], axis=0)[0:29759] + 1

    Hc800 = atlas.cifti_to_data(data_dir + '/dzhi/Indiv_par/Models/Models_03/' \
                                'asym_Hc_space-fs32k_K-17_HCPsubjects-800.dlabel.nii').reshape(-1)[0:29759]

    # Existing parcellations (from Kong collection)
    # GORDON_333 = nb.load(existing_atlas_dir + '/Gordon.32k.L.label.gii').darrays[0].data
    # YEO17 = spio.loadmat('/data/tge/dzhi/workspace/cbig_network_correspondence_data/atlases/fs_LR_32k/YeoLab/TY17.mat')['lh_labels'].reshape(-1)
    GORDON_286 = spio.loadmat(data_dir +
                              '/dzhi/workspace/cbig_network_correspondence_data/atlases/fs_LR_32k/WashU/EG286_12.mat')['lh_labels'].reshape(-1)
    GORDON_17 = spio.loadmat(data_dir +
                             '/dzhi/workspace/cbig_network_correspondence_data/atlases/fs_LR_32k/WashU/EG17.mat')['lh_labels'].reshape(-1)
    GLASSER = spio.loadmat(data_dir +
                           '/dzhi/workspace/cbig_network_correspondence_data/atlases/fs_LR_32k/Glasser/MG360J12.mat')['lh_labels'].reshape(-1)
    DU_15 = spio.loadmat(data_dir +
                         '/dzhi/workspace/cbig_network_correspondence_data/atlases/fs_LR_32k/Du/DU15NET.mat')['lh_labels'].reshape(-1)

    parcels = [Hc800, YEO2011, GORDON_17[atlas.vertex_mask[0]], GLASSER[atlas.vertex_mask[0]],
               DU_15[atlas.vertex_mask[0]]]
    names = ['HCP-800', 'Yeo17', 'Gordon17', 'Glasser', 'Du15']
    n_parcels = [17, 17, 17, 286, 15]

    return parcels, names, n_parcels


def make_eval_info(K, atlas='fs32k', train_info=['UKB'], train_sess='ses-2',
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
    minfo['K'] = K
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

def eval_group_DCBC(U_group, t_data, dist, minfo, subj_list=None):
    # convert tdata to tensor
    if type(t_data) is np.ndarray:
        t_data = pt.tensor(t_data, dtype=pt.get_default_dtype())
    # convert U_group and U_indiv to tensor
    if type(U_group) is np.ndarray:
        U_group = pt.tensor(U_group, dtype=pt.get_default_dtype())

    num_subj = t_data.shape[0]
    hut.report_cuda_memory()
    pt.cuda.empty_cache()
    # Now run the DCBC evaluation fo the group
    # zvalue_group = ev.calc_test_zvalue(U_group, t_data, return_single=False)
    dcbc_group = ev.calc_test_dcbc(U_group, t_data, dist, trim_nan=True)
    hut.report_cuda_memory()
    pt.cuda.empty_cache()
    inhomo_group = ev.calc_test_task_inhomogeneity(U_group, t_data,
                                                   return_single=True)
    # homo_group = ev.calc_test_homogeneity(U_group, t_data)
    hut.report_cuda_memory()
    pt.cuda.empty_cache()

    # ------------------------------------------
    # Collect the information from the evaluation
    # in a data frame
    ev_df = pd.DataFrame({'atlas': [minfo.atlas] * num_subj,
                          'K': [minfo.K] * num_subj,
                          'train_data': [minfo.datasets] * num_subj,
                          'train_sess': [minfo.train_sess] * num_subj,
                          'test_data': [minfo.test_data] * num_subj,
                          'test_sess': [minfo.test_sess] * num_subj,
                          'model_type': [minfo.model_type] * num_subj,
                          'group_map_name': [minfo.group_map_name] * num_subj,
                          'subj_num': np.arange(num_subj) if subj_list is None else subj_list,
                          'indiv_test_kappa': [minfo.indiv_test_kappa] * num_subj})
    # Add all the evaluations to the data frame
    ev_df['dcbc_group'] = dcbc_group.cpu()
    # ev_df['homo_group'] = homo_group.cpu()
    ev_df['inhomo_group'] = inhomo_group.cpu()
    # ev_df['zvalue_group'] = zvalue_group.cpu()

    return ev_df

def eval_parcel_DCBC(U_group, U_indiv, t_data, dist, minfo, subj_list=None):
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
    homo_group = ev.calc_test_homogeneity(Pgroup, t_data)
    homo_indiv = ev.calc_test_homogeneity(Pindiv, t_data)
    dcbc_group = ev.calc_test_dcbc(Pgroup, t_data, dist, max_dist=110, bin_width=5)
    dcbc_indiv = ev.calc_test_dcbc(Pindiv, t_data, dist, max_dist=110, bin_width=5)

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
                          'subj_num': np.arange(num_subj) if subj_list is None else subj_list,
                          'indiv_test_kappa': [minfo.indiv_test_kappa] * num_subj})
    # Add all the evaluations to the data frame
    ev_df['dcbc_group'] = dcbc_group.cpu()
    ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
    ev_df['homo_group'] = homo_group.cpu()
    ev_df['homo_indiv'] = homo_indiv.cpu()
    # ev_df.to_csv(out_file, index=False, sep='\t')
    return ev_df

def load_hcp_task(dataset_dir, subj_list, space='MNIAsymC2', run_list=[0,1,2,3],
                  type='Tseries', hemis=None, smooth=None, ext=None):

    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    
    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    
    data = []
    for i, run_id in enumerate(run_list):
        ses_data=[]
        for s in T.participant_id:
            # Assemble file name            
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_run{run_id}'
            else:
                file_name = f'/{s}_run{run_id}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dtseries.nii'

            # Load data / remove medial wall
            dat = nb.load(data_dir.format(s) + file_name)
            dat = dat.get_fdata().astype(np.float32)
            dat = dat[:, np.concatenate(this_at.vertex_mask)]

            if hemis is not None: # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:,this_at.indx_full[stru_idx]]
            
            ses_data.append(dat)
        
        data.append(np.stack(ses_data))

    return data

def load_msc_contrasts(ds_name, space='fs32k', sess='all', subj=None, smooth=None):
    atlas, _ = am.get_atlas(space)

    if sess == 'all':
        sess = ['motor','memory','mixed']
    elif isinstance(sess, str):
        sess = [sess]
    assert isinstance(sess, list)

    dataset = ds.DataSetMSC('/home/dzhi/eris_mount/Tian/MSC')
    T = dataset.get_participants()
    # Assemble the data
    Data = None
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a list of str or int'))

    # Loop again to assemble the data
    Data_list, info = [], []
    for i, s in enumerate(subj):
        subj_dat, subj_info = [], []
        for ses_id in sess:
            # Load the data
            if smooth is not None:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR_smooth{smooth}.dscalar.nii')
            else:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR.dscalar.nii')
            this_data = atlas.cifti_to_data(C)
            this_info = C.header.get_axis(0).name.tolist()

            # indices = [i for i, s in enumerate(this_info) if not re.search(r"[+-]", s)
            #            and not s.startswith("Alltasks")]
            indices = [i for i, s in enumerate(this_info) if not s.startswith("Alltasks")]

            subj_info.append([s for i, s in enumerate(this_info) if i in indices])
            subj_dat.append(this_data[indices])

        Data_list.append(np.vstack(subj_dat))
        info.append(np.concat(subj_info))

    # concatenate along the first dimension (subjects)
    Data = np.stack(Data_list)
    Data[np.isinf(Data)] = np.nan

    # Assemble info file
    # Check if all arrays are identical
    if all(np.array_equal(info[0], arr) for arr in info):
        # Convert to DataFrame with a single column
        info_com = pd.DataFrame(info[0], columns=['task_name'])
    else:
        # Convert to DataFrame with multiple columns
        info_com = pd.DataFrame(info).T
        info_com.columns = [f'subj_{i + 1}' for i in range(len(info))]

    return Data, info_com



def plot_multi_flat(data, atlas, grid, cmap='tab20b', dtype='label',
                    cscale=None, titles=None, colorbar=False,
                    save_fig=False):
    """ Plot multiple flatmaps in a grid

    Args:
        data: the input parcellations, shape(N, K, P) where N indicates
              the number of parcellations, K indicates the number of
              parcels, and P is the number of vertices.
        atlas: the atlas name used to plot the flatmap
        grid: the grid shape of the subplots
        cmap: the colormap used to plot the flatmap
        dtype: the data type of the input data, 'label' or 'prob'
        cscale: the color scale used to plot the flatmap
        titles: the titles of the subplots
        colorbar: whether to plot the colorbar
        save_fig: whether to save the figure, default format is png

    Returns:
        The plt figure plot
    """

    if isinstance(data, np.ndarray):
        n_subplots = data.shape[0]
    elif isinstance(data, list):
        n_subplots = len(data)

    if not isinstance(cmap, list):
        cmap = [cmap] * n_subplots

    for i in np.arange(n_subplots):
        plt.subplot(grid[0], grid[1], i + 1)
        futil.plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=None,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(titles[i])
        plt.tight_layout()

    if save_fig:
        plt.savefig('/indiv_parcellations.png')


def eval_UKB_group_indiv(smooth=[2,3,4,5,6], test_smooth=None, K=7, 
                         subj_list="participants_filtered_final.tsv", 
                         out_file=1):
    atlas, _ = am.get_atlas('MNIAsymC2')
    dist = ev.compute_dist(atlas.world.T, resolution=1)
    out_file = int(out_file)
    
    ## Load UKB 736 subjects test data
    print(f'Start loading data: UKBresting - ses-rest2 - Tseries ...')
    tic = time.perf_counter()
    t_data, _, _, _ = build_ukb_datasets(BASE_DIR,
                                         subj_list,
                                         space=atlas.name,
                                         ses_list=['ses-rest2'],
                                         type=['Tseries'],
                                         smooth=test_smooth)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')

    results = pd.DataFrame()
    for s in smooth:
        ######## Step 1. Load UKB 736 subjects training data
        print(f'Start evaluating UKB subjects bulk {out_file}')
        print(f'Start loading data: UKBresting - ses-rest1 - ICA25All ...')
        tic = time.perf_counter()
        data, cond_vec, part_vec, subj_ind = build_ukb_datasets(BASE_DIR,
                                                                subj_list,
                                                                space=atlas.name,
                                                                ses_list=['ses-rest1'],
                                                                type=['ICA25All'],
                                                                smooth=s)
        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')

        # Option 1: calculate indiv parcellations directly from fitted model
        # U, U_indv, M = get_indiv_parcellation_from_model(MODEL_DIR + 
        #                 f'/Models_04/asym_Uk_space-{atlas.name}_K-{K}_ses-rest1_sm{s}', data)
        
        # Option 2: calculate indiv parcellations from existing group map
        fname = MODEL_DIR + f'/Models_04/asym_Uk_space-{atlas.name}_K-{K}_ses-rest1_sm{s}'
        U_prior, _ = ar.load_group_parcellation(fname, device=DEVICE)
        # Vs, _ = em.load_emission_params(fname, 'V', device=DEVICE)
        
        for p in [5, 10, 20, 50]:
            U = U_prior*p
            ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                                sym_type='asym')
            U_indv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
                                                    cond_vec, part_vec, subj_ind,
                                                    sym_type='asym', n_iter=400,
                                                    em_params={'num_subj': data[0].shape[0],
                                                                'uniform_kappa': False,
                                                                'subjects_equal_weight':False,
                                                                'subject_specific_kappa': False,
                                                                'parcel_specific_kappa': True})
            
            ######## Step 3: Evaluate individual maps using DCBC
            # Step 3.2: Gatering all necessary information for evaluation
            eval_info = make_eval_info(K, train_info=['UKB'], train_sess='ses-rest1',
                                       tdata='UKB', test_sess='ses-rest2', 
                                       model_type='Models_04', group_map_name='UKB-736',
                                       test_kappa=None)
            # Step 3.3: Do DCBC evaluation on the second half data
            res = eval_parcel_DCBC(U, U_indv, t_data[0], dist, eval_info,
                                   subj_list=np.arange(50*(out_file-1), 50*(out_file-1)+data[0].shape[0]))
            dice = [hev.dice_coefficient(pt.argmax(U, dim=0), pt.argmax(U_indv, dim=1)[i],
                                         label_matching=True).item() 
                    for i in range(U_indv.shape[0])]
            # QC
            ari = [hev.ARI(pt.argmax(U, dim=0), pt.argmax(U_indv, dim=1)[i]).item() 
                   for i in range(U_indv.shape[0])]
            nmi = [1- hev.nmi(pt.argmax(U, dim=0).cpu(), pt.argmax(U_indv, dim=1)[i].cpu()) 
                   for i in range(U_indv.shape[0])]

            res['dice_group'] = dice
            res['ari_group'] = ari
            res['nmi_group'] = nmi
            res['train_smooth'] = s
            res['test_smooth'] = test_smooth
            res['group_strength'] = p
            results = pd.concat([results, res], ignore_index=True)
    
    results.to_csv(EVAL_DIR + f'/eval_all_UKB-736_K-7_split-{out_file}.tsv',
                    index=False, sep='\t')


def eval_existing_vs_UKB736(model_names=[2,3,4,5,6], test_smooth=None, K=7,
                            subj_list="participants_filtered_final.tsv", 
                            out_file=1):
    atlas, _ = am.get_atlas('MNIAsymC2')
    dist = ev.compute_dist(atlas.world.T, resolution=1)
    out_file = int(out_file)

    if not isinstance(model_names, list):
        model_names = [model_names]

    # Load atlas description json
    with open(ATLAS_DIR + '/atlas_description.json', 'r') as f:
        T = json.load(f)
    
    ## Load UKB 736 subjects test data
    print(f'Start loading data: UKBresting - ses-rest2 - Tseries ...')
    tic = time.perf_counter()
    t_data, _, _, _ = build_ukb_datasets(BASE_DIR,
                                         subj_list,
                                         space=atlas.name,
                                         ses_list=['ses-rest2'],
                                         type=['Tseries'],
                                         smooth=test_smooth)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    
    space_dir = '/tpl-MNI152NLin2009cSymC'
    results = pd.DataFrame()
    for i, model_name in enumerate(model_names):
        ######## Step 1. Load UKB 736 subjects training data
        print(f'Start evaluating UKB subjects bulk {out_file}')

        if model_name.startswith('Models'):
            # Our pre-trained model
            minfo, model = futil.load_batch_best(f"{model_name}", device=DEVICE)
            Prop = model.marginal_prob()
            Pgroup = pt.argmax(Prop, dim=0) + 1
        else:
            # load existing parcellation
            par = atlas.read_data(ATLAS_DIR +
                          f'/{space_dir}/atl-{model_name}_space-MNI152NLin2009cSymC_dseg.nii')
            Pgroup = pt.tensor(par, dtype=pt.get_default_dtype())
        
        Pgroup = pt.where(Pgroup==0, pt.tensor(float('nan')), Pgroup)
        ######## Step 3: Evaluate individual maps using DCBC
        # Step 3.2: Gatering all necessary information for evaluation
        eval_info = make_eval_info(K, train_info=['UKB'], train_sess='ses-rest1',
                                    tdata='UKB', test_sess='ses-rest2', 
                                    model_type='Models_04', group_map_name=model_name,
                                    test_kappa=None)
        # Step 3.3: Do DCBC evaluation on the second half data
        res = eval_group_DCBC(Pgroup, t_data[0], dist, eval_info,
                            subj_list=np.arange(50*(out_file-1), 50*(out_file-1)+t_data[0].shape[0]))

        res['train_smooth'] = 3
        res['test_smooth'] = test_smooth
        results = pd.concat([results, res], ignore_index=True)
    
    results.to_csv(EVAL_DIR + f'/eval_all_existing_vs_UKB736_split-{out_file}_sm-2.tsv',
                    index=False, sep='\t')
    

def eval_HCP_group_parcellation(parcels, names, space='fs32k', test_ses='all', K=[7],
                                train_smooth=None, test_smooth=None, subj_list="HCP80_training+validation_set.tsv",
                                out_file=1):
    out_file = int(out_file)
    hcp_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    if not isinstance(parcels, list):
        parcels = [parcels]

    space_sp = space.split('_')
    if len(space_sp) == 1:
        hemis = 'full'
        atlas, _ = am.get_atlas(space)
        atlas.calculate_symmetry()
    elif len(space_sp) == 2:
        hemis = 'half'
        space = space_sp[0]
        hem = space_sp[1]
        hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
        atlas, _ = am.get_atlas(space)
        stru_idx = atlas.structure.index(hemis_dict[hem])
        atlas.calculate_symmetry()
    else:
        raise NameError('Unrecognized `space` for atlasing!')

    # dist = futil.load_fs32k_dist(file_type=f'distGOD_mid_{hem}', hemis='half',
    #                              device=DEVICE if pt.cuda.is_available() else 'cpu')
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)
    ## Load HCP subjects test data
    print(f'Start loading data: HCP {space} resting - Tseries ...')
    tic = time.perf_counter()
    ## HCP task betas
    t_data, _, _, _, t_info = ut.build_hcp_datasets(HCP_DIR, subj_list,
                                                    atlas, ses_list=['ses-task'], join_sess=False, join_sess_part=False,
                                                    part_ind=['half'], part_num=None, cond_ind=['reg_id'],
                                                    type=['CondHalf'], hemis=None, smooth='6fwhm')
    t_info = np.array_split(t_info, 2)[0]
    # t_data = ut.load_hcp_timeseries(HCP_DIR, subj_list,
    #                                 space=atlas.name, run_list=[2, 3],
    #                                 type='Tseries', hemis=None, smooth=None)
    # t_data, t_info = ut.load_hcp_contrasts(HCP_DIR, subj_list, space=atlas.name,
    #                                     hemis=None, smooth='4_MSMAll')
    # t_data, t_info = ut.load_randy_contrasts(space=atlas.name, hemis=None, smooth='2',
    #                                          subj=[1,2,3,5,6,7,8,11,12,13,14])

    # t_data = ut.build_resting_data('RANDY15', space='fs32k',
    #                                         ses_list=[f'ses-rest{i}' for i in range(1,11)],
    #                                         type='Tseries', subj=None, hemis=hem, smooth=None)
    toc = time.perf_counter()
    print(f'Done loading. Used {toc - tic:0.4f} seconds!')
    
    results = pd.DataFrame()
    for i, par in enumerate(parcels):
        ######## Step 1. Load UKB 736 subjects training data
        print(f'Start evaluating {names[i]} on HCP subjects rest run {out_file}')
        
        Pgroup = np.where(par==0, np.nan, par)
        ######## Step 3: Evaluate individual maps using DCBC
        # Step 3.2: Gatering all necessary information for evaluation
        eval_info = make_eval_info(K[i], atlas=space, train_info=['group_train'], train_sess='all',
                                    tdata='HCP203_test', test_sess=None,
                                    model_type='Models_03', group_map_name=names[i],
                                    test_kappa=None)
        for r, td in enumerate(t_data):

            tasks_list = hcp_tasks + ['all']
            for task in tasks_list:
                if task == 'all':
                    idx = [True] * len(t_info)

                    # Individual evaluation
                    # homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:,idx,:])
                    zvalue_group = ev.calc_test_zvalue(Pgroup, td[:, idx, :], return_single=False)
                    np.save(RESULT_DIR + f'/section_4/HCP/zvalues' +
                            f'/zvalue_group_{names[i]}_on_HCPbetas_sm6_{out_file}.npy',
                            zvalue_group.cpu().numpy())
                    inhomo_nets = ev.calc_test_task_inhomogeneity(Pgroup, td[:, idx, :], return_single=False)
                    inhomo_nets = pt.where(inhomo_nets == 0, pt.nan, inhomo_nets)
                    np.save(RESULT_DIR + f'/section_4/HCP/inhomogeneity' +
                            f'/inhomo_nets_group_{names[i]}_on_HCPbetas_sm6_{out_file}.npy',
                            inhomo_nets.cpu().numpy())

                else:
                    idx = t_info['task_name'] == task

                hut.report_cuda_memory()
                pt.cuda.empty_cache()
                # Step 3.3: Do DCBC evaluation on the second half data
                res = eval_group_DCBC(Pgroup, td[:,idx,:], dist, eval_info,
                                      subj_list=pd.read_csv(HCP_DIR+subj_list, delimiter='\t')["participant_id"].to_list())

                pt.cuda.empty_cache()
                hut.report_cuda_memory()
                res['brain_wise'] = 'whole_brain'
                res['test_run'] = r
                res['task_name'] = task
                res['train_smooth'] = train_smooth[i]
                res['test_smooth'] = test_smooth
                res['test_type'] = 'betas'
                results = pd.concat([results, res], ignore_index=True)
    
    # results.to_csv(EVAL_DIR + f'/eval_group_rest_vs_task_vs_fusion_K-17_on-HCPtest-task-allcontrasts_sm4_{out_file}.tsv',
    #                 index=False, sep='\t')
    return results


def eval_fs32k_group_parcellation(parcels, names, t_data, task_nam=None, space='fs32k', train_smooth=None,
                                  test_smooth=None, K=[7]):

    if not isinstance(parcels, list):
        parcels = [parcels]

    # Correct for left or right hemisphere
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
        atlas.calculate_symmetry()
        stru_idx = atlas.structure.index(hemis_dict[hem])
        vert_indx = atlas.indx_full[stru_idx]
    else:
        raise NameError('Unrecognized `space` for atlasing!')
    
    dist = futil.load_fs32k_dist(file_type='distGOD_sp', hemis=hemis,
                                  device=DEVICE if pt.cuda.is_available() else 'cpu')
    
    results = pd.DataFrame()
    for i, par in enumerate(parcels):
        ######## Step 1. Load UKB 736 subjects training data
        print(f'Start evaluating group map {names[i]}')
        
        Pgroup = np.where(par==0, np.nan, par)
        ######## Step 3: Evaluate individual maps using DCBC
        # Step 3.2: Gatering all necessary information for evaluation
        eval_info = make_eval_info(K[i], atlas=space, train_info=None, train_sess='all',
                                    tdata='MSC_task-contrasts', test_sess=None,
                                    model_type='Models_03', group_map_name=names[i],
                                    test_kappa=None)
        for r, td in enumerate(t_data):
            # Step 3.3: Do DCBC evaluation on the second half data
            res = eval_group_DCBC(Pgroup, td[:,:,vert_indx], dist, eval_info, subj_list=None)
            # res['brain_wise'] = BRAIN_WISE[i]
            res['test_run'] = r
            res['task_name'] = task_nam[r]
            res['train_smooth'] = train_smooth[i]
            res['test_smooth'] = test_smooth
            res['test_type'] = 'contrast'
            results = pd.concat([results, res], ignore_index=True)
    
    return results


def make_md_smoothing_group_atlas_specs(model_dir=None, K=17):
    """Return the Md K=17 smoothing/masking atlas families to evaluate."""
    model_dir = Path(model_dir) if model_dir is not None else Path(MODEL_DIR) / 'Models_03'
    smooth_levels = [0, 2, 4, 6, 8, 10]

    specs = []
    for smooth in smooth_levels:
        suffix = '' if smooth == 0 else f'_sm{smooth}fwhm'
        specs.append({
            'family': 'smoothed_only',
            'processing_order': 'smooth_only',
            'smooth_fwhm': smooth,
            'name': f'Md_smooth-only_sm{smooth}',
            'path': model_dir / f'asym_Md_space-fs32k_K-{K}{suffix}.dlabel.nii',
        })

    for smooth in smooth_levels:
        suffix = '_masked-hi0.1lo0.1' if smooth == 0 else f'_masked-hi0.1lo0.1_desc-sm{smooth}fwhm'
        specs.append({
            'family': 'mask_then_smooth',
            'processing_order': 'mask_then_smooth',
            'smooth_fwhm': smooth,
            'name': f'Md_mask-then-smooth_sm{smooth}',
            'path': model_dir / f'asym_Md_space-fs32k_K-{K}{suffix}.dlabel.nii',
        })

    for smooth in [2, 4, 6, 8, 10]:
        specs.append({
            'family': 'smooth_then_mask',
            'processing_order': 'smooth_then_mask',
            'smooth_fwhm': smooth,
            'name': f'Md_smooth-then-mask_sm{smooth}',
            'path': model_dir / f'asym_Md_space-fs32k_K-{K}_sm{smooth}fwhm_masked-hi0.1lo0.1.dlabel.nii',
        })

    missing = [str(spec['path']) for spec in specs if not spec['path'].exists()]
    if missing:
        raise FileNotFoundError('Missing group atlas file(s):\n' + '\n'.join(missing))
    return specs


def load_group_atlas_from_dlabel(atlas, path):
    """Load one dlabel group atlas and return a 1D parcel label vector."""
    parcel = atlas.cifti_to_data(str(path))
    parcel = np.asarray(parcel).squeeze()
    if parcel.ndim > 1:
        parcel = parcel[0]
    return parcel.reshape(-1)


def load_hcp_task_contrast_eval_data(atlas, subj_list_file, smooth='4_MSMAll',
                                     positive_only=False):
    """Load HCP task contrast dscalars using the indiv_eval_hcp path."""
    t_data, t_info = ut.load_hcp_contrasts(
        HCP_DIR, f'/subj_list/{subj_list_file}', space=atlas.name,
        return_positive=positive_only, hemis=None, smooth=smooth)

    if 'task_name' in t_info.columns:
        t_info['task_name'] = [str(task).rstrip('2') for task in t_info.task_name]
    return t_data, t_info


def eval_md_smoothing_group_atlases_on_hcp_task(
        subj_list_file='HCP200_test_1.tsv',
        model_dir=None,
        out_file=None,
        test_smooth='4_MSMAll',
        positive_only=False,
        K=17):
    """Evaluate Md smoothing/masking group atlases on HCP task contrasts."""
    subj_list_file = Path(subj_list_file).name
    atlas, _ = am.get_atlas('fs32k')
    atlas.calculate_symmetry()
    dist = pt.load(BASE_DIR + '/Atlases/tpl-fs32k/distGOD_fs32k.pt', weights_only=True)
    specs = make_md_smoothing_group_atlas_specs(model_dir=model_dir, K=K)

    print(f'Loading HCP task contrasts: {subj_list_file}, smooth={test_smooth} ...')
    tic = time.perf_counter()
    t_data, t_info = load_hcp_task_contrast_eval_data(
        atlas, subj_list_file, smooth=test_smooth, positive_only=positive_only)
    toc = time.perf_counter()
    print(f'Done loading HCP task contrasts. Used {toc - tic:0.4f} seconds.')

    subj_table = pd.read_csv(Path(HCP_DIR) / 'subj_list' / subj_list_file, sep='\t')
    subj_ids = subj_table['participant_id'].to_list()
    tasks = ['all']

    results = pd.DataFrame()
    for spec in specs:
        print(f'Evaluating {spec["name"]}: {spec["path"].name}')
        parcel = load_group_atlas_from_dlabel(atlas, spec['path'])
        Pgroup = np.where(parcel == 0, np.nan, parcel)
        eval_info = make_eval_info(
            K, atlas='fs32k', train_info=['MDTB'], train_sess='all',
            tdata='HCP', test_sess='task-contrasts',
            model_type='Models_03', group_map_name=spec['name'],
            test_kappa=None)

        for test_idx, td in enumerate(t_data):
            if type(td) is np.ndarray:
                td = pt.tensor(td, dtype=pt.get_default_dtype())
            for task in tasks:
                if task == 'all':
                    task_idx = np.ones(len(t_info), dtype=bool)
                else:
                    task_idx = (t_info['task_name'] == task).to_numpy()
                if not np.any(task_idx):
                    continue

                res = eval_group_DCBC(Pgroup, td[:, task_idx, :], dist, eval_info,
                                      subj_list=subj_ids)
                res['atlas_family'] = spec['family']
                res['processing_order'] = spec['processing_order']
                res['smooth_fwhm'] = spec['smooth_fwhm']
                res['atlas_file'] = spec['path'].name
                res['train_smooth'] = spec['smooth_fwhm']
                res['test_smooth'] = test_smooth
                res['test_type'] = 'contrast'
                res['test_run'] = test_idx
                res['task_name'] = task
                results = pd.concat([results, res], ignore_index=True)

    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_file, index=False, sep='\t')
        print(f'Saved evaluation results to {out_file}')
    return results


def find_md_smoothing_eval_files(out_dir=EVAL_DIR, test_smooth='4_MSMAll'):
    """Find saved Md smoothing-family HCP contrast evaluation TSVs."""
    pattern = f'eval_group-Md_smoothing-families_K-17_on-HCPtask-contrast-{test_smooth}_*.tsv'
    return sorted(Path(out_dir).glob(pattern))


def make_task_contrast_plot_input(result_files=None, out_dir=EVAL_DIR,
                                  test_smooth='4_MSMAll', task_name='all'):
    """Return the minimal raw rows needed for the HCP task-contrast plot."""
    if result_files is None or len(result_files) == 0:
        result_files = find_md_smoothing_eval_files(out_dir=out_dir, test_smooth=test_smooth)
    else:
        result_files = [Path(path) for path in result_files]
    if not result_files:
        raise FileNotFoundError(
            f'No Md smoothing-family evaluation TSVs found in {out_dir!r} '
            f'for test_smooth={test_smooth!r}.')

    frames = []
    for path in result_files:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, sep='\t'))
    results = pd.concat(frames, ignore_index=True)
    results = results.loc[results['task_name'] == task_name].copy()
    if results.empty:
        raise ValueError(f'No rows found for task_name={task_name!r}.')

    hue_order = ['Smoothed only', 'Smooth -> mask']
    results['atlas_family_label'] = pd.Series(index=results.index, dtype=object)
    results.loc[results['atlas_family'] == 'smoothed_only', 'atlas_family_label'] = 'Smoothed only'
    results.loc[results['atlas_family'] == 'smooth_then_mask', 'atlas_family_label'] = 'Smooth -> mask'
    results.loc[
        (results['atlas_family'] == 'mask_then_smooth') & (results['smooth_fwhm'] == 0),
        'atlas_family_label'
    ] = 'Smooth -> mask'
    results['atlas_family_label'] = pd.Categorical(
        results['atlas_family_label'], hue_order, ordered=True)
    results = results.loc[results['atlas_family_label'].isin(hue_order)].copy()
    smooth_order = sorted(
        results.groupby('smooth_fwhm', observed=True)['atlas_family_label'].nunique()
        .loc[lambda counts: counts == len(hue_order)]
        .index
        .to_list()
    )
    results = results.loc[results['smooth_fwhm'].isin(smooth_order)].copy()
    return results[
        ['smooth_fwhm', 'atlas_family_label', 'subj_num', 'dcbc_group', 'inhomo_group']
    ].sort_values(['smooth_fwhm', 'atlas_family_label', 'subj_num']).reset_index(drop=True)


def plot_grouped_bars_upper_sem(ax, summary, x_col, hue_col, mean_col, sem_col,
                                x_order, hue_order, palette, total_width=0.76,
                                capsize=4, edge_color='black',
                                edge_width=1.0):
    """Draw grouped bars with upper-only SEM error bars."""
    x_positions = np.arange(len(x_order))
    bar_width = total_width / len(hue_order)
    for hue_idx, hue in enumerate(hue_order):
        offsets = x_positions + (hue_idx - (len(hue_order) - 1) / 2) * bar_width
        sub = summary.loc[summary[hue_col] == hue].set_index(x_col)
        means = np.array([sub.loc[x, mean_col] if x in sub.index else np.nan for x in x_order], dtype=float)
        sems = np.array([sub.loc[x, sem_col] if x in sub.index else np.nan for x in x_order], dtype=float)
        yerr = np.vstack([np.zeros_like(sems), sems])
        ax.bar(
            offsets,
            means,
            width=bar_width,
            label=hue,
            color=palette[hue],
            edgecolor=edge_color,
            linewidth=edge_width,
            yerr=yerr,
            capsize=capsize,
            error_kw={'elinewidth': edge_width, 'capthick': edge_width},
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_order)


def plot_md_smoothing_group_eval(result_files=None, out_dir=EVAL_DIR,
                                 test_smooth='4_MSMAll', task_name='all',
                                 save_file=None):
    """Plot group DCBC and task inhomogeneity for Md atlas families."""
    results = make_task_contrast_plot_input(
        result_files=result_files, out_dir=out_dir,
        test_smooth=test_smooth, task_name=task_name)

    family_order = ['Smoothed only', 'Smooth -> mask']
    palette = {
        'Smoothed only': '#C8C8C8',
        'Smooth -> mask': '#595959',
    }
    hue_order = family_order
    smooth_order = sorted(results['smooth_fwhm'].dropna().unique())

    summary = (
        results.groupby(['smooth_fwhm', 'atlas_family_label'], observed=True)
        .agg(
            n_subjects=('subj_num', 'nunique'),
            dcbc_group_mean=('dcbc_group', 'mean'),
            dcbc_group_sem=('dcbc_group', 'sem'),
            inhomo_group_mean=('inhomo_group', 'mean'),
            inhomo_group_sem=('inhomo_group', 'sem'),
        )
        .reset_index()
        .sort_values(['smooth_fwhm', 'atlas_family_label'])
    )
    print(summary.to_string(index=False))

    edge_color = 'black'
    edge_width = 1.0
    capsize = 4

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True)
    panels = [
        ('dcbc_group_mean', 'dcbc_group_sem', 'Group DCBC'),
        ('inhomo_group_mean', 'inhomo_group_sem', 'Group Task Inhomogeneity'),
    ]

    for ax, (mean_col, sem_col, title) in zip(axes, panels):
        plot_grouped_bars_upper_sem(
            ax, summary,
            x_col='smooth_fwhm',
            hue_col='atlas_family_label',
            mean_col=mean_col,
            sem_col=sem_col,
            x_order=smooth_order,
            hue_order=hue_order,
            palette=palette,
            total_width=0.76,
            capsize=capsize,
            edge_color=edge_color,
            edge_width=edge_width,
        )
        ax.set_title(title)
        ax.set_xlabel('Training smoothing (FWHM)')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=0)

    axes[0].set_ylim(0, 0.07)
    axes[1].set_ylim(0.92, 0.96)
    axes[0].legend(title='Atlas family', frameon=False)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    fig.suptitle(
        f'Md group atlas smoothing/masking families\n'
        f'Tested on HCP task contrasts ({test_smooth}, task={task_name})'
    )
    fig.tight_layout()

    if save_file is None:
        save_file = (
            Path(out_dir) /
            f'plot_group-Md_smoothing-families_K-17_on-HCPtask-contrast-{test_smooth}_task-{task_name}.pdf'
        )
    if save_file:
        save_file = Path(save_file)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, format=save_file.suffix.lstrip('.') or 'pdf')
        print(f'Saved plot to {save_file}')
    plt.show()
    return fig, axes, results, summary


def load_existing_md_hcp_rest_eval(out_dir=EVAL_DIR):
    """Load the existing Md smoothing-family evaluations on HCP resting-state data."""
    out_dir = Path(out_dir)
    subj_table = pd.read_csv(Path(HCP_DIR) / 'subj_list' / 'HCP203_test_set.tsv', sep='\t')
    subj_id_map = dict(enumerate(subj_table['participant_id'].to_list()))
    family_specs = [
        (
            'eval_Md_K-17_sm0-10_on-HCPtest_split-{split}.tsv',
            {
                'Md_K-17_': 0,
                'Md_K-17__sm2fwhm': 2,
                'Md_K-17__sm4fwhm': 4,
                'Md_K-17__sm6fwhm': 6,
                'Md_K-17__sm8fwhm': 8,
                'Md_K-17__sm10fwhm': 10,
            },
            'smooth_only',
        ),
        (
            'eval_Md_K-17_sm0-10_masked_on-HCPtest_split-{split}.tsv',
            {
                'Md_K-17_sm2fwhm_masked': 2,
                'Md_K-17_sm4fwhm_masked': 4,
                'Md_K-17_sm6fwhm_masked': 6,
                'Md_K-17_sm8fwhm_masked': 8,
                'Md_K-17_sm10fwhm_masked': 10,
            },
            'smooth-mask',
        ),
        (
            'eval_Md_K-17_masked_sm0-10_on-HCPtest_split-{split}.tsv',
            {
                'Md_K-17': 0,
                'Md_K-17_desc-sm2fwhm': 2,
                'Md_K-17_desc-sm4fwhm': 4,
                'Md_K-17_desc-sm6fwhm': 6,
                'Md_K-17_desc-sm8fwhm': 8,
                'Md_K-17_desc-sm10fwhm': 10,
            },
            'mask-smooth',
        ),
    ]

    frames = []
    for file_template, smooth_map, atlas_type in family_specs:
        for split in [1, 2, 3, 4]:
            path = out_dir / file_template.format(split=split)
            if not path.exists():
                raise FileNotFoundError(path)
            res = pd.read_csv(path, sep='\t')
            res['train_smooth'] = res['group_map_name'].map(smooth_map)
            if res['train_smooth'].isna().any():
                missing_names = res.loc[res['train_smooth'].isna(), 'group_map_name'].unique()
                raise ValueError(f'Unmapped group_map_name values in {path.name}: {missing_names}')
            res['train_smooth'] = pd.to_numeric(res['train_smooth'], errors='raise')
            global_subj_num = res['subj_num'] + (split - 1) * 50
            res['subj_num'] = global_subj_num.map(subj_id_map)
            if res['subj_num'].isna().any():
                missing_ids = global_subj_num.loc[res['subj_num'].isna()].unique()
                raise ValueError(f'Unmapped subject indices in {path.name}: {missing_ids}')
            res['subj_num'] = res['subj_num'].astype(int)
            res['type'] = atlas_type
            res['source_file'] = path.name
            frames.append(res)

    return pd.concat(frames, ignore_index=True)


def make_existing_hcp_rest_plot_input(out_dir=EVAL_DIR):
    """Return the minimal raw rows needed for the HCP resting-state plot."""
    results = load_existing_md_hcp_rest_eval(out_dir=out_dir)
    type_order = ['smooth_only', 'smooth-mask']
    results['plot_type'] = results['type'].astype(str)
    results.loc[
        (results['type'].astype(str) == 'mask-smooth') & (results['train_smooth'] == 0),
        'plot_type'
    ] = 'smooth-mask'
    results['plot_type'] = pd.Categorical(results['plot_type'], type_order, ordered=True)
    results = results.loc[results['plot_type'].isin(type_order)].copy()
    smooth_order = sorted(
        results.groupby('train_smooth', observed=True)['plot_type'].nunique()
        .loc[lambda counts: counts == len(type_order)]
        .index
        .to_list()
    )
    results = results.loc[results['train_smooth'].isin(smooth_order)].copy()
    return results[
        ['train_smooth', 'plot_type', 'subj_num', 'test_run', 'dcbc_group', 'homo_group']
    ].sort_values(['train_smooth', 'plot_type', 'subj_num', 'test_run']).reset_index(drop=True)


def plot_existing_hcp_rest_group_eval(
        result_files=None,
        out_dir=EVAL_DIR,
        save_file=None):
    """Plot the existing Md group-atlas evaluations on HCP resting-state data."""
    if result_files not in (None, []):
        raise ValueError('This plot uses the fixed eval_Md_K-17 split files; do not pass --plot-files.')

    results = make_existing_hcp_rest_plot_input(out_dir=out_dir)
    type_order = ['smooth_only', 'smooth-mask']
    smooth_order = sorted(results['train_smooth'].dropna().unique())
    palette = {
        'smooth_only': '#C8C8C8',
        'smooth-mask': '#595959',
    }
    metrics = [
        ('dcbc_group_mean', 'dcbc_group_sem', 'Group DCBC'),
        ('homo_group_mean', 'homo_group_sem', 'Group Homogeneity'),
    ]

    summary = (
        results.groupby(['train_smooth', 'plot_type'], observed=True)
        .agg(**{
            'n_observations': ('dcbc_group', 'count'),
            'n_subjects': ('subj_num', 'nunique'),
            'dcbc_group_mean': ('dcbc_group', 'mean'),
            'dcbc_group_sem': ('dcbc_group', 'sem'),
            'homo_group_mean': ('homo_group', 'mean'),
            'homo_group_sem': ('homo_group', 'sem'),
        })
        .reset_index()
        .sort_values(['train_smooth', 'plot_type'])
    )
    print(summary.to_string(index=False))

    edge_color = 'black'
    edge_width = 1.0
    capsize = 4

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharex=True)
    for ax, (mean_col, sem_col, title) in zip(axes, metrics):
        plot_grouped_bars_upper_sem(
            ax, summary,
            x_col='train_smooth',
            hue_col='plot_type',
            mean_col=mean_col,
            sem_col=sem_col,
            x_order=smooth_order,
            hue_order=type_order,
            palette=palette,
            total_width=0.76,
            capsize=capsize,
            edge_color=edge_color,
            edge_width=edge_width,
        )
        ax.set_title(title)
        ax.set_xlabel('Training smoothing (FWHM)')
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=0)

    axes[0].set_ylim(0, 0.03)
    axes[1].set_ylim(0.04, 0.06)
    axes[0].legend(title='Atlas family', frameon=False)
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()

    fig.suptitle('Existing Md group atlas smoothing/masking families\nTested on HCP resting-state data')
    fig.tight_layout()

    if save_file is None:
        save_file = Path(out_dir) / 'plot_existing-Md-smoothing-families_on-HCPrest-Tseries.pdf'
    if save_file:
        save_file = Path(save_file)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, format=save_file.suffix.lstrip('.') or 'pdf')
        print(f'Saved plot to {save_file}')
    plt.show()
    return fig, axes, results, summary


def parse_group_eval_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate Md smoothing/masking group atlases on HCP task contrasts.')
    parser.add_argument(
        '--eval-md-smoothing-hcp-task', action='store_true',
        help='Run the Md smoothing/masking atlas-family evaluation.')
    parser.add_argument(
        '--plot-md-smoothing-hcp-task', action='store_true',
        help='Plot saved Md smoothing/masking atlas-family HCP evaluation TSVs.')
    parser.add_argument(
        '--plot-existing-hcp-rest', action='store_true',
        help='Plot existing group-atlas HCP resting-state evaluation TSVs.')
    parser.add_argument(
        '--export-plot-inputs', action='store_true',
        help='Export compact TSV files containing the raw rows needed to recreate both plots.')
    parser.add_argument(
        '--split', default='4',
        help="HCP200 test split to evaluate: 1, 2, 3, 4, or 'all'.")
    parser.add_argument(
        '--subj-list-file', default=None,
        help='Explicit HCP subject-list TSV filename under HCP_img/subj_list.')
    parser.add_argument(
        '--model-dir', default=str(Path(MODEL_DIR) / 'Models_03'),
        help='Directory containing the Md K=17 dlabel atlas files.')
    parser.add_argument(
        '--out-dir', default=EVAL_DIR,
        help='Directory where evaluation TSV files are written.')
    parser.add_argument(
        '--test-smooth', default='4_MSMAll',
        help='Smoothing tag for HCP task contrast dscalars.')
    parser.add_argument(
        '--positive-only', action='store_true',
        help='Evaluate only positive HCP task contrasts.')
    parser.add_argument(
        '--plot-files', nargs='*', default=None,
        help='Specific evaluation TSV files to plot. Defaults to all matching files in --out-dir.')
    parser.add_argument(
        '--plot-task', default='all',
        help="Task name to plot from the evaluation TSVs. Default: 'all'.")
    parser.add_argument(
        '--plot-out', default=None,
        help='Output plot path. Defaults to a PDF in --out-dir.')
    parser.add_argument(
        '--task-plot-data-out', default=None,
        help='Output TSV path for compact HCP task-contrast plot data.')
    parser.add_argument(
        '--rest-plot-data-out', default=None,
        help='Output TSV path for compact HCP resting-state plot data.')
    return parser.parse_args(argv)


def run_md_smoothing_group_eval_from_args(args):
    if args.subj_list_file is not None:
        subj_list_files = [Path(args.subj_list_file).name]
    elif str(args.split).lower() == 'all':
        subj_list_files = [f'HCP200_test_{split}.tsv' for split in [1, 2, 3, 4]]
    else:
        subj_list_files = [f'HCP200_test_{int(args.split)}.tsv']

    for subj_list_file in subj_list_files:
        split_tag = Path(subj_list_file).stem.replace('HCP200_test_', 'split-')
        if split_tag == 'HCP200_test':
            split_tag = 'all-subjects'
        out_file = (
            Path(args.out_dir) /
            f'eval_group-Md_smoothing-families_K-17_on-HCPtask-contrast-{args.test_smooth}_{split_tag}.tsv'
        )
        eval_md_smoothing_group_atlases_on_hcp_task(
            subj_list_file=subj_list_file,
            model_dir=args.model_dir,
            out_file=out_file,
            test_smooth=args.test_smooth,
            positive_only=args.positive_only,
            K=17)


def run_md_smoothing_group_plot_from_args(args):
    plot_md_smoothing_group_eval(
        result_files=args.plot_files,
        out_dir=args.out_dir,
        test_smooth=args.test_smooth,
        task_name=args.plot_task,
        save_file=args.plot_out)


def run_existing_hcp_rest_group_plot_from_args(args):
    plot_existing_hcp_rest_group_eval(
        result_files=args.plot_files,
        out_dir=args.out_dir,
        save_file=args.plot_out)


def export_plot_input_files(out_dir=EVAL_DIR, test_smooth='4_MSMAll', task_name='all',
                            task_out=None, rest_out=None):
    """Export the compact raw rows used by the task and rest plots."""
    out_dir = Path(out_dir)
    task_data = make_task_contrast_plot_input(
        out_dir=out_dir, test_smooth=test_smooth, task_name=task_name)
    rest_data = make_existing_hcp_rest_plot_input(out_dir=out_dir)

    if task_out is None:
        task_out = (
            out_dir /
            f'plotdata_group-Md_smoothing-families_K-17_on-HCPtask-contrast-{test_smooth}_task-{task_name}.tsv'
        )
    else:
        task_out = Path(task_out)
    if rest_out is None:
        rest_out = out_dir / 'plotdata_existing-Md-smoothing-families_on-HCPrest-Tseries.tsv'
    else:
        rest_out = Path(rest_out)

    task_out.parent.mkdir(parents=True, exist_ok=True)
    rest_out.parent.mkdir(parents=True, exist_ok=True)
    task_data.to_csv(task_out, sep='\t', index=False)
    rest_data.to_csv(rest_out, sep='\t', index=False)

    print(f'Saved task contrast plot data to {task_out} ({task_data.shape[0]} rows)')
    print(f'Saved resting-state plot data to {rest_out} ({rest_data.shape[0]} rows)')
    return task_data, rest_data


def run_export_plot_inputs_from_args(args):
    export_plot_input_files(
        out_dir=args.out_dir,
        test_smooth=args.test_smooth,
        task_name=args.plot_task,
        task_out=args.task_plot_data_out,
        rest_out=args.rest_plot_data_out)


def make_rand_par_L(K, num_parcellation=100, mesh='sphere'):
    # Define the atlas to generate random parcellation
    atlas, _ = am.get_atlas('fs32k', ATLAS_DIR)
    # Load surface and mask files
    surf_file_L = ATLAS_DIR + f'/tpl-fs32k/tpl-fs32k_hemi-L_{mesh}.surf.gii'
    mask_file_L = ATLAS_DIR + '/tpl-fs32k/tpl-fs32k_hemi-L_mask.label.gii'

    # Generate 100 random parcellation given the resolution
    rand_par, names = [], []
    for i in range(num_parcellation):
        this_par = hut.make_random_parcellation(K, surf_file_L, mask_file_L)
        this_par = this_par[atlas.vertex_mask[0]]
        rand_par.append(this_par)
        names.append(f'random_K-{K}_{i+1}')

    return rand_par, names

if __name__ == "__main__":
    args = parse_group_eval_args()
    if args.export_plot_inputs:
        run_export_plot_inputs_from_args(args)
        sys.exit(0)

    if len(sys.argv) == 1:
        run_md_smoothing_group_plot_from_args(args)
        run_existing_hcp_rest_group_plot_from_args(args)
        sys.exit(0)

    if args.plot_existing_hcp_rest:
        run_existing_hcp_rest_group_plot_from_args(args)
        sys.exit(0)

    if args.plot_md_smoothing_hcp_task:
        run_md_smoothing_group_plot_from_args(args)
        sys.exit(0)

    if args.eval_md_smoothing_hcp_task:
        run_md_smoothing_group_eval_from_args(args)
        sys.exit(0)

    # if len(sys.argv) != 3:
    #     print("Usage: python group_eval.py <K> <i>")
    #     sys.exit(1)

    atlas, _ = am.get_atlas('fs32k')
    RES_DIR = '/home/dzhi/eris_mount/dzhi/Indiv_par/Models/Models_03'
    results = pd.DataFrame()
    parcels, names, n_parcels = load_existing_atlas()
    smoothes = [0] * len(parcels)
    # for K in range(10,11):
    #     print('Evaluating K={} ...'.format(K))
    #     # info, model = futil.load_batch_best(f'/Models_03/task_fusion/asym_MdPoNiIbWmDeSo_space-fs32k_K-{K}', device=DEVICE)
    #     # task_fusion_baseline = pt.argmax(model.marginal_prob()[:,0:29759], dim=0)+1
    #     # parcels.append(task_fusion_baseline.cpu().numpy())
    #     # names.append('baseline')
    #     # smoothes.append(0)
    #     # n_parcels.append(K)
    #
    #     for s in [6]:
    #         info, model2 = futil.load_batch_best(
    #             f'/Models_03/task_fusion/asym_MdPoNiIbWmDeSo_space-fs32k_K-{K}_sm{s}fwhm_zstat_masked-hi0.1lo0.1', device=DEVICE)
    #
    #         parcel2 = pt.argmax(model2.marginal_prob()[:, 0:29759], dim=0) + 1
    #         parcels.append(parcel2.cpu().numpy())
    #         names.append(f'smooth{s}fwhm_mask')
    #         smoothes.append(s)
    #         n_parcels.append(K)
    #
    #     # Load random parcellation
    #     pars, nams = make_rand_par_L(K)
    #     parcels += pars
    #     names += nams
    #     n_parcels += [K] * len(pars)
    #     smoothes += [0] * len(pars)

    # 1. DU15 rest-only baseline
    DU15, net_name, colors = gp.get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k')
    # 2. HBP15 rest-only baseline
    # HBP_15_Hc = atlas.cifti_to_data(RES_DIR + '/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized.dlabel.nii').reshape(-1)
    HBP_15_Hc = atlas.cifti_to_data(
        RES_DIR + '/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_DU15-inits.dlabel.nii').reshape(-1)
    # HBP_15_HcMd = atlas.cifti_to_data(RES_DIR + '/task_fusion/asym_MdHc_space-fs32k_K-15_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1.dlabel.nii').reshape(-1)
    # HBP_15_Hc = atlas.cifti_to_data(RES_DIR + '/asym_Hc_space-fs32k_K-15_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_all.dlabel.nii')
    # HBP_15_MdNiIbHc = atlas.cifti_to_data(RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_all.dlabel.nii')[0]

    ############ 17 networks ############
    YEO2011 = nb.load('/home/dzhi/eris_mount/dzhi/workspace/res/group/Yeo2011_17.dlabel.nii').get_fdata().reshape(-1)
    KONG2019 = np.argmax(ut.get_kong2019_group_parcellation()[0], axis=0) + 1
    Hc_1 = atlas.cifti_to_data(RES_DIR + '/task_fusion/asym_Hc_space-fs32k_K-17_HCP40-Kong_ROI1483Run_sm6fwhm_binarized_all.dlabel.nii')[32].reshape(-1)
    # Hc_2 = atlas.cifti_to_data(
    #     RES_DIR + '/asym_Hc_space-fs32k_K-17_HCP40subjects_ROI1483Run_desc-sm4fwhm_binarized.dlabel.nii').reshape(-1)
    MbNiIb_1 = atlas.cifti_to_data(
        RES_DIR + '/task_fusion/asym_MdNiIb_space-fs32k_K-17_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1_all.dlabel.nii')[0]
    MbNiIbHc_1 = atlas.cifti_to_data(
        RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess_all.dlabel.nii')[0]

    ############ TASK ############
    # MbNiIb = atlas.cifti_to_data(RES_DIR + '/task_fusion/asym_MdNiIb_space-fs32k_K-17_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1_all.dlabel.nii')[0]
    # 3. Fusion(MbNiIbHc) weighting N-feature, DU15 inits
    # MbNiIbHc_1 = atlas.cifti_to_data(RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_DU15-inits.dlabel.nii')[0]
    # # 4. Fusion(MbNiIbHc) weighting equal, DU15 inits
    # MbNiIbHc_2 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_equalweights_all.dlabel.nii')[0]
    # # 5. Fusion(MbNiIbHc) weighting rest1task1, DU15 inits
    # MbNiIbHc_3 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_task0.5rest0.5_DU15-inits_all.dlabel.nii')[0]


    # MdNiIbWmDeSo_1 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbWmDeSo_space-fs32k_K-17_arrange-independent_sm6fwhm_zstat_masked-hi0.1lo0.1.dlabel.nii')[0]
    #
    # MdNiIbWmDeSoHc_1 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbWmDeSoHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess_all.dlabel.nii')[0]
    # MdNiIbWmDeSoHc_2 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbWmDeSoHc_space-fs32k_K-17_sm6fwhm_binarized_Ib-jointsess_equalweights_all.dlabel.nii')[0]


    parcels = [YEO2011, KONG2019, Hc_1, MbNiIb_1, MbNiIbHc_1]
    names = ['YEO2011', 'KONG2019', 'HBP17_rest', '3Task', 'Fusion(Nfeature 3+1)']

    # parcels = [DU15, HBP_15_Hc, MbNiIbHc_1, MbNiIb]
    # names = ['DU15', 'HBP15_rest', 'Fusion(N-feature) DU15init', 'Task (MbNiIb)']

    # 6. Fusion(MbNiIbHc) weighting equal, random inits
    # MbNiIbHc_4 = atlas.cifti_to_data(
    #     RES_DIR + '/task_fusion/asym_MdNiIbHc_space-fs32k_K-15_sm6fwhm_binarized_Ib-jointsess_equalweights_random-inits_all.dlabel.nii')
    # parcels += [MbNiIbHc_4[i] for i in range(MbNiIbHc_4.shape[0])]
    # names += [f'Fusion(equal) randinit{i}' for i in range(MbNiIbHc_4.shape[0])]
    n_parcels = [17] * len(parcels)
    smoothes = [0] * len(parcels)

    ### Evaluation on the task-based datasets
    # dat1, info1, _ = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, sess='ses-s1',
    #                                 type='CondHalf', subj=None, smooth=None)
    # dat2, info2, _ = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, sess='ses-s2',
    #                                 type='CondHalf', subj=None, smooth=None)
    #
    # data = np.concatenate([dat1, dat2], axis=1)
    # info = pd.concat([info1, info2], ignore_index=True)
    # info['task_name']=[s.rstrip('2') for s in info.task_name]
    #
    # results = pd.DataFrame()
    # for task in info.task_name.unique():
    #     idx = info['task_name'] == task
    #     res = eval_fs32k_group_parcellation(parcels, names, [data[:,idx,:]],
    #                                 task_nam=task, space='fs32k_L', test_smooth=None, K=17)
    #     results = pd.concat([results, res], ignore_index=True)

    # data, _ = load_msc_contrasts('MSC', sess='all', subj=None, smooth=2.55)
    # data1, _ = load_msc_contrasts('MSC', sess=['motor'], subj=None, smooth=2.55)
    # data2, _ = load_msc_contrasts('MSC', sess=['mixed'], subj=None, smooth=2.55)
    # data3, _ = load_msc_contrasts('MSC', sess=['memory'], subj=None, smooth=2.55)
    #
    # results = eval_fs32k_group_parcellation(parcels, names, [data], task_nam=['all'],
    #                                         space='fs32k_L', train_smooth=smoothes, test_smooth=None, K=n_parcels)
    # results.to_csv(EVAL_DIR + f'/eval_group-random_K-10-30_sm6-masked_on-MSC-task-contrasts.tsv',
    #                 index=False, sep='\t')

    ### Evaluation on the HCP resting-state data
    results = pd.DataFrame()
    for r in [1]:
        res = eval_HCP_group_parcellation(parcels, names, space='fs32k', K=n_parcels, train_smooth=smoothes,
                                    test_smooth=None, subj_list=f'/subj_list/HCP200_test.tsv', out_file=r)
        results = pd.concat([results, res], ignore_index=True)

    results = pd.DataFrame()
    for i in [1,2,3,4]:
        df = pd.read_csv(EVAL_DIR + f'/eval_group_rest_vs_task_vs_fusion_K-17_on-HCPtest-task-contrasts-sm4_{i}.tsv',
                         sep='\t')
        results = pd.concat([results, df], ignore_index=True)

    ## Plot results
    results = pd.read_csv(EVAL_DIR + f'/eval_group-random_K-10-30_sm6-masked_on-MSC-task-contrasts.tsv', sep='\t')
    results = results[(results['task_name']=='all')]

    res1 = pd.read_csv(EVAL_DIR + f'/eval_group-fusion_K-10-30_sm8-10-masked_on-MSC-task-contrasts.tsv', sep='\t')

    df_random = results[results['group_map_name'].str.startswith('random', na=False)]
    df_existing = results[~results['group_map_name'].str.startswith('random', na=False) &
                          ~results['group_map_name'].str.startswith('smooth', na=False)]

    df_smooth_masked = res1[res1['group_map_name'].str.startswith('smooth10', na=False) & (res1['task_name']=='all')]
    df_random_group_inhomo = df_random.groupby([df_random.K, df_random.subj_num])['inhomo_group'].mean().reset_index()
    df_random_group_dcbc = df_random.groupby([df_random.K, df_random.subj_num])['dcbc_group'].mean().reset_index()
    df_random_group = df_random_group_inhomo.merge(df_random_group_dcbc, on=['K', 'subj_num'])

    df_random_group.loc[:,'type'] = 'random'

    df_sm = df_smooth_masked[['K', 'subj_num', 'inhomo_group','dcbc_group']].reset_index(drop=True)
    df_sm['type'] = 'smooth_masked'
    df_new = pd.concat([df_sm, df_random_group], ignore_index=True)
    df_sm['diff_inhomo'] = df_sm['inhomo_group'] - df_random_group['inhomo_group']
    df_sm['diff_dcbc'] = df_sm['dcbc_group'] - df_random_group['dcbc_group']

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sb.barplot(data=df_new, x='K', y='inhomo_group', hue='type', errorbar='se')
    plt.ylim(0.8, 0.95)
    plt.subplot(1, 3, 2)
    sb.barplot(data=df_sm, x='K', y='diff_inhomo', errorbar='se')

    plt.subplot(1, 3, 3)
    sb.barplot(data=df_sm, x='K', y='dcbc_group', errorbar='se')



    plt.suptitle('Group parcellation: K=10-30, smooth=10, tested on MSC (task contrasts)')
    plt.tight_layout()
    plt.show()

    ######## Step 2. Generate group / indiv parcellations
    # Option 1: calculate indiv parcellations directly from fitted model
    # U, U_indv, M = get_indiv_parcellation_from_model(MODEL_DIR + 
    #                   f'/Models_07/asym_Uk_space-MNIAsymC2_K-7_ses-rest1', data)

    # Option 2: calculate indiv parcellations from existing group map
    # atlas_dir = ATLAS_DIR + '/tpl-MNI152NLin2009cSymC'
    # model_name = f'/atl-Buckner7_space-MNI152NLin2009cSymC_dseg.nii'
    # U_hard = atlas.read_data(atlas_dir + model_name)
    # conf_dir = '/data/tge/dzhi/Indiv_par/Buckner_JNeurophysiol11_MNI152'
    # conf_name = f'/Buckner2011_7NetworksConfidence_MNI152_FreeSurferConformed1mm_LooseMask.nii.gz'
    # conf_map = atlas.read_data(conf_dir + conf_name)
    # U = convert_hard_to_prob(U_hard, strength=1, confidence=conf_map)

    # ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
    #                                       sym_type='asym')
    # U_indv, _, M = fm.get_indiv_parcellation(ar_model, atlas, data,
    #                                          cond_vec, part_vec, subj_ind,
    #                                          sym_type='asym',
    #                                          em_params={'num_subj': data[0].shape[0],
    #                                                     'uniform_kappa': None,
    #                                                     'subjects_equal_weight':True,
    #                                                     'subject_specific_kappa': True,
    #                                                     'parcel_specific_kappa': True})

    # em_params={'subjects_equal_weight':True,
    #             'uniform_kappa': None,
    #             'subject_specific_kappa': False,
    #             'parcel_specific_kappa': True}

    # del data
    # pt.cuda.empty_cache()
    # fm.report_cuda_memory()

    # ######## Step 3: Evaluate individual maps using DCBC
    # # Step 3.1: compute the distance matrix
    # dist = ev.compute_dist(atlas.world.T, resolution=1)
    # # Step 3.2: Gatering all necessary information for evaluation
    # eval_info = make_eval_info(M, train_info=['UKB'], train_sess='ses-rest1',
    #                         tdata='UKB', test_sess='ses-rest2', 
    #                         model_type='Models_04', group_map_name='Buckner7',
    #                         test_kappa=None)
    # # Step 3.3: Do DCBC evaluation on the second half data
    # res = eval_parcel_DCBC(U, U_indv, t_data[0], dist, eval_info,
    #                         out_file='eval_dcbc_indiv_Buckner7_k-7_model-04_test.tsv')
    # dice = [hev.dice_coefficient(pt.tensor(U_hard), pt.argmax(U_indv, dim=1)[i]) 
    #         for i in range(U_indv.shape[0])]
    # # res.to_csv(f'eval_dcbc_indiv_Buckner7_k-7_model-04_test2_prior.tsv', index=False, sep='\t')

    # ######## Step 4: Visualization
    # # Step 4.1 (optional): plot the DCBC results
    # ev_df = pd.read_csv('eval_dcbc_indiv_parcellations.tsv', sep='\t')
    # plt.figure(figsize=(5, 5))
    # df = pd.melt(ev_df, var_name='group', value_name='value')
    # df = df.loc[(df['group'] == 'dcbc_group') | (df['group'] == 'dcbc_indiv')]
    # sb.barplot(x='group', y='value', errorbar="se", width=0.7, data=df)
    # plt.show()

    # # Step 4.2: plot group parcellation
    # plt.figure(figsize=(10, 10))
    # plot_multi_flat(U.unsqueeze(0).cpu().numpy(), 'MNIAsymC2', grid=(1, 1),
    #                 cmap='tab20', dtype='prob', titles=['group prior'])
    # plt.show()

    # # Step 4.3: plot individual parcellation
    # plt.figure(figsize=(40,20))
    # plot_multi_flat(U_indv.cpu().numpy(), 'MNIAsymC2', grid=(2, 5),
    #                 cmap='tab20', dtype='prob',
    #                 titles=["subj_{}".format(i+1) for i in range(U_indv.shape[0])])
    # plt.show()

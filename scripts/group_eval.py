#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: dzhi
"""
import time, sys, json
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
import HierarchBayesParcel.util as ut
import FusionModel.util as futil
import FusionModel.evaluate as ev

from group_parcellation import build_ukb_datasets, build_hcp_datasets, load_hcp_timeseries, load_hcp_task_contrast
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
HCP_DIR = '/home/dzhi/eris_mount/Tian/HCP_img'
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
    KONG2019 = nb.load(data_dir + '/dzhi/Indiv_par/Kong_2019/group_prior' \
                       '/HCP_40/Kong-2019_MSHBM_HCP40_hard.dlabel.nii').get_fdata().reshape(-1)[0:29759]

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


def make_eval_info(K, atlas='MNIAsymC2', train_info=['UKB'], train_sess='ses-2',
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
    # Now run the DCBC evaluation fo the group
    zvalue_group = ev.calc_test_zvalue(U_group, t_data, return_single=False)
    dcbc_group = ev.calc_test_dcbc(U_group, t_data, dist, trim_nan=True)
    inhomo_group = ev.calc_test_task_inhomogeneity(U_group, t_data,
                                                   return_single=True)
    # homo_group = ev.calc_test_homogeneity(U_group, t_data)

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
    ev_df['zvalue_group'] = zvalue_group.cpu()

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
    
    results.to_csv(f'/data/tge/dzhi/Indiv_par/Evaluations/eval_all_UKB-736_K-7_split-{out_file}.tsv',
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
    
    results.to_csv(f'/data/tge/dzhi/Indiv_par/Evaluations/eval_all_existing_vs_UKB736_split-{out_file}_sm-2.tsv',
                    index=False, sep='\t')
    

def eval_HCP_group_parcellation(parcels, names, space='fs32k', test_ses='all', K=[7],
                                train_smooth=None, test_smooth=None, subj_list="HCP80_training+validation_set.tsv",
                                out_file=1):
    out_file = int(out_file)

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
    
    dist = futil.load_fs32k_dist(file_type='distGOD_sp', hemis=hemis,
                                  device=DEVICE if pt.cuda.is_available() else 'cpu')
    
    ## Load HCP subjects test data
    print(f'Start loading data: HCP {space} resting - Tseries ...')
    tic = time.perf_counter()
    # t_data, _, _, _, _ = build_hcp_datasets(HCP_DIR, subj_list, atlas, ses_list=['ses-task'],
    #                                     join_sess=False, join_sess_part=False,
    #                                     part_ind=['half'], part_num=None,cond_ind=['reg_id'],
    #                                     type=['CondAll'], hemis=hem, smooth=test_smooth)
    
    # t_data = load_hcp_timeseries(HCP_DIR, subj_list, atlas, run_list=[out_file],
    #                             type='Tseries', hemis=hem, smooth=test_smooth)
    from indiv_eval_hcp import load_hcp_contrasts
    t_data, t_info = load_hcp_contrasts(HCP_DIR, subj_list, space=atlas.name,
                                        hemis=hem, smooth='2_MSMAll')
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
                                    tdata='HCP_test', test_sess=None, 
                                    model_type='Models_03', group_map_name=names[i],
                                    test_kappa=None)
        for r, td in enumerate(t_data):
            # Step 3.3: Do DCBC evaluation on the second half data
            res = eval_group_DCBC(Pgroup, td, dist, eval_info, subj_list=None)
            res['brain_wise'] = 'whole_brain'
            res['test_run'] = r
            res['train_smooth'] = train_smooth[i]
            res['test_smooth'] = test_smooth
            res['test_type'] = 'contrast'
            results = pd.concat([results, res], ignore_index=True)
    
    results.to_csv(f'/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations/eval_group-7taskfusion-random_K-10-30_sm6-masked_on-HCP203test-task-contrast.tsv',
                    index=False, sep='\t')
    

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


def make_rand_par_L(K, num_parcellation=100, mesh='sphere'):
    # Define the atlas to generate random parcellation
    atlas, _ = am.get_atlas('fs32k', ATLAS_DIR)
    # Load surface and mask files
    surf_file_L = ATLAS_DIR + f'/tpl-fs32k/tpl-fs32k_hemi-L_{mesh}.surf.gii'
    mask_file_L = ATLAS_DIR + '/tpl-fs32k/tpl-fs32k_hemi-L_mask.label.gii'

    # Generate 100 random parcellation given the resolution
    rand_par, names = [], []
    for i in range(num_parcellation):
        this_par = ut.make_random_parcellation(K, surf_file_L, mask_file_L)
        this_par = this_par[atlas.vertex_mask[0]]
        rand_par.append(this_par)
        names.append(f'random_K-{K}_{i+1}')

    return rand_par, names

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: python group_eval.py <K> <i>")
    #     sys.exit(1)

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
    # results.to_csv(f'/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations/eval_group-random_K-10-30_sm6-masked_on-MSC-task-contrasts.tsv',
    #                 index=False, sep='\t')

    ### Evaluation on the HCP resting-state data
    eval_HCP_group_parcellation(parcels, names, space='fs32k_L', K=n_parcels, train_smooth=smoothes,
                                test_smooth=None, subj_list=f'/subj_list/HCP203_test_set_filtered_1.tsv', out_file=1)

    ## Plot results
    results = pd.read_csv(f'/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations/eval_group-random_K-10-30_sm6-masked_on-MSC-task-contrasts.tsv', sep='\t')
    results = results[(results['task_name']=='all')]

    res1 = pd.read_csv(f'/home/dzhi/eris_mount/dzhi/Indiv_par/Evaluations/eval_group-fusion_K-10-30_sm8-10-masked_on-MSC-task-contrasts.tsv', sep='\t')

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
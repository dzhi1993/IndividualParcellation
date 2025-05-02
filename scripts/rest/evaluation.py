#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: Caro Nettekoven
"""
import numpy as np
import torch as pt
import nitools as nt
import pandas as pd
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm
from global_config import DEVICE
import IndividualParcellation.scripts.paths as paths
import time
from FusionModel.util import plot_data_flat, plot_multi_flat
from scripts.rest.fit import get_mdtb
import pickle
from IndividualParcellation.scripts.indiv_evaluation import make_eval_info, eval_parcel_DCBC
from scripts.localizer_batteries import domain_masks

# Settings
symmetric = True # Whether to get a symmetric or asymmetric parcellation
if symmetric:
    sym_type = 'sym'

space, _ = am.get_atlas('MNISymC2')
if space.name == 'MNISymC2':
    space_folder = 'tpl-MNI152NLin2009cSymC'

atlas_dir = paths.set_atlas_dir()
data_dir = paths.set_fusion_dir()
dataset_dir = data_dir + '/MDTB'
figure_dir = paths.set_figure_dir()
base_dir = paths.set_base_dir()
results_dir = paths.set_results_dir(base_dir)

v_types = {                    
            'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
            'general_eq':            {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
            'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
            'subject_specific_eq':   {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
            }

vmodel_types = {
            'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
            'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
            }

dtypes = ['Net67Run', 'Fus06Run']



def compute_var_cov(data, cond='all', mean_centering=True):
    """
        Compute the affinity matrix by given kernel type,
        default to calculate Pearson's correlation between all vertex pairs

        :param data: subject's connectivity profile, shape [N * k]
                     N - the size of vertices (voxel)
                     k - the size of activation conditions
        :param cond: specify the subset of activation conditions to evaluation
                    (e.g condition column [1,2,3,4]),
                     if not given, default to use all conditions
        :param mean_centering: boolean value to determine whether the given subject data
                               should be mean centered

        :return: cov - the covariance matrix of current subject data. shape [N * N]
                 var - the variance matrix of current subject data. shape [N * N]
    """
    if mean_centering:
        data = data - pt.mean(data, dim=1, keepdim=True)  # mean centering

    # specify the condition index used to compute correlation, otherwise use all conditions
    if cond != 'all':
        data = data[:, cond]
    elif cond == 'all':
        data = data
    else:
        raise TypeError("Invalid condition type input! cond must be either 'all'"
                        " or the column indices of expected task conditions")

    k = data.shape[1]
    cov = pt.matmul(data, data.T) / (k - 1)
    # sd = data.std(dim=1).reshape(-1, 1)  # standard deviation
    sd = pt.sqrt(pt.sum(data ** 2, dim=1, keepdim=True) / (k - 1))
    var = pt.matmul(sd, sd.T)

    return cov, var

def compute_dist(coord, resolution=2):
    """
    calculate the distance matrix between each of the voxel pairs by given mask file

    :param coord: the ndarray of all N voxels coordinates x,y,z. Shape N * 3
    :param resolution: the resolution of .nii file. Default 2*2*2 mm

    :return: a distance matrix of N * N, where N represents the number of masked voxels
    """
    if type(coord) is np.ndarray:
        coord = pt.tensor(coord, dtype=pt.get_default_dtype())

    num_points = coord.shape[0]
    D = pt.zeros((num_points, num_points))
    for i in range(3):
        D = D + (coord[:, i].reshape(-1, 1) - coord[:, i]) ** 2
    return pt.sqrt(D) * resolution



def make_eval_info(M, train_info=None, tdata='MDTB',
                   group_type='Models_03', indivtrain_ind='half',
                   indivtrain_values=1, indivtest_values=2,
                   test_kappa=None):
    """ Collects all the information from the model and the
        training and test data sets into a single dictionary

    Args:
        M (fm.Model): model object
        train_info (dict): training data information
        test_info (dict): test data information

    Returns:
        minfo (dict): model information
    """
    if train_info is None:
        train_info = pd.Series()
    minfo = train_info
    minfo['test_data'] = tdata
    minfo['group_type'] = group_type
    minfo['indivtrain_ind'] = indivtrain_ind
    minfo['indivtrain_val'] = indivtrain_values
    minfo['indivtest_val'] = indivtest_values
    minfo['indiv_test_kappa'] = test_kappa
    return minfo


def compute_DCBC(maxDist=35, binWidth=1, parcellation=np.empty([]),
                 func=None, dist=None, weighting=True):
    """
    The main entry of DCBC calculation for volume space
    :param hems:        Hemisphere to test. 'L' - left hemisphere; 'R' - right hemisphere; 'all' - both hemispheres
    :param maxDist:     The maximum distance for vertices pairs
    :param binWidth:    The spatial binning width in mm, default 1 mm
    :param parcellation:
    :param dist_file:   The path of distance metric of vertices pairs, for example Dijkstra's distance, GOD distance
                        Euclidean distance. Dijkstra's distance as default
    :param weighting:   Boolean value. True - add weighting scheme to DCBC (default)
                                       False - no weighting scheme to DCBC
    """
    numBins = int(np.floor(maxDist / binWidth))
    cov, var = compute_var_cov(func)
    # cor = np.corrcoef(func)
    if not dist.is_sparse:
        dist = dist.to_sparse()

    row = dist._indices()[0]
    col = dist._indices()[1]
    distance = dist._values()
    # row, col, distance = sp.sparse.find(dist)

    # making parcellation matrix without medial wall and nan value
    par = parcellation
    num_within, num_between, corr_within, corr_between = [], [], [], []
    for i in range(numBins):
        inBin = pt.where((distance > i * binWidth) &
                         (distance <= (i + 1) * binWidth))[0]

        # lookup the row/col index of within and between vertices
        within = pt.where((par[row[inBin]] == par[col[inBin]]) == True)[0]
        between = pt.where((par[row[inBin]] == par[col[inBin]]) == False)[0]

        # retrieve and append the number of vertices for within/between in current bin
        num_within.append(
            pt.tensor(within.numel(), dtype=pt.get_default_dtype()))
        num_between.append(
            pt.tensor(between.numel(), dtype=pt.get_default_dtype()))

        # Compute and append averaged within- and between-parcel correlations in current bin
        this_corr_within = pt.nanmean(cov[row[inBin[within]], col[inBin[within]]]) \
            / pt.nanmean(var[row[inBin[within]], col[inBin[within]]])
        this_corr_between = pt.nanmean(cov[row[inBin[between]], col[inBin[between]]]) \
            / pt.nanmean(var[row[inBin[between]], col[inBin[between]]])

        corr_within.append(this_corr_within)
        corr_between.append(this_corr_between)

        del inBin

    if weighting:
        weight = 1 / (1 / pt.stack(num_within) + 1 / pt.stack(num_between))
        weight = weight / pt.sum(weight)
        DCBC = pt.nansum(pt.multiply(
            (pt.stack(corr_within) - pt.stack(corr_between)), weight))
    else:
        DCBC = pt.nansum(pt.stack(corr_within) - pt.stack(corr_between))
        weight = pt.nan

    D = {
        "binWidth": binWidth,
        "maxDist": maxDist,
        "num_within": num_within,
        "num_between": num_between,
        "corr_within": corr_within,
        "corr_between": corr_between,
        "weight": weight,
        "DCBC": DCBC
    }

    return D

def calc_test_dcbc(parcels, testdata, dist, max_dist=35, bin_width=1,
                   trim_nan=False, return_wb_corr=False, verbose=True):
    """DCBC: evaluate the resultant parcellation using DCBC
    Args:
        parcels (np.ndarray): the input parcellation:
            either group parcellation (1-dimensional: P)
            individual parcellation (num_subj x P )
        dist (pt.Tensor): the distance metric
        testdata (np.ndarray): the functional test dataset,
                                shape (num_sub, N, P)
        trim_nan (boolean): if true, make the nan voxel label will be
                            removed from DCBC calculation. Otherwise,
                            we treat nan voxels are in the same parcel
                            which is label 0 by default.
        masks (dict) : Masks for spatially separatate DCBC calculation
    Returns:
        dcbc_values (np.ndarray): the DCBC values of subjects
    """
    if trim_nan:
        # mask the nan voxel pairs distance to nan for non-sparse tensor
        dist[pt.where(pt.isnan(parcels))[0], :] = 0
        dist[:, pt.where(pt.isnan(parcels))[0]] = 0

    dcbc_values, D_all = [], []
    

    for sub in range(testdata.shape[0]):
        print(f'Subject {sub}', end=':')
        tic = time.perf_counter()
        if parcels.ndim == 1:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                            parcellation=parcels,
                            dist=dist, func=testdata[sub].T)
        else:
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                            parcellation=parcels[sub],
                            dist=dist, func=testdata[sub].T)
        dcbc_values.append(D['DCBC'])
        D_all.append(D)
        toc = time.perf_counter()
        print(f"{toc-tic:0.4f}s")

    if return_wb_corr:
        return pt.stack(dcbc_values), D_all
    else:
        return pt.stack(dcbc_values)
    
def eval_parcel_DCBC(U_group, U_indiv, U_data, t_data, dist, minfo, masks={}, out_file=None):
    """Evaluate the individual parcellation using DCBC
    Args:
        U_group (np.ndarray): the group parcellation
        U_indiv (np.ndarray): the individual parcellation
        U_data (np.ndarray): the individual parcellation
        t_data (np.ndarray): the functional test dataset,
                            shape (num_sub, N, P)
        dist (pt.Tensor): the distance metric
        minfo (dict): the model information
        masks (dict) : Masks for spatially separatate DCBC calculation"""
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
    Pdata = pt.argmax(U_data, dim=1) + 1
    
    if masks is None:
        masks['whole'] = np.ones(t_data.shape[-1], dtype=bool)
    eval = []
    for mask_name in masks.keys():
        mask = masks[mask_name]
        print(f'Calculating DCBC for {mask_name}')
        dcbc_group = calc_test_dcbc(Pgroup[mask], t_data[:,:,mask], dist[mask][:,mask])
        dcbc_indiv = calc_test_dcbc(Pindiv[:,mask], t_data[:,:,mask], dist[mask][:,mask])
        dcbc_data = calc_test_dcbc(Pdata[:,mask], t_data[:,:,mask], dist[mask][:,mask])

        # ------------------------------------------
        # Collect the information from the evaluation
        # in a data frame

        ev_df = pd.DataFrame({
                            'test_data': [minfo.test_data] * num_subj,
                            'group_type': [minfo.group_type] * num_subj,
                            'indivtrain_ind': [minfo.indivtrain_ind] * num_subj,
                            'indivtrain_val': [minfo.indivtrain_val] * num_subj,
                            'indivtest_val': [minfo.indivtest_val] * num_subj,
                            'subj_num': np.arange(num_subj),
                            'mask': np.repeat(mask_name, num_subj),
                            })
        # Add all the evaluations to the data frame
        ev_df['dcbc_group'] = dcbc_group.cpu()
        ev_df['dcbc_indiv'] = dcbc_indiv.cpu()
        ev_df['dcbc_data'] = dcbc_data.cpu()
        eval.append(ev_df)
    
    # Concatenate the evaluation results
    eval = pd.concat(eval, axis=0)
    eval.to_csv(out_file, index=False, sep='\t')
    return eval

def evaluate_mdtb_rest():
    data_dir = paths.set_fusion_dir()
    atlas_dir = paths.set_atlas_dir()

    space, _ = am.get_atlas('MNISymC2')
    if space.name == 'MNISymC2':
        space_folder = 'tpl-MNI152NLin2009cSymC'

    
    # Load group prior and construct arrangement model
    sym_type = 'sym'
    atlas_dir = paths.set_atlas_dir()
    atlas = 'NettekovenSym32'
    model_name = f'/{space_folder}/atl-{atlas}_space-{space_folder.split("tpl-")[1]}_probseg.nii'
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/{space_folder}/atl-{atlas}.lut')
    U = space.read_data(atlas_dir + model_name)
    U = U.T
    ar_model = ar.build_arrangement_model(U, prior_type='prob', sym_type=sym_type, atlas=space)

    # -- Get connectivity fingerprints of regions --
    # Load HCP data
    hcp, hcp_info, hcp_tds = ds.get_dataset(data_dir, 'HCP', atlas=space.name, type='Net67Run')

    tdata_hcp, cond_v_hcp, part_v_hcp, sub_ind = fm.prep_datasets(hcp, hcp_info.sess,
                                                    hcp_info['net_id'].values,
                                                    hcp_info['half'].values,
                                                    join_sess=False,
                                                    join_sess_part=False)
    
    _, _, model_hcp = fm.get_indiv_parcellation(ar_model, space, tdata_hcp, cond_v_hcp, part_v_hcp, sub_ind, Vs=None, sym_type = sym_type)

    # -- Get MDTB subject parcellations --
    # Get subjects
    T = pd.read_csv(f'{data_dir}/MDTB/participants.tsv', delimiter='\t')
    participants = T.participant_id
    rest_subjects = participants[T['ses-rest'] == 1].tolist()

    # Load MDTB rest data
    data67, info67, tds = ds.get_dataset(data_dir, 'MDTB', atlas=space.name, sess='ses-rest', type='Net67Run', subj=rest_subjects)
    tdata67, cond_v67, part_v67, sub_ind_v67 = fm.prep_datasets(data67, info67.sess,
                                                    info67['net_id'].values,
                                                    info67['half'].values,
                                                    join_sess=False,
                                                    join_sess_part=False)

    # Parcellate using connectivity fingerprints learned on MDTB rest data
    indiv_par_rest_net67, _, model_rest67 = fm.get_indiv_parcellation(ar_model, space, tdata67, cond_v67, part_v67, sub_ind_v67, Vs=None, sym_type = sym_type)

    # Parcellate using connectivity fingerprints from HCP data
    indiv_par_rest_net67_vs, _, model_rest67_vs = fm.get_indiv_parcellation(ar_model, space, tdata67, cond_v67, part_v67, sub_ind_v67, Vs=[model_hcp.emissions[0].V], sym_type = sym_type)


        
    # Load MDTB task data
    data, info, tds = ds.get_dataset(data_dir, 'MDTB', atlas=space.name, subj=rest_subjects)
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                      info['cond_num_uni'].values,
                                                      info['half'].values,
                                                      join_sess=False,
                                                      join_sess_part=False)
    # Parcellate using task session 1
    indiv_par_task, _, model_task = fm.get_indiv_parcellation(ar_model, space, [tdata[0]],[cond_v[0]], [part_v[0]],[sub_ind[0]], sym_type = sym_type)

    # -- Evaluate the individual parcellations --
    # Load MDTB session sc2 to evaluate
    dist = compute_dist(space.world.T, resolution=1)
    # Make evaluation info
    model_name = f'/Models_03/NettekovenSym32_space-MNISymC2'
    model_dir = paths.set_model_dir()
    _, minfo = ar.load_group_parcellation(model_dir + model_name, device=DEVICE)
    eval_info = make_eval_info(model_rest67_vs, minfo, tdata='MDTB', group_type='Models_03',
                               indivtrain_ind='half', indivtrain_values=1,
                               indivtest_values=2, test_kappa=None)
    # Step 2.3: Do DCBC evaluation on the second half data
    ev_task = eval_parcel_DCBC(U, indiv_par_task, tdata[1], dist, eval_info,
                     out_file='eval_dcbc_indiv_parcellations.tsv')
    ev_task['type'] = ['task']*ev_task.shape[0]
    ev_rest = eval_parcel_DCBC(U, indiv_par_rest_net67, tdata[1], dist, eval_info,
                     out_file='eval_dcbc_indiv_parcellations_rest.tsv')
    ev_rest['type'] = ['rest']*ev_task.shape[0]
    ev_rest_vs = eval_parcel_DCBC(U, indiv_par_rest_net67_vs, tdata[1], dist, eval_info,
                        out_file='eval_dcbc_indiv_parcellations_rest_vs.tsv')
    ev_rest_vs['type'] = ['rest_Vs']*ev_task.shape[0]
    
    # Concate the results
    ev_df = pd.concat([ev_task, ev_rest, ev_rest_vs], axis=0)
    ev_df.to_csv('eval_dcbc_indiv_parcellations_all.tsv', sep='\t')

def evaluate_all_model_types(dtype='Net67Run'):
    
    mdtb_Vmodels = {f'{mtype}_{vtype}': pickle.load(open(f'{results_dir}/fitted_emissions/fitted_rest-mdtb_dtype-{dtype}_mtype-{mtype}_vtype-{vtype}.pkl', 'rb')) for mtype in vmodel_types.keys() for vtype in v_types.keys()}
    # Get Prior
    atlas = 'NettekovenSym32'
    model_name = f'/{space_folder}/atl-{atlas}_space-{space_folder.split("tpl-")[1]}_probseg.nii'
    U = space.read_data(atlas_dir + model_name)
    U = U.T
    
    # get domain masks
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/{space_folder}/atl-{atlas}.lut') 
    masks = domain_masks(atlas, U, labels)

    # Get Uhat
    Uhats = {}
    Uhats_data = {}
    for model_type in mdtb_Vmodels.keys():
        selected_vmodel = mdtb_Vmodels[model_type]
        emloglik_c = [e.Estep() for e in selected_vmodel.emissions]
        emloglik_comb = selected_vmodel.collect_evidence(emloglik_c)
        Uhats_data[model_type] = pt.softmax(emloglik_comb, dim=1)# get data only parcellation
        Uhat, ll_A = selected_vmodel.arrange.Estep(emloglik_comb) # get integrated parcellation
        Uhats[model_type] = Uhat

    # Load MDTB task data
    T = pd.read_csv(
    base_dir + '/FunctionalFusion/MDTB/participants.tsv', delimiter='\t')
    mdtb_rest_subjects = T.participant_id[T['ses-rest'] == 1].tolist()
    data, info, tds = ds.get_dataset(data_dir, 'MDTB', atlas=space.name, subj=mdtb_rest_subjects)
    tdata, cond_v, part_v, sub_ind = fm.prep_datasets(data, info.sess,
                                                        info['cond_num_uni'].values,
                                                        info['half'].values,
                                                        join_sess=False,
                                                        join_sess_part=False)
    dist = ev.compute_dist(space.world.T, resolution=1)

    eval_info = make_eval_info(mdtb_Vmodels[list(mdtb_Vmodels.keys())[0]], tdata='MDTB', group_type='Models_03',
                            indivtrain_ind='half', indivtrain_values=1,
                            indivtest_values=2, test_kappa=None)
    
    
    results = [] 
    for mtype in mdtb_Vmodels.keys():
        res = eval_parcel_DCBC(U,
                        Uhats[mtype],
                        Uhats_data[mtype],
                        tdata[1],
                        dist,
                        eval_info,
                        masks=masks,
                        out_file=f'{results_dir}/eval_dcbc_dtype-{dtype}_mtype-{mtype}_regionwise.tsv')
        res['mtype'] = mtype
        results.append(res)
    # Concate the results
    ev_df = pd.concat(results, axis=0)
    ev_df.to_csv(f'{results_dir}/eval_dcbc_dtype-{dtype}_all_modeltypes_regionwise.tsv', sep='\t')


def calculate_kappa_curve(M, edata, num_runs=2,kappas = [0,0.5,1,3,5,8,200], masks=None):
    """
    Calculate the DCBC values for different kappa values
    
    Args:
        M (fm.Model): model object
        edata : evaluation data
        num_runs (int): number of runs
        kappas (list): list of kappa values to test
        masks (dict) : Masks for spatially separatate DCBC calculation
    """
    space, _ = am.get_atlas('MNISymC2')

    # Load the evaluation data:
    max_dist=40
    bin_width=1.5
    T = []

    print('Computing distances...')
    dist = compute_dist(space.world.T, resolution=1)

    if masks is None:
        masks['whole'] = np.ones(edata.shape[-1], dtype=bool)

    for mask_name in masks.keys():
        mask = masks[mask_name]
        print(f'Calculating DCBC for {mask_name}')
        for i,kappa in enumerate(kappas):
            print(f'Running Kappa {kappa}...')
            M.emissions[0].kappa = pt.tensor(kappa)
            U,_ = M.Estep()

            for s in range(U.shape[0]):
                parcel = pt.argmax(U[s],dim=0)+1
                D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                        parcellation=parcel[mask],
                        dist=dist[mask][:,mask],
                        func=edata[s].T[mask])
                d = {'sn':[s],
                    'num_runs':[num_runs],
                    'max_dist':max_dist,
                    'bin_width':bin_width,
                    'kappa':kappa,
                    'DCBC':D['DCBC'].item(),
                    'mask':mask_name}
                T.append(pd.DataFrame(d))
    T = pd.concat(T)
    return T

if __name__ == "__main__":
    # evaluate_mdtb_rest()
    # evaluate_all_model_types(dtype='Fus06Run')
    # Get Prior
    atlas = 'NettekovenSym32'
    model_name = f'/{space_folder}/atl-{atlas}_space-{space_folder.split("tpl-")[1]}_probseg.nii'
    U = space.read_data(atlas_dir + model_name)
    U = U.T
    # get domain masks
    _, cmap, labels = nt.read_lut(f'{atlas_dir}/{space_folder}/atl-{atlas}.lut') 
    masks = domain_masks(atlas, U, labels)

    # Get evaluation data
    print('Loading Evaluation data...')
    test_session = 's2'
    ttdata, ttinfo, _ = ds.get_dataset(data_dir, 'MDTB', atlas=space.name, sess=f'ses-{test_session}', type='CondHalf')
    # Make ttdata a tensor
    ttdata = pt.tensor(ttdata, dtype=pt.get_default_dtype())
    
    models = {(dtype,vtype): pickle.load(open(f'{results_dir}/fitted_emissions/fitted_rest-mdtb_dtype-{dtype}_mtype-general_vtype-{vtype}.pkl', 'rb')) for vtype in v_types.keys() for dtype in dtypes}
    results = []
    for model in models.keys():
        T = calculate_kappa_curve(models[model], edata=ttdata, num_runs=2,kappas = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 3, 5, 8], masks=masks)
        T['mtype'] = model[1]
        T['dtype'] = model[0]
        T.to_csv(f'{results_dir}/evaluation/kappa_curve-fine_test-{test_session}_vtype-{model[1]}_dtype-{model[0]}.tsv', sep='\t', index=False)
        results.append(T)
    results = pd.concat(results)
    results.to_csv(f'{results_dir}/evaluation/kappa_curve-fine_test-{test_session}_all_models.tsv', sep='\t', index=False)
    pass

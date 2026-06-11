#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirically checks different settings for Kappa for the MDTB dataset
"""
import numpy as np
import torch as pt
import pandas as pd
import nibabel as nb
import nitools as nt
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.matrix as mm

import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
from DCBC.DCBC_vol import compute_DCBC, compute_dist
from copy import deepcopy,copy

from global_config import MODEL_DIR, BASE_DIR, DEVICE

def get_arrangement_model():
    ## Step 1: Load the atlas
    atlas, _ = am.get_atlas('MNISymC3')

    ##  Load the group prior from a pre-trained model
    model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'
    fname = MODEL_DIR + model_name
    U, _ = ar.load_group_parcellation(fname, device=DEVICE)
    ar_model = ar.build_arrangement_model(U, prior_type='logpi', atlas=atlas,
                                          sym_type='asym')
    return ar_model, atlas





def fit_V(atlas, ar_model, train_sess='ses-s1'):
        ## Step 4a: Load the individual localizing data / info from Fusion project
    tdata, tinfo, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, sess=train_sess, type='CondRun')
    cond_v = tinfo['cond_num_uni'].to_numpy()
    part_v = tinfo['run'].to_numpy()

    # Initialize emission models
    i=0
    em_model = em.MixVMF(K=ar_model.K, P=atlas.P, X=mm.indicator(cond_v),part_vec=part_v)
    M = fm.FullMultiModel(ar_model, [em_model])
    M.initialize([tdata], subj_ind=[np.arange(24)])

    # Learn the V's on the data
    M, ll, _, U_indiv = M.fit_em(iter=100, tol=0.01,
                                     fit_arrangement=False,
                                     fit_emission=True,
                                     first_evidence=False)
    return M ,tdata, cond_v, part_v

def fit_individual_kappa(train_sess='ses-s1', num_runs=8, kappa=1):
    T = []
    ar_model, atlas = get_arrangement_model()
    M ,tdata, cond_v, part_v = fit_V(atlas, ar_model, train_sess=train_sess)
    M.clear()
    # Make alernative common and subject specific kappa models
    MF=[]
    for m in range(2):
        MF.append(deepcopy(M))
    MF[0].emissions[0].subject_specific_kappa = True
    MF[0].emissions[0].kappa = pt.tensor([5]*24)
    MF[0].emissions[0].set_param_list(['kappa'])
    MF[1].emissions[0].set_param_list(['kappa'])

    # Change the amount of data used for individual parcellation
    for num_runs in np.arange(2,17):
        for i in range(30):
            print(f'Running {num_runs} runs, iter {i}...')
            runs = np.random.choice(np.arange(16),size=num_runs,replace=False)
            indx = np.array([v in runs for v in part_v])
            for m in range(2):
                MF[m].emissions[0].part_vec = pt.tensor(part_v[indx],dtype=pt.float32)
                MF[m].emissions[0].X = pt.tensor(mm.indicator(cond_v[indx]),dtype=pt.float32)
                MF[m].initialize([tdata[:,indx,:]])
                MF[m], ll, _, U_indiv = MF[m].fit_em(iter=100, tol=0.01,
                                        fit_arrangement=False,
                                        fit_emission=True,
                                        first_evidence=False)
                for s in range(24):
                    d = {'sn':[s],
                        'iter':[i],
                        'num_runs':[num_runs],
                        'kappa_indiv':MF[0].emissions[0].kappa[s].numpy(),
                        'kappa_group':MF[1].emissions[0].kappa.numpy()}
                    T.append(pd.DataFrame(d))
    T = pd.concat(T)
    return T

def calculate_kappa_curve(train_sess='ses-s1', num_runs=8,kappas = [0,0.5,1,3,5,8,200]):
    ar_model, atlas = get_arrangement_model()
    M ,tdata, cond_v, part_v = fit_V(atlas, ar_model, train_sess=train_sess)

    # REduce the amount of data used for individual parcellation
    indx = part_v<=num_runs
    M.emissions[0].part_vec = pt.tensor(part_v[indx],dtype=pt.float32)
    M.emissions[0].X = pt.tensor(mm.indicator(cond_v[indx]),dtype=pt.float32)
    M.initialize([tdata[:,indx,:]])

    # Load the evaluation data:
    max_dist=40
    bin_width=1.5
    T = []
    print('Loading Evaluation data...')
    ttdata, ttinfo, _ = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, sess='ses-s2', type='CondHalf')
    print('Computing distances...')
    dist = compute_dist(atlas.world.T, resolution=1)
    for i,kappa in enumerate(kappas):
        print(f'Running Kappa {kappa}...')
        M.emissions[0].kappa = pt.tensor(kappa)
        U,_ = M.Estep()

        for s in range(U.shape[0]):
            parcel = pt.argmax(U[s],dim=0)+1
            D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                    parcellation=parcel,
                    dist=dist,
                    func=ttdata[s].T)
            d = {'sn':[s],
                'train':[train_sess],
                'num_runs':[num_runs],
                'max_dist':max_dist,
                'bin_width':bin_width,
                'kappa':kappa,
                'DCBC':D['DCBC']}
            T.append(pd.DataFrame(d))
    T = pd.concat(T)
    return T

if __name__ == "__main__":
    T = calculate_kappa_curve(train_sess='ses-s1', num_runs=2,kappas = [0,0.2,0.5,0.8,1,2,3,4,5,6,8,10,15,20,200])
    T.to_csv(f'results/kappa_individ_runs-02_train-s1.tsv',sep='\t',index=False)
    # T = fit_individual_kappa()
    # T.to_csv(f'results/kappa_individ_estimation_s1.tsv',sep='\t',index=False)
    pass


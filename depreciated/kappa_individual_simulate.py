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
import HierarchBayesParcel.spatial as sp
from DCBC.DCBC_vol import compute_DCBC, compute_dist
from copy import deepcopy,copy


def get_true_model(kappa=[10,10,10,10,10,10],
            K=5,
            n_part=10,
            n_cond=15,
            width=30,
            height=30,
            theta_mu = 60):
        # Arrangement model
    n_subj = len(kappa)
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.ArrangeIndependent(K=K, P=grid.P)
    arrangeT.logpi = grid.random_smooth_pi(K=K, theta_mu=theta_mu,centroids=[0 , 29, 434, 870, 899])

    # Emission model
    part_vec = np.kron(np.arange(n_part),np.ones(n_cond,)).astype(int)
    X = np.kron(np.ones((n_part,1)),np.eye(n_cond))
    emissionT = em.MixVMF(K=K, P=grid.P,
                          X=X,part_vec=part_vec,
                          num_subj=n_subj,
                          parcel_specific_kappa=False,
                          subject_specific_kappa=True)
    emissionT.kappa = pt.tensor(kappa)
    return arrangeT,emissionT,grid,n_subj

def get_fit_models(arrangeT,emissionT,n_subj):
    MF=[]
    for m in range(2):
        MF.append(fm.FullMultiModel(arrangeT, [deepcopy(emissionT)]))
    MF[0].emissions[0].subject_specific_kappa = True
    MF[0].emissions[0].kappa = pt.tensor([5]*n_subj)
    MF[0].emissions[0].set_param_list(['kappa'])
    MF[1].emissions[0].subject_specific_kappa = False
    MF[1].emissions[0].kappa = pt.tensor([5])
    MF[1].emissions[0].set_param_list(['kappa'])
    return MF

def fit_models(MF,Y_train,kappa_true):
    # Change the amount of data used for individual parcellation
    for m in range(2):
        MF[m].initialize([Y_train])
        MF[m], ll, _, _ = MF[m].fit_em(iter=100, tol=0.01,
                                fit_arrangement=False,
                                fit_emission=True,
                                first_evidence=False)
    T=[]
    for s in range(Y_train.shape[0]):
        d = {'sn':[s],
            'num_runs':[MF[0].emissions[0].num_part[0,0,0].item()],
            'num_cond':[MF[0].emissions[0].M],
            'kappa_true':kappa_true[s],
            'kappa_indiv':MF[0].emissions[0].kappa[s].numpy(),
            'kappa_group':MF[1].emissions[0].kappa.numpy()}
        T.append(pd.DataFrame(d))
    T = pd.concat(T)
    return T


def make_gaussian_data(U,em,grid):
    num_subj = U.shape[0]
    Y = pt.zeros((num_subj, em.N, em.P))
    for s in range(num_subj):
        Y[s,:,:] = pt.normal(0,pt.sqrt(1/em.kappa[s]),(em.N,em.P))
        Y[s, :, :] = Y[s, :, :] + pt.matmul(em.X, em.V[:, U[s, :].long()])
    return Y


def do_sim1():
    T = []
    kappa=[0,1,2,3,4,5,6,7,8,9,10]
    for i in [2,3,4,5,6,7,8,9,10]:
        print(f'Running {i}')
        arrangeT,emissionT,grid,n_subj = get_true_model(kappa=kappa,n_part=i)
        U_true = arrangeT.sample(num_subj=n_subj)
        Y_train = emissionT.sample(U_true)
        MF = get_fit_models(arrangeT,emissionT,n_subj)
        T.append(fit_models(MF,Y_train,kappa))
    T = pd.concat(T)
    return T

def do_sim2():
    T = []
    kappa=[1,5,10,15,20]
    for i in [2,3,4,5,6,7,8,9,10]:
        print(f'Running {i}')
        arrangeT,emissionT,grid,n_subj = get_true_model(kappa=kappa,n_part=i,n_cond=5)
        U_true = arrangeT.sample(num_subj=n_subj)
        Y_train = make_gaussian_data(U_true,emissionT,grid)
        MF = get_fit_models(arrangeT,emissionT,n_subj)
        T.append(fit_models(MF,Y_train,kappa))
    T = pd.concat(T)
    return T





if __name__ == "__main__":
    # T = calculate_kappa_curve(train_sess='ses-s1', num_runs=16,kappas = [0,0.2,0.5,0.8,1,2,3,4,5,6,8,10,15,20,200])
    T=do_sim2()
    T.to_csv(f'results/kappa_individ_simulation2.tsv',sep='\t',index=False)

    pass








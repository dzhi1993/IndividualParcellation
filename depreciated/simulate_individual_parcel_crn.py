# Simulates individual parcellation and VMF model estimation 


import numpy as np
import torch as pt
import matplotlib.pyplot as plt

import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.spatial as sp


def do_sim_basic(theta_mu = 80, 
                         kappa=10,
                         K =5, 
                         n_cond=15, 
                         n_part=3,
                         n_subj=6,
                         subject_specific_kappa=False,
                         parcel_specific_kappa=False):
    width =30 
    height = 30 
    # Step 1: Create the true arrangement
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.ArrangeIndependent(K=K, P=grid.P)    
    arrangeT.logpi = grid.random_smooth_pi(K=K, theta_mu=theta_mu,centroids=[0 , 29, 434, 870, 899])
    #plt.figure(figsize=(10, 4))
    # grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, K])
    
    # Step 2: Create the true emission
    part_vec = np.kron(np.arange(n_part),np.ones(n_cond,)).astype(int)
    X = np.kron(np.ones((n_part,1)),np.eye(n_cond))
    emissionT = em.MixVMF(K=K, P=grid.P,
                          X=X,part_vec=part_vec,
                          num_subj=n_subj,
                          parcel_specific_kappa=parcel_specific_kappa,
                          subject_specific_kappa=subject_specific_kappa)
    emissionT.kappa = pt.tensor(kappa)

    # Make a full model for training
    emissionF = em.MixVMF(K=K, P=grid.P,
                            X=X,part_vec=part_vec,
                            num_subj=n_subj,
                            parcel_specific_kappa=parcel_specific_kappa,
                            subject_specific_kappa=subject_specific_kappa)

    # Step 3: Sample subjects and data 
    U_true = arrangeT.sample(num_subj=n_subj)
    # plt.figure(figsize=(20, 2))
    # grid.plot_maps(U, cmap='tab20', vmax=19, grid=[1, int(n_sub)])
    Y_train = emissionT.sample(U_true)

    M = fm.FullMultiModel(arrangeT, [emissionF])
    M.initialize([Y_train])

    M,ll,theta,U_hat = M.fit_em(iter=100, seperate_ll=False, fit_emission=True,
               fit_arrangement=False, first_evidence=False)
    
    plt.figure(figsize=(12, 7))
    plt.subplot(2,3,1)
    plt.imshow(emissionT.V,vmin=-0.2,vmax=0.2)
    plt.title('True V')
    plt.subplot(2,3,2)
    plt.imshow(M.emissions[0].V,vmin=-0.2,vmax=0.2)
    plt.title('Estimated V')
    plt.subplot(2,3,3)
    ind = M.get_param_indices('emissions.0.V')
    plt.plot(theta[:,ind])
    plt.title('V')

    plt.subplot(2,3,4)
    ind = M.get_param_indices('emissions.0.kappa')
    plt.plot(theta[:,ind])
    plt.title('kappa')

    plt.subplot(2,3,5)
    P = pt.argmax(arrangeT.marginal_prob(),dim=0)
    plt.imshow(P.reshape(grid.dim),cmap='tab20',vmax=19)
    plt.title('True')

    plt.subplot(2,3,6)
    U_hat_w = pt.argmax(U_hat.mean(dim=0),dim=0)
    plt.imshow(U_hat_w.reshape(grid.dim),cmap='tab20',vmax=19)
    plt.title('avrg Uhat')
    pass 


def do_sim_weirdsubj(theta_mu = 80, 
                         kappa=[10,10,10],
                         K = 5, 
                         n_cond=15, 
                         n_part=3,
                         n_subj=3,
                         subject_specific_kappa=True):
    width =30 
    height = 30 
    # Step 1: Create the true arrangement
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.ArrangeIndependent(K=K, P=grid.P)    
    arrangeT.logpi = grid.random_smooth_pi(K=K, theta_mu=theta_mu,centroids=[0 , 29, 434, 870, 899])
    #plt.figure(figsize=(10, 4))
    # grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, K])
    
    # Step 2: Create the true emission
    part_vec = np.kron(np.arange(n_part),np.ones(n_cond,)).astype(int)
    X = np.kron(np.ones((n_part,1)),np.eye(n_cond))
    emissionT = em.MixVMF(K=K, P=grid.P,
                          X=X,part_vec=part_vec,
                          num_subj=n_subj,
                          subject_specific_kappa=True)
    emissionT.kappa = pt.tensor(kappa)

    # Step 3: Sample subjects and data 
    U_true = arrangeT.sample(num_subj=n_subj)
    # plt.figure(figsize=(20, 2))
    # grid.plot_maps(U, cmap='tab20', vmax=19, grid=[1, int(n_sub)])
    Y_train = emissionT.sample(U_true)

    # Now insert a single subject that has a different V - and similar for all parcels
    # This outlying subject is 0 
    emissionOutlier = em.MixVMF(K=K, P=grid.P,
                          X=X,part_vec=part_vec,
                          num_subj=1)
    emissionOutlier.kappa = pt.tensor(kappa[0])
    v = pt.randn(n_cond, 1)
    v= v / pt.sqrt(pt.sum(v** 2, dim=0))
    emissionOutlier.V = v.repeat((1,K))
    Y_train[0] = emissionOutlier.sample(U_true[0:1])
    
    # Make a full model for training
    emissionF = em.MixVMF(K=K, P=grid.P,
                            X=X,part_vec=part_vec,
                            num_subj=n_subj,
                            subject_specific_kappa=subject_specific_kappa)

    M = fm.FullMultiModel(arrangeT, [emissionF])
    M.initialize([Y_train])

    M,ll,theta,U_hat = M.fit_em(iter=100, seperate_ll=False, fit_emission=True,
               fit_arrangement=False, first_evidence=False)
    
    plt.figure(figsize=(12, 7))
    plt.subplot(2,3,1)
    plt.imshow(emissionT.V,vmin=-0.2,vmax=0.2)
    plt.title('True V ')
    plt.subplot(2,3,2)
    plt.imshow(emissionOutlier.V,vmin=-0.2,vmax=0.2)
    plt.title('OutlierV')
    plt.subplot(2,3,3)
    plt.imshow(M.emissions[0].V,vmin=-0.2,vmax=0.2)
    plt.title('Estimated V')

    plt.subplot(2,3,4)
    plt.imshow(U_hat.sum(dim=2))
    plt.xlabel('component')
    plt.ylabel('subject')

    badDim = pt.argmax(U_hat[0].sum(dim=1))
    print(f'Similarity to outlier vector:{v.squeeze() @ emissionF.V[:,badDim]:2f}')
    print(f'Similarity to true vevtor:{emissionT.V[:,badDim]@emissionF.V[:,badDim]:.2f}')
    
    pass 


if __name__=="__main__":
    # Simple uniform kappa: 
    # do_sim_basic(kappa=10)    

    # Parcel-specific kappa:
    # do_sim_basic(kappa=[10,80,80,80,20],parcel_specific_kappa=True)    
    
    # Subjects-specific kappa:
    # do_sim_basic(kappa=[8,20,10,20,5,10],subject_specific_kappa=True)    
    do_sim_weirdsubj(kappa=[50,20,10],subject_specific_kappa=True,n_subj=3)
    pass 
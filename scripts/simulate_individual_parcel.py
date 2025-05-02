# Simulates individual parcellation and VMF model estimation 


import numpy as np
import torch as pt
import matplotlib.pyplot as plt

import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.spatial as sp
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.spatial as sp
import IndividualParcellation.simulate as sim


def do_sim_basic(theta_mu = 80, 
                         kappa=10,
                         K =5, 
                         n_cond=15, 
                         n_part=3,
                         n_subj=6,
                         subject_specific_kappa=False,
                         parcel_specific_kappa=False):
    """Simulate a basic model with uniform kappa and no subject or parcel specific kappa.
    Args:
        theta_mu: Dispersion of the probabilitic map: Larger vaklues lead to more uncertain maps;
        kappa: Concentration parameter of the VMF distribution: Larger values less noisy artifical data
        K: Number of parcels
        n_cond: Number of conditions
        n_part: Number of partitions (runs)
        n_subj: Number of subjects
        subject_specific_kappa: If True, each subject has a different kappa
        parcel_specific_kappa: If True, each parcel has a different kappa
    """
    width =30 
    height = 30 
    # Step 1: Create the true arrangement
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.ArrangeIndependent(K=K, P=grid.P)    
    arrangeT.logpi = grid.random_smooth_pi(K=K, theta_mu=theta_mu,centroids=[0 , 29, 434, 870, 899])
    # Plot the group map optionally
    plt.figure(figsize=(10, 4))
    grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, K])
    
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
                         subject_specific_kappa=True,
                         subjects_equal_weight=False):
    """Simulated a basic dataset for n_subj, where the first subject has a different V (same v for all parcels)
    
    Args:
        theta_mu: Dispersion of the probabilitic map: Larger vaklues lead to more uncertain maps;
        kappa: Concentration parameter of the VMF distribution: Larger values less noisy artifical data
        K: Number of parcels
        n_cond: Number of conditions
        n_part: Number of partitions (runs)
        n_subj: Number of subjects
        subject_specific_kappa: If True, each subject will be estimated with a different kappa
        subjects_equal_weight: For V estimation: False (default): average all voxels in region across subject 
                        True: Average and normalize in each subject fristm then average across subjects
    """


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
                            subject_specific_kappa=subject_specific_kappa,
                            subjects_equal_weight=subjects_equal_weight)

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


def run_sim_estV():
    # ----- Simulation parameters -----
    K=5
    n_subj=6
    simulations = 5
    N = 10
    kappa = [0.2, 3]
    plot = False

    # N.B.:Parcel specific kappa is not implemented for subjects_equal_weight and Subject + Regions specific kappa is not implemented yet
    model_types = {                    
                'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'general_eq':            {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'subject_specific_eq':   {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
    }
    

    # ----- Run estimation -----
    data_types = {'general_low': {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': kappa[0]},
                    'general_high': {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': kappa[1]},
                    'subject_specific_one': {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': [kappa[0]] + [kappa[1]]*int(n_subj-1)},
                    'subject_specific_half': {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': [kappa[0]]*int(n_subj/2) + [kappa[1]]*int(n_subj/2)},
    }
    results = sim.run_simulation(data_types,
                                    model_types, 
                                    simulations=simulations, 
                                    K=K, 
                                    n_subj=n_subj, 
                                    plot=plot,
                                    estimate='V')
            
    filename = f"simVest_K-{K}_nsub-{n_subj}_kappa-{kappa[0]}-{kappa[1]}"
    results.to_csv(f"./simulations/{filename}.tsv", sep='\t')
    print(f'Evaluation saved to simulations/{filename}.tsv')



if __name__=="__main__":
    # Simple uniform kappa: 
    # do_sim_basic(kappa=10,theta_mu=10)    

    # Parcel-specific kappa:
    # do_sim_basic(kappa=[10,80,80,80,20],parcel_specific_kappa=True)    
    
    # Subjects-specific kappa:
   # do_sim_basic(kappa=[8,20,10,20,5,10],n_subj=6,subject_specific_kappa=True)    
    run_sim_estV()
    pass 
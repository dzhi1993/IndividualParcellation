import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.spatial as sp
import pandas as pd


def generate_data(data_type,
                theta_mu = 80, 
                K = 5, 
                n_cond=15, 
                n_part=3,
                n_subj=6,
                outlier_subject=False
                ):
    """Generate simulation data for individual regions and functional responses of a number of subjects.
    Args:
        data_type: Dictionary with the following keys:
            - 'subject_specific_kappa' (bool): If True, each subject will be estimated with a different kappa
            - 'parcel_specific_kappa' (bool): If True, each region or subject has a different kappa
            - 'kappa' (float or list): Concentration parameter of the VMF distribution. If float, all regions or subjects have the same kappa. If list, each region or subject has a different kappa
        theta_mu: Dispersion of the probabilistic map: Larger values lead to more uncertain maps;
        K: Number of parcels
        n_cond: Number of conditions
        n_part: Number of partitions (runs)
        n_subj: Number of subjects
        subject_specific_kappa: If True, each subject will be estimated with a different kappa
        parcel_specific_kappa: If True, each region or subject has a different kappa
        outlier_subject: If True, first subject has different V than others (all regions have same V)
    
    Returns:
        U_true: True group map
        Y_train: Simulated data
        emissionT: True emission model
        arrangeT: True arrangement model
        grid: Spatial grid
        X: Design matrix
        part_vec: Partition vector
        V_outlier: Outlier subject's V if outlier_subject is True, None otherwise
    """
    # create a grid with approximate dimensions of cerebellar MNISymC2 template
    width = 135
    height = 135 
    
    # Step 1: Create the true group map
    grid = sp.SpatialGrid(width=width, height=height)
    arrangeT = ar.ArrangeIndependent(K=K, P=grid.P)    
    if K==5:
        centroids = [0 , 29, 434, 870, 899]
    else:
        rand = np.random.RandomState(42)
        centroids = rand.randint(0, width*height, K)
    arrangeT.logpi = grid.random_smooth_pi(K=K, theta_mu=theta_mu,centroids=centroids)
    if False:
        plt.figure(figsize=(10, 4))
        grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, K])
    
    # Step 2: Create the true emission
    part_vec = np.kron(np.arange(n_part),np.ones(n_cond,)).astype(int)
    X = np.kron(np.ones((n_part,1)),np.eye(n_cond))
    emissionT = em.MixVMF(K=K, P=grid.P,
                          X=X,part_vec=part_vec,
                          num_subj=n_subj,
                          parcel_specific_kappa=data_type['parcel_specific_kappa'],
                          subject_specific_kappa=data_type['subject_specific_kappa'])
    emissionT.kappa = pt.tensor(data_type['kappa'])

    # Step 3: Sample subjects and data 
    U_true = arrangeT.sample(num_subj=n_subj)
    Y_train = emissionT.sample(U_true)

    # Step 3b: If outlier subject is requested, replace the data of the first subject with subject that has a different V (similar V for all parcels)
    V_outlier = None
    if outlier_subject:
        emissionOutlier = em.MixVMF(K=K, P=grid.P,
                            X=X,part_vec=part_vec,
                            num_subj=1)
        if isinstance(data_type['kappa'],list):
            subject_kappa = data_type['kappa'][0]
        else:
            subject_kappa = data_type['kappa']
        emissionOutlier.kappa = pt.tensor(subject_kappa)
        V_outlier = pt.randn(n_cond, 1)
        V_outlier= V_outlier / pt.sqrt(pt.sum(V_outlier** 2, dim=0))
        emissionOutlier.V = V_outlier.repeat((1,K))
        Y_train[0] = emissionOutlier.sample(U_true[0:1])
        V_outlier = V_outlier.squeeze()
    
    return U_true, Y_train, emissionT, arrangeT, grid, X, part_vec, V_outlier


def estimate_Vs(model_type, arrangeT, K, Y_train, X, part_vec):
    """Simulate estimating Vs with a specific model type.
    Args:
        model_type: Dictionary with the following keys:
            - 'subject_specific_kappa' (bool): If True, each subject will be estimated with a different kappa
            - 'parcel_specific_kappa' (bool): If True, each region or subject has a different kappa
            - 'subject_equal_weight' (bool): For V estimation: If True, averages all voxels in within subject first, then average across subjects
        arrangeT: True arrangement
        K: Number of parcels
        Y_train: Data of subjects
        part_vec: Partition vector
        X: Design matrix of dimension n_part*n_cond x n_cond
    """
    
    # Make a full model for training
    emissionF = em.MixVMF(K=K, P=Y_train.shape[2],
                            X=X,part_vec=part_vec,
                            num_subj=Y_train.shape[0],
                            parcel_specific_kappa=model_type['parcel_specific_kappa'],
                            subject_specific_kappa=model_type['subject_specific_kappa'], 
                            subjects_equal_weight=model_type['subjects_equal_weight'])
    M = fm.FullMultiModel(arrangeT, [emissionF])
    M.initialize([Y_train])

    M,ll,theta,U_hat = M.fit_em(iter=100, seperate_ll=False, fit_emission=True,
               fit_arrangement=False, first_evidence=False)
    
    return M, theta, U_hat

def estimate_Uhat(model_type, Vs, arrangeT, K, Y_train, X, part_vec, kappa=None):
    """Given the true Vs, estimate the Uhats.
    Args:
        model_type: Dictionary with the following keys
            - 'subject_specific_kappa' (bool): If True, each subject will be estimated with a different kappa
            - 'parcel_specific_kappa' (bool): If True, each region or subject has a different kappa
            - 'subject_equal_weight' (bool): For V estimation: If True, averages all voxels in within subject first, then average across subjects
        Vs: True Vs
        arrangeT: True arrangement
        K: Number of parcels
        Y_train: Data of subjects
        part_vec: Partition vector
        X: Design matrix of dimension n_part*n_cond x n_cond
        kappa: kappas to use for estimation (can be true kappas or estimated kappas from V estimation)
    """
    # Make a full model for training
    emissionF = em.MixVMF(K=K, P=Y_train.shape[2],
                            X=X,part_vec=part_vec,
                            num_subj=Y_train.shape[0],
                            parcel_specific_kappa=model_type['parcel_specific_kappa'],
                            subject_specific_kappa=model_type['subject_specific_kappa'],
                            subjects_equal_weight=model_type['subjects_equal_weight'])
    emissionF.V = Vs
    new_param_list = emissionF.param_list.copy()
    new_param_list.remove('V')

    if kappa is not None:
        if model_type['subject_specific_kappa'] and kappa.dim()==0:
            kappa = [kappa]*Y_train.shape[0]
        elif model_type['parcel_specific_kappa'] and kappa.dim()==0:
            kappa = [kappa]*K
        elif not model_type['parcel_specific_kappa'] and not model_type['subject_specific_kappa'] and kappa.dim()!=0:
            kappa = pt.mean(kappa)
        emissionF.kappa = pt.tensor(kappa)
        new_param_list.remove('kappa')
    
    emissionF.set_param_list(new_param_list)

    M = fm.FullMultiModel(arrangeT, [emissionF])
    M.initialize([Y_train])

    M,ll,theta,U_hat = M.fit_em(iter=100, seperate_ll=False, fit_emission=False,
               fit_arrangement=False, first_evidence=False)
    
    return M, theta, U_hat


def initialize_results(estimate='Uhat'):
    """
    Initializes a dictionary to store the evaluation results.

    Args:
        estimate (str, optional): The type of estimate to evaluate ('V' or 'Uhat'). Defaults to 'Uhat'.

    Returns:
        dict: The initialized results dictionary with the following keys:
            - 'simulation': A list to store the simulation numbers.
            - 'model_type': A list to store the model types.
            - 'data_type': A list to store the data types.
            - 'outlier': A list to store the outlier indicators.
            - 'K': A list to store the number of components in the model.
            - 'kappa_true_lowest': A list to store the lowest true kappa values.
            - 'kappa_true_highest': A list to store the highest true kappa values.
            - 'kappa_true': A list to store the true kappa values.
            - 'kappa_estimated': A list to store the estimated kappa values.
            - 'correct_mean': A list to store the percentage of correct mean assignments.
            - 'correct_outlier': A list to store the percentage of correct outlier assignments.
            - 'correct_mean_normal': A list to store the percentage of correct mean assignments for normal subjects.
            - 'similarity_true' (optional): A list to store the true similarity values (only if 'V' is in estimate).
            - 'similarity_outlier' (optional): A list to store the outlier similarity values (only if 'V' is in estimate).
    """
    results ={
            'simulation': [],
            'model_type': [],
            'data_type': [],
            'outlier': [],
            'K': [],
            'kappa_true_lowest': [],
            'kappa_true_highest': [],
            'kappa_true': [],
            'kappa_estimated': [],
            'correct_mean': [],
            'correct_outlier': [],
            'correct_mean_normal': []
        }
    # Add optional keys for storing V similarity (estimated to true) if 'V' will be estimated
    if 'V' in estimate:
        results['similarity_true'] = []
        results['similarity_outlier'] = []

    return results



def evaluate_simulation(results,
                        simulation_no,
                        estimate,
                        model_type,
                        data_type,
                        outlier,
                        K,
                        M,
                        emissionT,
                        grid,
                        kappa,
                        U_true,
                        Uhat,
                        V_outlier):
    """
    Evaluates a simulation by calculating various metrics and storing the results in a dictionary.

    Args:
        results (dict): A dictionary to store the evaluation results.
        simulation_no (int): The simulation number.
        estimate (str): The type of estimate to evaluate ('V' or 'Uhat').
        model_type (str): The type of model used in the simulation.
        data_type (str): The type of data used in the simulation.
        outlier (bool): Indicates whether the simulation includes an outlier.
        K (int): The number of components in the model.
        M (object): The model object.
        emissionT (object): The emission object.
        grid (object): The grid object.
        kappa (float): The true kappa value.
        U_true (ndarray): The true U matrix.
        Uhat (ndarray): The estimated U matrix.
        V_outlier (ndarray): The outlier V vector.

    Returns:
        results (dict): The updated results dictionary.
    """

    if 'V' in estimate:
        true_sim = pt.mean(pt.sum(emissionT.V * M.emissions[0].V, dim=0))
        results['similarity_true'].append(true_sim.item())
        # store the similarity of the estimated V to the V of the outlier subject
        if outlier:
            badDim = pt.argmax(Uhat[0].sum(dim=1))
            outlier_sim = pt.mean(pt.sum(V_outlier * M.emissions[0].V[:,badDim], dim=0)).item()
        else:
            outlier_sim = np.nan
        results['similarity_outlier'].append(outlier_sim)
    
    # For all estimation types, calculate the percentage of correct Uhat assignments
    match = (U_true == np.argmax(Uhat, axis=1))
    percentage_correct = (pt.sum(match, axis=1)/grid.P*100).numpy()
    results['correct_mean'].append(percentage_correct.mean())
    results['correct_outlier'].append(percentage_correct[0] if outlier else np.nan)
    results['correct_mean_normal'].append(percentage_correct[1:].mean() if outlier else np.nan)
    # For all estimation types, store the simulation parameters
    results['simulation'].append(simulation_no)
    results['model_type'].append(model_type)
    results['data_type'].append(data_type)
    results['outlier'].append(outlier)
    results['K'].append(K)
    results['kappa_true'].append(kappa)
    results['kappa_estimated'].append(M.emissions[0].kappa.numpy())
    results['kappa_true_lowest'].append(kappa[0] if isinstance(kappa, list) else kappa)
    results['kappa_true_highest'].append(kappa[-1] if isinstance(kappa, list) else kappa)
    

    return results

def run_simulation(data_types, model_types, estimate='Uhat', simulations=50, K=5, n_subj=6, plot=False):
    """
    Run simulations for individual parcellation.

    Args:
        data_types (dict): A dictionary containing different types of data.
        model_types (dict): A dictionary containing different types of models.
        estimate (str, optional): The type of estimation to perform. Defaults to 'Uhat'.
        simulations (int, optional): The number of simulations to run. Defaults to 50.
        K (int, optional): The number of regions. Defaults to 5.
        n_subj (int, optional): The number of subjects. Defaults to 6.
        plot (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        pd.DataFrame: The results of the simulations.

    """
    simulation_no = 0
    results = initialize_results(estimate=estimate)
    for sim in range(simulations):
        print(f'Simulation {sim+1}/{simulations}')

        # ======== Generate the data ========        
        for data_type in data_types:
            for outlier in [True, False]:
                
                # --- Data ---
                # increase simulation number for every type of data generated
                simulation_no += 1
                # Generate data
                U_true, Y_train, emissionT, arrangeT, grid, X, part_vec, V_outlier = generate_data(
                        data_types[data_type],
                        K=K,
                        n_subj=n_subj,
                        outlier_subject=outlier,
                        ) 
                # --- Plots ---
                if plot:
                    # Group parcellation
                    plt.figure(figsize=(10, 4))
                    grid.plot_maps(pt.exp(arrangeT.logpi), cmap='jet', vmax=1, grid=[1, arrangeT.K])
                    # Individual parcellations
                    plt.figure(figsize=(20, 2))
                    grid.plot_maps(U_true, cmap='tab20', vmax=19, grid=[1, int(Y_train.shape[0])])
                
                # ======== Estimate Parameters ========
                for model_type in model_types:

                    # --- Estimation ---
                    if estimate=='V':
                        # Estimate Vs
                        M, theta, Uhat = estimate_Vs(
                                    model_types[model_type],
                                    arrangeT,
                                    K,
                                    Y_train,
                                    X,
                                    part_vec)
                    elif estimate == 'Uhat':
                        # Estimate Uhat
                        M, theta, Uhat = estimate_Uhat(
                                    model_types[model_type],
                                    emissionT.V,
                                    arrangeT,
                                    K,
                                    Y_train,
                                    X,
                                    part_vec)
                        
                    # --- Evaluation ---
                    results = evaluate_simulation(results,
                                                    simulation_no,
                                                    estimate,
                                                    model_type,
                                                    data_type,
                                                    outlier,
                                                    K,
                                                    M,
                                                    emissionT,
                                                    grid,
                                                    data_types[data_type]['kappa'],
                                                    U_true,
                                                    Uhat,
                                                    V_outlier)

                
    results = pd.DataFrame(results)
    return results




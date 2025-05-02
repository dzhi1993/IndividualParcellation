import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import seaborn as sb
import IndividualParcellation.scripts.paths as paths
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.spatial as sp
import ProbabilisticParcellation.util as ut
import pandas as pd
from IndividualParcellation.plot import plot_simulation
import IndividualParcellation.simulate as sim

if __name__=="__main__":
    base_dir = paths.set_base_dir()
    results_dir = paths.set_results_dir(base_dir)

    # ----- Simulation parameters -----
    K=32
    n_subj=8
    plot=False
    simulations = 1
    noise_levels = [(0.2, 1)]
    # noise_levels = [(0.1, 10), (4, 20), (10, 80)]

    # N.B.:Parcel specific kappa is not implemented for subjects_equal_weight and Subject + Regions specific kappa is not implemented yet
    model_types = {                    
                'general':               {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'general_eq':            {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                'subject_specific':      {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False},
                'subject_specific_eq':   {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':True},
                }
    

    # ----- Run estimation -----
    estimate_parameters = ['V', 'Uhat']
    for low, high in noise_levels:
        data_types = {'general_low': {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': low},
                    'general_high': {'subject_specific_kappa':False, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': high},
                    'subject_specific_one': {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': [low] + [high]*int(n_subj-1)},
                    'subject_specific_half': {'subject_specific_kappa':True, 'parcel_specific_kappa':False, 'subjects_equal_weight':False, 'kappa': [low]*int(n_subj/2) + [high]*int(n_subj/2)},
        }
        for estimate in estimate_parameters:
            results = sim.run_simulation(data_types,
                                    model_types, 
                                    simulations=simulations, 
                                    K=K, 
                                    n_subj=n_subj, 
                                    plot=plot,
                                    estimate=estimate)
            
            filename = f'simulate_{estimate}-estimation_K-{K}_nsub-{n_subj}_kappa-{low}-{high}_sim-{simulations}'
            results.to_csv(results_dir + f'evaluation/{filename}.csv')
            print(f'Evaluation saved to {results_dir}evaluation/{filename}.csv')


    


    pass
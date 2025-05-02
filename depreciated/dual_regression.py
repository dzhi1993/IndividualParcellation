"""
Dual regression to estimate individual parcellations using
the Functional-Fusion framework

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: January 13, 2024
last update: February 2024

Compatibility: Python 3.11.5
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import softmax
from sklearn.linear_model import LinearRegression

import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds

import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.full_model as fm

from global_config import MODEL_DIR, BASE_DIR
from utils import plot_multi_flat


# ###################### FUNCTIONS  ####################################

def get_iparcel_dualreg(Ug, Y):
    """
    Dual regression to estimate individual parcellations

    Args:
        Ug (torch.Tensor or np.ndarray): Group map (n_voxels, n_parcels)
        Y (np.ndarray): Individual fMRI data (n_subjects, n_obs, n_voxels)

    Returns:
        Ui (np.ndarray): Individual parcellation (n_subjects, n_parcels, n_voxels)
    """

    # Set nan's to 0 in the input data
    Y[np.isnan(Y)] = 0.

    # Move the tensor to CPU and convert it into a NumPy array
    if torch.is_tensor(Ug):
        Ug = Ug.cpu().numpy()

    # Normalization (compute probability distribution of group map)
    prob_Ug = softmax(Ug)

    # For every individual:
    Ui = []
    for y in Y:
        # Initialize V
        V = np.empty((prob_Ug.shape[1], y.shape[0]))

        # Fit the OLS to generate V --> (n_obs, n_parcels)
        reg_ols = LinearRegression()
        V = reg_ols.fit(prob_Ug, y.T).coef_

        # Fit the Non-Negative Least Squares to compute Ui -- >
        # (n_parcels, n_voxels)
        reg_nnls = LinearRegression(positive=True)
        ui = reg_nnls.fit(V, y).coef_.T

        # Append
        Ui.append(ui)

    return np.array(Ui)


# ######################### INPUTS ######################################

## Load the atlas
atlas, _ = am.get_atlas('MNISymC3')
# sym_type = 'sym'
sym_type = 'asym'

## Load the group map from functional-fusion pre-trained model
model_name = f'/Models_03/asym_Md_space-MNISymC3_K-17'
fname = MODEL_DIR + model_name
U, _ = ar.load_group_parcellation(fname, device='cuda')

# #######################################################################

if __name__ == "__main__":

    # Load the individual localizing data from the Functional-Fusion framework
    data, info, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name, subj=None)

    # Split the data
    tdata, _, _, _ = fm.prep_datasets(data, info.sess,
                                      info['cond_num_uni'].values,
                                      info['half'].values,
                                      join_sess=False,
                                      join_sess_part=False)

    # Stack data across observations (halfs or sessions)
    # stacker function in utils
    # stacked_Y = stacker(tdata)

    # Estimate individual parcellations
    U_i = get_iparcel_dualreg(U.T, tdata[0])

    # Save individual parcellations
    np.save('iparcel_mdtb-sc1_mni_asym-k17_dual-regression.npy', U_i)

    # Visualization: plot Ui, i.e. the probabilistic individual parcellations
    plt.figure(figsize=(20, 20))
    plot_multi_flat(U_i, 'MNISymC3', grid=(6, 4), cmap='tab20', dtype='prob',
                    titles=["subj_{}".format(i+1) for i in range(U_i.shape[0])],
                    fig_path='iparcel_mdtb-sc1_mni_asym-k17_dual-regression.png')
    plt.show()

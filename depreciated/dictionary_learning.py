"""
Sparse Dictionary-Learning with different regularization schemes to estimate
individual parcellations using the Functional-Fusion framework

author: Ana Luisa Pinho
email: agrilopi@uwo.ca

created: January 29, 2024
last update: February 2024

Compatibility: Python 3.11.5
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from joblib import Memory

from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import spectral_embedding
from sklearn.decomposition import dict_learning_online, sparse_encode
from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNet

import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds

import HierarchBayesParcel.full_model as fm

from global_config import BASE_DIR
from utils import plot_multi_flat


# ###################### FUNCTIONS  ####################################

def initial_dictionary(n_clusters, Y,):
    """
    Create the initial dictionary

    Args:
        Y (np.ndarray): fMRI data per subject (n_subjects, n_obs, n_voxels)

    Returns:
        components (np.ndarray): dictionary (n_components, n_conditions)
    """

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0,
                             batch_size=200, n_init=10)
    kmeans = kmeans.fit(Y.T)

    dictionary_ = kmeans.cluster_centers_
    dictionary = (dictionary_.T / np.sqrt((dictionary_ ** 2).sum(1))).T
    similarity = np.exp(np.corrcoef(dictionary))
    embedding = spectral_embedding(similarity, n_components=1)
    order = np.argsort(embedding.T).ravel()
    dictionary = dictionary[order]

    return dictionary


def get_iparcel_dictlearning(Y, n_components=17, method='online', alpha=.01,
                             l1_ratio=.5, write_dir='/tmp'):
    """
    Create the sparse-encoded individual parcellations

    Args:
        Y (np.ndarray): fMRI data per subject (n_subjects, n_obs, n_voxels)

    Returns:
        components (np.ndarray): Individual parcellation (n_parcels, n_voxels)
    """

    # Set nan's to 0 in the input data
    Y[np.isnan(Y)] = 0.

    # Retrieve number of subjects and number of voxels
    n_subjects = Y.shape[0]
    n_voxels = Y.shape[2]

    # Reshape input data --> (n_obs, n_subjects * n_voxels)
    y = Y.swapaxes(0, 1)
    y = np.reshape(y, (y.shape[0], -1))

    # Compute the initial dictionary across subjects
    mem = Memory(write_dir, verbose=0)
    dictionary = mem.cache(initial_dictionary)(n_components, y)

    if method == 'online':
        components, _, _ = dict_learning_online(
            y.T,
            n_components,
            alpha=alpha,
            dict_init=dictionary,
            batch_size=256,
            method='cd',
            return_code=True,
            shuffle=True,
            n_jobs=1,
            positive_code=True,
            return_n_iter=True
        )
    elif method in ['multitask_lasso', 'multitask_elasticnet']:
        # components shape --> (n_subjects * n_voxels, n_components)
        components = np.zeros((y.shape[1], n_components))
        if method == 'multitask_lasso':
            clf = MultiTaskLasso(alpha=alpha)
        else:
            assert method == 'multitask_elasticnet'
            clf = MultiTaskElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        for i in np.arange(n_voxels):
            x = y[:, i: i + n_subjects * n_voxels: n_voxels]
            components[i: i + n_subjects * n_voxels: n_voxels] = \
                clf.fit(dictionary.T, x).coef_
    else:
        assert method == 'sparse'
        components = sparse_encode(
            y.T, dictionary, alpha=alpha, max_iter=100, n_jobs=1,
            check_input=True, positive=True)

    # Reshape components --> (n_subjects, n_components, n_voxels)
    Ui = components.T.reshape(n_components, components.shape[0] // n_voxels,
                              n_voxels)
    Ui = np.swapaxes(Ui, 0, 1)

    return Ui


# ######################### INPUTS ######################################

# Home dir (cross-platform valid)
home = str(Path.home())

## Load the atlas
atlas, _ = am.get_atlas('MNISymC3')
# sym_type = 'sym'
sym_type = 'asym'

# Method's parameters
method = 'online'
alpha = .01
l1_ratio = .5 # for multitask_elasticnet

# #######################################################################

if __name__ == "__main__":

    # Load the individual localizing data from the Functional-Fusion framework
    data, info, tds = ds.get_dataset(BASE_DIR, 'MDTB', atlas=atlas.name,
                                     subj=None)

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
    U_i = get_iparcel_dictlearning(tdata[0], n_components=17, method=method,
                                   alpha=alpha)

    # Save individual parcellations
    np.save(
        str(Path(home, 'iparcel_mdtb-sc1_mni_asym-k17_' + method + '.npy')),
        U_i)

    # Visualization: plot Ui, i.e. the probabilistic individual parcellations
    plt.figure(figsize=(20, 20))
    plot_multi_flat(
        U_i, 'MNISymC3', grid=(6, 4), cmap='tab20', dtype='prob',
        titles=["subj_{}".format(i+1) for i in range(U_i.shape[0])],
        fig_path=str(Path(
            home, 'iparcel_mdtb-sc1_mni_asym-k17_' + method + '.png')))
    plt.show()

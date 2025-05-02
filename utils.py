import pickle
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import pandas as pd
import HierarchBayesParcel.arrangements as ar

from FusionModel.util import plot_data_flat

def stacker(data_list):
    """
    Stack data across sessions

    Args:
        data (list): Individual fMRI data (n_obs, )

    Returns:
        Ui (np.ndarray): Individual fMRI data
                         (n_subjects, n_obs * n_conditions, n_voxels)
    """

    # Stack data
    for i in np.arange(len(data_list)):
        if i == 0:
            stacked_data = data_list[0]
        else:
            stacked_data = np.append(stacked_data, data_list[i], axis=1)

    return stacked_data

def plot_multi_flat(data, atlas, grid, cmap='tab20b', dtype='label',
                    cscale=None, titles=None, colorbar=False, fig_path=None):
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
        fig_path: path of output figure, default format is png

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
        plot_data_flat(data[i], atlas,
                       cmap=cmap[i],
                       dtype=dtype,
                       cscale=None,
                       render='matplotlib',
                       colorbar=(i == 0) & colorbar)

        plt.title(titles[i])
        plt.tight_layout()

    if isinstance(fig_path, str):
        plt.savefig(fig_path)

def convert_hard_to_prob(U, strength=7.0, confidence=None):
    ''' Convert a hard parcellation to probabilistic

    Args:
        U (np.ndarray): P-long vector of hard parcellation
        strength (float): the strength of the parcellation (prior)

    Returns:
        the probabilistic parcellation, either a (K, P) group
        map or a (n_subj, K, P) individual map. In whichever case,
        the sum along dimension K equals to 1, making it a
        probabilistic parcellation.

    Notes:
        In some exisiting hard parcellations, a voxel may have a
        label 0 to indicate unassigned parcel. For these voxels,
        we give a flat distribution - so that the probability of
        such a voxel to assign one parcel is 1/K.
    '''
    assert np.all((U >= 0) & (U.astype(int) == U)), \
        "The input U must be non-negative integer numpy array!"
    _, U = np.unique(U, return_inverse=True)
    K = np.unique(U).size

    if U.ndim == 1:
        logpi = ar.expand_mn_1d(U, K) * strength
        if confidence is not None:
            if type(confidence) is np.ndarray:
                confidence = pt.tensor(confidence, dtype=pt.get_default_dtype())

            logpi = logpi * confidence
        # Set parcel 0 to unassigned
        logpi = logpi[1:, :] if np.any(np.unique(U) == 0) else logpi
        return pt.softmax(logpi, dim=0)
    elif U.ndim == 2:
        logpi = ar.expand_mn(U, K) * strength
        # Set parcel 0 to unassigned
        logpi = logpi[:, 1:, :] if np.any(np.unique(U) == 0) else logpi
        return pt.softmax(logpi, dim=1)
    else:
        raise ValueError('The input U must be (P,) or (n_subj, P) integer ndarray!')

def load_batch_fit(fname, device=None):
    """ Loads a batch of fitted models
    Args:
        fname (str): File directory
        device: model model to which device, 'cpu', 'cuda'..
    Returns:
        info: Data Frame with information
        models: List of models
    """
    info = pd.read_csv(fname + '.tsv', sep='\t')
    with open(fname + '.pickle', 'rb') as file:
        models = pickle.load(file)

    if device is not None:
        for m in models:
            m.move_to(device)

    return info, models

def load_batch_best(fname, device=None):
    """ Loads a batch of model fits and selects the best one

    Args:
        fname (str): File name
        device: model model to which device, 'cpu', 'cuda'..

    Returns:
        info_reduced: Data Frame with reduced information
        best_model: the model with highest likelihood
    """
    info, models = load_batch_fit(fname)

    j = info.loglik.argmax()

    best_model = models[j]
    if device is not None:
        best_model.move_to(device)

    info_reduced = info.iloc[j]
    return info_reduced, best_model
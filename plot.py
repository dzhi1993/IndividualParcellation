#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: Caro Nettekoven
"""
import numpy as np
import torch as pt
import nibabel as nb
import nitools as nt
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import FusionModel.evaluate as ev
from IndividualParcellation.scripts.indiv_evaluation import make_eval_info, eval_parcel_DCBC
from FusionModel.util import plot_multi_flat, plot_data_flat

from global_config import DEVICE
import IndividualParcellation.scripts.paths as paths
import imageio
import os

def plot_diagnostics(indiv_par, V, cmap=None, labels=None, subject_list=None):
    """
    Function to plot diagnostic plots for individual parcellation procedure
        

    Parameters:
    - indiv_par (numpy.ndarray): (Subj x Regions x Voxels) Array representing individual parcellations.
    - V (numpy.ndarray): Mean functional profile (Vs of fitted emission model).
    - cmap (str, optional): Colormap to use for the plots. Defaults to None.
    - labels (list, optional): List of regions. Defaults to None.
    - subject_list (list, optional): List of subject names. Defaults to None.
    
    Returns:
    - fig (matplotlib.figure.Figure): Figure object
    """
    if labels is None:
        labels = [f'Region {i}' for i in range(V.shape[1])]
    
    # -- Plot subject contribution to each region --
    fig, axs = plt.subplots(1, 3, figsize=(25, 10))
    img = axs[0].imshow(indiv_par.sum(axis=2))
    plt.colorbar(img, orientation='horizontal')

    # Set labels and ticks
    axs[0].set_xlabel('Regions')
    axs[0].set_ylabel('Subjects')
    if subject_list is not None:
        axs[0].set_yticks(np.arange(indiv_par.shape[0]), labels=subject_list)
    axs[0].set_xticks(np.arange(V.shape[1])[::4], labels=labels[1::4])
    axs[0].set_title('Sum of individual parcellations')

    # -- Plot Vs --
    im = axs[1].imshow(V)
    plt.colorbar(im, orientation='horizontal')
    axs[1].set_xlabel('Regions')
    axs[1].set_ylabel('Conditions')
    axs[1].set_xticks(np.arange(V.shape[1])[::4], labels=labels[1::4])
    axs[1].set_title('V')

    # -- Plot mean individual parcellation --
    plot_data_flat(indiv_par.mean(dim=0).cpu().numpy(), atlas='MNISymC2',
                    cmap=cmap, dtype='prob')
    # Plot title underneath the subplot
    axs[2].annotate('Mean individual parcellation', xy=(0.5, 0.1), xycoords='axes fraction',ha='center')
    return fig


def plot_evolution(indiv_par, space='MNISymC2', cmap=None, outname='indiv_par_evolution', subject_list=None):
    """Function to plot the evolution of the individual parcellations over training iterations. Writes out a gif of the evolution of the individual parcellations over training iterations
    
    Parameters
    ----------
    indiv_par : torch.tensor
        Individual parcellations. Shape (n_subj, n_par, n_cond)
    space : str
        Name of the space to plot the parcellations in
    cmap : colormap
        Name of the colormap to use
    outname : str
        Name of the output file
    subject_list : list
        List of subject names

    """

    if subject_list is None:
        subject_list = [f'Subject No. {i}' for i in range(indiv_par.shape[0])]

    for i in np.arange(indiv_par.shape[0]):
        plt.figure(figsize=(20, 20))
        plot_multi_flat(indiv_par.cpu().numpy()[i],
                        atlas=space,
                        grid=(int(np.ceil(indiv_par.shape[1]/4)), 4),
                        cmap=cmap,
                        dtype='prob',
                        titles=subject_list)
        plt.savefig(f'{outname}_iter-{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

    filenames = [f'{outname}_iter-{i}.png' for i in range(indiv_par.shape[0])]
    # images = []
    writer = imageio.get_writer(outname + '.gif', fps = 30, codec='libx264',
                quality=10, pixelformat='yuvj444p')
    for filename in filenames:
        writer.append_data(imageio.imread(filename))
    writer.close()
    for filename in filenames:
        os.remove(filename)
    

def plot_simulation(emissionT, M, theta, Uhat, arrangeT, grid):
    """
    Plot diagnostic plots for simulated estimation of the emission model
    
    Args:
        emissionT (type): True emission model.
        M (type): Description of M.
        theta (type): Description of theta.
        Uhat (type): Description of Uhat.
        arrangeT (type): Description of arrangeT.
        grid (type): Description of grid.
    
    Returns:
        None
    """
    
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
    plt.xlabel('iteration')
    plt.title('V evolution')

    plt.subplot(2,3,4)
    ind = M.get_param_indices('emissions.0.kappa')
    plt.plot(theta[:,ind])
    plt.title('kappa')

    plt.subplot(2,3,5)
    P = pt.argmax(arrangeT.marginal_prob(),dim=0)
    plt.imshow(P.reshape(grid.dim),cmap='tab20',vmax=19)
    plt.title('True')

    plt.subplot(2,3,6)
    U_hat_w = pt.argmax(Uhat.mean(dim=0),dim=0)
    plt.imshow(U_hat_w.reshape(grid.dim),cmap='tab20',vmax=19)
    plt.title('avrg Uhat')
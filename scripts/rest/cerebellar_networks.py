#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script of evaluate the individual parcellation results

Created on 12/4/2023 at 4:22 PM
Author: Caro Nettekoven
"""
import numpy as np
import nibabel as nb
import nitools as nt
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import Functional_Fusion.connectivity as conn
from IndividualParcellation.scripts.indiv_evaluation import make_eval_info, eval_parcel_DCBC
from FusionModel.util import plot_multi_flat, plot_data_flat
import SUITPy as suit
import IndividualParcellation.scripts.paths as paths
from nilearn import plotting
import PcmPy as pcm


base_dir = paths.set_base_dir()
atlas_dir = paths.set_atlas_dir(base_dir)
data_dir = paths.set_fusion_dir(base_dir)
figure_dir = paths.set_figure_dir()


# Import resting-state group networks
# For each network, get the cerebellar component
# Regress the cerebellar component into the subject's cerebellar data to get the timecourse of that cerebellar component
# Correlate the cerebellar timecourse with the timecourse of the subject's fs32k data
# Average the correlation pattern of the cortical connectivity fingerprint across subjects
# Display it on the cortical surface for the two cerebellar-only networks, and compare it to fingerprint of the cortico-cerebellar networks
# --> Is the pattern overall more dispersed for the cerebellar-only networks? --> Could indicate orchestrating role of the cerebellum
# --> Is the fingerprint overall stronger for the cerebellar-only networks?

def plot_all_networks(networks, atlas, ainfo):
    """
    Plot the cerebellar components of all networks.

    Args:
        networks (ndarray): The networks data.
        atlas (Atlas): The atlas object.
        ainfo (dict): Atlas info.

    Returns:
        None
    """

    networks = networks.T
    nifti = atlas.data_to_nifti(networks)

    surf_data = suit.flatmap.vol_to_surf(nifti, stats='nanmean',
                                         space=ainfo['normspace'])
    grid = (int(np.ceil(networks.shape[0]/4)), 4)
    titles = [f'Network {i+1}' for i in range(networks.shape[0])]

    plt.figure(figsize=(20, 80))
    for i in np.arange(networks.shape[0]):
        plt.subplot(grid[0], grid[1], i + 1)
        
        ax = suit.flatmap.plot(surf_data[:,i],
                               cmap='RdBu_r',
                               new_figure=False,
                               overlay_type='func',
                               colorbar=False,
                               cscale=(-3, 3)
                              )
        plt.title(titles[i])
        plt.tight_layout()

    plt.savefig(figure_dir + '/group_networks_cerebellar.png', dpi=300, bbox_inches='tight')

def cerebellar_cortical_correlation(networks, cerebellar_data, cortical_data):
    """Function to correlate the cerebellar network timecourse with the cortical timecourse of each vertex/voxel"""
    # Loop through subjects
    Corr = np.zeros((cortical_data.shape[0], networks.shape[1], cortical_data.shape[2]))
    for s, sub in enumerate(subject_list):
        cerebellar_timecourse = conn.regress_networks(networks.T, cerebellar_data[s,:,:])
        # Correlate cerebellar timecourse with cortical timecourse of each vertex
        cortical_timecourse = cortical_data[s,:,:]

        corr = np.corrcoef(cerebellar_timecourse, cortical_timecourse.T)
        Corr[s,:,:] = corr[:cerebellar_timecourse.shape[0], cerebellar_timecourse.shape[0]:]

    return Corr


if __name__ == "__main__":
    # Import resting-state group networks
    # Get atlas
    space='MNISymC2'
    average_within_ico = True
    cortical_res = 1442 # 1002
    atlas, ainfo = am.get_atlas(space)
    networks = atlas.read_data(data_dir + f'/HCP/group_ica/dim_auto/signal/signal_components.nii.gz')

    # plot_all_networks(networks, atlas, ainfo)

    # Regress group networks into cerebellar data
    # Get cerebellar data
    T = pd.read_csv(f'{data_dir}/MDTB/participants.tsv', delimiter='\t')
    participants = T.participant_id
    subject_list = participants[T['ses-rest'] == 1].tolist()

    # Get cerebellar data
    cerebellar_data, info, mdtb = ds.get_dataset(data_dir, 'MDTB', atlas=space, sess=['ses-rest'], type='Tseries', subj=subject_list)
    
    # Get cortical data and average within Icosahedrons
    cortical_data, _ = mdtb.get_data(
            space='fs32k', ses_id='ses-rest', type='Tseries', subj=subject_list)
    if average_within_ico:
        ico = [atlas_dir + f'/tpl-fs32k/Icosahedron{cortical_res}.L.label.gii',
            atlas_dir + f'/tpl-fs32k/Icosahedron{cortical_res}.R.label.gii']
        cortical_data, names = conn.average_within_Icos(
                ico, cortical_data)
    cortical_data = np.nan_to_num(cortical_data)

    # Get the correlation between cerebellar network timecourses and cortical timecourses
    Corr = cerebellar_cortical_correlation(networks, cerebellar_data, cortical_data)

    # plot correlation matrix
    # plt.figure()
    # plt.imshow(Corr.mean(axis=0), cmap='RdBu_r', vmin=-0.6, vmax=0.6)

    

    # Barplot of average correlation pattern
    Corr_avg = Corr.mean(axis=2)
    Corr_sem = Corr_avg.std(axis=0) / np.sqrt(Corr_avg.shape[0])
    plt.figure()
    plt.bar(np.arange(Corr_avg.shape[1]), Corr_avg.mean(axis=0), yerr=Corr_sem)
    plt.ylim(-0.17, 0.17)
    # colour cerebellar-only networks
    for c in [0, 10]:
        plt.bar(c, Corr_avg.mean(axis=0)[c], yerr=Corr_sem[c], color='r')
    plt.savefig(figure_dir + '/cerebellar_cortical_correlation.png', dpi=300, bbox_inches='tight')

    
    # --- Bring connectivity fingerprint to surface ---
    Corr_group = Corr.mean(axis=0)
    # Get fs32k surface
    atlas_fs, _ = am.get_atlas("fs32k", atlas_dir)
    atlas_fs.get_parcel(ico, unite_struct=False)
    # Project to surace map
    indicator_matrix = pcm.indicator(atlas_fs.label_vector)
    cortical_map = Corr_group @ indicator_matrix.T
    cifti_img = atlas_fs.data_to_cifti(cortical_map)
    cortical_map_surf = nt.surf_from_cifti(cifti_img)

    # Plot the cortical map
    n=3
    hemis = ["left", "right"]
    cortical_map_surf_selected = [cortical_map_surf[h][n,:] for h,_ in enumerate(hemis)]
    
    view = "lateral"
    cmap = "bwr"
    colorbar = False
    surfs = [atlas_dir + f"/tpl-fs32k/tpl-fs32k_hemi-{hemi}_inflated.surf.gii" for hemi in ['L', 'R']]
    ax = []
    for h, hemi in enumerate(hemis):
        fig = plotting.plot_surf_stat_map(
                                    surfs[h], cortical_map_surf_selected[h], hemi=hemi,
                                    # title='Surface left hemisphere',
                                    colorbar=colorbar, 
                                    view = view,
                                    cmap=cmap,
                                    engine='plotly',
                                    symmetric_cbar = True,
                                )
    ax.append(fig.figure)


    # setting up cameras to get a better view of the hand area in the lateral view
    # rotate to get a better view of M1 only on lateral view
    camera_params = []
    camera_params.append(dict( #left hemi
        center=dict(x=0,y=0,z=0),
        eye=dict(x=-1.5, y=0,z=0.9),
        up=dict(x=0,y=0,z=1),
    ))
    camera_params.append(dict( # right hemi
        center=dict(x=0,y=0,z=0),
        eye=dict(x=1.5, y=0,z=0.9),
        up=dict(x=0,y=0,z=1),
    ))
    ax[0].show()


    # for h, hemi in enumerate(hemis):
    #     if view == "lateral":
    #         ax[h].update_layout(scene_camera=camera_params[h])
    #     ax[h].show()


        
        
        

        
    pass
        


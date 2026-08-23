#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal example for training an individual parcellation

Author: dzhi
"""
import numpy as np
import pandas as pd
import torch as pt
import nibabel as nb
import nitools as nt
import Functional_Fusion.atlas_map as am
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.evaluation as hev
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.util as hut
import utils as ut

from global_config import (DEVICE, EXAMPLE_DIR, REPLICATION_DIR, RESULTS_DIR)

RESULT_DIR = RESULTS_DIR / 'train_individual'

# Example data and model files included in the repository.
SPACE = 'fs32k'
EXAMPLE_NAME = 'example_subject'
DATA_TYPE = 'Ico642Run'
DATA_FILE = EXAMPLE_DIR / f'example_rest_space-{SPACE}_{DATA_TYPE}_desc-sm4fwhm_binarized.dscalar.nii'
INFO_FILE = EXAMPLE_DIR / f'example_rest_{DATA_TYPE}.tsv'
GROUP_PRIOR_FILE = REPLICATION_DIR / 'group_parcellations' / '17Networks' / 'HBP17_FUSION_networks_prob.dscalar.nii'

# Basic training settings. Edit these values directly for another example.
GROUP_STRENGTH = 10
SPATIAL_WEIGHT = 5
N_ITER = 200
OUTPUT_PREFIX = f'{EXAMPLE_NAME}_HBP17_FUSION_indiv'


def load_group_prior(file_name, device=DEVICE):
    U = nb.load(file_name).get_fdata().astype(np.float32)
    U = np.nan_to_num(U)
    U = pt.tensor(U, dtype=pt.get_default_dtype(), device=device)

    # Normalize defensively in case the stored probabilities have minor
    # numerical drift or medial-wall zeros.
    col_sum = U.sum(dim=0, keepdim=True)
    U = pt.where(col_sum > 0, U / col_sum.clamp(min=1e-8),
                 pt.ones_like(U) / U.shape[0])
    return U


def load_example_data(atlas, data_file, info_file):
    info = pd.read_csv(info_file, sep='\t')
    data = atlas.cifti_to_data(str(data_file)).astype(np.float32)
    data = np.nan_to_num(data)

    # Match the existing HCP individual-training scripts: remove the mean
    # profile across observations before fitting the individual map.
    data = data - np.mean(data, axis=0, keepdims=True)

    data = [data[None, :, :]]
    cond_vec = [info['net_id'].values.reshape(-1, )]
    part_vec = [info['run'].values.reshape(-1, )]
    subj_ind = [np.array([0])]
    return data, cond_vec, part_vec, subj_ind, info


def align_group_prior_to_kong2019(U, device=DEVICE):
    align, network_names, colors = ut.get_kong2019_group_parcellation()
    align = pt.tensor(align, dtype=pt.get_default_dtype(), device=device)
    indx = hev.matching_greedy(align, U)
    return U[indx, :], network_names, colors


def make_empty_connectivity(num_vertices, device=DEVICE):
    """
    Build a sparse Wc placeholder for the mRBM/cRBM_Wc code path.

    The full paper analyses use a fs32k neighborhood matrix. This minimal
    release example does not ship that large file, so the spatial term is
    disabled while preserving the same arrangement-model API.
    """
    indices = pt.empty((2, 0), dtype=pt.long, device=device)
    values = pt.empty((0,), dtype=pt.get_default_dtype(), device=device)
    return pt.sparse_coo_tensor(indices, values,
                                (num_vertices, num_vertices),
                                device=device).coalesce()


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load the atlas.
    atlas, _ = am.get_atlas(SPACE)
    atlas.calculate_symmetry()

    # Step 2: Load the repository-local HBP17 FUSION group prior and align it
    # to the Kong2019 17-network naming/color convention.
    U = load_group_prior(GROUP_PRIOR_FILE)
    U, network_names, colors = align_group_prior_to_kong2019(U)

    # Step 3: Load the repository-local example subject training data.
    data, cond_vec, part_vec, subj_ind, info = load_example_data(atlas,
                                                                DATA_FILE,
                                                                INFO_FILE)
    n_subj = np.unique(np.concatenate(subj_ind, axis=0)).size

    # Step 4: Build the mRBM arrangement model, matching indiv_eval_hcp.py.
    Wc = make_empty_connectivity(atlas.P)
    ar_model = ar.build_arrangement_model(U.clone(), prior_type='prob',
                                          atlas=atlas, sym_type='asym',
                                          model_type='cRBM_Wc',
                                          Wc=Wc, theta=SPATIAL_WEIGHT,
                                          epos_iter=20, num_chain=n_subj)
    ar_model.bu = ar_model.bu * GROUP_STRENGTH

    # Step 5: Estimate the individual parcellation.
    U_indiv, _, M = fm.get_indiv_parcellation(
        ar_model,
        atlas,
        data,
        cond_vec,
        part_vec,
        subj_ind,
        Vs=None,
        sym_type='asym',
        n_iter=N_ITER,
        em_params={'num_subj': n_subj,
                   'subjects_equal_weight': True,
                   'subject_specific_kappa': False,
                   'parcel_specific_kappa': False})

    # Step 6: Save probability and hard-label outputs.
    prob_file = RESULT_DIR / f'{OUTPUT_PREFIX}_prob.npy'
    np.save(prob_file, U_indiv.cpu().numpy())

    Pindiv = pt.argmax(U_indiv, dim=1) + 1
    img = nt.make_label_cifti(Pindiv.T.cpu().numpy(),
                              atlas.get_brain_model_axis(),
                              column_names=[EXAMPLE_NAME],
                              label_names=network_names,
                              label_RGBA=colors)
    label_file = RESULT_DIR / f'{OUTPUT_PREFIX}.dlabel.nii'
    nb.save(img, label_file)

    del M
    pt.cuda.empty_cache()
    hut.report_cuda_memory()

    print(f'Saved individual probabilities to {prob_file}')
    print(f'Saved individual label map to {label_file}')

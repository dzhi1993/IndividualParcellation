#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script of optimal task battery selection

Created on 9/2/2025 at 3:54 PM
Author: dzhi
"""
import numpy as np
import torch as pt
import nibabel as nb
import pandas as pd
from itertools import combinations

from global_config import DATA_ROOT_PATH, DEVICE

device = pt.device(DEVICE)

def pairwise_overlap(maps, selected):
    """
       Compute average pairwise overlap among selected maps. Given
       two maps, the overlap rate is calculated by number of intersect
       vertices / union vertices.

    Args:
        maps (np.ndarray or np.bool): NxP numpy array of task maps.
        selected (list): a list of index integer of selected maps.

    Returns:
        overlap (float): the average overlap rate among pairwise
            selected maps.

    """
    if len(selected) < 2:
        return 0.0

    overlaps = []
    for i, j in combinations(selected, 2):
        inter = pt.sum(maps[i] & maps[j]).item()
        union = pt.sum(maps[i] | maps[j]).item()
        if union > 0:
            overlaps.append(inter / union)

    return sum(overlaps) / len(overlaps) if overlaps else 0.0

def task_beta_selection(maps, M, lamda=0.5, selected=None):
    """
       Select M maps greedily by maximizing the objective score. The
       objective score is calculated as current marginal brain coverage
       penalized by pairwise overlapping of the selected maps. lamda is
       a hyperparameter tunes how much you care about independence vs
       coverage.

       Args:
           maps (np.ndarray): NxP numpy array of maps.
           M (np.int): the number of maps to select.
           lamda (float): Lamda parameter.
           selected (list): a list of indicator integers of
               pre-selected maps. For example, if selected=[0,2], then
               the first and thrid maps will be included in the final
               optimal task battery. Therefore, this function seeks to
               only find the M-2 maps into the list. This is useful if
               user has prior knowledge of which task condition "must"
               have in the model training.

       Returns:
           selected (list): a list of index of the selected task maps.
           final_cov (float): the total brain coverage rate.
           final_overlap (float): the average overlap rate among the
               selected maps.
       """
    N, P = maps.shape
    selected = [] if selected is None else selected
    covered = pt.any(maps[selected], dim=0) if selected \
        else pt.zeros(P, dtype=pt.bool, device=device)

    for _ in range(M - len(selected)):
        best_idx, best_score = None, -float("inf")

        for i in range(N):
            if i in selected:
                continue

            trial = selected + [i]
            trial_cov = pt.any(maps[trial], dim=0).float().mean().item()
            trial_overlap = pairwise_overlap(maps, trial)
            score = trial_cov - lamda * trial_overlap

            if score > best_score:
                best_idx, best_score = i, score

        selected.append(best_idx)
        covered = pt.any(maps[selected], dim=0)

    final_cov = covered.float().mean().item()
    final_overlap = pairwise_overlap(maps, selected)
    return sorted(selected), final_cov, final_overlap

if __name__ == '__main__':
    ## Load task data (N task condition, P)
    ds_name = 'MDTB'
    data_dir = DATA_ROOT_PATH / 'Tian' / ds_name / 'derivatives/group/data'

    # 1. masked task condition (betas) map from group-level analysis
    # The reason of using group task beta maps is we here wanted to
    # find the optimal battery based on population-level activation pattern,
    # we will not do this optimal search on each individual's task map.
    maps = nb.load(data_dir / 'group_space-fs32k_ses-01_CondAll_masked-hi0.1lo0.1_binarized.dscalar.nii').get_fdata()
    # 2. Unmasked task condition map
    # maps = nb.load(data_dir + '/group_space-fs32k_ses-task_CondAll.dscalar.nii').get_fdata()

    # convert to torch bool on GPU
    maps = pt.from_numpy(maps).to(device)
    maps = maps.bool()

    ## Find optimal task battery
    all_tasks, all_coverage = [], []
    for num_tasks in range(10,11):
        best_score, task_idx = -float("inf"), None

        for i in range(maps.shape[0]):
            best, cover, overlap = task_beta_selection(maps, num_tasks, lamda=0, selected=[i])
            score = cover - 0 * overlap

            if score > best_score:
                best_score = score
                task_idx = best

        all_tasks.append(task_idx)
        all_coverage.append(best_score)
        print(task_idx, best_score)

    # Load condition labels
    T = pd.read_csv(data_dir + '/group_ses-01_CondAll.tsv', sep='\t')
    print(T.iloc[task_idx])

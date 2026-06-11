#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merged HCP benchmarking evaluation for individual parcellations.

This script evaluates individual maps from both the MSHBM baseline and the
mRBM-HBP pipeline with the same metric logic:
    - task contrasts: DCBC on all contrasts, task inhomogeneity by task domain
    - rest time series: resting-state homogeneity

The original evaluation_MSHBM_indiv.py and evaluation_HBP_indiv.py scripts are
kept unchanged for reference.

Created for the paper-release repository cleanup.
Author: dzhi
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as pt
import nibabel as nb
import Functional_Fusion.atlas_map as am
import Functional_Fusion.dataset as ds
import FusionModel.evaluate as ev
import HierarchBayesParcel.util as hut

import IndividualParcellation.utils as ut
from global_config import BASE_DIR


hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
HCP_TASKS = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR',
             'RELATIONAL', 'SOCIAL', 'WM']

HCP_DIR = '/home/dzhi/eris_mount/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    HCP_DIR = '/data/tge/Tian/HCP_img'
if not Path(HCP_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

ERIS_DIR = Path('/home/dzhi/eris_mount')
if not ERIS_DIR.exists():
    ERIS_DIR = Path('/data/tge')
if not ERIS_DIR.exists():
    raise (NameError('Could not find eris_dir'))

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'
REPLICATION_DIR = REPO_ROOT / 'replication'
SUBJECT_LIST_DIR = REPLICATION_DIR / 'subject_list'
RES_DIR = RESULTS_DIR / Path(__file__).resolve().parent.name
RES_DIR.mkdir(parents=True, exist_ok=True)

# pytorch cuda global flag: True - cuda; False - cpu
pt.cuda.is_available = lambda : False
if pt.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
pt.set_default_device(DEVICE)
pt.set_default_dtype(pt.float32)


# Basic evaluation settings. Edit these values directly for another run.
SPACE = 'fs32k'
K = 17
TEST_DATA_TYPE = 'rest'      # 'task' or 'rest'
SUBJECT_LIST_FILES = ['test.tsv']
TASK_SMOOTH = '4_MSMAll'
TASK_POSITIVE_ONLY = False
REST_RUN_LIST = [2, 3]
REST_SMOOTH = None
REST_TYPE = 'Tseries'
DIST_FILE = Path(BASE_DIR) / 'Atlases' / 'tpl-fs32k' / 'distGOD_fs32k.pt'

MSHBM_INDIV_DIR = ERIS_DIR / 'dzhi' / 'workspace' / 'res' / \
                  'ind_parcellation' / 'HCP203_test_set'
HBP_INDIV_FILE = ERIS_DIR / 'dzhi' / 'Indiv_par' / 'Results' / \
                 'section_4' / 'HCP' / \
                 'RestPrior+HCPrest-2run-indiv_space-fs32k_K-17_Ico642Run.dlabel.nii'

MODEL_CONFIGS = [
    {'name': 'MSHBM17',
     'source': 'MSHBM',
     'train_runs': 2,
     'strength': 100,
     'spatial_w': 30,
     'train_smooth': '6fwhm',
     'parcellation_dir': MSHBM_INDIV_DIR},
    {'name': 'HBP17_rest',
     'source': 'HBP',
     'train_runs': 2,
     'train_smooth': '6fwhm',
     'parcellation_file': HBP_INDIV_FILE,
     'reference_subject_list': 'HCP200_test.tsv'},
]


def make_eval_info(model_config, test_sess):
    minfo = pd.Series()
    minfo['atlas'] = SPACE
    minfo['K'] = K
    minfo['datasets'] = ['HCP']
    minfo['train_sess'] = f"run-{model_config['train_runs']}"
    minfo['test_data'] = 'HCP'
    minfo['test_sess'] = test_sess
    minfo['model_type'] = model_config['source']
    minfo['group_map_name'] = model_config['name']
    minfo['indiv_test_kappa'] = None
    return minfo


def get_subject_list_path(file_name):
    subj_list_path = SUBJECT_LIST_DIR / file_name
    if not subj_list_path.exists():
        subj_list_path = Path(HCP_DIR) / 'subj_list' / file_name
    if not subj_list_path.exists():
        raise FileNotFoundError(f'Could not find subject list: {file_name}')
    return subj_list_path


def load_subject_table(file_name):
    subj_list_path = get_subject_list_path(file_name)
    return subj_list_path, pd.read_csv(subj_list_path, sep='\t')


def tensorize(data):
    if type(data) is np.ndarray:
        data = pt.tensor(data, dtype=pt.get_default_dtype(), device=DEVICE)
    return data


def load_hcp_task_contrasts(subjects, atlas, smooth=TASK_SMOOTH,
                            positive_only=TASK_POSITIVE_ONLY):
    hcp_ds = ds.DataSetHcpTask(HCP_DIR)
    data, info = [], []
    for s in subjects.participant_id:
        print(f'Loading task contrasts for {s}')
        file_name = f'/{s}_tfMRI_contrasts_level2_hp200_s{smooth}.dscalar.nii'
        dat = nb.load(hcp_ds.func_dir.format(s) + file_name).get_fdata().astype(np.float32)
        dat = dat[:, np.concatenate(atlas.indx_full)]
        this_info = pd.read_csv(hcp_ds.func_dir.format(s) +
                                f'/{s}_tfMRI_contrasts_level2_hp200.tsv',
                                sep='\t')
        data.append(dat)
        info.append(this_info)

    data = np.stack(data)
    assert all(df.shape == info[0].shape for df in info[1:]), \
        'Not all subjects have the same task-info table shape.'
    info = info[0]

    if positive_only:
        contrast_idx = info['positive'] == 1
        info = info[contrast_idx].reset_index(drop=True)
    else:
        contrast_idx = np.arange(data.shape[1])

    data = data[:, contrast_idx, :]
    info['task_name'] = [s.rstrip('2') for s in info.task_name]
    return [data], info


def load_hcp_rest_timeseries(subjects, atlas, run_list=REST_RUN_LIST,
                             smooth=REST_SMOOTH, data_type=REST_TYPE):
    data_dir = '/mnt/sda/HCP_rfMRI/fix_32k/{0}'
    data = []
    for run_id in run_list:
        ses_data = []
        for s in subjects.participant_id:
            if smooth is None or smooth == 0:
                file_name = f'/{s}_run{run_id}'
            else:
                file_name = f'/{s}_run{run_id}_desc-sm{smooth}'
            file_name += '.dtseries.nii' if data_type == 'Tseries' else f'.{data_type}.nii'

            print(f'Loading rest run {run_id} for {s}')
            dat = nb.load(data_dir.format(s) + file_name).get_fdata().astype(np.float32)
            dat = dat[:, np.concatenate(atlas.vertex_mask)]
            ses_data.append(dat)
        data.append(np.stack(ses_data))

    info = pd.DataFrame({'task_name': ['REST'] * data[0].shape[1]})
    return data, info


def load_test_data(subjects, atlas):
    if TEST_DATA_TYPE == 'task':
        return load_hcp_task_contrasts(subjects, atlas)
    elif TEST_DATA_TYPE == 'rest':
        return load_hcp_rest_timeseries(subjects, atlas)
    else:
        raise ValueError("TEST_DATA_TYPE must be 'task' or 'rest'.")


def load_mshbm_parcellation(model_config, subj_list_path):
    Pindiv = ut.get_kong2019_indiv_parcellations(
        str(model_config['parcellation_dir']),
        str(subj_list_path),
        w=model_config['strength'],
        c=model_config['spatial_w'],
        num_sess=model_config['train_runs'])
    return Pindiv.to(dtype=pt.get_default_dtype(), device=DEVICE)


def load_hbp_parcellation(model_config, subjects):
    Pindiv = nb.load(model_config['parcellation_file']).get_fdata()[:]
    Pindiv = pt.tensor(Pindiv, dtype=pt.get_default_dtype(), device=DEVICE)

    ref_path = get_subject_list_path(model_config['reference_subject_list'])
    ref_table = pd.read_csv(ref_path, sep='\t')
    ref_index = pd.Series(np.arange(len(ref_table)),
                          index=ref_table.participant_id.astype(str))
    idx = [ref_index[str(s)] for s in subjects.participant_id.astype(str)]
    return Pindiv[idx]


def load_individual_parcellation(model_config, subj_list_path, subjects):
    if model_config['source'] == 'MSHBM':
        Pindiv = load_mshbm_parcellation(model_config, subj_list_path)
    elif model_config['source'] == 'HBP':
        Pindiv = load_hbp_parcellation(model_config, subjects)
    else:
        raise ValueError(f"Unknown parcellation source: {model_config['source']}")

    if Pindiv.ndim == 1:
        Pindiv = Pindiv.unsqueeze(0)
    return pt.where(Pindiv == 0, pt.nan, Pindiv)


def get_task_indices(info):
    if TEST_DATA_TYPE == 'task':
        tasks = HCP_TASKS + ['all']
    else:
        tasks = ['all']

    task_indices = {}
    for task in tasks:
        if task == 'all':
            task_indices[task] = [True] * len(info)
        else:
            task_indices[task] = list(info['task_name'] == task)
    return task_indices


def make_result_frame(minfo, subjects, model_config, task, test_run,
                      num_contrasts, num_vertices):
    n_subj = len(subjects)
    return pd.DataFrame({'atlas': [minfo.atlas] * n_subj,
                         'K': [minfo.K] * n_subj,
                         'train_data': [minfo.datasets] * n_subj,
                         'train_sess': [minfo.train_sess] * n_subj,
                         'test_data': [minfo.test_data] * n_subj,
                         'test_sess': [minfo.test_sess] * n_subj,
                         'model_type': [minfo.model_type] * n_subj,
                         'group_map_name': [minfo.group_map_name] * n_subj,
                         'subj_num': [f'{i}' for i in subjects.participant_id],
                         'indiv_test_kappa': [minfo.indiv_test_kappa] * n_subj,
                         'task_name': [task] * n_subj,
                         'test_run': [test_run] * n_subj,
                         'train_smooth': [model_config['train_smooth']] * n_subj,
                         'test_smooth': [TASK_SMOOTH if TEST_DATA_TYPE == 'task' else REST_SMOOTH] * n_subj,
                         'test_type': [TEST_DATA_TYPE] * n_subj,
                         'num_contrasts': [num_contrasts] * n_subj,
                         'num_vertices': [num_vertices] * n_subj})


def evaluate_metric_pair(Pindiv, td, dist, idx, task):
    hut.report_cuda_memory()
    dcbc_indiv = None
    if TEST_DATA_TYPE == 'task' and task == 'all':
        dcbc_indiv = ev.calc_test_dcbc(Pindiv, td[:, idx, :], dist, trim_nan=True)
        dcbc_indiv = pt.where(dcbc_indiv == 0, pt.nan, dcbc_indiv).cpu().numpy()
    pt.cuda.empty_cache()
    hut.report_cuda_memory()

    hut.report_cuda_memory()
    if TEST_DATA_TYPE == 'task':
        homo_indiv = None
        inhomo_indiv = ev.calc_test_task_inhomogeneity(Pindiv, td[:, idx, :],
                                                       return_single=True)
        inhomo_indiv = inhomo_indiv.cpu().numpy()
    else:
        homo_indiv = ev.calc_test_homogeneity(Pindiv, td[:, idx, :])
        homo_indiv = homo_indiv.cpu().numpy()
        inhomo_indiv = None
    pt.cuda.empty_cache()
    hut.report_cuda_memory()

    return dcbc_indiv, homo_indiv, inhomo_indiv


def evaluate_one_model(model_config, subj_list_path, subjects, test_data,
                       test_info, dist):
    Pindiv = load_individual_parcellation(model_config, subj_list_path, subjects)
    test_sess = 'contrasts' if TEST_DATA_TYPE == 'task' else REST_TYPE
    minfo = make_eval_info(model_config, test_sess=test_sess)
    task_indices = get_task_indices(test_info)

    results = pd.DataFrame()
    for r, td in enumerate(test_data):
        td = tensorize(td)
        for task, idx in task_indices.items():
            num_contrasts = int(np.sum(idx))
            if num_contrasts == 0:
                continue

            res = make_result_frame(minfo, subjects, model_config, task,
                                    test_run=r + 1,
                                    num_contrasts=num_contrasts,
                                    num_vertices=td.shape[2])

            dcbc_indiv, homo_indiv, inhomo_indiv = evaluate_metric_pair(
                Pindiv, td, dist, idx, task)

            res['dcbc_indiv'] = dcbc_indiv if dcbc_indiv is not None else np.nan
            res['homo_indiv'] = homo_indiv if homo_indiv is not None else np.nan
            res['inhomo_indiv'] = inhomo_indiv if inhomo_indiv is not None else np.nan
            results = pd.concat([results, res], ignore_index=True)
    return results


if __name__ == "__main__":
    atlas, _ = am.get_atlas(SPACE)
    atlas.calculate_symmetry()
    dist = pt.load(DIST_FILE, weights_only=True)

    results = pd.DataFrame()
    for subj_list_file in SUBJECT_LIST_FILES:
        subj_list_path, subjects = load_subject_table(subj_list_file)

        print(f'Start loading HCP {TEST_DATA_TYPE} test data for {subj_list_file} ...')
        tic = time.perf_counter()
        test_data, test_info = load_test_data(subjects, atlas)
        toc = time.perf_counter()
        print(f'Done loading. Used {toc - tic:0.4f} seconds!')
        hut.report_cuda_memory()

        for model_config in MODEL_CONFIGS:
            print(f"Evaluating {model_config['name']} on {TEST_DATA_TYPE} data ...")
            this_res = evaluate_one_model(model_config, subj_list_path, subjects,
                                          test_data, test_info, dist)
            this_res['subject_list'] = subj_list_file
            results = pd.concat([results, this_res], ignore_index=True)

    out_file = RES_DIR / \
        f'eval_indiv_MSHBM-HBP_HCP_K-{K}_test_on-HCP-{TEST_DATA_TYPE}.tsv'
    results.to_csv(out_file, index=False, sep='\t')
    print(f'Saved merged benchmarking results to {out_file}')

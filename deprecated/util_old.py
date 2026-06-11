import pickle, os, warnings, math, subprocess
import numpy as np
import torch as pt
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nb
import nitools as nt
import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.evaluation as ev
import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import scipy.io as spio

from FusionModel.util import plot_data_flat
from pathlib import Path
from itertools import combinations
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR

ERIS_DIR = '/home/dzhi/eris_mount'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    raise (NameError('Could not find hcp_dir'))


def load_hcp_contrasts(dataset_dir, subj_list, space='MNIAsymC2', sess='all',
                        type='Tseries', beta_include=True, hemis=None, smooth=None):
    hcp_ds = ds.DataSetHcpTask(dataset_dir)
    T = hcp_ds.get_participants(subj_list)
    if sess == 'all':
        sess = hcp_ds.task_domain

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    data, info, domains = [],[],[]
    for s in T.participant_id:
        ses_data, ses_info, ses_domains = [],[],[]
        for ses_id in sess:
            # Assemble file name
            if smooth is None:
                file_name = f'/ses-{ses_id}/{s}_tfMRI_{ses_id}_level2_hp200_s2.dscalar.nii'
            else:
                file_name = f'/ses-{ses_id}/{s}_tfMRI_{ses_id}_level2_hp200_s{smooth}.dscalar.nii'

            # Load data / info
            dat = nb.load(hcp_ds.func_dir.format(s) + file_name)
            this_info = dat.header.get_axis(0).name.tolist()
            prefix = os.path.commonprefix(this_info)
            this_info = [s[len(prefix):] for s in this_info]
            dat = dat.get_fdata().astype(np.float32)

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]
            else:
                dat = dat[:, np.concatenate(this_at.indx_full)]

            if beta_include:
                contrast_idx = np.arange(dat.shape[0])
            else:
                # Remove the betas (contrast) from this session
                reg_info = pd.read_csv(hcp_ds.estimates_dir.format(s) +
                                       f'/ses-task/{s}_ses-task_reginfo.tsv', sep='\t')
                beta_names = reg_info.loc[reg_info.task_name == ses_id].cond_name.unique()
                beta_names = [s + '_' for s in beta_names]
                contrast_idx = [i for i, s in enumerate(this_info) if not s.startswith(tuple(beta_names))]

            ses_data.append(dat[contrast_idx,:])
            ses_info.append([this_info[i] for i in contrast_idx])
            ses_domains.append([ses_id] * len(contrast_idx))

        data.append(np.vstack(ses_data))
        info.append(np.concatenate(ses_info))
        domains.append(np.concatenate(ses_domains))

    # Check if all arrays are identical
    assert all(np.array_equal(info[0], arr) for arr in info)
    assert all(np.array_equal(domains[0], arr) for arr in domains)

    info_com = pd.DataFrame({'contrast_name': info[0],
                             'task_name': domains[0]})

    data = np.stack(data)
    return [data], info_com
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

try:
    from IndividualParcellation.global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
except (ImportError, ModuleNotFoundError, NameError):
    try:
        from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR
    except (ImportError, ModuleNotFoundError, NameError):
        MODEL_DIR = None
        BASE_DIR = None
        ATLAS_DIR = None

REPO_ROOT = Path(__file__).resolve().parent
REPLICATION_DIR = REPO_ROOT / 'replication'
MSHBM_17NETWORK_DIR = REPLICATION_DIR / 'MSHBM_17networks'

ERIS_DIR = '/home/dzhi/eris_mount'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = None

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

def get_DU15_parcellation(file_name='DU15NET_Prior', atlas_space='fs32k'):
    atlas, _ = am.get_atlas(atlas_space)
    DU15_dir = ERIS_DIR + '/dzhi/workspace/DU15NET'
    file = nb.load(DU15_dir + f'/HCP/fsLR_32k/{file_name}_fsLR_32k.dlabel.nii')
    DU15 = atlas.cifti_to_data(file)
    DU15 = np.nan_to_num(DU15)

    info = pd.read_csv(DU15_dir + '/DU15NET_ColorLUT.csv')
    network_names = list(info['Abbreviation'])
    colors = info[["R","G","B","A"]].to_numpy().astype(float)
    colors[:, :3] = colors[:, :3] / 255

    return DU15, network_names, colors

def get_kong2019_group_parcellation():
    network_names = spio.loadmat(MSHBM_17NETWORK_DIR / '17network_labels.mat')['network_name']
    network_names = ['???'] + [network_names[0][i][0] for i in range(17)]

    colors = spio.loadmat(MSHBM_17NETWORK_DIR / 'group.mat')['colors']/255
    colors = colors[1:,:]
    colors = np.hstack((colors, np.ones((17, 1))))
    colors = np.vstack((np.zeros(4), colors))
    KONG2019 = nb.load(MSHBM_17NETWORK_DIR / 'Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii').get_fdata()[:]

    return KONG2019, network_names, colors

def get_kong2019_indiv_parcellations(dir, subj_list, w=80, c=40, num_sess=1, save_file=False):
    atlas, _ = am.get_atlas('fs32k')
    KONG2019, nets, colors = get_kong2019_group_parcellation()
    if type(KONG2019) is np.ndarray:
        KONG2019 = pt.tensor(KONG2019, dtype=pt.get_default_dtype())

    parcellations = []
    sub_name = []
    T = pd.read_csv(subj_list, delimiter='\t')
    for i, s in enumerate(T.participant_id):
        mat_file = dir + f'/Ind_parcellation_MSHBM_sub{s}_w{w}_MRF{c}_num-sess{num_sess}.mat'

        left = spio.loadmat(mat_file)['lh_labels'].reshape(-1)
        right = spio.loadmat(mat_file)['rh_labels'].reshape(-1)
        left_labels = left[atlas.vertex_mask[0]]
        right_labels = right[atlas.vertex_mask[1]]
        parcel = np.concatenate([left_labels, right_labels])
        Prob = ar.expand_mn_1d(parcel, K=18)[1:, :]
        # Align colors with Kong 2019
        mask = parcel == 0
        indx = ev.matching_greedy(KONG2019, Prob)
        new_parcel = pt.argmax(Prob[indx,:], dim=0) + 1
        new_parcel[mask] = 0

        parcellations.append(new_parcel)
        sub_name.append(f'sub_{s}')

    if save_file:
        img = nt.make_label_cifti(pt.stack(parcellations).T.cpu().numpy(), atlas.get_brain_model_axis(),
                                column_names=sub_name, label_names=nets, label_RGBA=colors)
        outdir = ERIS_DIR + '/dzhi/Indiv_par/Kong_2019/indiv_par'
        nb.save(img, outdir + f'/{os.path.basename(dir)}_indiv_par_w{w}_MRF{c}_n-sess{num_sess}_1.dlabel.nii')

    return pt.stack(parcellations)

def make_per_network_cifti(parcel, space='fs32k', type='hard', outfile=None):
    atlas, am_info = am.get_atlas(space)
    align, net_name, colors = ut.get_kong2019_group_parcellation()

    # Find argmax per column
    max_idx = pt.argmax(parcel, dim=0)
    one_hot = pt.zeros_like(parcel)
    one_hot[max_idx, pt.arange(parcel.shape[1])] = 1

    img = nt.make_label_cifti(parcel.T, atlas.get_brain_model_axis(),
                              column_names=net_name,
                              label_names=net_name,
                              label_RGBA=colors)
    pass

def make_border(in_file, column_name, hem='L', outfile='L.border'):

    surf = ATLAS_DIR + f'/tpl-fs32k/tpl-fs32k_hemi-{hem}_inflated.surf.gii'
    smooth_cmd = (f"wb_command -cifti-label-to-border {in_file} "
                  f"-column {column_name} "
                  f"-border {surf} {outfile}")
    subprocess.run(smooth_cmd, shell=True)
    print(f'Done generating border {column_name} hemisphere {hem}')


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


def build_resting_data(dataset, space='fs32k', ses_list='all', type='Tseries',
                       subj=None, hemis=None, smooth=None):
    # set up
    my_dataset = ds.get_dataset_class(BASE_DIR, dataset)
    T = my_dataset.get_participants()
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

    if ses_list == 'all':
        ses_list = my_dataset.sessions

    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, str):
        subj = [subj]
    elif isinstance(subj, (int, np.integer)):
        subj = [T.participant_id.iloc[subj]]
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a str, int, list or ndarray'))

    data = []
    for i, ses_id in enumerate(ses_list):
        print(f'Loading {ses_id}...')
        ses_data = []
        for s in subj:
            try:
                # Load the data
                if smooth is not None:
                    C = nb.load(my_dataset.data_dir.format(s)
                                + f'/{s}_space-{space}_{ses_id}_{type}_{smooth}.dscalar.nii')
                else:
                    C = nb.load(my_dataset.data_dir.format(s)
                                + f'/{s}_space-{space}_{ses_id}_{type}.dscalar.nii')
                dat = C.get_fdata().astype(np.float32)
                if smooth is not None and ("binarized" in smooth):
                    dat = dat.astype(np.int8)
            except FileNotFoundError:
                dat = np.nan
            ses_data.append(dat)

        ref_shape = next(m.shape for m in ses_data if isinstance(m, np.ndarray))
        ses_data = [np.full(ref_shape, np.nan) if not isinstance(m, np.ndarray) else m for m in ses_data]
        ses_data = np.stack(ses_data)

        if hemis is not None:  # if cortical data
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            ses_data = ses_data[:,:,this_at.indx_full[stru_idx]]

        # Make the zero voxels to nan
        # zero_cols = np.all(ses_data == 0, axis=1, keepdims=True)
        # data.append(np.where(zero_cols, np.nan, ses_data))
        data.append(ses_data)

    return data


def build_hcp_datasets(dataset_dir, subj_list, this_at, ses_list=['ses-rest1'],
                       type=['Tseries'], join_sess=False, join_sess_part=False,
                       part_ind='half', part_num=None, cond_ind='net_id', hemis=None,
                       smooth=None, ext=None):
    '''Build datasets for functional fusion framework, each dataset is
    supposed to follow BIDS filing structure. Where each subject's data
    is located in <root of your dataset folder>/derivatives/sub-XXX/data
    folder. A <participant.tsv> file is expected in the root directory

    Args:
        dataset_dir: the dataset root directory
        subj_list: a csv file contains all subjects id

    Returns:

    '''
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    data_dir = dataset_dir + '/derivatives/{0}/data'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    space = this_at.name
    assert len(ses_list) == len(type), "session list and type list must equal length!"

    if ses_list[0] == 'all':
        ses_list = ['ses-rest1', 'ses-rest2']
        type = np.repeat(type, 2)

    data = []
    info_l = []
    for i, ses_id in enumerate(ses_list):
        ses_data = []
        for s in T.participant_id:
            info_raw = pd.read_csv(data_dir.format(s)
                                   + f'/{s}_{ses_id}_{type[i]}.tsv', sep='\t')

            # Assemble file name
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}'
            else:
                file_name = f'/{s}_space-{space}_{ses_id}_{type[i]}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dscalar.nii'

            # Load data
            # dat = nb.load(data_dir.format(s) + file_name)
            # # this_data.append(atlas.read_data(data_dir.format(s) + file_name).T)
            # dat = dat.get_fdata().astype(np.float32)
            dat = this_at.cifti_to_data(data_dir.format(s) + file_name).astype(np.float16)
            if "binarized" in file_name:
                dat = dat.astype(np.int8)

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]

            ses_data.append(dat)

        data.append(np.stack(ses_data))
        info_l.append(info_raw)

    dat = np.concatenate(data, axis=1)
    info = pd.concat(info_l, ignore_index=True, sort=False)
    n_subj = dat.shape[0]

    # Step 2: Assemble condition and partition vectors
    data, cond_vec, part_vec, subj_ind = [], [], [], []
    # Make different sessions either the same or different
    if join_sess:
        if part_num is not None:
            indx = info[part_ind] == part_num
        else:
            indx = np.full(info[part_ind].shape, True)
        # Check if we want to set no partition after join sessions
        if join_sess_part:
            part_vec.append(np.ones(indx.shape))
        else:
            part_vec.append(info[part_ind].values[indx].reshape(-1, ))

        data.append(dat[:, indx, :])
        cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
        subj_ind.append(np.arange(0, n_subj))
    else:
        splitter = 'sess' if info.get('sess') is not None else 'half'
        sessions = np.unique(info[splitter])
        # Now build and split across the correct sessions:
        for s in sessions:
            if part_num is None:
                indx_list = [info[splitter] == s]
            else:
                indx_list = [(info[splitter] == s) & (info[part_ind] == pn) for pn in part_num]

            for indx in indx_list:
                data.append(dat[:, indx, :])
                cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(0, 0 + n_subj))

    return data, cond_vec, part_vec, subj_ind, info


def load_hcp_timeseries(dataset_dir, subj_list, space='MNIAsymC2', run_list=[0,1,2,3],
                       type='Tseries', hemis=None, smooth=None, ext=None):

    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')

    data_dir = '/mnt/sda/HCP_rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)

    data = []
    for i, run_id in enumerate(run_list):
        ses_data=[]
        for s in T.participant_id:
            # Assemble file name
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_run{run_id}'
            else:
                file_name = f'/{s}_run{run_id}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dtseries.nii'

            # Load data / remove medial wall
            dat = nb.load(data_dir.format(s) + file_name)
            dat = dat.get_fdata().astype(np.float32)
            dat = dat[:, np.concatenate(this_at.vertex_mask)]

            if hemis is not None: # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:,this_at.indx_full[stru_idx]]

            ses_data.append(dat)

        data.append(np.stack(ses_data))

    return data


def load_hcp_contrasts(dataset_dir, subj_list, space='fs32k', sess='all',
                       return_positive=True, beta_only=False, hemis=None, smooth=None):
    hcp_ds = ds.DataSetHcpTask(dataset_dir)
    T = hcp_ds.get_participants(subj_list)
    if sess == 'all':
        sess = hcp_ds.task_domain

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    data, info = [],[]
    for s in T.participant_id:
        # Load data / info
        print(f'Loading contrasts for {s}')
        file_name = f'/{s}_tfMRI_contrasts_level2_hp200_s{smooth}.dscalar.nii'
        dat = nb.load(hcp_ds.func_dir.format(s) +
                      file_name).get_fdata().astype(np.float32)
        this_info = pd.read_csv(hcp_ds.func_dir.format(s) +
                                f'/{s}_tfMRI_contrasts_level2_hp200.tsv', sep='\t')

        if hemis is not None:  # if cortical data
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            dat = dat[:, this_at.indx_full[stru_idx]]
        else:
            dat = dat[:, np.concatenate(this_at.indx_full)]

        data.append(dat)
        info.append(this_info)

    data = np.stack(data)
    # Check if all arrays are identical
    assert all(df.shape == info[0].shape for df in info[1:]), \
        "Not all subjects have the same DataFrame shape!"
    info_com = info[0]

    if return_positive:
        contrast_idx = info_com['positive'] == 1
        info_com = info_com[contrast_idx].reset_index(drop=True)
    else:
        contrast_idx = np.arange(data.shape[1])

    data = data[:, contrast_idx, :]
    return [data], info_com

def load_msc_contrasts(ds_name, space='fs32k', sess='all', subj=None, smooth=None):
    atlas, _ = am.get_atlas(space)

    if sess == 'all':
        sess = ['motor','memory','mixed']
    elif isinstance(sess, str):
        sess = [sess]
    assert isinstance(sess, list)

    dataset = ds.DataSetMSC('/home/dzhi/eris_mount/Tian/MSC')
    T = dataset.get_participants()
    # Assemble the data
    Data = None
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a list of str or int'))

    # Loop again to assemble the data
    Data_list, info = [], []
    for i, s in enumerate(subj):
        subj_dat, subj_info = [], pd.DataFrame()
        for ses_id in sess:
            # Load the data
            if smooth is not None:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR_smooth{smooth}.dscalar.nii')
            else:
                C = nb.load(dataset.contrast_dir.format(s) +
                             f'/{ses_id}/{s}-{ses_id}_contrasts_32k_fsLR.dscalar.nii')
            this_data = atlas.cifti_to_data(C)
            this_info = C.header.get_axis(0).name.tolist()

            # indices = [i for i, s in enumerate(this_info) if not re.search(r"[+-]", s)
            #            and not s.startswith("Alltasks")]
            indices = [i for i, s in enumerate(this_info) if not s.startswith("Alltasks")]

            this_df = pd.DataFrame({'cond_name': [s for i, s in enumerate(this_info) if i in indices],
                               'task_name': ses_id})
            subj_info = pd.concat([subj_info, this_df], ignore_index=True)
            subj_dat.append(this_data[indices])

        Data_list.append(np.vstack(subj_dat))
        info.append(subj_info)

    # concatenate along the first dimension (subjects)
    Data = np.stack(Data_list)
    Data[np.isinf(Data)] = np.nan

    # Assemble info file
    # Check if all arrays are identical
    assert all(info[0].equals(df) for df in info[1:]), \
        "subjects must have same number of task contrasts!"
    info_com = info[0]

    return Data, info_com


def build_msc_resting_data(dataset_dir, subj_list, this_at, ses_list=['ses-rest1'],
                       type=['Tseries'], join_sess=False, join_sess_part=False,
                       part_ind='half', part_num=None, cond_ind='net_id', hemis=None,
                       smooth=None, ext=None):
    '''Build datasets for functional fusion framework, each dataset is
    supposed to follow BIDS filing structure. Where each subject's data
    is located in <root of your dataset folder>/derivatives/sub-XXX/data
    folder. A <participant.tsv> file is expected in the root directory

    Args:
        dataset_dir: the dataset root directory
        subj_list: a csv file contains all subjects id

    Returns:

    '''
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')
    data_dir = dataset_dir + '/derivatives/{0}/data'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    space = this_at.name

    if ses_list[0] == 'all':
        ses_list = ['ses-rest1', 'ses-rest2']

    data = []
    info_l = []
    for i, ses_id in enumerate(ses_list):
        ses_data = []
        for s in T.participant_id:
            info_raw = pd.read_csv(data_dir.format(s)
                                   + f'/{s}_{ses_id}_{type}.tsv', sep='\t')

            # Assemble file name
            if smooth is None or (smooth == 0):
                file_name = f'/{s}_space-{space}_{ses_id}_{type}'
            else:
                file_name = f'/{s}_space-{space}_{ses_id}_{type}_desc-sm{smooth}'

            file_name = file_name + ext if ext is not None else file_name
            file_name += '.dscalar.nii'

            # Load data
            # dat = nb.load(data_dir.format(s) + file_name)
            # # this_data.append(atlas.read_data(data_dir.format(s) + file_name).T)
            # dat = dat.get_fdata().astype(np.float32)
            dat = this_at.cifti_to_data(data_dir.format(s) + file_name).astype(np.float16)
            if "binarized" in file_name:
                dat = dat.astype(np.int8)

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]

            ses_data.append(dat)

        data.append(np.stack(ses_data))
        info_l.append(info_raw)

    dat = np.concatenate(data, axis=1)
    info = pd.concat(info_l, ignore_index=True, sort=False)
    n_subj = dat.shape[0]

    # Step 2: Assemble condition and partition vectors
    data, cond_vec, part_vec, subj_ind = [], [], [], []
    # Make different sessions either the same or different
    if join_sess:
        if part_num is not None:
            indx = info[part_ind] == part_num
        else:
            indx = np.full(info[part_ind].shape, True)
        # Check if we want to set no partition after join sessions
        if join_sess_part:
            part_vec.append(np.ones(indx.shape))
        else:
            part_vec.append(info[part_ind].values[indx].reshape(-1, ))

        data.append(dat[:, indx, :])
        cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
        subj_ind.append(np.arange(0, n_subj))
    else:
        splitter = 'sess' if info.get('sess') is not None else 'half'
        sessions = np.unique(info[splitter])
        # Now build and split across the correct sessions:
        for s in sessions:
            if part_num is None:
                indx_list = [info[splitter] == s]
            else:
                indx_list = [(info[splitter] == s) & (info[part_ind] == pn) for pn in part_num]

            for indx in indx_list:
                data.append(dat[:, indx, :])
                cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
                part_vec.append(info[part_ind].values[indx].reshape(-1, ))
                subj_ind.append(np.arange(0, 0 + n_subj))

    return data, cond_vec, part_vec, subj_ind, info

def load_randy_contrasts(space='fs32k', subj=None, hemis=None, smooth=2,
                         verbose=False):
    """Loads all the CIFTI files in the data directory of a certain space
     / type and returns they content as a Numpy array

    Args:
        space (str): Atlas space (Defaults to 'SUIT3').
        subj (ndarray, str, or list):  Subject numbers /names to get
            [None = all]
    Returns:
        Data (ndarray): (n_subj, n_contrast, n_voxel) array of data
        info (DataFramw): Data frame with common descriptor
    """
    dataset = ds.DataSetRANDY15(ERIS_DIR + '/Tian/RANDY15')
    T = dataset.get_participants()
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, str):
        subj = [subj]
    elif isinstance(subj, (int, np.integer)):
        subj = [T.participant_id.iloc[subj]]
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a str, int, list or ndarray'))

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    max = 0
    # Loop over the different subjects to find the most complete info
    for s in subj:
        # Get an check the information
        info_raw = pd.read_csv(dataset.contrast_dir.format(s)
                               + f'/{s}_AllContrasts.tsv', sep='\t')
        # Keep the most complete info
        if info_raw.shape[0] > max:
            info_com = info_raw
            max = info_raw.shape[0]
    base = np.asarray(info_com['contrast_name'])

    # Loop again to assemble the data
    Data_list = []
    for i, s in enumerate(subj):
        # If you add verbose printout, make it so
        # that by default it is suppressed by a verbose=False option
        if verbose:
            print(f'- Getting data for {s} in {space}')
        # Load the data
        if smooth is not None:
            C = nb.load(dataset.contrast_dir.format(s)
                        + f'/{s}_AllContrasts_{space}_sm{smooth}_Zmap.dscalar.nii')
        else:
            C = nb.load(dataset.contrast_dir.format(s)
                        + f'/{s}_AllContrasts_{space}_Zmap.dscalar.nii')
        this_data = C.get_fdata()

        if hemis is not None:  # if cortical data
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            this_data = this_data[:, this_at.indx_full[stru_idx]]
        else:
            this_data = this_data[:, np.concatenate(this_at.indx_full)]

        # Check if this subject data in incomplete
        if this_data.shape[0] != info_com.shape[0]:
            this_info = pd.read_csv(dataset.contrast_dir.format(s)
                                    + f'/{s}_AllContrasts.tsv', sep='\t')
            incomplete = np.asarray(this_info['contrast_name'])
            contrast_to_row = {name: i for i, name in enumerate(incomplete)}
            aligned_data = np.full((len(base), this_data.shape[1]), np.nan)

            for j, name in enumerate(base):
                if name in contrast_to_row:
                    aligned_data[j] = this_data[contrast_to_row[name]]
                else:
                    warnings.warn(f'{s} - missing contrast {name}')
            this_data = aligned_data

        Data_list.append(this_data[np.newaxis, ...])
    # concatenate along the first dimension (subjects)
    Data = np.concatenate(Data_list, axis=0)
    # Ensure that infinite values (from div / 0) show up as NaNs
    Data[np.isinf(Data)] = np.nan
    return [Data], info_com

def load_randy_contrasts_wo_run(run_exclude=1, space='fs32k', subj=None,
                                hemis=None, smooth=2):
    dict = ['EPROJ_PASTselfmPRESself', 'EPROJ_FUTselfmPRESself',
            'NBACK_2BACKm0BACK',
            'VODDK_TARG',
            'LANG_SENTmCTRL',
            'TOM_EMOmPHYS', 'TOM_BELIEFmPHOTO']

    data = []
    for d in dict:
        domain = d.split('_')[0]
        con = d.split('_')[1]
        this_data = load_randy_betas_runwise(domain, con, space=space,
                                                 subj=subj, hemis=hemis, smooth=smooth)

        print(f"Loading {domain}, {con}, total {this_data.shape[1]} runs..")
        if run_exclude is not None:
            this_data = np.delete(this_data, run_exclude-1, axis=1)
        data.append(np.nanmean(this_data, axis=1))

    data = np.stack(data, axis=1)

    ## Making the contrast of interest
    ## Note, the combination of the two contrasts is AVERAGE
    ## e.g. PASTFUTmPRES is 0.5 * (PASTselfmPRESself + FUTselfmPRESself)
    Y = np.empty((data.shape[0], 5, data.shape[2]))
    Y[:, 0, :] = (data[:, 0, :] + data[:, 1, :]) / 2   # PASTselfmPRESself + FUTselfmPRESself
    Y[:, 1:4, :] = data[:, 2:5, :]  # keep middle three (2,3,4)
    Y[:, 4, :] = (data[:, 5, :] + data[:, 6, :]) / 2  # EMOmPHYS + BELIEFmPHOTO

    ## Hard coded info
    info = pd.DataFrame({'task_name': ['EPROJ','NBACK','VODDK','LANG','TOM'],
                        'contrast_name': ['PASTFUTUREmPRESENT','2BACKm0BACK',
                                          'TARGETmBASELINE','SENTmCTRL',
                                          'EMOmPHYS+BELmPHO']})

    return [Y], info

def load_randy_contrasts_runwise(run_idx=[0,1,2], space='fs32k', subj=None,
                                hemis=None, smooth=2):
    dict = ['EPROJ_PASTselfmPRESself', 'EPROJ_FUTselfmPRESself',
            'NBACK_2BACKm0BACK',
            'VODDK_TARG',
            'LANG_SENTmCTRL',
            'TOM_EMOmPHYS', 'TOM_BELIEFmPHOTO']

    data = []
    for d in dict:
        domain = d.split('_')[0]
        con = d.split('_')[1]
        this_data = load_randy_betas_runwise(domain, con, space=space,
                                                 subj=subj, hemis=hemis, smooth=smooth)

        print(f"Loading {domain}, {con}, total {this_data.shape[1]} runs. "
              f"and only extract first 3 runs for consistency!")
        data.append(this_data[:,run_idx,:])

    data = np.stack(data, axis=1) # (n_subj, n_contrast, runs, P)
    data = np.transpose(data, (2, 0, 1, 3)) # (runs, n_subj, n_contrast, P)
    num_subj = data.shape[1]
    P = data.shape[3]

    ## Making the contrast of interest
    ## Note, the combination of the two contrasts is AVERAGE
    ## e.g. PASTFUTmPRES is 0.5 * (PASTselfmPRESself + FUTselfmPRESself)
    total_Y = []
    for run in range(len(run_idx)):
        Y = np.empty((num_subj, 5, P))
        this_data = data[run]
        Y[:, 0, :] = (this_data[:, 0, :] + this_data[:, 1, :]) / 2   # PASTselfmPRESself + FUTselfmPRESself
        Y[:, 1:4, :] = this_data[:, 2:5, :]  # keep middle three (2,3,4)
        Y[:, 4, :] = (this_data[:, 5, :] + this_data[:, 6, :]) / 2  # EMOmPHYS + BELIEFmPHOTO
        total_Y.append(Y)

    ## Hard coded info
    info = pd.DataFrame({'task_name': ['EPROJ','NBACK','VODDK','LANG','TOM'],
                        'contrast_name': ['PASTFUTUREmPRESENT','2BACKm0BACK',
                                          'TARGETmBASELINE','SENTmCTRL',
                                          'EMOmPHYS+BELmPHO']})
    info['cond_name'] = info['contrast_name']

    return total_Y, info

def load_randy_betas_runwise(domain, con_name, space='fs32k', subj=None,
                                 hemis=None, smooth=2, verbose=False):
    """Loads all the CIFTI files in the data directory of a certain space
     / type and returns they content as a Numpy array

    Args:
        space (str): Atlas space (Defaults to 'SUIT3').
        subj (ndarray, str, or list):  Subject numbers /names to get
            [None = all]
    Returns:
        Data (ndarray): (n_subj, n_contrast, n_voxel) array of data
        info (DataFramw): Data frame with common descriptor
    """
    dataset = ds.DataSetRANDY15(ERIS_DIR + '/Tian/RANDY15')
    T = dataset.get_participants()
    # Deal with subset of subject option
    if subj is None:
        subj = T.participant_id
    elif isinstance(subj, str):
        subj = [subj]
    elif isinstance(subj, (int, np.integer)):
        subj = [T.participant_id.iloc[subj]]
    elif isinstance(subj, (list, np.ndarray)):
        if isinstance(subj[0], (int, np.integer)):
            subj = T.participant_id.iloc[subj]
        elif isinstance(subj[0], str):
            subj = subj
        else:
            raise (NameError('subj must be a list of strings or integers'))
    else:
        raise (NameError('subj must be a str, int, list or ndarray'))

    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(space)
    this_at.calculate_symmetry()

    max = 0
    # Loop over the different subjects to find the most complete info
    for s in subj:
        # Get an check the information
        info_raw = pd.read_csv(dataset.contrast_dir.format(s)
                               + f'/{s}_{domain}_{con_name}_Contrasts.tsv', sep='\t')
        # Keep the most complete info
        if info_raw.shape[0] > max:
            info_com = info_raw
            max = info_raw.shape[0]
    base = np.asarray(info_com['contrast_name'])

    Data = np.full((len(subj), len(base), this_at.P), np.nan)
    # Loop again to assemble the data
    Data_list = []
    for i, s in enumerate(subj):
        if verbose:
            print(f'- Getting data for {s} in {space}')
        # Load the data
        C = nb.load(dataset.contrast_dir.format(s)
                    + f'/{s}_{domain}_{con_name}_fs32k_sm{smooth}_Zmap.dscalar.nii')
        this_data = C.get_fdata()

        # Make sure it fits into Data[i]
        n1 = min(this_data.shape[0], Data.shape[1])
        n2 = min(this_data.shape[1], Data.shape[2])
        Data[i, :n1, :n2] = this_data[:n1, :n2]

    if hemis is not None:  # if cortical data
        stru_idx = this_at.structure.index(hemis_dict[hemis])
        Data = Data[:,:, this_at.indx_full[stru_idx]]
    else:
        Data = Data[:, :, np.concatenate(this_at.indx_full)]

    return Data


def build_dataset(datasets, atlas='fs32k', sess=None, cond_ind=None, type=None,
                    part_ind=None, part_num=None, subj=None, join_sess=True,
                    join_sess_part=False, smooth=None, hemis=None):
    """Builds list of datasets, cond_vec, part_vec, subj_ind
    from different data sets
    Args:
        datasets (list): Names of datasets to include
        atlas (str): Atlas indicator
        sess (list): list of 'all' or list of sessions
        design_ind (list, optional): _description_. Defaults to None.
        part_ind (list, optional): _description_. Defaults to None.
        subj (list, optional): _description_. Defaults to None.
        join_sess (bool, optional): Model the sessions with a single model.
            Defaults to True.
    Returns:
        data, cond_vec, part_vec, subj_ind
    """
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}
    this_at, _ = am.get_atlas(atlas, ATLAS_DIR)
    data, cond_vec, part_vec, subj_ind, infos = [],[],[],[],[]

    # Set defaults for data sets:
    if sess is None:
        sess = 'all'

    sub = 0
    # Run over datasets get data + design
    dat, info, dts = ds.get_dataset(BASE_DIR, datasets,
                                atlas=atlas, sess='ses-task',
                                type=type, subj=subj, smooth=smooth)
    # dat = np.nan_to_num(dat)

    if hemis is not None:
        stru_idx = this_at.structure.index(hemis_dict[hemis])
        dat = dat[:,:,this_at.indx_full[stru_idx]]

    n_subj = dat.shape[0]

    # Find correct indices
    if cond_ind is None:
        cond_ind = dts.cond_ind
    if part_ind is None:
        part_ind = dts.part_ind
    # Make different sessions either the same or different
    if join_sess:
        sessions = dts.sessions if sess == 'all' else sess
        if part_num is None:
            indx = np.full(info[part_ind].shape, True)
        else:
            indx = (info["task_name"].isin(sessions)) & (info[part_ind].isin(part_num))

        # Check if we want to set no partition after join sessions
        if join_sess_part:
            part_vec.append(np.ones(indx.shape))
        else:
            part_vec.append(info[part_ind].values[indx].reshape(-1, ))

        # Make the zero voxels to nan
        this_dat = dat[:, indx, :]
        zero_cols = np.all(this_dat == 0, axis=1, keepdims=True)
        data.append(np.where(zero_cols, np.nan, this_dat))

        cond_vec.append(info[cond_ind].values[indx].reshape(-1, ))
        subj_ind.append(np.arange(sub, sub + n_subj))
        infos.append(info[indx])
    else:
        sessions = dts.sessions if sess == 'all' else sess
        # Now build and split across the correct sessions:
        indices = []
        for parts in part_num:
            indx = (info[part_ind] == parts) & (info["task_name"].isin(sessions))
            indices.append(indx)

        for indx in indices:
            this_dat = dat[:, indx, :]
            this_cond_vec = info[cond_ind].values[indx].reshape(-1, )
            this_part_vec = info[part_ind].values[indx].reshape(-1, )

            # QC
            # rw = rel.within_subj(np.nan_to_num(this_dat), cond_vec=this_cond_vec, part_vec=this_part_vec,
            #                      separate='condition_wise', subtract_mean=True)
            # cond_mask = np.where(rw <= 0.5, False, True)
            # cond_mask = np.hstack((cond_mask, cond_mask.copy()))
            # this_dat[~cond_mask, :] = np.nan

            # print(f'{datasets[i]} session {s} - data drop {np.sum(~cond_mask)/2} conditions from {n_subj} subjects at cut-off > 0.5' )
            # print(f'{np.sum(~cond_mask)}/{cond_mask.size} contrasts, drop rate {np.sum(~cond_mask)/cond_mask.size}')

            # Make the zero voxels to nan
            zero_cols = np.all(this_dat == 0, axis=1, keepdims=True)
            data.append(np.where(zero_cols, np.nan, this_dat))
            # indx = pt.tensor(np.where(idx == True)[0])
            # data.append(pt.index_select(dat, 1, indx))
            cond_vec.append(this_cond_vec)
            part_vec.append(this_part_vec)
            subj_ind.append(np.arange(sub, sub + n_subj))
            infos.append(info[indx])

        sub += n_subj
    return data, cond_vec, part_vec, subj_ind, infos

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
    if isinstance(maps, np.ndarray):
        maps = pt.tensor(maps, dtype=pt.get_default_dtype())

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

def greedy_task_beta_selection(maps, num_task, lamda=0.5):
    if isinstance(maps, np.ndarray):
        maps = pt.tensor(maps, dtype=pt.get_default_dtype())

    maps = maps.bool()
    best_score, task_idx = -np.inf, None
    for i in range(maps.shape[0]):
        best, cover, overlap = task_beta_selection(maps, num_task, lamda=lamda, selected=[i])
        score = cover - lamda * overlap

        if score > best_score:
            best_score = score
            task_idx = best

    print(f"Best subsets are {task_idx}, with coverage {best_score}")
    return task_idx, best_score


def greedy_select_max_trace_inverse_cov(maps, M, selected=None, z_transfer=True,
                                        eps=1e-6):
    """
    Greedy selection of task maps maximizing trace of the inverse covariance matrix.

    Parameters
    ----------
    maps : ndarray of shape (N, P)
        Task activation maps (continuous values, z-scored per map).
        N = number of tasks, P = number of vertices.
    M : int
        Number of maps to select.
    selected : list[int] or None
        Optional initial pre-selected indices.
    eps : float
        Small regularization added to covariance diagonal for numerical stability.

    Returns
    -------
    best_subset : list[int]
        Selected task map indices.
    scores : list[float]
        Trace of inverse covariance at each step.
    """
    valid_voxels = ~np.any(np.isnan(maps), axis=0)  # shape (P,)
    maps = maps[:, valid_voxels]
    if z_transfer:
        maps = (maps - maps.mean(axis=1, keepdims=True)) / maps.std(axis=1, keepdims=True)

    N, P = maps.shape
    selected = [] if selected is None else list(selected)
    scores = []

    for _ in range(M - len(selected)):
        best_idx, best_score = None, -np.inf

        for i in range(N):
            if i in selected:
                continue
            trial = selected + [i]
            X = maps[trial]  # shape: (len(trial), P)

            # compute task-task covariance (tasks × tasks)
            C = np.cov(X)  # shape: (len(trial), len(trial))

            # regularize diagonal for numerical stability
            C += eps * np.eye(len(trial))

            # compute trace of inverse covariance
            try:
                C_inv = np.linalg.inv(C)
                score = - np.trace(C_inv)
            except np.linalg.LinAlgError:
                score = -np.inf  # in case of singular matrix

            if score > best_score:
                best_idx, best_score = i, score

        selected.append(best_idx)
        scores.append(best_score)

    return sorted(selected), scores[-1]

import pickle, os
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
from global_config import MODEL_DIR, BASE_DIR, ATLAS_DIR

ERIS_DIR = '/home/dzhi/eris_mount'
if not Path(ERIS_DIR).exists():
    ERIS_DIR = '/data/tge'
if not Path(ERIS_DIR).exists():
    raise (NameError('Could not find hcp_dir'))

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

def get_kong2019_group_parcellation():
    network_names = spio.loadmat(ERIS_DIR + '/dzhi/workspace/CBIG/stable_projects/'
                                 'brain_parcellation/Kong2019_MSHBM/lib/'
                                 'group_priors/HCP_40/17network_labels.mat')['network_name']
    network_names = ['???'] + [network_names[0][i][0] for i in range(17)]

    colors = spio.loadmat(ERIS_DIR + '/dzhi/workspace/CBIG/stable_projects/'
                     'brain_parcellation/Kong2019_MSHBM/lib/'
                     'group_priors/HCP_40/group.mat')['colors']/255
    colors = colors[1:,:]
    colors = np.hstack((colors, np.ones((17, 1))))
    colors = np.vstack((np.zeros(4), colors))
    KONG2019 = nb.load(ERIS_DIR + '/dzhi/Indiv_par/Kong_2019/group_prior' \
                       '/HCP_40/Kong-2019_MSHBM_HCP40_prob_prior.dscalar.nii').get_fdata()[:]

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
                                + f'/{s}_space-{space}_{ses_id}_{type}_desc-sm{smooth}.dscalar.nii')
                else:
                    C = nb.load(my_dataset.data_dir.format(s)
                                + f'/{s}_space-{space}_{ses_id}_{type}.dscalar.nii')
                dat = C.get_fdata()
            except FileNotFoundError:
                dat = np.nan
            ses_data.append(dat)

        ref_shape = next(m.shape for m in ses_data if isinstance(m, np.ndarray))
        ses_data = [np.full(ref_shape, np.nan) if not isinstance(m, np.ndarray) else m for m in ses_data]
        ses_data = np.stack(ses_data)

        if hemis is not None:  # if cortical data
            stru_idx = this_at.structure.index(hemis_dict[hemis])
            ses_data = ses_data[:,:,this_at.indx_full[stru_idx]]
        data.append(ses_data)

    return data

def load_hcp_timeseries(dataset_dir, subj_list, space='MNIAsymC2', run_list=[0,1,2,3],
                       type='Tseries', hemis=None, smooth=None, ext=None):

    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')

    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
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


def load_hcp_contrasts(dataset_dir, subj_list, space='MNIAsymC2', sess='all',
                        type='Tseries', hemis=None, smooth=None):
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
        subj_dat, subj_info = [], []
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

            subj_info.append([s for i, s in enumerate(this_info) if i in indices])
            subj_dat.append(this_data[indices])

        Data_list.append(np.vstack(subj_dat))
        info.append(np.concat(subj_info))

    # concatenate along the first dimension (subjects)
    Data = np.stack(Data_list)
    Data[np.isinf(Data)] = np.nan

    # Assemble info file
    # Check if all arrays are identical
    if all(np.array_equal(info[0], arr) for arr in info):
        # Convert to DataFrame with a single column
        info_com = pd.DataFrame(info[0], columns=['task_name'])
    else:
        # Convert to DataFrame with multiple columns
        info_com = pd.DataFrame(info).T
        info_com.columns = [f'subj_{i + 1}' for i in range(len(info))]

    return Data, info_com


def build_msc_resting_data(dataset_dir, subj_list, this_at, ses_list='all',
                        type='Tseries', hemis=None, smooth=None, ext=None):
    # Step 1: Build the data into list of 3d tensor
    T = pd.read_csv(dataset_dir + f'/{subj_list}', sep='\t')

    data_dir = dataset_dir + '/rfMRI/fix_32k/{0}'
    hemis_dict = {'L': 'cortex_left', 'R': 'cortex_right'}

    data = []
    for i, run_id in enumerate(ses_list):
        ses_data = []
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

            if hemis is not None:  # if cortical data
                stru_idx = this_at.structure.index(hemis_dict[hemis])
                dat = dat[:, this_at.indx_full[stru_idx]]

            ses_data.append(dat)

        data.append(np.stack(ses_data))

    return data

def load_randy_contrasts(space='fs32k', ses_id='ses-s1', type=None,
                         subj=None, hemis=None, smooth=2, verbose=False):
    """Loads all the CIFTI files in the data directory of a certain space / type and returns they content as a Numpy array

    Args:
        space (str): Atlas space (Defaults to 'SUIT3').
        ses_id (str): Session ID (Defaults to 'ses-s1').
        type (str): Type of data (Defaults to 'CondHalf').
        subj (ndarray, str, or list):  Subject numbers /names to get [None = all]
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
    if type is None:
        type = dataset.default_type

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
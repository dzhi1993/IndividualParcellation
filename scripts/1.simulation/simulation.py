import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch as pt
import torchvision.transforms as transforms
import pandas as pd
import seaborn as sb
import copy, time

import HierarchBayesParcel.arrangements as ar
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import HierarchBayesParcel.spatial as sp
import HierarchBayesParcel.evaluation as ev
from FusionModel.evaluate import calc_test_dcbc


class FullModel:
    """Minimal simulation container for one arrangement and one emission model."""

    def __init__(self, arrange, emission):
        self.arrange = arrange
        self.emission = emission

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / 'results' / '1.simulation'
# pytorch cuda global flag
# pt.cuda.is_available = lambda : False
pt.set_default_tensor_type(pt.cuda.FloatTensor
                           if pt.cuda.is_available() else
                           pt.FloatTensor)


def gaussian_kernel(size, sigma=1):
    """Generate a Gaussian kernel of the given size and standard deviation"""
    size = int(size) // 2
    coords = pt.meshgrid(*[pt.arange(-size, size + 1)] * N)
    distances_sq = sum([(x ** 2) for x in coords])
    g = torch.exp(-distances_sq / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_smoothing_Nd(image):
    """Apply a 3x3 Gaussian kernel smoothing on an Nd image"""
    kernel = gaussian_kernel(3, sigma=1).unsqueeze(0).unsqueeze(0).to(image.device)
    smoothed = F.conv2d(image.unsqueeze(0), kernel.repeat(image.shape[1], 1, 1, 1), padding=1)
    return smoothed.squeeze(0)

def make_cmpRBM_data(width=10, K=5, N=20, num_subj=20, theta_mu=20,
                     theta_w=1.0, emission_model=None, do_plot=1):
    """Generates (and plots Markov random field data)
    Args:
        width (int, optional): [description]. Defaults to 10.
        K (int, optional): [description]. Defaults to 5.
        N (int, optional): [description]. Defaults to 200.
        theta_mu (int, optional): [description]. Defaults to 20.
        theta_w (int, optional): [description]. Defaults to 2.
        sigma2 (float, optional): [description]. Defaults to 0.5.
        do_plot (int): 1: Plot of the first 10 samples 2: + sample path
    """
    P = width * width
    # Step 1: Create the true model
    grid = sp.SpatialGrid(width=width, height=width)
    W = grid.get_neighbour_connectivity()
    W += pt.eye(W.shape[0])

    # Step 2: Initialize the parameters of the true model
    arrangeT = ar.cmpRBM(K, grid.P, Wc=W, theta=theta_w)
    arrangeT.name = 'cmpRBM_true'
    # 4 corners + 1 centriod
    arrangeT.bu = grid.random_smooth_pi(K=K, theta_mu=theta_mu,
                                        centroids=[0,
                                                   width - 1,
                                                   int(P / 2 + width / 2),
                                                   P - width,
                                                   P - 1])
    if emission_model is None:
        # Default sigma=0.2 for GMM
        emission_model = em.MixGaussian(5, 10, 2500)
        emission_model.sigma2 = pt.tensor(0.2)

    MT = FullModel(arrangeT, emission_model)

    # Step 3: Plot the prior of the true mode
    # grid.plot_maps(pt.softmax(arrangeT.bu, 0), cmap='jet', vmax=1, grid=[1, 5])
    # plt.show()

    # Step 4: Generate data by sampling from the above model
    U = ar.sample_multinomial(pt.softmax(arrangeT.bu, 0), shape=(num_subj, K, grid.P))
    if do_plot > 0:
        plt.figure(figsize=(20, 10))

    # Burn-in process
    for i in range(10):
        ph, H = arrangeT.sample_h(U)
        pu, U = arrangeT.sample_U(H)
        if do_plot > 0:
            u = ar.compress_mn(U)
            grid.plot_maps(u[8], cmap='tab10', vmax=K, grid=[2, 5], offset=i + 1)

    if do_plot > 0:
        plt.suptitle("B. Burn-in process")
        plt.show()

    Utrue = ar.compress_mn(U)
    MT.arrange.gibbs_U = U
    # This is the training data
    Ytrain = MT.emission.sample(Utrue)
    Ytest = MT.emission.sample(Utrue)  # Testing data

    return Ytrain, Ytest, Utrue, MT, grid

def make_train_model(model_name='cmpRBM', K=3, P=5, num_subj=20, eneg_iter=10,
                     epos_iter=10, Wc=None, theta=None, fit_W=True, fit_bu=False, lr=1):
    if model_name.startswith('idenp'):
        # 1 - Independent spatial arrangement model
        M = ar.ArrangeIndependent(K=K, P=P, spatial_specific=True,
                                  remove_redundancy=False)
        M.random_params()
        M.name = model_name
    elif model_name == 'cRBM_W':
        # Boltzmann with a arbitrary fully connected model - P hiden nodes
        n_hidden = P
        M = ar.cmpRBM(K, P, nh=n_hidden, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = f'cRBM_{n_hidden}'
        M.W = pt.randn(n_hidden, P) * 0.1
        M.alpha = lr
        M.fit_W = fit_W
        M.fit_bu = fit_bu
    elif model_name == 'cRBM_Wc':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        M = ar.cmpRBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc'
        M.fit_W = fit_W
        M.fit_bu = fit_bu
        M.alpha = lr
    elif model_name == 'cRBM_Wc_true':
        # Covolutional Boltzman machine with the true neighbourhood matrix
        # theta_w in this case is not fit.
        M = ar.cmpRBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc_true'
        M.fit_W = False
        M.fit_bu = False
        M.alpha = lr
    elif model_name == 'cRBM_Wc2':
        if Wc is None:
            raise ValueError('Wc must be provided to create wcmDBM arrangement model')

        M = ar.wcmDBM(K, P, Wc=Wc, theta=theta, eneg_iter=eneg_iter,
                      epos_iter=epos_iter, eneg_numchains=num_subj)
        M.name = 'cRBM_Wc2'
        M.fit_W = fit_W
        M.fit_bu = fit_bu
        M.alpha = lr
    else:
        raise ValueError('Unknown model name')

    return M


def train_sml(arM, emM, Ytrain, Ytest, part, crit='Ecos_err',
              n_epoch=20, batch_size=20, verbose=False):
    """Trains only arrangement model, given a fixed emission
    likelhoood.

    Args:
        arM (ArrangementMode):
        emM (EmissionModel )
        Y_train (tensor): Y_testing log likelihood (KxP)
        Y_test (tensor): Y_training log likelihood test (KxP)
        part (tensor): 1xP partition number for completion test
        crit (str): _description_. Defaults to 'logpY'.
        n_epoch (int): _description_. Defaults to 20.
        batch_size (int): _description_. Defaults to 20.
        verbose (bool): _description_. Defaults to False.

    Returns:
        model: Fitted model
        T: Pandas data frame with epoch level performance metrics
        thetaH: History of fitted thetas
    """
    emlog_train = emM.Estep(Ytrain)
    emlog_test = emM.Estep(Ytest)
    num_subj = emlog_train.shape[0]
    Utrain = pt.softmax(emlog_train, dim=1)

    crit_types = ['train', 'marg', 'test']  # different evaluation types
    CR = np.zeros((len(crit_types), n_epoch))
    theta_list = pt.zeros((arM.nparams, n_epoch))
    marginals = pt.zeros((n_epoch, arM.K, arM.P))
    CE = pt.zeros((n_epoch,))
    # Intialize negative sampling
    for epoch in range(n_epoch):
        # Get test error
        EU, _ = arM.Estep(emlog_train, gather_ss=False)
        for i, ct in enumerate(crit_types):
            # Training emission logliklihood:
            if ct == 'train':
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytrain, EU, crit=crit)
            elif ct == 'marg':
                pi = arM.marginal_prob()
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytest, pi, crit=crit)
            elif ct == 'test':
                CR[i, epoch] = ev.evaluate_full_arr(emM, Ytest, EU, crit=crit)
            elif ct == 'compl':
                CR[i, epoch] = ev.evaluate_completion_arr(arM, emM, Ytest, part, crit=crit)
        if (verbose):
            print(f'epoch {epoch:2d} Test: {crit[2, epoch]:.4f}')

        theta_list[:, epoch] = arM.get_params()
        marginals[epoch, :, :] = arM.marginal_prob()
        # Update the model in batches
        for b in range(0, num_subj - batch_size + 1, batch_size):
            ind = range(b, b + batch_size)
            arM.Estep(emlog_train[ind, :, :])
            if hasattr(arM, 'Eneg'):
                arM.Eneg(use_chains=ind,
                         emission_model=emM)
            arM.Mstep()

        # Record the cross entropy parameters
        if arM.name.startswith('idenp'):
            CE[epoch] = 0
        else:
            CE[epoch] = ev.cross_entropy(pt.softmax(emlog_train, dim=1),
                                         arM.eneg_U)
            # CE[epoch] = pt.abs(pt.softmax(emlog_train, dim=1) - arM.eneg_U).sum()

    # Make a data frame for the results
    T = pd.DataFrame()
    for i, ct in enumerate(crit_types):
        T1 = pd.DataFrame({'model': [arM.name] * n_epoch,
                           'type': [ct] * n_epoch,
                           'iter': np.arange(n_epoch),
                           'crit': CR[i]})
        T = pd.concat([T, T1], ignore_index=True)

    return arM, T, theta_list, CE, marginals


def eval_dcbc(models, emM, Ytrain, Ytest, grid, Utrue_group, Utrue_indiv, SD,
              max_dist=10, bin_width=1):
    D = pd.DataFrame()
    group_par, indiv_par = [], []
    nsubj = Utrue_indiv.shape[0]

    for m in models:
        smooth = 0
        if isinstance(m, str) and m.startswith('data'):
            if m == 'data':
                ind = 0
            else:
                ind = int(m.split('_')[1])
            emloglik_train = emM.Estep(Ytrain[ind])
            this_Ugroup = pt.softmax(emloglik_train.sum(dim=0), dim=0).argmax(dim=0)
            this_Uindiv = pt.softmax(emloglik_train, 1).argmax(dim=1)
            name = m
            smooth = SD[ind]
            model_type = 'data'
        elif m == 'Utrue':
            this_Ugroup = Utrue_group
            this_Uindiv = Utrue_indiv
            name = m
            model_type = 'true'
        else:
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                this_Ugroup = m.marginal_prob().argmax(dim=0)
                this_Uindiv = m.estep_Uhat.argmax(dim=1)
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                this_Ugroup = pt.softmax(m.bu, dim=0).argmax(dim=0)
                this_Uindiv = m.epos_Uhat.argmax(dim=1)
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name

        dcbc_group = calc_test_dcbc(this_Ugroup, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)
        dcbc_indiv = calc_test_dcbc(this_Uindiv, Ytest, grid.Dist,
                                    max_dist=int(max_dist), bin_width=bin_width)

        group_par.append(this_Ugroup)
        indiv_par.append(this_Uindiv)

        dict = {'model': [name] * nsubj,
                'type': ['test'] * nsubj,
                'smooth': [smooth] * nsubj,
                'arrangement': [model_type] * nsubj,
                'dcbc_group': dcbc_group.cpu(),
                'dcbc_indiv': dcbc_indiv.cpu()}
        D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)

    return D, group_par, indiv_par


def eval_arrange(models, emM, Ytrain, Ytest, SD, Utrue):
    D = pd.DataFrame()
    Utrue_mn = ar.expand_mn(Utrue, emM.K)
    nsubj = Utrue.shape[0]

    for m in models:
        smooth = 0
        if isinstance(m, str) and m.startswith('data'):
            if m == 'data':
                ind = 0
            else:
                ind = int(m.split('_')[1])
            emloglik_train = emM.Estep(Ytrain[ind])
            EU = pt.softmax(emloglik_train, 1)
            smooth = SD[ind]
            name = m
            model_type = 'data'
        elif m == 'Utrue':
            EU = Utrue_mn
            name = m
            model_type = 'true'
        else:
            # EU,_ = m.Estep(emloglik_train, gather_ss=False)
            if m.name.startswith('idenp'):
                EU = m.estep_Uhat
                smooth = float(m.name.split('_')[1])
                model_type = 'idenp'
            elif m.name.startswith('cRBM'):
                EU = m.epos_Uhat
                model_type = 'cRBM'
            else:
                raise NameError('Unknown model name')
            name = m.name
        uerr_test1 = pt.mean(pt.abs(Utrue_mn - EU), dim=(1, 2)).cpu()
        cos_err = ev.coserr(Ytest, emM.V, EU, adjusted=False,
                            soft_assign=False).cpu()
        Ecos_err = ev.coserr(Ytest, emM.V, EU, adjusted=False,
                             soft_assign=True).cpu()

        dict = {'model': [name] * nsubj,
                'type': ['test'] * nsubj,
                'smooth': [smooth] * nsubj,
                'arrangement': [model_type] * nsubj,
                'uerr': uerr_test1,
                'cos_err': cos_err,
                'Ecos_err': Ecos_err}
        D = pd.concat([D, pd.DataFrame(dict)], ignore_index=True)
    return D



def plot_Uhat_maps(models, emloglik, grid):
    plt.figure(figsize=(10, 7))
    n_models = len(models)
    K = emloglik.shape[1]
    for i, m in enumerate(models):
        if m is None:
            Uh = pt.softmax(emloglik, dim=1)
        else:
            Uh, _ = m.Estep(emloglik)
        grid.plot_maps(Uh[0], cmap='jet', vmax=1, grid=(n_models, K), offset=i * K + 1)


def plot_P_maps(pmaps, grid):
    n_models = len(pmaps)
    K = pmaps[0].shape[0]

    plt.figure(figsize=(K * 3, n_models * 3))
    for i, m in enumerate(pmaps):
        grid.plot_maps(m, cmap='jet', vmax=1, grid=(n_models, K), offset=i * K + 1)

    plt.show()


def plot_U_maps(pmaps, grid, title):
    n_models = len(pmaps)

    plt.figure(figsize=(n_models * 3, 4))
    for i, m in enumerate(pmaps):
        grid.plot_maps(m, cmap='tab20', vmax=19, grid=(1, n_models), offset=i + 1)
        plt.title(title[i])

    plt.show()


def plot_individual_Uhat(models, Utrue, emloglik, grid, style='prob'):
    # Get the expectation
    n_models = len(models) + 2
    K = emloglik.shape[1]
    P = emloglik.shape[2]

    Uh = []
    height = 2 if style == 'mixed' else 1
    plt.figure(figsize=(n_models * 3, height * 4))

    # Uh order: data -> models -> Utrue
    Uh.append(ar.expand_mn(Utrue[0:1], K))
    Uh.append(pt.softmax(emloglik[0:1], dim=1))
    for i, m in enumerate(models):
        A, _ = m.Estep(emloglik[0:1])
        Uh.append(A)

    if style == 'prob':
        for i, uh in enumerate(Uh):
            grid.plot_maps(uh[0], cmap='jet', vmax=1,
                           grid=(n_models, K),
                           offset=K * i + 1)
    elif style == 'argmax':
        ArgM = pt.zeros(n_models, P)
        for i, uh in enumerate(Uh):
            ArgM[i, :] = pt.argmax(uh[0], dim=0)
        grid.plot_maps(ArgM, cmap='tab10', vmax=K,
                       grid=(1, n_models))
    elif style == 'mixed':
        ArgM = pt.zeros(n_models, P)
        Prob = pt.zeros(n_models, P)

        for i, uh in enumerate(Uh):
            ArgM[i, :] = pt.argmax(uh[0], dim=0)
            Prob[i, :] = uh[0][2, :]
        grid.plot_maps(ArgM, cmap='tab10', vmax=K,
                       grid=(2, n_models))
        grid.plot_maps(Prob, cmap='jet', vmax=1,
                       grid=(2, n_models),
                       offset=n_models + 1)

    plt.show()


def plot_evaluation(D, criteria=['uerr', 'cos_err', 'Ecos_err', 'dcbc_group', 'dcbc_indiv'],
                    types=['test', 'compl']):
    def plot_metric_by_smooth(frame, metric, label=None):
        if frame.empty:
            return
        stats = (frame.groupby('smooth', as_index=True)[metric]
                 .agg(['mean', 'sem'])
                 .sort_index())
        plt.errorbar(stats.index.to_numpy(),
                     stats['mean'].to_numpy(),
                     yerr=stats['sem'].fillna(0).to_numpy(),
                     label=label,
                     capsize=3)

    # Get the final error and the true pott models
    ncrit = len(criteria)
    ntypes = len(types)
    plt.figure(figsize=(5 * ncrit, 5 * ntypes))
    for j in range(ntypes):
        for i in range(ncrit):
            plt.subplot(ntypes, ncrit, i + j * ncrit + 1)
            # sb.barplot(data=D[D.type==types[j]], x='model', y=criteria[i])

            df = D[(D.type == types[j]) & (D.arrangement == 'idenp')]
            plot_metric_by_smooth(df, criteria[i], label='idenp')

            emlog = D[(D.type == types[j]) & (D.arrangement == 'data')]
            if not emlog.empty:
                plt.axhline(emlog[criteria[i]].mean().item(), color='k', ls=':',
                            label='data')
                plot_metric_by_smooth(emlog, criteria[i])

            rbm_wc = D[(D.type == types[j]) & (D.model == 'cRBM_Wc')]
            if not rbm_wc.empty:
                plt.axhline(rbm_wc[criteria[i]].mean().item(), color='r', ls=':',
                            label='cRBM_Wc')

            rbm_wc = D[(D.type == types[j]) & (D.model == 'Utrue')]
            if not rbm_wc.empty:
                plt.axhline(rbm_wc[criteria[i]].mean().item(), color='b', ls=':',
                            label='Utrue')

            plt.title(f'{criteria[i]}{types[j]}')
            plt.legend()
            # plt.xticks(rotation=45)

    plt.suptitle(f'final errors')
    plt.tight_layout()

    plt.savefig(RESULT_DIR / 'test_errs.pdf', format='pdf')
    plt.show()





def simulation(K=5, width=50, num_subj=30, batch_size=30, n_epoch=200, theta=1.5,
                 theta_mu=180, emission='gmm', epos_iter=20, eneg_iter=20, num_sim=10):
    P = width * width
    if emission == 'gmm':  # MixGaussian
        sigma2 = 0.2
        N = 10
        emissionM = em.MixGaussian(K, N, P)
        emissionM.sigma2 = pt.tensor(sigma2)
    elif emission == 'mn':  # Multinomial
        w = 2.0
        emissionM = em.MultiNomial(K=K, P=P)
        emissionM.w = pt.tensor(w)

    # Record the results
    TT = pd.DataFrame()
    DD = pd.DataFrame()
    HH = pt.zeros((num_sim, n_epoch))
    BU_all = pt.zeros((num_sim, n_epoch))
    BU_all_1 = pt.zeros((num_sim, n_epoch))
    BU_all_2 = pt.zeros((num_sim, n_epoch))
    BU_all_3 = pt.zeros((num_sim, n_epoch))
    CE_rbm1 = pt.zeros((num_sim, n_epoch))
    CE_rbm2 = pt.zeros((num_sim, n_epoch))
    GM, IM, BUs = [], [], []

    # REcorded bias parameter
    # SD = np.concatenate((np.linspace(0.1,1,10), np.linspace(1.5,3,4)))
    # SD = np.round(SD, decimals=2)
    SD = [0.5]
    Rec = pt.zeros((len(SD) + 4, num_sim, K, P))  # unsmooth + 2 rbms + 1 emloglik

    # Generate partitions for region-completion testing
    num_part = 4
    p = pt.ones(num_part) / num_part
    part = pt.multinomial(p, P, replacement=True)
    for s in range(num_sim):
        start = time.perf_counter()
        Ytrain, Ytest, Utrue, Mtrue, grid = make_cmpRBM_data(width, K, N=N,
                                                             num_subj=num_subj, theta_mu=theta_mu,
                                                             theta_w=theta, emission_model=emissionM,
                                                             do_plot=0)

        # Get the smoothed training data
        Ytrain_smooth = []
        for smooth in SD:
            blur_transform = transforms.GaussianBlur(kernel_size=5, sigma=smooth)
            Ys = blur_transform(Ytrain.view(Ytrain.shape[0], -1, width, width))
            Ys = Ys.view(Ytrain.shape[0], Ytrain.shape[1], -1)
            Ytrain_smooth.append(Ys)

        emloglik_train = Mtrue.emission.Estep(Ytrain)
        emloglik_test = Mtrue.emission.Estep(Ytest)
        P = Mtrue.emission.P

        # Get the true arrangement model and its loglik
        rbm = Mtrue.arrange
        rbm.name = 'true'

        # Make list of fitting models
        Models, fitted_M = [], []
        fitting_names = ['idenp_0'] + [f'idenp_{s}' for s in SD] + ['cRBM_Wc', 'cRBM_Wc_true']
        Y_fit = [Ytrain] + Ytrain_smooth + [Ytrain, Ytrain]
        for nam in fitting_names:
            Models.append(make_train_model(model_name=nam, K=K, P=P,
                                           num_subj=num_subj, eneg_iter=eneg_iter,
                                           epos_iter=epos_iter, Wc=rbm.Wc.squeeze(2),
                                           theta=None,
                                           fit_W=True, fit_bu=True, lr=0.5))

        Models[-1].bu = rbm.bu.detach().clone()
        Models[-1].theta = rbm.theta
        # Train different arrangement model
        TH, CE, MG = [], [], []
        T = pd.DataFrame()
        for i, m in enumerate(Models):
            # Give the model the true bias/W for rbms
            if m.name.startswith('cRBM') or m.name.startswith('wcmDBM'):
                # m.W = rbm.W.detach().clone()
                # m.bu = rbm.bu.detach().clone()
                pass

            m, T1, theta_hist, ce, marginals = train_sml(m, Mtrue.emission, Y_fit[i],
                                                         Ytest, part, batch_size=batch_size,
                                                         n_epoch=n_epoch)
            fitted_M.append(m)
            TH.append(theta_hist)
            CE.append(ce)
            MG.append(marginals)
            T = pd.concat([T, T1], ignore_index=True)

        # Evaluate overall
        # 1. u_absolute error, cos_err, and expected cos_err
        D = eval_arrange(['data'] + fitted_M + ['Utrue'],
                         Mtrue.emission, [Ytrain], Ytest, np.insert(SD, 0, 0),
                         Utrue=Utrue)

        # 2. DCBC
        binWidth = 5
        max_dist = binWidth * pt.ceil(grid.Dist.max() / binWidth)
        D1, group_map, indiv_map = eval_dcbc(['data']
                                             + fitted_M + ['Utrue'],
                                             Mtrue.emission,
                                             [Ytrain], Ytest, grid,
                                             pt.softmax(rbm.bu, dim=0).argmax(dim=0),
                                             Utrue, np.insert(SD, 0, 0),
                                             max_dist=max_dist, bin_width=binWidth)

        GM.append(group_map)
        IM.append(indiv_map)
        res = pd.concat([D, D1.iloc[:, 4:]], axis=1)

        res['sim'] = s
        DD = pd.concat([DD, res], ignore_index=True)
        TT = pd.concat([TT, T], ignore_index=True)

        # Record the theta for rbm_Wc model only
        HH[s, :] = TH[-1][fitted_M[-1].get_param_indices('theta'), :]
        # Record the distance measure of |bias - true bu|
        fit_bu = TH[-1][fitted_M[-1].get_param_indices('bu'), :]
        fit_bu = fit_bu.T.view(-1, rbm.bu.shape[0], rbm.bu.shape[1])
        for counter in range(fit_bu.shape[0]):
            # 2. marginals L2-norm
            BU_all_1[s, counter] = pt.norm(rbm.marginal_prob() - MG[-1][counter], p=2)
            BU_all_2[s, counter] = pt.norm(rbm.marginal_prob() - MG[0][counter], p=2)
            BU_all_3[s, counter] = pt.norm(MG[0][counter] - MG[-1][counter], p=2)

            # 3. BU L2-norm
            this_fb = fit_bu - fit_bu.mean(dim=1, keepdim=True)
            this_bu = rbm.bu - rbm.bu.mean(dim=0, keepdim=True)
            BU_all[s, counter] = pt.norm(this_fb[counter, :, :] - this_bu, p=2)

        # Record cross entropy for rbms
        CE_rbm1[s, :] = CE[-2]
        CE_rbm2[s, :] = CE[-1]
        BUs.append(fit_bu)

        # record the different fitting runs into structure
        Rec[0, s, :, :] = pt.softmax(emloglik_train, 1).mean(dim=0)  # first is data
        for j, fm in enumerate(fitted_M):
            if fm.name.startswith('idenp'):
                Rec[j + 1, s, :, :] = pt.softmax(fm.logpi, 0)
            elif fm.name.startswith('cRBM'):
                Rec[j + 1, s, :, :] = pt.softmax(fm.bu, 0)
            else:
                raise ValueError('Unknown model name')
        # Rec[-1,s,:,:] = ar.expand_mn(Utrue, K).mean(dim=0)

        finish = time.perf_counter()
        elapse = time.strftime('%H:%M:%S', time.gmtime(finish - start))
        print(f"   Done - time {elapse}")

    # Plot learning curves by epoch
    fig = plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(HH.T.cpu().numpy())
    plt.axhline(y=HH[:, -1].cpu().numpy().mean(), color='r', linestyle='-')
    plt.axhline(y=theta, color='k', linestyle='-')
    plt.ylabel('Theta')
    plt.subplot(1, 3, 2)
    plt.plot(BU_all.T.cpu().numpy())
    plt.ylabel('L2-norm - |bu - true bu|')
    plt.subplot(1, 3, 3)
    plt.plot(BU_all_1.T.cpu().numpy())
    plt.axhline(y=BU_all_2[:, -1].cpu().numpy().mean(), color='r', linestyle='-')
    plt.axhline(y=0, color='k', linestyle=':')
    plt.ylabel('L2-norm - marginals')

    plt.tight_layout()
    # plt.savefig(result_path('learning_curves_2.pdf'), format='pdf')
    plt.show()

    return grid, DD, Rec, rbm, fitted_M, Utrue, emloglik_train, GM, IM



if __name__ == '__main__':
    ##------------------Supplementary Figure 1------------------##
    TM = [60, 240, 680]
    TH = [0.2, 0.5, 1, 1.5, 5]
    prior, samples = [],[]
    for i, theta_mu in enumerate(TM):
        for j, theta in enumerate(TH):
            _, _, Utrue, Mtrue, grid = make_cmpRBM_data(50, 5, N=10,
                                                        num_subj=10,
                                                        theta_mu=theta_mu,
                                                        theta_w=theta,
                                                        emission_model=None,
                                                        do_plot=0)
            samples.append(Utrue[0])
        prior.append(pt.softmax(Mtrue.arrange.bu, 0))

    ##-- Supp Fig.1 panel a
    plt.figure(figsize=(15, 5))
    grid.plot_maps(pt.vstack(prior), cmap='jet', vmax=1, grid=[3, 5])
    plt.suptitle('A. Group-level priors for the 5 parcels')
    # plt.savefig('true_maps.pdf', format='pdf')
    plt.show()

    ##-- Supp Fig.1 panel c
    plt.figure(figsize=(15, 5))
    grid.plot_maps(pt.stack(samples), cmap='tab10', vmax=5, grid=[3, 5])
    plt.suptitle('C. Example individual parcellation in different settings')
    # plt.savefig('true_maps.pdf', format='pdf')
    plt.show()

    ##-- Supp Fig.1 panel b and d
    num_subj = 10
    _, _, Utrue, MT, grid = make_cmpRBM_data(50, 5, N=num_subj,
                                        num_subj=10,
                                        theta_mu=240,
                                        theta_w=1.2,
                                        emission_model=None,
                                        do_plot=1)
    U_ind = ar.sample_multinomial(MT.arrange.marginal_prob(), shape=(num_subj, 5, grid.P))
    grid.plot_maps(Utrue, cmap='tab10', vmax=5, grid=[2, 5])
    plt.suptitle('D. Example 10 individual maps')
    # plt.savefig('true_maps.pdf', format='pdf')
    plt.show()

    ##------------------Supplementary Figure 2------------------##
    # This is to simulate 100 times independently
    grid, DD, _, _, Models, Utrue, emloglik_train, _, _ = simulation(theta_mu=240,
                                                                       num_sim=1)

    # Supp Fig. 2a
    plot_individual_Uhat(Models[:-1], Utrue[0:1], emloglik_train[0:1],
                         grid, style='mixed')

    # Supp Fig. 2b,c,d
    output_file = RESULT_DIR / 'eval_cpmRBM_fit.tsv'
    # DD.to_csv(output_file, index=False, sep='\t')
    DD = pd.read_csv(output_file, delimiter='\t')
    plot_evaluation(DD, types=['test'])

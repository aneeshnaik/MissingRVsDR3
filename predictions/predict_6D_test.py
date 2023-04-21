#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guess RVs for DR3 6D test set.

Created: April 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
from batchgauss import sample
from banyan import BNN
from tqdm import trange

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
import src.utils as u
import src.params as p
from src.coords import convert_pos


def get_obs_cov(df):

    # number of stars
    N = len(df)

    # obs array
    obs = np.stack(
        [df['ra'], df['dec'],
         df['pmra'], df['pmdec'], df['v_los']],
        axis=-1
    )

    # standard deviations
    s0 = (df['ra_err']).to_numpy()
    s1 = (df['dec_err']).to_numpy()
    s2 = (df['pmra_err']).to_numpy()
    s3 = (df['pmdec_err']).to_numpy()
    s4 = (df['v_los_err']).to_numpy()
    s_list = [s0, s1, s2, s3, s4]

    # construct correlation matrix
    r = np.zeros((N, 5, 5))
    for i in range(5):
        r[:, i, i] = 1
    r[:, 0, 1] = r[:, 1, 0] = (df['ra_dec_corr']).to_numpy()
    r[:, 0, 2] = r[:, 2, 0] = (df['ra_pmra_corr']).to_numpy()
    r[:, 0, 3] = r[:, 3, 0] = (df['ra_pmdec_corr']).to_numpy()
    r[:, 1, 2] = r[:, 2, 1] = (df['dec_pmra_corr']).to_numpy()
    r[:, 1, 3] = r[:, 3, 1] = (df['dec_pmdec_corr']).to_numpy()
    r[:, 2, 3] = r[:, 3, 2] = (df['pmra_pmdec_corr']).to_numpy()

    # covariance matrix
    cov = np.zeros_like(r)
    for i in range(5):
        for j in range(5):
            si = s_list[i]
            sj = s_list[j]
            cov[:, i, j] = r[:, i, j] * si * sj

    return obs, cov


def get_loader(d_matrix, obs, cov, rng, N_batch, x_mu, x_sig, y_mu, y_sig):

    # sample distance from distance matrix
    N = len(d_matrix)
    q = d_matrix[np.arange(N), rng.integers(low=0, high=10, size=N)]

    # sample ra/dec/pmra/pmdec/v_los from covariance matrix, stack with d
    q = np.hstack((q[:, None], sample(means=obs, covs=cov, rng=rng)))

    # convert ra/dec/d to X/Y/Z (note d = 1/parallax)
    X, Y, Z = convert_pos(q[:, 1], q[:, 2], q[:, 0]).T

    # create torch tensors
    x = torch.tensor(np.stack([X, Y, Z, q[:, 3], q[:, 4]], axis=-1)).double()
    y = torch.tensor(q[:, 5])[:, None].double()

    # rescale units
    x = u.rescale_data(x, mu=x_mu, sig=x_sig)
    y = u.rescale_data(y, mu=y_mu, sig=y_sig)

    # construct torch loader
    dset = TensorDataset(x, y)
    loader = DataLoader(dset, batch_size=N_batch)
    return loader


if __name__ == "__main__":

    N_samples_per_model = 50
    N_ensemble = 16

    # find GPU, otherwise use CPU
    print("Searching for GPU:", flush=True)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU not found. Using CPU:", flush=True)
        device = torch.device("cpu")
    print(">>>Done.\n", flush=True)

    # initialise RNG
    print("Initialising random number generator:", flush=True)
    rng = np.random.default_rng(42)
    print(">>>Done.\n", flush=True)

    # load data
    ddir = u.get_datadir()
    print(f"Found data directory: {ddir} || Loading data:", flush=True)
    df = pd.read_hdf(ddir + "DR3_6D/test.hdf5")
    print(">>>Done.\n", flush=True)

    # get truths
    v_los_true = df['v_los'].to_numpy()
    v_los_err = df['v_los_err'].to_numpy()

    # construct distance matrix
    print("Constructing distance matrix:", flush=True)
    d_matrix = np.array([df[f'd{i}'] for i in range(10)]).T
    print(">>>Done.\n", flush=True)

    # construct covariance matrix
    print("Constructing covariance matrices:", flush=True)
    obs, cov = get_obs_cov(df)
    print(">>>Done.\n", flush=True)

    # construct models
    print("Constructing BNN ensemble:", flush=True)
    models = []
    for i in range(16):
        model = BNN(N_in=5, N_out=1, N_hidden=p.N_hidden, N_units=p.N_units)
        model.to(device)
        model.double()
        models.append(model)
    print(">>>Done.\n", flush=True)

    # construct models
    print("Reading trained models:", flush=True)
    for i in range(16):
        models[i].load(f"../models/{i}.pth", device)
    print(">>>Done.\n", flush=True)

    # data loader args
    largs = dict(
        x_mu=p.x_mu, x_sig=p.x_sig, y_mu=p.y_mu, y_sig=p.y_sig,
        N_batch=100000, rng=rng
    )

    # loop over samples
    print("Commencing loop:", flush=True)
    y = torch.zeros((len(df), 16, N_samples_per_model))
    tr = trange(N_samples_per_model)
    for i in tr:

        # get loader
        loader = get_loader(d_matrix, obs, cov, **largs)

        # get preds
        model.eval()
        with torch.no_grad():
            filled = 0
            for j, (x, _) in enumerate(loader):
                x = x.to(device)
                for k in range(16):
                    y[filled:filled + len(x), k, i] = models[k](x, 1).squeeze().to('cpu')
                filled += len(x)
    print(">>>Done.\n", flush=True)

    # reshape, rescale and convert to numpy
    print("Reshaping and rescaling predictions:", flush=True)
    y = y.reshape((len(df), N_ensemble * N_samples_per_model))
    v_los_preds = (p.y_sig * y.cpu() + p.y_mu).detach().numpy().squeeze()
    print(">>>Done.\n", flush=True)

    # save
    print("Saving:", flush=True)
    np.savez(
        ddir + "DR3_predictions/6D_test.npz",
        v_los_preds=v_los_preds,
        v_los_true=v_los_true,
        v_los_err=v_los_err
    )
    print(">>>Done.\n", flush=True)

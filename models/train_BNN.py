#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train BNN on Gaia DR3.

Created: August 2022
Author: A. P. Naik
"""
import sys
import pandas as pd
import numpy as np
from time import perf_counter as time
from tqdm import trange
from batchgauss import sample

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLR

sys.path.append("..")
import src.utils as u
import src.params as p
from src.ml import train_epoch as train
from src.bnn import BNN
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
    loader = u.construct_data_loader(x, y, N_batch)
    return loader


if __name__ == "__main__":

    # script arg is random seed
    assert len(sys.argv) == 2
    seed = int(sys.argv[1])

    # start message
    print("\n", flush=True)
    print("%%%%%%%%%%%%%%%%", flush=True)
    print(f"Training a BNN with random seed {seed}.", flush=True)
    print("%%%%%%%%%%%%%%%%\n", flush=True)

    # data directory
    ddir = u.get_datadir()
    print(f"Found data directory: {ddir}\n", flush=True)

    # various run parameters
    datafile = ddir + "train.hdf5"
    Nh = p.N_hidden
    Nu = p.N_units
    Ns = p.N_samples
    Nb = p.N_batch
    x_mu = p.x_mu
    x_sig = p.x_sig
    y_mu = p.y_mu
    y_sig = p.y_sig
    device = 'gpu'
    print("Specified parameters:\n"
          f"N_hidden = {Nh}\n"
          f"N_units = {Nu}\n"
          f"N_samples = {Ns}\n"
          f"N_batch = {Nb}\n"
          f"Training data file: {datafile}\n"
          f"x_mu = {x_mu}\n"
          f"x_sig = {x_sig}\n"
          f"y_mu = {y_mu}\n"
          f"y_sig = {y_sig}\n"
          f"Device: {device}\n", flush=True)

    # set device
    device = u.find_torch_device(device, verbose=True)

    # initialise random number generator
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # load datasets
    print("Loading data:", flush=True)
    df = pd.read_hdf(datafile)
    print(">>>Done.\n", flush=True)

    # construct distance matrix
    print("Constructing distance matrix:", flush=True)
    d_matrix = np.array([df[f'd{i}'] for i in range(10)]).T
    print(">>>Done.\n", flush=True)

    # construct covariance matrix
    print("Constructing covariance matrices:", flush=True)
    obs, cov = get_obs_cov(df)
    print(">>>Done.\n", flush=True)

    # set up model
    print("Constructing BNN:", flush=True)
    print(">>>Setting up architecture:", flush=True)
    model = BNN(N_in=5, N_out=1, N_hidden=Nh, N_units=Nu)
    print(f"{model}", flush=True)
    print(">>>Moving model to requested device", flush=True)
    model.to(device)
    print(">>>Switching to double precision", flush=True)
    model.double()
    print(">>>Done.\n", flush=True)

    # loader args
    largs = dict(
        x_mu=x_mu, x_sig=x_sig, y_mu=y_mu, y_sig=y_sig,
        N_batch=p.N_batch, rng=rng
    )

    # set up other training objects
    optim = Adam(model.parameters(), lr=p.lr0)
    scheduler = ReduceLR(optim, factor=p.lr_fac, min_lr=p.min_lr)

    # training loop
    print("Commencing training:", flush=True)
    training_data = []
    tr = trange(p.N_epochs_max)
    for i in tr:

        # start stopclock
        t0 = time()

        # get loaders
        loader = get_loader(d_matrix, obs, cov, **largs)

        # train
        model.train()
        loss, lr = train(model, device, loader, optim, scheduler, Ns)

        # stop stopclock
        t1 = time()
        t = t1 - t0

        # store training data
        row = {
            'loss': loss,
            'lr': lr,
            't': t
        }
        training_data.append(row)

        # add loss and learning rate to progress bar
        s = f'loss={loss:.4f}, lr={lr:.4e}'
        tr.set_postfix_str(s)

        # end early if min lr reached
        if lr <= p.min_lr:
            break

    # save
    print("Training complete. Saving...", flush=True)
    model.save(f'{seed}.pth')
    df = pd.DataFrame(training_data)
    df.to_csv(f'{seed}_training_data.csv')
    print("Done.", flush=True)

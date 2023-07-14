#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intermediate script for generating v_los map: loop over prediction files and
get X/Y coordinates for each star along with single predicted v_los, save whole
thing to file (which can be loaded to make plot).

Created: May 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
from tqdm import trange
from os.path import exists

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_pos
from src.gmm import gmm_batchsample


def data_cut(df, X_min, X_max, Y_min, Y_max, f_cut, sig_cut=None):

    # get distances and errors
    d_matrix = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = np.mean(d_matrix, axis=-1)
    f = np.std(d_matrix, axis=-1) / d

    # rv error
    sig = (0.5 * (df['q841'] - df['q159'])).to_numpy()

    # convert coords
    X, Y, Z = convert_pos(df['ra'], df['dec'], d).T

    # quality cut
    if sig_cut is None:
        m = (f < f_cut) \
            & (X > X_min) & (X < X_max) \
            & (Y > Y_min) & (Y < Y_max)
    else:
        m = (f < f_cut) & (sig < sig_cut) \
            & (X > X_min) & (X < X_max) \
            & (Y > Y_min) & (Y < Y_max)
    return df[m]


def get_XYV_6D(X_min, X_max, Y_min, Y_max, f_cut):

    # load data, merge w/ GMM catalogue
    ddir = get_datadir()
    gaia_file = ddir + "DR3_6D/test.hdf5"
    gmm_file = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    df = pd.read_hdf(gaia_file)
    df = df.merge(pd.read_hdf(gmm_file), on='source_id')

    # quality cut
    df = data_cut(df, X_min, X_max, Y_min, Y_max, f_cut)

    # get distances and errors
    d_matrix = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = np.mean(d_matrix, axis=-1)

    # convert coordinates
    X, Y, Z = convert_pos(df['ra'], df['dec'], d).T

    # reduce precision
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    V = df['v_los'].to_numpy()
    return X, Y, V


def get_XYV_5D(i, X_min, X_max, Y_min, Y_max, f_cut, sig_cut):

    # load data, merge w/ GMM catalogue
    ddir = get_datadir()
    gaia_file = ddir + f"DR3_5D/{i}.hdf5"
    gmm_file = ddir + f"DR3_predictions/5D_{i}_GMM_catalogue.hdf5"
    df = pd.read_hdf(gaia_file)
    df = df.merge(pd.read_hdf(gmm_file), on='source_id')

    # quality cut
    df = data_cut(df, X_min, X_max, Y_min, Y_max, f_cut, sig_cut)

    # get distances and errors
    d_matrix = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = np.mean(d_matrix, axis=-1)

    # convert coordinates
    X, Y, Z = convert_pos(df['ra'], df['dec'], d).T

    # GMM params
    means = np.stack([df[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([df[f'var{i}'] for i in range(4)], axis=-1)
    weights = np.stack([df[f'w{i}'] for i in range(4)], axis=-1)

    # sample vel
    V = gmm_batchsample(weights, means, variances)

    # reduce precision
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    V = V.astype(np.float32)

    return X, Y, V


if __name__ == "__main__":

    # params
    X_min = -17
    X_max = 1
    Y_min = -9
    Y_max = 9
    f_cut = 0.25   # fractional distance error cut
    sig_cut = 80   # posterior width cut, only on 5D set

    # get 6D data
    print("Getting 6D data:", flush=True)
    X_6D, Y_6D, V_6D = get_XYV_6D(X_min, X_max, Y_min, Y_max, f_cut)
    print(">>>Done.\n", flush=True)

    # loop over 5D datasets
    X_5D = np.array([], dtype=np.float32)
    Y_5D = np.array([], dtype=np.float32)
    V_5D = np.array([], dtype=np.float32)
    print("Getting 5D data:", flush=True)
    for i in trange(32):
        X, Y, V = get_XYV_5D(i, X_min, X_max, Y_min, Y_max, f_cut, sig_cut)
        X_5D = np.append(X_5D, X)
        Y_5D = np.append(Y_5D, Y)
        V_5D = np.append(V_5D, V)
    print(">>>Done.\n", flush=True)

    # save
    print("Saving:", flush=True)
    np.savez(
        get_datadir() + 'figures/fig8_median_vlos_map_XYV_data.npz',
        X_6D=X_6D, Y_6D=Y_6D, V_6D=V_6D,
        X_5D=X_5D, Y_5D=Y_5D, V_5D=V_5D,
    )
    print(">>>Done.\n", flush=True)

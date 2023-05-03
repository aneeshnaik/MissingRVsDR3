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


def get_XYV_6D():

    # load DR3 data
    ddir = get_datadir()
    cols = ['radial_velocity', 'ra', 'dec']
    df = pd.read_csv(ddir + 'DR3_6D/DR3_6D.csv', usecols=cols)

    # read distance samples
    d = np.load(ddir + 'DR3_6D/distance_samples.npy')[:, 0]

    # convert coordinates
    X, Y, Z = convert_pos(df['ra'], df['dec'], d).T

    # reduce precision
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    V = df['radial_velocity'].to_numpy().astype(np.float32)

    return X, Y, V


def get_XYV_5D(i):

    # load Gaia data
    ddir = get_datadir()
    df = pd.read_hdf(ddir + f"DR3_5D/{i}.hdf5")
    N = len(df)

    # convert coordinates
    X, Y, Z = convert_pos(df['ra'], df['dec'], df['d0']).T

    # load predictions
    preds = np.load(ddir + f"DR3_predictions/5D_{i}.npy")
    assert len(preds) == N

    # random column for each row
    indices = np.random.randint(0, preds.shape[1], size=N)
    V = preds[np.arange(N), indices]

    # reduce precision
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    V = V.astype(np.float32)

    return X, Y, V


if __name__ == "__main__":

    # get 6D data
    print("Getting 6D data:", flush=True)
    X_6D, Y_6D, V_6D = get_XYV_6D()
    print(">>>Done.\n", flush=True)

    # loop over 5D datasets
    X_5D = np.array([], dtype=np.float32)
    Y_5D = np.array([], dtype=np.float32)
    V_5D = np.array([], dtype=np.float32)
    print("Getting 5D data:", flush=True)
    for i in trange(32):
        if not exists(get_datadir() + f"DR3_5D/{i}.hdf5"):
            continue
        X, Y, V = get_XYV_5D(i)
        X_5D = np.append(X_5D, X)
        Y_5D = np.append(Y_5D, Y)
        V_5D = np.append(V_5D, V)
    print(">>>Done.\n", flush=True)

    # save
    print("Saving:", flush=True)
    np.savez(get_datadir() + 'figures/figX_median_vlos_map_XYV_data.npz',
             X_6D=X_6D, Y_6D=Y_6D, V_6D=V_6D,
             X_5D=X_5D, Y_5D=Y_5D, V_5D=V_5D)
    print(">>>Done.\n", flush=True)
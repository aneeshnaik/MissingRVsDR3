#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertically stacked EDR3-derived posterior means and measured DR3 values.

Created: May 2023
Author: A. P. Naik
"""
import sys
import pandas as pd
import numpy as np
from os.path import exists
from h5py import File as hFile
from tqdm import trange

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')
from matplotlib.colors import LinearSegmentedColormap as LSCmap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append("..")
from src.utils import get_datadir


def get_v_data(N):

    # filepaths
    ddir = get_datadir()
    match_cat = ddir + "EDR3_predictions/EDR3_prediction_match.hdf5"

    # load match data
    print(">>>Loading match data:")
    df = pd.read_hdf(match_cat)

    # cut by radial velocity error
    print(">>>RV error cut")
    m = (df['radial_velocity_error'] < 5)
    df = df[m]

    # random subset
    print(">>>Random subset")
    df = df.iloc[np.random.choice(np.arange(len(df)), N, replace=False)]

    v_true = df['radial_velocity'].to_numpy()
    mu_pred = df['mu_pred'].to_numpy()
    sig_pred = df['sig_pred'].to_numpy()
    return v_true, mu_pred, sig_pred


def create_plot_data(dfile, N_plot):

    # filepaths
    ddir = get_datadir()
    match_cat = ddir + "EDR3_predictions/EDR3_prediction_match.hdf5"

    # load match data
    print(">>>Loading match data:")
    df = pd.read_hdf(match_cat)

    # cut by radial velocity error
    print(">>>RV error cut")
    m = (df['radial_velocity_error'] < 5)
    df = df[m]

    # random subset
    print(">>>Random subset")
    df = df.iloc[np.random.choice(np.arange(len(df)), N_plot, replace=False)]

    # to numpy
    v_true = df['radial_velocity'].to_numpy()
    mu_pred = df['mu_pred'].to_numpy()
    sig_pred = df['sig_pred'].to_numpy()

    # sort by mean pred
    inds = np.argsort(mu_pred)
    v_true = v_true[inds]
    mu_pred = mu_pred[inds]
    sig_pred = sig_pred[inds]

    # save
    print(">>>Saving")
    np.savez(dfile, v_true=v_true, mu_pred=mu_pred, sig_pred=sig_pred)
    return


if __name__ == "__main__":

    # plot params
    N_plot = 10000

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig1_EDR3_posteriors_REATTEMPT_3_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, N_plot)
    data = np.load(dfile)
    v_true = data['v_true']
    mu_pred = data['mu_pred']
    sig_pred = data['sig_pred']

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'

    # setup figure
    fig = plt.figure(figsize=(7, 4.5))
    X0 = 0.09
    X1 = 0.96
    Y0 = 0.085
    Y1 = 0.94
    dX = X1 - X0
    dY = Y1 - Y0
    ax = fig.add_axes([X0, Y0, dX, dY])

    # plot
    ax.scatter(v_true, np.arange(N_plot) + 1, fc=c2, ec='none', s=1.6, alpha=0.9, rasterized=True)
    ax.plot(mu_pred, np.arange(N_plot) + 1, c=c1)
    ax.fill_betweenx(np.arange(N_plot) + 1, mu_pred - sig_pred, mu_pred + sig_pred, fc=c1, alpha=0.3)

    # labels ticks etc.
    ax.grid(True, c='k', ls='dotted')
    ax.set_xlabel(r"Radial Velocity [km/s]")
    ax.set_ylabel(r"Star #")
    ax.tick_params(top=True, right=True, direction='inout')
    ax.set_xlim(-200, 200)
    ax.set_ylim(1, N_plot + 1)

    # legend
    obs_handle = Line2D([], [], color=c2, marker='o', linestyle='None', markersize=6, label=r'Measured radial velocity')
    mu_handle = Line2D([], [], color=c1, label=r'Predictions: mean')
    sig_handle = Patch(color=c1, alpha=0.3, label=r"Predictions: $1\sigma$ region")
    handles = [obs_handle, mu_handle, sig_handle]
    ax.legend(handles=handles, facecolor='w', edgecolor='k')

    # fig title
    fig.suptitle("Pre-DR3 predictions: predictive distributions compared with DR3 measurements")

    # save
    fig.savefig("fig1_EDR3_posteriors.pdf", dpi=400)

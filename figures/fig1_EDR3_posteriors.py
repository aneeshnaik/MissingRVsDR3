#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vertically stacked EDR3-derived posteriors against measured DR3 values. 

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

sys.path.append("..")
from src.utils import get_datadir


def get_v_data(N):

    # filepaths
    ddir = get_datadir()
    match_cat = ddir + "EDR3_predictions/EDR3_prediction_match.hdf5"
    pred_cat = ddir + "EDR3_predictions/EDR3MissingRVCatalogue.hdf5"

    # load match data
    print(">>>Loading match data:")
    df = pd.read_hdf(match_cat)

    # load predicted catalogue
    print(">>>Loading prediction catalogue")
    with hFile(pred_cat, 'r') as hf:
        ids = hf["ids"][:]
        v_pred = hf["v_samples"][:]

    # random subset
    print(">>>Match")
    df = df.iloc[np.random.choice(np.arange(len(df)), N, replace=False)]
    m = np.in1d(ids, df['source_id'].to_numpy())
    ids = ids[m]
    v_pred = v_pred[m]
    v_pred = v_pred[np.where(ids == df['source_id'].to_numpy()[:, None])[1]]
    v_true = df['radial_velocity'].to_numpy()
    return v_true, v_pred


def create_plot_data(dfile, N_plot):

    # download data (random subset)
    v_true, v_pred = get_v_data(N_plot)

    # get kernel width
    print(">>>Computing kernel width")
    N_samples = v_pred.shape[1]
    h = 0.6 * np.std(v_pred, axis=-1) * np.power(N_samples, -0.2)

    # v bins
    print(">>>Setting up v bins")
    N_bins = 100
    x = np.linspace(-120, 120, N_bins)

    # loop over stars
    print(">>>Looping over stars")
    pdfs = np.zeros((N_plot, N_bins))
    for i in trange(N_plot):

        # get logistic kernel
        z = 0.5 * (x[:, None] - v_pred[i][None]) / h[i]
        sech2 = 1 / (4 * h[i] * np.cosh(z)**2)

        # sum over samples for PDFs
        pdfs[i] = np.sum(sech2, axis=-1) / N_samples

    # sort by v_true
    print(">>>Sorting")
    inds = np.argsort(v_true)
    trues = v_true[inds]
    pdfs = pdfs[inds]

    # save
    print(">>>Saving")
    np.savez(dfile, trues=trues, pdfs=pdfs)
    return


if __name__ == "__main__":

    # plot params
    N_plot = 100000

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig1_EDR3_posteriors_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, N_plot)
    data = np.load(dfile)
    trues = data['trues']
    pdfs = data['pdfs']
    N_plot = len(trues)

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'
    clist = ['#ffffff', '#f0f6f6', '#e2eded', '#d4e5e4',
              '#c5dcdb', '#b7d3d3', '#a9cbca', '#9ac2c1',
              '#8cbab9', '#7db1b0', '#6ea9a8', '#5fa09f',
              '#4f9897', '#3d908f', '#288787', '#007f7f']
    cmap = LSCmap.from_list("", clist)

    # setup figure
    fig = plt.figure(figsize=(6.9, 4.5))
    X0 = 0.1
    X3 = 0.94
    Y0 = 0.085
    Y1 = 0.98
    Xgap = 0.01
    cdX = 0.035
    X2 = X3 - cdX
    X1 = X2 - Xgap
    dX = X1 - X0
    dY = Y1 - Y0
    ax = fig.add_axes([X0, Y0, dX, dY])
    cax = fig.add_axes([X2, Y0, cdX, dY])

    # plot
    label = r'True $v_\mathrm{los}$'
    extent = [-120, 120, 0.5, N_plot + 0.5]
    line = ax.plot(trues, np.arange(N_plot) + 1, c=c2, lw=2, label=label)[0]
    vmax = 0.025
    im = ax.imshow(
        pdfs / vmax, interpolation='none', cmap=cmap, vmax=1,
        origin='lower', extent=extent, aspect='auto', rasterized=True)
    plt.colorbar(im, cax=cax)

    # labels ticks etc.
    ax.grid(True, c='k', ls='dotted')
    ax.set_xlabel(r"$v_\mathrm{los}$ [km/s]")
    ax.set_ylabel(r"Star \#")
    cax.set_ylabel("Frequency Density [arbitrary units]")
    handles = [line, Patch(color=c1, label="Predictions")]
    ax.legend(handles=handles, facecolor='w', edgecolor='k')
    ax.tick_params(top=True, right=True, direction='inout')

    # save
    plt.savefig("fig1_EDR3_posteriors.pdf", dpi=800)
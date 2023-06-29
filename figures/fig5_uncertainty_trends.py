#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot showing how posterior widths vary with distance and distance error.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
from os.path import exists

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir


def load_6D_set():
    """Load 6D test set and corresponding GMM catalogue as pd DataFrame."""
    ddir = get_datadir()
    gmm = pd.read_hdf(ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5")
    df = pd.read_hdf(ddir + "DR3_6D/test.hdf5")
    df = df.merge(gmm, on='source_id')
    return df


def create_plot_data(dfile):
    # load 6D set
    df = load_6D_set()

    # get posterior widths
    sig = 0.5 * (df['q841'] - df['q159']).to_numpy()

    # get distances and errors
    d_matrix = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = np.mean(d_matrix, axis=-1)
    r = np.std(d_matrix, axis=-1) / d

    # construct bins
    N_bins = 16
    d_edges = np.linspace(0, 8, N_bins + 1)
    d_cens = 0.5 * (d_edges[1:] + d_edges[:-1])
    r_edges = np.logspace(-2, -0.5, N_bins + 1)
    r_cens = 10**(0.5 * (np.log10(r_edges[1:]) + np.log10(r_edges[:-1])))

    # d bins
    inds = np.digitize(d, d_edges) - 1
    sig_d_median = [np.median(sig[inds == i]) for i in range(N_bins)]
    sig_d_err = [np.std(sig[inds == i]) for i in range(N_bins)]

    # r bins
    inds = np.digitize(r, r_edges) - 1
    sig_r_median = [np.median(sig[inds == i]) for i in range(N_bins)]
    sig_r_err = [np.std(sig[inds == i]) for i in range(N_bins)]

    # save
    np.savez(
        dfile,
        x0=d_cens, y0=sig_d_median, err0=sig_d_err,
        x1=r_cens, y1=sig_r_median, err1=sig_r_err
    )
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig5_uncertainty_trends_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    x0 = data['x0']
    y0 = data['y0']
    err0 = data['err0']
    x1 = data['x1']
    y1 = data['y1']
    err1 = data['err1']

    # colours
    c1 = 'teal'
    c2 = 'goldenrod'

    # set up figure
    left = 0.135
    right = 0.99
    top = 0.9
    bottom = 0.19
    hgap = 0.05
    dX = (right - left - hgap) / 2
    X0 = left
    X1 = left + dX + hgap
    dY = top - bottom
    Y = bottom
    fig = plt.figure(figsize=(3.35, 2.25))
    ax0 = fig.add_axes([X0, Y, dX, dY])
    ax1 = fig.add_axes([X1, Y, dX, dY])

    # plot
    eargs = {'fmt': '.', 'color': 'teal'}
    ax0.errorbar(x0, y0, err1, **eargs)
    ax1.errorbar(x1, y1, err1, **eargs)

    # panel 1 distance line
    ax0.plot([2.97, 2.97], [0, 160], c=c2, ls='dashed')
    ax0.arrow(2.97, 80, -1.5, 0, head_length=0.5, width=3, fc=c2, ec='none')
    ax0.text(1.4, 90, r"95\%of" + "\ntraining set", ha='left', rotation=90)

    # limits, ticks, labels
    for ax in [ax0, ax1]:
        ax.set_ylim(0, 160)
        ax.tick_params(direction='inout', which='both')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax1.tick_params(labelleft=False)
    ax1.set_xscale('log')
    ax0.set_xlabel(r"Distance [kpc]")
    ax1.set_xlabel(r"Fractional distance error")
    ax0.set_ylabel(r"Prediction uncertainty [km/s]")
    ax0.xaxis.set_label_coords(0.5, -0.175, transform=ax0.transAxes)
    ax1.xaxis.set_label_coords(0.5, -0.175, transform=ax1.transAxes)
    fig.suptitle("6D test set: prediction uncertainty trends", x=left + 0.5 * (right - left))

    # save figure
    fig.savefig("fig5_uncertainty_trends.pdf", dpi=800)

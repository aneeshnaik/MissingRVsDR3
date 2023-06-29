#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantile distribution of DR3 RVs within EDR3-derived posteriors.

Created: June 2022
Author: A. P. Naik
"""
import pandas as pd
import numpy as np
import sys
from os.path import exists

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir


def create_plot_data(dfile):

    print("Creating plot data file:")

    # data directory
    ddir = get_datadir()

    # load match data
    print(">>>Loading match data:")
    df = pd.read_hdf(ddir + "EDR3_predictions/EDR3_prediction_match.hdf5")

    # cut by radial velocity error
    m = (df['radial_velocity_error'] < 5)
    df = df[m]

    # PANEL 1: F values
    print(">>>Panel 1")
    F = df['F'].to_numpy()

    # save plot data
    print(">>>Saving")
    np.save(dfile, F)
    print(">>>Done.\n")

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig2_EDR3_quantiles_data.npy"
    if not exists(dfile):
        create_plot_data(dfile)
    F = np.load(dfile)

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=1.2)

    # set up figure
    fig = plt.figure(figsize=(3.35, 2.7))
    X0 = 0.07
    X1 = 0.97
    dX = X1 - X0
    Y0 = 0.17
    Y1 = 0.92
    dY = Y1 - Y0
    ax = fig.add_axes([X0, Y0, dX, dY])

    # plot
    N_bins = 100
    edges = np.linspace(0, 1, N_bins + 1)
    cens = 0.5 * (edges[1:] + edges[:-1])
    n = ax.hist(F, edges, **hargs)[0]
    ax.plot([0, 1], [1, 1], c=c2, lw=1.5, ls='dashed')

    # hatch region
    xf = np.array([edges[i // 2 + i % 2] for i in range(2 * N_bins)])
    yf0 = np.array([n[i // 2] for i in range(2 * N_bins)])
    yf1 = np.ones_like(yf0)
    ax.fill_between(xf, yf0, yf1, hatch='xxxx', fc='none', linewidth=0, alpha=0.5, rasterized=True)

    # limits, labels, ticks etc
    ax.set_xlim(0, 1)

    # axis labels
    t = r'$F(v_\mathrm{true}|\mathrm{model})' \
        r'= \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$'
    ax.set_xlabel(t, usetex=True)
    ax.set_ylabel("Probability Density [arbitrary units]")
    ax.tick_params(direction='inout', right=False, labelleft=False, left=False, top=True)
    fig.suptitle("Pre-DR3 predictions: quantile distribution")

    # save figure
    fig.savefig("fig2_EDR3_quantiles.pdf", dpi=800)

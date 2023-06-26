#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Test predictions from EDR3-trained BNN.

Created: June 2022
Author: A. P. Naik
"""
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

from os.path import exists
from h5py import File as hFile
from scipy.stats import norm
from matplotlib.lines import Line2D
from tqdm import trange

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
    ax1_y = df['F'].to_numpy()

    # PANEL 2: error histogram
    print(">>>Panel 2")
    ax2_y = df['sig_pred'].to_numpy()
    q16t, q84t = np.percentile(df['radial_velocity'], q=[16, 84])
    ax2_y_true = (q84t - q16t) / 2

    # PANEL 3: single generation (for each star, choose first sample)
    print(">>>Panel 3")
    ax3_y0 = df['radial_velocity'].to_numpy()
    ax3_y1 = df['single_prediction'].to_numpy()

    # save plot data
    print(">>>Saving")
    np.savez(
        dfile,
        ax1_y=ax1_y,
        ax2_y=ax2_y, ax2_y_true=ax2_y_true,
        ax3_y0=ax3_y0, ax3_y1=ax3_y1
    )
    print(">>>Done.\n")

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig2_test_EDR3_predictions_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    ax1_y = data['ax1_y']
    ax2_y = data['ax2_y']
    ax2_y_true = data['ax2_y_true']
    ax3_y0 = data['ax3_y0']
    ax3_y1 = data['ax3_y1']

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=1.2)

    # set up figure
    fig = plt.figure(figsize=(6.9, 2.7), dpi=150)
    X0 = 0.035
    X5 = 0.985
    Xgap = 0.03
    dX = (X5 - X0 - 2 * Xgap) / 3
    X1 = X0 + dX
    X2 = X1 + Xgap
    X3 = X2 + dX
    X4 = X3 + Xgap
    Y0 = 0.17
    Y1 = 0.9
    dY = Y1 - Y0
    ax1 = fig.add_axes([X0, Y0, dX, dY])
    ax2 = fig.add_axes([X2, Y0, dX, dY])
    ax3 = fig.add_axes([X4, Y0, dX, dY])

    # plot panel 1
    bins = np.linspace(0, 1, 100)
    ax1.hist(ax1_y, bins, **hargs)
    ax1.plot([0, 1], [1, 1], c='k', lw=1.5, ls='dotted')

    # plot panel 2
    ax2.hist(ax2_y, np.linspace(0, 80, 250), **hargs)
    ax2lim = ax2.get_ylim()
    med = np.median(ax2_y)
    ax2.plot([ax2_y_true, ax2_y_true], [0, 1], c=c2, ls='dashed')
    ax2.plot([med, med], [0, 1], c='k', ls='dashed')

    # plot panel 3
    bins = np.linspace(-200, 200, 250)
    hargs = dict(
        bins=bins, density=True,
        histtype='stepfilled', lw=1.2, ec='k', alpha=0.7
    )
    ax3.hist(ax3_y1, fc=c1, label='Predictions', **hargs)
    ax3.hist(ax3_y0, fc=c2, label='True', **hargs)
    ax3.legend(frameon=False, loc='upper left')

    # arrows+labels on panel 2
    tr = ax2.transAxes
    targs = {
        'transform': tr,
        'ha': 'center',
        'va': 'center'
    }
    arrargs = {
        'width': 0.01,
        'length_includes_head': True,
        'transform': tr,
        'lw': 0.5
    }
    lab1 = 'True $v_\mathrm{los}$\n dist. width\n' + rf'$\sigma=${ax2_y_true:.1f} km/s'
    lab2 = 'Median pred.\nuncertainty\n' + rf'$\sigma=${med:.1f} km/s'
    ax2.text(0.80, 0.85, lab1, **targs)
    ax2.text(0.77, 0.25, lab2, **targs)
    ax2.arrow(0.62, 0.85, -0.11, 0, fc='goldenrod', **arrargs)
    ax2.arrow(0.59, 0.25, -0.24, 0, fc='k', **arrargs)

    # axis limits
    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 80)
    ax2.set_ylim(ax2lim)
    ax3.set_xlim(-200, 200)
    ax3.set_ylim(0, ax3.get_ylim()[1] * 1.1)

    # axis labels
    ax1.set_xlabel(r'$F(v_\mathrm{true}|\mathrm{model}) = \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$')
    ax2.set_xlabel("Prediction uncertainties [km/s]")
    ax3.set_xlabel(r"$v_\mathrm{los}\ [\mathrm{km/s}]$")
    ax1.set_ylabel("Probability Density [arbitrary units]")

    # panel titles
    ax1.set_title("Quantile distribution")
    ax2.set_title("Prediction uncertainties")
    ax3.set_title("Single prediction for each star")

    # ticks
    for ax in fig.axes:
        ax.tick_params(direction='inout', right=False, labelleft=False, left=False, top=True)

    # save figure
    plt.savefig("fig2_test_EDR3_predictions.pdf")

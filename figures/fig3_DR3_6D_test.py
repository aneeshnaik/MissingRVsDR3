#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Test predictions from EDR3-trained BNN.

Created: June 2022
Author: A. P. Naik
"""
import pandas as pd
import numpy as np
import sys
from tqdm import trange

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

from os.path import exists
from h5py import File as hFile
from scipy.stats import norm
from matplotlib.lines import Line2D

sys.path.append("..")
from src.utils import get_datadir, batch_calculate


def get_CDF(v_true, v_pred, h):

    x = np.zeros_like(v_pred)

    batch_size = 10000
    N = len(v_true)
    N_batches = N // batch_size

    # loop over batches
    for i in trange(N_batches):
        i0 = i * batch_size
        i1 = (i + 1) * batch_size
        x[i0:i1] = 0.5 * (v_true[i0:i1, None] - v_pred[i0:i1]) / h[i0:i1, None]

    # remainder data
    if N % batch_size != 0:
        i0 = N_batches * batch_size
        x[i0:] = 0.5 * (v_true[i0:, None] - v_pred[i0:]) / h[i0:, None]
    return np.sum(0.5 * (np.tanh(x) + 1), axis=-1) / v_pred.shape[1]


def create_plot_data(dfile):

    print("Creating plot data file:")

    print(">>>Loading data:")
    ddir = get_datadir()
    data = np.load(ddir + 'DR3_predictions/6D_test.npz')
    v_pred = data['v_los_preds']
    v_true = data['v_los_true']

    print(">>>Loading summary stats:")
    data = np.load(ddir + 'DR3_predictions/6D_test_summary_stats.npz')
    mu = data['mu']
    sig = data['sig']
    h = data['h']

    # PANEL 1: residuals
    print(">>>Panel 1")
    ax1_y = (mu - v_true) / sig

    # PANEL 2: calibration curve
    print(">>>Panel 2")
    ax2_y = get_CDF(v_true, v_pred, h)

    # PANEL 3: error histogram
    print(">>>Panel 3")
    ax3_err = np.copy(sig)
    q16t, q84t = np.percentile(v_true, q=[16, 84])
    ax3_err_true = (q84t - q16t) / 2

    # PANEL 4: single generation
    # for each star, choose first sample
    print(">>>Panel 4")
    ax4_y0 = np.copy(v_true)
    ax4_y1 = v_pred[:, 0]

    # save plot data
    print(">>>Saving")
    np.savez(
        dfile,
        ax1_y=ax1_y, ax2_y=ax2_y,
        ax3_err=ax3_err, ax3_err_true=ax3_err_true,
        ax4_y0=ax4_y0, ax4_y1=ax4_y1
    )
    print(">>>Done.\n")

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig3_DR3_6D_test_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    ax1_y = data['ax1_y']
    ax2_y = data['ax2_y']
    ax3_err = data['ax3_err']
    ax3_err_true = data['ax3_err_true']
    ax4_y0 = data['ax4_y0']
    ax4_y1 = data['ax4_y1']

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=1.2)

    # set up figure
    fig = plt.figure(figsize=(6.9, 5.5), dpi=150)
    X0 = 0.08
    X3 = 0.92
    Xgap = 0.03
    dX = (X3 - X0 - Xgap) / 2
    X1 = X0 + dX
    X2 = X1 + Xgap
    Y0 = 0.065
    Y3 = 0.96
    Ygap = 0.15
    dY = (Y3 - Y0 - Ygap) / 2
    Y1 = Y0 + dY
    Y2 = Y1 + Ygap
    ax1 = fig.add_axes([X0, Y2, dX, dY])
    ax2 = fig.add_axes([X2, Y2, dX, dY])
    ax3 = fig.add_axes([X0, Y0, dX, dY])
    ax4 = fig.add_axes([X2, Y0, dX, dY])

    # plot panel 1
    bins = np.linspace(-4.25, 4.25, 250)
    p = norm().pdf(bins)
    ax1.plot(bins, p, lw=2, ls='dashed', c=c2, label='Standard normal')
    ax1.hist(ax1_y, bins, **hargs, label='Residuals')

    # plot panel 2
    l1 = r'$F(v_\mathrm{true}|\mathrm{model})$'
    l2 = r'$x=y$'
    ax2.hist(ax2_y, bins=1000, cumulative=True, density=True, histtype='step', ec=c1, lw=1.5, label=l1)
    line1 = Line2D([0], [0], label=l1, c=c1, lw=1.5)
    line2 = ax2.plot([0, 1], [0, 1], c='k', ls='dotted', label=l2)[0]
    handles = [line1, line2]

    # plot panel 3
    ax3.hist(ax3_err, np.linspace(0, 80, 250), **hargs)
    ax3lim = ax3.get_ylim()
    med = np.median(ax3_err)
    ax3.plot([ax3_err_true, ax3_err_true], [0, 1], c=c2, ls='dashed')
    ax3.plot([med, med], [0, 1], c='k', ls='dashed')

    # plot panel 4
    bins = np.linspace(-200, 200, 250)
    hargs = dict(
        bins=bins, density=True,
        histtype='stepfilled', lw=1.2, ec='k', alpha=0.75
    )
    ax4.hist(ax4_y0, fc=c2, label='True', **hargs)
    ax4.hist(ax4_y1, fc=c1, label='Predictions', **hargs)

    # arrows+labels on panel 3
    tr = ax3.transAxes
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
    lab1 = 'True $v_\mathrm{los}$\n distribution width\n' + rf'$\sigma=${ax3_err_true:.1f} km/s'
    lab2 = 'Median prediction\nuncertainty\n' + rf'$\sigma=${med:.1f} km/s'
    ax3.text(0.81, 0.85, lab1, **targs)
    ax3.text(0.77, 0.25, lab2, **targs)
    ax3.arrow(0.62, 0.85, -0.08, 0, fc=c2, **arrargs)
    ax3.arrow(0.59, 0.25, -0.22, 0, fc='k', **arrargs)

    # axis limits
    ax1.set_ylim(0, 0.45)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax3.set_ylim(ax3lim)

    # legends
    ax1.legend(frameon=False, loc='upper left')
    ax2.legend(handles=handles, frameon=False)
    ax4.legend(frameon=False)

    # x-labels
    lab = (r"$\displaystyle\frac"
           r"{v_\mathrm{true} - \mu_\mathrm{pred.}}"
           r"{\sigma_\mathrm{pred.}}$")
    ax1.set_xlabel(lab)
    ax2.set_xlabel(r'$F(v_\mathrm{true}|\mathrm{model}) = \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$')
    ax3.set_xlabel("Prediction uncertainties [km/s]")
    ax4.set_xlabel(r"$v_\mathrm{los}\ [\mathrm{km/s}]$")

    # y-labels
    ax1.set_ylabel("Probability Density [arbitrary units]")
    ax2.set_ylabel("CDF")
    ax3.set_ylabel("Probability Density [arbitrary units]")
    ax4.set_ylabel("Probability Density [arbitrary units]")

    # panel titles
    ax1.set_title("Residuals")
    ax2.set_title("Calibration curve")
    ax3.set_title("Prediction uncertainties")
    ax4.set_title("Single prediction for each star")

    # ticks
    for ax in [ax2, ax4]:
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
    for ax in fig.axes:
        ax.tick_params(direction='inout', right=True, left=True, top=True)

    # save figure
    plt.savefig("fig3_DR3_6D_test.pdf")

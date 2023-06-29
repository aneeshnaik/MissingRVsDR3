#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare BNN predictions against DR3 6D test set.

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
from src.gmm import gmm_cdf_batch, gmm_batchsample


def create_plot_data(dfile):

    print("Creating plot data file:")

    # load data
    print(">>>Loading data:")
    ddir = get_datadir()
    gaia_file = ddir + "DR3_6D/test.hdf5"
    gmm_file = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    test_set = pd.read_hdf(gaia_file)
    df = test_set.merge(pd.read_hdf(gmm_file), on='source_id')
    weights = np.stack([df[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([df[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([df[f'var{i}'] for i in range(4)], axis=-1)

    # get quantiles (panel 0)
    print(">>>Calculate quantile distribution (panel 0):")
    v_true = df['v_los'].to_numpy()
    F = gmm_cdf_batch(v_true, weights, means, variances)

    # draw single sample of each star (panel 1)
    print(">>>Draw single sample (panel 1)")
    v_sample = gmm_batchsample(weights, means, variances)

    # prediction uncertainties (panel 2)
    print(">>>Get uncertainties (panel 2)")
    sig = 0.5 * (df['q841'] - df['q159']).to_numpy()
    q16t, q84t = np.percentile(v_true, q=[16, 84])
    sig_true = (q84t - q16t) / 2

    # save plot data
    print(">>>Saving")
    np.savez(
        dfile,
        ax0_F=F,
        ax1_v_true=v_true, ax1_v_sample=v_sample,
        ax2_sig=sig, ax2_sig_true=sig_true,
    )
    print(">>>Done.\n")

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig4_DR3_6D_test_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)

    # colours
    c1 = 'teal'
    c2 = 'goldenrod'

    # set up figure
    fig = plt.figure(figsize=(7, 2.7))
    X0 = 0.03
    X5 = 1 - X0
    Xgap = 0.03
    dX = (X5 - X0 - 2 * Xgap) / 3
    X1 = X0 + dX
    X2 = X1 + Xgap
    X3 = X2 + dX
    X4 = X3 + Xgap
    Y0 = 0.16
    Y1 = 0.86
    dY = Y1 - Y0
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X2, Y0, dX, dY])
    ax2 = fig.add_axes([X4, Y0, dX, dY])

    # plot 1st panel
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=1.2)
    bins = np.linspace(0, 1, 100)
    F = data['ax0_F']
    ax0.hist(F, bins, **hargs)
    ax0.plot([0, 1], [1, 1], c=c2, lw=1.5, ls='dashed')

    # plot 2nd panel
    bins = np.linspace(-200, 200, 250)
    hargs = dict(
        bins=bins, density=True,
        histtype='stepfilled', lw=1.2, ec='k', alpha=0.7
    )
    v0 = data['ax1_v_true']
    v1 = data['ax1_v_sample']
    ax1.hist(v1, fc=c1, label='Predictions', **hargs)
    ax1.hist(v0, fc=c2, label='True', **hargs)
    ax1.legend(frameon=False, loc='upper left')

    # plot 3rd panel
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=1.2)
    sig = data['ax2_sig']
    sigt = data['ax2_sig_true']
    ax2.hist(sig, np.linspace(0, 80, 250), **hargs)
    ax2lim = ax2.get_ylim()
    med = np.median(sig)
    ax2.plot([sigt, sigt], [0, 1], c=c2, ls='dashed')
    ax2.plot([med, med], [0, 1], c='k', ls='dashed')

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
    lab1 = 'True radial vel.\n dist. width\n' + f'{sigt:.1f} km/s'
    lab2 = 'Median pred.\nuncertainty\n' + f'{med:.1f} km/s'
    ax2.text(0.81, 0.85, lab1, **targs)
    ax2.text(0.77, 0.25, lab2, **targs)
    ax2.arrow(0.65, 0.85, -0.10, 0, fc='goldenrod', **arrargs)
    ax2.arrow(0.59, 0.25, -0.22, 0, fc='k', **arrargs)

    # axis limits
    ax0.set_xlim(0, 1)
    ax1.set_xlim(-200, 200)
    ax1.set_ylim(0, ax1.get_ylim()[1] * 1.1)
    ax2.set_xlim(0, 80)
    ax2.set_ylim(ax2lim)

    # axis labels
    ax0.set_xlabel(r'$F(v_\mathrm{true}|\mathrm{model}) = \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$', usetex=True)
    ax1.set_xlabel("Radial velocity [km/s]")
    ax2.set_xlabel("Prediction uncertainties [km/s]")
    ax0.set_ylabel("Probability Density [arbitrary units]")

    # panel titles
    ax0.set_title("Quantile distribution")
    ax1.set_title("Single prediction for each star")
    ax2.set_title("Prediction uncertainties")
    fig.suptitle("6D test set: BNN prediction summaries")

    # ticks
    for ax in fig.axes:
        ax.tick_params(direction='inout', right=False, labelleft=False, left=False, top=True)

    # save figure
    fig.savefig("fig4_DR3_6D_test.pdf", dpi=800)

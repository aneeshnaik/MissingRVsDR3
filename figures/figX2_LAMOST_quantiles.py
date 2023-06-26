#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure showing quantile plot for Gaia 5D stars with LAMOST RVs.

Created: May 2023
Author: A. P. Naik
"""
import sys
import pandas as pd
import numpy as np
from os.path import exists
from tqdm import trange

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir
from src.gmm import gmm_cdf_batch as calc_F


def get_lamost_5D_xmatch():

    ddir = get_datadir()
    xmatch_file = ddir + 'external/LAMOST_xmatch.hdf5'
    lamost = pd.read_hdf(xmatch_file)

    # make empty match dataframe
    xmatch = pd.DataFrame()

    # loop over GMM catalogues
    for i in trange(32):

        # load GMM catalogue
        gaia_5D_file = ddir + f"DR3_predictions/5D_{i}_GMM_catalogue.hdf5"
        gaia_5D = pd.read_hdf(gaia_5D_file)

        # which LAMOST entries are in GMM catalogue and v.v.:
        m1 = np.isin(lamost['gaia_source_id'], gaia_5D['source_id'])
        m2 = np.isin(gaia_5D['source_id'], lamost['gaia_source_id'])
        l_sub = lamost[m1].reset_index(drop=True)
        g_sub = gaia_5D[m2].reset_index(drop=True)

        # merge (and rename and stuff)
        mi = g_sub.merge(l_sub, left_on='source_id', right_on='gaia_source_id')
        mi.drop('gaia_source_id', axis=1, inplace=True)
        names = {'v': 'lamost_v', 'v_err': 'lamost_v_err'}
        mi.rename(names, axis=1, inplace=True)

        # append to match dataframe
        xmatch = pd.concat((xmatch, mi))

    return xmatch


def get_lamost_6D_xmatch():

    ddir = get_datadir()
    xmatch_file = ddir + 'external/LAMOST_xmatch.hdf5'
    lamost = pd.read_hdf(xmatch_file)

    # make empty match dataframe
    xmatch = pd.DataFrame()

    # load GMM catalogue
    gaia_6D_file = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    gaia_6D = pd.read_hdf(gaia_6D_file)

    # add column for gaia RVs
    gaia_6D['v_gaia'] = pd.read_hdf(ddir + f"DR3_6D/test.hdf5")['v_los'].to_numpy()

    # which LAMOST entries are in GMM catalogue and v.v.:
    m1 = np.isin(lamost['gaia_source_id'], gaia_6D['source_id'])
    m2 = np.isin(gaia_6D['source_id'], lamost['gaia_source_id'])
    l_sub = lamost[m1].reset_index(drop=True)
    g_sub = gaia_6D[m2].reset_index(drop=True)

    # merge (and rename and stuff)
    mi = g_sub.merge(l_sub, left_on='source_id', right_on='gaia_source_id')
    mi.drop('gaia_source_id', axis=1, inplace=True)
    names = {'v': 'lamost_v', 'v_err': 'lamost_v_err'}
    mi.rename(names, axis=1, inplace=True)

    # append to match dataframe
    xmatch = pd.concat((xmatch, mi))

    return xmatch


def create_plot_data(dfile, diffcut):

    print("5D")
    xmatch = get_lamost_5D_xmatch()
    weights = np.stack([xmatch[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([xmatch[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([xmatch[f'var{i}'] for i in range(4)], axis=-1)
    F_5D = calc_F(xmatch['lamost_v'].to_numpy(), weights, means, variances)

    print("6D")
    xmatch = get_lamost_6D_xmatch()
    weights = np.stack([xmatch[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([xmatch[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([xmatch[f'var{i}'] for i in range(4)], axis=-1)
    F_6D = calc_F(xmatch['lamost_v'].to_numpy(), weights, means, variances)

    m = np.abs(xmatch['lamost_v'] - xmatch['v_gaia']) < diffcut
    F_6D_cut = F_6D[m]

    print("Saving")
    np.savez(dfile, F_5D=F_5D, F_6D=F_6D, F_6D_cut=F_6D_cut)
    return


if __name__ == "__main__":

    # plot params
    diffcut = 5

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figX2_LAMOST_quantiles_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, diffcut)
    data = np.load(dfile)
    F_5D = data['F_5D']
    F_6D = data['F_6D']
    F_6D_cut = data['F_6D_cut']

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'
    hargs = dict(density=True, fc=c1, histtype='stepfilled', ec='k', lw=0.8)

    # create figure
    fig = plt.figure(figsize=(3.35, 4.5))
    bottom = 0.1
    top = 0.99
    left = 0.08
    right = 0.97
    dX = right - left
    dY = (top - bottom) / 3
    Y0 = top - 1 * dY
    Y1 = top - 2 * dY
    Y2 = top - 3 * dY
    ax0 = fig.add_axes([left, Y0, dX, dY])
    ax1 = fig.add_axes([left, Y1, dX, dY])
    ax2 = fig.add_axes([left, Y2, dX, dY])

    # plot
    bins = np.linspace(0, 1, 300)
    ax0.hist(F_5D, bins, **hargs)
    ax1.hist(F_6D, bins, **hargs)
    ax2.hist(F_6D_cut, bins, **hargs)
    for ax in [ax0, ax1, ax2]:
        ax.plot([-0.02, 1], [1, 1], c=c2, lw=1.5, ls='dotted')

    # limits, ticks, labels
    for ax in [ax0, ax1, ax2]:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 3)
        ax.tick_params(
            direction='inout',
            right=False,
            labelleft=False,
            left=True,
            top=False
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.patch.set_alpha(0)
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    xlabel = r"$F(v_\mathrm{true}|\mathrm{model})" \
             r" = \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$"
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel("Probability Density [arbitrary units]")
    ax0.text(0.5, 0.7, r"LAMOST $\times$ Gaia 5D", ha='center', va='top', transform=ax0.transAxes)
    ax1.text(0.5, 0.7, r"LAMOST $\times$ Gaia 6D (test set)", ha='center', va='top', transform=ax1.transAxes)
    ax2.text(0.5, 0.7, r"LAMOST $\times$ Gaia 6D (test set)", ha='center', va='top', transform=ax2.transAxes)
    ax2.text(0.5, 0.6, r"(mismatched vels. removed)", ha='center', va='top', transform=ax2.transAxes)

    # save figure
    plt.savefig("figX2_LAMOST_quantiles.pdf")

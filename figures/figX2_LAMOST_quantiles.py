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
    for i in range(1):

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


def create_plot_data(dfile):
    xmatch = get_lamost_5D_xmatch()
    weights = np.stack([xmatch[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([xmatch[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([xmatch[f'var{i}'] for i in range(4)], axis=-1)
    F = calc_F(xmatch['lamost_v'].to_numpy(), weights, means, variances)
    np.savez(dfile, F=F)
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figX2_LAMOST_quantiles_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    F = data['F']

    # plot settings
    c = 'teal'
    hargs = dict(density=True, fc=c, histtype='stepfilled', ec='k', lw=1.2)

    # create figure
    fig = plt.figure(figsize=(3.35, 3), dpi=150)
    bottom = 0.15
    top = 0.98
    left = 0.08
    right = 0.97
    dX = right - left
    dY = top - bottom
    ax = fig.add_axes([left, bottom, dX, dY])

    # plot
    bins = np.linspace(0, 1, 100)
    ax.hist(F, bins, **hargs)
    ax.plot([0, 1], [1, 1], c='k', lw=1.5, ls='dotted')

    # limits, ticks, labels
    ax.set_xlim(0, 1)
    ax.tick_params(
        direction='inout', right=False, labelleft=False, left=False, top=True
    )
    xlabel = r"$F(v_\mathrm{true}|\mathrm{model})" \
             r" = \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density [arbitrary units]")

    # save figure
    plt.savefig("figX2_LAMOST_quantiles.pdf")

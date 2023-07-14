#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure comparing LAMOST and Gaia velocities.

Created: June 2023
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


def create_plot_data(dfile):

    # load cross-match
    xmatch = get_lamost_6D_xmatch()

    # read velocity cols
    v_gaia = xmatch['v_gaia'].to_numpy()
    v_lamost = xmatch['lamost_v'].to_numpy()

    # calc quantiles
    weights = np.stack([xmatch[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([xmatch[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([xmatch[f'var{i}'] for i in range(4)], axis=-1)
    F = calc_F(v_lamost, weights, means, variances)

    # save
    np.savez(dfile, F=F, v_gaia=v_gaia, v_lamost=v_lamost)
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figB2_LAMOST_gaia_vels_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    F = data['F']
    v0 = data['v_gaia']
    v1 = data['v_lamost']

    # plot settings
    c2 = 'teal'
    c1 = 'goldenrod'

    # create figure
    fig = plt.figure(figsize=(3.35, 3))
    left = 0.16
    right = 0.99
    bottom = 0.13
    top = 0.93
    dX = right - left
    dY = top - bottom
    ax = fig.add_axes([left, bottom, dX, dY])
    fig.suptitle("LAMOST versus Gaia radial velocity measurements", x=left + 0.5 * dX)

    # p-cut
    p = 0.00001
    m = ((F < p) | (F > 1 - p))

    # plot
    ax.plot([-500, 500], [-500, 500], c='k', ls='dashed', label=r'$y=x$', lw=0.5)
    sargs = dict(s=0.2, zorder=2, alpha=0.8, rasterized=True)
    ax.scatter(v0[~m], v1[~m], c=c1, label=r'$p \geq 10^{-5}$', **sargs)
    ax.scatter(v0[m], v1[m], c=c2, label=r'$p < 10^{-5}$', **sargs)

    # limits, ticks, spines, labels
    leg = ax.legend(
        markerscale=15, handletextpad=0.5, handlelength=1.5, frameon=False,
    )
    for t in leg.get_texts():
        t.set_usetex(True)
    ax.set_xlim(-500, 500)
    ax.set_ylim(-500, 500)
    ax.tick_params(direction='inout')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r"Gaia radial velocity [km/s]")
    ax.set_ylabel(r"LAMOST radial velocity [km/s]")

    # save figure
    fig.savefig("figB2_LAMOST_gaia_vels.pdf", dpi=800)

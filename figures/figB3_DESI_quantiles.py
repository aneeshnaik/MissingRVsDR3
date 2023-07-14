#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure showing quantile plot for Gaia 5D stars with DESI RVs.

Created: July 2023
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys
from tqdm import trange
from os.path import exists
from astropy.io import fits
from astropy.table import Table

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir
from src.gmm import gmm_cdf_batch as calc_F


def load_DESI_data():
    """Load DESI MWS catalogue, apply cuts, return as pandas dataframe."""
    # Open FITS file
    filename = get_datadir() + "external/mwsall-pix-fuji.fits"
    hdulist = fits.open(filename)

    # Access the data in HDUs 1 (main) and 5 (gaia)
    d1 = Table(hdulist[1].data)
    d5 = Table(hdulist[5].data)

    # Close the FITS file
    hdulist.close()

    # quality cut
    m = (d1['RVS_WARN'] == 0) \
        & (d1['RR_SPECTYPE'] == 'STAR') \
        & (np.abs(d1['VRAD']) < 600) \
        & (d1['PRIMARY']) \
        & (d5['SOURCE_ID'] != 999999)
    d1 = d1[m]
    d5 = d5[m]

    # put RVs and Gaia IDs into dataframe
    data = {
        'v_desi': d1['VRAD'].astype(np.float32),
        'source_id': d5['SOURCE_ID'].astype(np.int64),
    }
    return pd.DataFrame(data)


def get_desi_5D_xmatch():
    """Get DESI x Gaia 5D match, return as pandas dataframe."""
    # load DESI data
    ddir = get_datadir()
    desi = load_DESI_data()

    # make empty match dataframe
    xmatch = pd.DataFrame()

    # loop over GMM catalogues
    for i in trange(32):

        # load GMM catalogue
        gaia_5D_file = ddir + f"DR3_predictions/5D_{i}_GMM_catalogue.hdf5"
        gaia_5D = pd.read_hdf(gaia_5D_file)

        # which LAMOST entries are in GMM catalogue and v.v.:
        m1 = np.isin(desi['source_id'], gaia_5D['source_id'])
        m2 = np.isin(gaia_5D['source_id'], desi['source_id'])
        d_sub = desi[m1].reset_index(drop=True)
        g_sub = gaia_5D[m2].reset_index(drop=True)

        # merge
        mi = g_sub.merge(d_sub, on='source_id')

        # append to match dataframe
        xmatch = pd.concat((xmatch, mi))

    return xmatch


def create_plot_data(dfile):

    # load DESI xmatch
    df = get_desi_5D_xmatch()

    # calculate quantiles
    weights = np.stack([df[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([df[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([df[f'var{i}'] for i in range(4)], axis=-1)
    F = calc_F(df['v_desi'].to_numpy(), weights, means, variances)

    # save
    np.save(dfile, F)
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figB3_DESI_quantiles_data.npy"
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

    # limits, labels, ticks etc
    ax.set_xlim(0, 1)

    # axis labels
    t = r'$F(v_\mathrm{true}|\mathrm{model})' \
        r'= \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$'
    ax.set_xlabel(t, usetex=True)
    ax.set_ylabel("Probability Density [arbitrary units]")
    ax.tick_params(direction='inout', right=False, labelleft=False, left=False, top=True)
    fig.suptitle("Quantile distribution of DESI measurements")

    # save figure
    fig.savefig("figB3_DESI_quantiles.pdf", dpi=800)

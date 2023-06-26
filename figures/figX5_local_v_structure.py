#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D histograms of local velocity (U/V) field. Moving groups etc. Adapted from
a script made by A. Widmark.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
from os.path import exists

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_posvel
from src.constants import U_SUN, V_SUN


def create_plot_data(dfile, d_cut, N_bins, u_min, u_max, v_min, v_max):
    # load data
    print(">>>Loading data:")
    ddir = get_datadir()
    df = pd.read_hdf(ddir + 'DR3_6D/test.hdf5')
    gmm = pd.read_hdf(ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5")
    df = df.merge(gmm, on='source_id')
    N = len(df)

    # get random distance sample
    print(">>>Getting distance samples:")
    d = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = d[np.arange(N), np.random.choice(np.arange(10), replace=True, size=N)]

    # cut to local stars
    print(">>>Making distance cut:")
    df = df[d < d_cut]
    N = len(df)
    d = d[d < d_cut]

    # sample from GMMs
    print(">>>Getting GMM sample:")
    N_mix = 4
    means = np.stack([df[f'mu{i}'] for i in range(N_mix)], axis=-1)
    variances = np.stack([df[f'var{i}'] for i in range(N_mix)], axis=-1)
    weights = np.stack([df[f'w{i}'] for i in range(N_mix)], axis=-1)
    inds = np.apply_along_axis(lambda x: np.random.choice(4, p=x), axis=1, arr=weights)
    means = means[np.arange(N), inds]
    variances = variances[np.arange(N), inds]
    v_pred = np.random.normal(loc=means, scale=np.sqrt(variances))

    # convert to Cartesians
    print(">>>Converting coordinates:")
    ra = df['ra'].to_numpy()
    dec = df['dec'].to_numpy()
    pmra = df['pmra'].to_numpy()
    pmdec = df['pmdec'].to_numpy()
    v_true = df['v_los'].to_numpy()
    u0, v0, w0 = convert_posvel(ra, dec, d, pmra, pmdec, v_true)[1].T
    u1, v1, w1 = convert_posvel(ra, dec, d, pmra, pmdec, v_pred)[1].T

    # histograms
    print(">>>Generating histograms:")
    u_bins = np.linspace(u_min, u_max, N_bins + 1)
    v_bins = np.linspace(v_min, v_max, N_bins + 1)
    h0 = np.histogram2d(u0 - U_SUN, v0 - V_SUN, bins=(u_bins, v_bins))[0]
    h1 = np.histogram2d(u1 - U_SUN, v1 - V_SUN, bins=(u_bins, v_bins))[0]

    # save
    np.savez(dfile, u_bins=u_bins, v_bins=v_bins, h0=h0, h1=h1)
    return


if __name__ == "__main__":

    # plot params
    d_cut = 0.2
    N_bins = 40
    u_min = -65
    u_max = 45
    v_min = -75
    v_max = 35

    # load plot data
    ddir = get_datadir()
    dfile = ddir + 'figures/figX5_local_v_structure_data.npz'
    if not exists(dfile):
        create_plot_data(dfile, d_cut, N_bins, u_min, u_max, v_min, v_max)
    data = np.load(dfile)
    u_bins = data['u_bins']
    v_bins = data['v_bins']
    h0 = data['h0']
    h1 = data['h1']

    # custom colour map
    clist = [
        '#ffffff', '#f0f6f6', '#e2eded', '#d4e5e4',
        '#c5dcdb', '#b7d3d3', '#a9cbca', '#9ac2c1',
        '#8cbab9', '#7db1b0', '#6ea9a8', '#5fa09f',
        '#4f9897', '#3d908f', '#288787', '#007f7f'
    ]
    cmap = LSCmap.from_list("", clist)

    # set up figure
    aspect = 3.35 / 5
    fig = plt.figure(figsize=(3.35, 3.35 / aspect))
    left = 0.15
    right = 0.85
    bottom = 0.08
    top = 0.9
    CdX = 0.035
    dX = right - CdX - left
    dY = aspect * dX
    CdY = 2 * dY
    X0 = left
    Y0 = bottom + dY
    Y1 = bottom
    CX = right - CdX
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    cax = fig.add_axes([CX, Y1, CdX, CdY])

    # plot
    cmargs = {'vmin': 0, 'vmax': 500, 'cmap': cmap, 'rasterized': True}
    cm0 = ax0.pcolormesh(u_bins, v_bins, h0.T, **cmargs)
    cm1 = ax1.pcolormesh(u_bins, v_bins, h1.T, **cmargs)

    # colorbar
    plt.colorbar(cm0, cax=cax)

    # ticks, labels, etc.
    ax1.set_xlabel(r'$U$ [km/s]')
    ax0.set_ylabel(r'$V$ [km/s]')
    ax1.set_ylabel(r'$V$ [km/s]')
    for ax in [ax0, ax1]:
        ax.tick_params(right=True, top=True, direction='inout')
    ax0.tick_params(labelbottom=False)
    ax0.text(0.05, 0.95, "True $v_\mathrm{los}$", ha='left', va='top', transform=ax0.transAxes)
    ax1.text(0.05, 0.95, "Predicted $v_\mathrm{los}$", ha='left', va='top', transform=ax1.transAxes)
    cax.set_ylabel("Density [arbitrary units]")

    # save
    fig.savefig("figX5_local_v_structure.pdf", dpi=800)

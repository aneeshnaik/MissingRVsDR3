#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map of mean vertical velocity (W) in Galactic disc.
Adapted from script by Axel Widmark.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d as bin2d
from scipy.ndimage import gaussian_filter
from tqdm import trange
from os.path import exists

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_posvel
from src.gmm import gmm_batchsample
from src.constants import D_GC


def load_merged_5D_set(i):
    """Load Gaia and GMM catalogue for dataset i, merge, return."""
    ddir = get_datadir()
    gaia_file = ddir + f"DR3_5D/{i}.hdf5"
    gmm_file = ddir + f"DR3_predictions/5D_{i}_GMM_catalogue.hdf5"
    df = pd.read_hdf(gaia_file)
    df = df.merge(pd.read_hdf(gmm_file), on='source_id')
    return df


def quality_cut(df, f_cut, sig_cut):
    """Return DataFrame after applying distance/RV error cuts."""
    # get distances and errors
    d_matrix = np.stack([df[f'd{i}'] for i in range(10)], axis=-1)
    d = np.mean(d_matrix, axis=-1)
    f = np.std(d_matrix, axis=-1) / d

    # rv error
    sig = (0.5 * (df['q841'] - df['q159'])).to_numpy()

    # cut by distance and RV error
    m = (f < f_cut) & (sig < sig_cut)
    return df[m]


def get_XYZW_df(df, X_min, X_max, Y_min, Y_max, Z_lim):
    """From merged df, construct new df with only X, Y, Z, W, spatially cut."""

    # get distances
    d = np.mean(np.stack([df[f'd{i}'] for i in range(10)]), axis=0)

    # GMM params
    means = np.stack([df[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([df[f'var{i}'] for i in range(4)], axis=-1)
    weights = np.stack([df[f'w{i}'] for i in range(4)], axis=-1)

    # sample vel
    v_los = gmm_batchsample(weights, means, variances)

    # coordinate conversion
    ra = df['ra'].to_numpy()
    dec = df['dec'].to_numpy()
    pmra = df['pmra'].to_numpy()
    pmdec = df['pmdec'].to_numpy()
    pos, vel = convert_posvel(ra, dec, d, pmra, pmdec, v_los)

    # construct new dataframe w/ only X, Y, Z, W
    data = {'X': pos[:, 0], 'Y': pos[:, 1], 'Z': pos[:, 2], 'W': vel[:, 2]}
    df = pd.DataFrame(data)

    # spatial cut
    Z_min = -Z_lim
    Z_max = Z_lim
    m = ((df['X'] > X_min) & (df['X'] < X_max)
         & (df['Y'] > Y_min) & (df['Y'] < Y_max)
         & (df['Z'] > Z_min) & (df['Z'] < Z_max))
    return df[m]


def create_plot_data(dfile, f_cut, sig_cut, X_min, X_max, Y_min, Y_max, Z_lim):

    # construct X/Y bins
    N_bins = 28
    X_edges = np.linspace(X_min, X_max, N_bins + 1)
    X_cens = 0.5 * (X_edges[1:] + X_edges[:-1])
    Y_edges = np.linspace(Y_min, Y_max, N_bins + 1)
    Y_cens = 0.5 * (Y_edges[1:] + Y_edges[:-1])
    bins = (X_edges, Y_edges)
    W_tot = np.zeros((N_bins, N_bins))
    WN_tot = np.zeros((N_bins, N_bins))
    WS_tot = np.zeros((N_bins, N_bins))
    N = np.zeros((N_bins, N_bins))
    NN = np.zeros((N_bins, N_bins))
    NS = np.zeros((N_bins, N_bins))

    # loop over datasets
    for i in trange(32):

        # load quality-cut data
        df = quality_cut(load_merged_5D_set(i), f_cut, sig_cut)

        # convert to XYZW (w/ spatial cut)
        df = get_XYZW_df(df, X_min, X_max, Y_min, Y_max, Z_lim)
        Z = df['Z']
        X = df['X']
        XN = X[Z > 0]
        XS = X[Z < 0]
        Y = df['Y']
        YN = Y[Z > 0]
        YS = Y[Z < 0]
        W = df['W']
        WN = W[Z > 0]
        WS = W[Z < 0]

        # get total W and N in bins
        W_tot += bin2d(X, Y, W, statistic='sum', bins=bins)[0]
        WN_tot += bin2d(XN, YN, WN, statistic='sum', bins=bins)[0]
        WS_tot += bin2d(XS, YS, WS, statistic='sum', bins=bins)[0]
        N += bin2d(X, Y, W, statistic='count', bins=bins)[0]
        NN += bin2d(XN, YN, WN, statistic='count', bins=bins)[0]
        NS += bin2d(XS, YS, WS, statistic='count', bins=bins)[0]

    # save
    np.savez(
        dfile,
        W_tot=W_tot, WN_tot=WN_tot, WS_tot=WS_tot,
        N=N, NN=NN, NS=NS
    )
    return


if __name__ == "__main__":

    # plot params
    f_cut = 0.2
    sig_cut = 80
    X_min = -18
    X_max = -4
    Y_min = -7
    Y_max = 7
    Z_lim = 0.6
    N_cut = 50

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig9_W_map_data.npz"
    if not exists(dfile):
        create_plot_data(
            dfile, f_cut, sig_cut, X_min, X_max, Y_min, Y_max, Z_lim
        )
    data = np.load(dfile)
    W_tot = data['W_tot']
    WN_tot = data['WN_tot']
    WS_tot = data['WS_tot']
    N = data['N']
    NN = data['NN']
    NS = data['NS']

    # get mean from W_tot / N
    with np.errstate(divide='ignore', invalid='ignore'):
        V0 = W_tot / N
        VN = WN_tot / NN
        VS = WS_tot / NS

    # nan bins with <N_cut stars
    V0[N < N_cut] = np.nan
    VN[NN < N_cut] = np.nan
    VS[NS < N_cut] = np.nan

    # apply gaussian filter
    V0 = gaussian_filter(V0, sigma=0.6)
    V1 = gaussian_filter(VN - VS, sigma=0.6)

    # set up figure
    asp = 3.35 / 5.4
    left = 0.13
    right = 0.83
    bottom = 0.08
    X0 = left
    dX = right - left
    dY = asp * dX
    Y0 = bottom + dY
    Y1 = bottom
    CX = X0 + dX
    CY = Y1
    CdX = 0.035
    CdY = 2 * dY
    fig = plt.figure(figsize=(3.35, 3.35 / asp))
    tX = X0 + 0.5 * dX
    fig.suptitle("5D predictions: vertical velocity and N/S asymmetry", x=tX)
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    cax = fig.add_axes([CX, CY, CdX, CdY])

    # plot
    clist = ['teal', '#ffffff', 'goldenrod']
    cmap = LSCmap.from_list("", clist)
    extent = (X_min, X_max, Y_min, Y_max)
    imargs = {
        'extent': extent,
        'vmin': -6, 'vmax': 6,
        'cmap': cmap,
        'interpolation': 'none',
        'origin': 'lower'
    }
    im0 = ax0.imshow(V0.T, **imargs)
    im1 = ax1.imshow(V1.T, **imargs)

    # limits, ticks
    for ax in [ax0, ax1]:
        ax.set_xlim(X_min, X_max)
        ax.set_ylim(Y_min, Y_max)
        ax.tick_params(direction='inout', top=True, right=True)

    # aximuthal and radial grid lines
    gridargs = dict(c='k', ls='dotted', lw=0.5)
    for R in np.arange(2, 20, 2):
        phi = np.linspace(0, 2 * np.pi, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)
        ax1.plot(X, Y, **gridargs)
    for phi in np.linspace(0, 2 * np.pi, 17):
        R = np.linspace(0, 20, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)
        ax1.plot(X, Y, **gridargs)

    # solar position
    ax0.scatter([-8.122], [0], c='k', marker='x', s=20)
    ax1.scatter([-8.122], [0], c='k', marker='x', s=20)

    # titles
    bbox_props = dict(
        boxstyle=patches.BoxStyle.Round(pad=0.4, rounding_size=0.2),
        facecolor='w',
        alpha=0.6,
        edgecolor='k'
    )
    targs = {
        'fontsize': 12,
        'va': 'top', 'ha': 'left',
        'bbox': bbox_props,
        'usetex': True
    }
    t0 = r"$\overline{W} + W_\odot$"
    t1 = r"$\overline{W}_N - \overline{W}_S$"
    ax0.text(0.05, 0.95, t0, transform=ax0.transAxes, **targs)
    ax1.text(0.05, 0.95, t1, transform=ax1.transAxes, **targs)

    # axis labels
    ax0.set_xlabel(r"$X$ [kpc]", usetex=True)
    ax1.set_xlabel(r"$X$ [kpc]", usetex=True)
    ax0.set_ylabel(r"$Y$ [kpc]", usetex=True)
    ax1.set_ylabel(r"$Y$ [kpc]", usetex=True)

    # colourbar
    plt.colorbar(im0, cax=cax)
    cax.set_ylabel("Velocity [km/s]")

    # save
    plt.savefig("fig9_W_map.pdf", dpi=800)
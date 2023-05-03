#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY.

Created: MONTH YEAR
Author: A. P. Naik
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

from os.path import exists
from scipy.ndimage import gaussian_filter

sys.path.append("..")
from src.utils import get_datadir


def calc_vlos_map(X, Y, V, SIGD, SIGV, N_bins, X_min, X_max, Y_min, Y_max,
                  SIGD_cut, SIGV_cut):

    # bin edges
    X_edges = np.linspace(X_min, X_max, N_bins + 1)
    Y_edges = np.linspace(Y_min, Y_max, N_bins + 1)

    # cut to region and quality
    m = ((X > X_min)
         & (X < X_max)
         & (Y > Y_min)
         & (Y < Y_max)
         & (SIGV < SIGV_cut)
         & (SIGD < SIGD_cut))
    X = X[m]
    Y = Y[m]
    V = V[m]

    # bin
    inds_X = np.digitize(X, X_edges) - 1
    inds_Y = np.digitize(Y, Y_edges) - 1
    inds = inds_X * N_bins + inds_Y

    # order by bin
    order = np.argsort(inds)
    inds = inds[order]
    X = X[order]
    Y = Y[order]
    V = V[order]

    # bin counts
    counts = np.zeros(N_bins**2, dtype=int)
    unq_inds, unq_counts = np.unique(inds, return_counts=True)
    for i, ind in enumerate(unq_inds):
        counts[ind] = unq_counts[i]

    # loop over bins, get median
    counter = 0
    mu = np.zeros((N_bins, N_bins))
    for i in range(N_bins):
        for j in range(N_bins):
            ind = i * N_bins + j
            count = counts[ind]
            i0 = counter
            i1 = counter + count
            if count == 0:
                mu[i, j] = np.nan
            else:
                mu[i, j] = np.median(V[i0:i1])
            counter = i1

    return mu


def create_plot_data(
        dfile, N_bins, X_min, X_max, Y_min, Y_max, SIGD_cut, SIGV_cut
):

    # load data
    print("Loading data:")
    ddir = get_datadir()
    data = np.load(ddir + "figures/figX_median_vlos_map_XYV_data.npz")
    X5 = data['X_5D']
    Y5 = data['Y_5D']
    V5 = data['V_5D']
    SIGD5 = data['SIGD_5D']
    SIGV5 = data['SIGV_5D']
    X6 = data['X_6D']
    Y6 = data['Y_6D']
    V6 = data['V_6D']
    SIGD6 = data['SIGD_6D']
    SIGV6 = data['SIGV_6D']
    print(">>>Done.\n")

    # construct maps
    print("5D map:")
    mu5 = calc_vlos_map(
        X5, Y5, V5, SIGD5, SIGV5,
        N_bins, X_min, X_max, Y_min, Y_max, SIGD_cut, SIGV_cut
    )
    print(">>>Done.\n")
    print("6D map:")
    mu6 = calc_vlos_map(
        X6, Y6, V6, SIGD6, SIGV6,
        N_bins, X_min, X_max, Y_min, Y_max, SIGD_cut, SIGV_cut)
    print(">>>Done.\n")

    np.savez(dfile, mu5=mu5, mu6=mu6)
    return


if __name__ == "__main__":

    # plot region
    N_bins = 360
    X_min = -17
    X_max = 1
    Y_min = -9
    Y_max = 9
    SIGD_cut = 1.5
    SIGV_cut = 80

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figX_median_vlos_map_data.npz"
    if not exists(dfile):
        create_plot_data(
            dfile, N_bins, X_min, X_max, Y_min, Y_max, SIGD_cut, SIGV_cut
        )
    data = np.load(dfile)
    mu5 = data['mu5']
    mu6 = data['mu6']

    # smooth data
    mu5 = gaussian_filter(data['mu5'], sigma=2)
    mu6 = gaussian_filter(data['mu6'], sigma=2)

    # bin edges
    X_edges = np.linspace(X_min, X_max, N_bins + 1)
    Y_edges = np.linspace(Y_min, Y_max, N_bins + 1)
    X_cens = 0.5 * (X_edges[1:] + X_edges[:-1])
    Y_cens = 0.5 * (Y_edges[1:] + Y_edges[:-1])

    # plot settings
    extent = [X_max, X_min, Y_max, Y_min]
    imargs = dict(
        origin='lower', vmax=110, vmin=-110, extent=extent,
        cmap='Spectral_r', interpolation='none'
    )
    gridargs = dict(c='k', ls='dotted', lw=0.5)

    # set up figure
    asp = 3.35 / 6.4
    fig = plt.figure(figsize=(3.35, 3.35 / asp), dpi=150)
    X0 = 0.13
    X1 = 0.95
    dX = X1 - X0
    dY = asp * dX
    Y0 = 0.06
    Y1 = Y0 + dY
    Y2 = Y1 + dY
    cdY = 0.025
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    cax = fig.add_axes([X0, Y2, dX, cdY])

    # plot maps
    im0 = ax0.imshow(np.flip(mu5).T, **imargs)
    im1 = ax1.imshow(np.flip(mu6).T, **imargs)

    # contours
    ax0.contour(X_cens, Y_cens, mu5.T, levels=[0], colors='k', linewidths=0.75)
    ax1.contour(X_cens, Y_cens, mu6.T, levels=[0], colors='k', linewidths=0.75)

    # colourbar
    plt.colorbar(im0, cax=cax, orientation='horizontal')

    # solar position
    ax0.scatter([-8.122], [0], c='k', marker='x', s=20)
    ax1.scatter([-8.122], [0], c='k', marker='x', s=20)

    # aximuthal grid lines
    for R in np.arange(2, 20, 2):
        phi = np.linspace(0, 2 * np.pi, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)
        ax1.plot(X, Y, **gridargs)

    # radial grid lines
    for phi in np.linspace(0, 2 * np.pi, 17):
        R = np.linspace(0, 20, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)
        ax1.plot(X, Y, **gridargs)

    # axis limits
    for ax in [ax0, ax1]:
        ax.set_xlim(X_max, X_min)
        ax.set_ylim(Y_max, Y_min)

    # axis labels
    ax0.set_xlabel(r'$X\ [\mathrm{kpc}]$')
    ax0.set_ylabel(r'$Y\ [\mathrm{kpc}]$')
    ax1.set_ylabel(r'$Y\ [\mathrm{kpc}]$')
    cax.set_xlabel(r"Median $v_\mathrm{los}$ [km/s]")
    cax.xaxis.set_label_position('top')

    # ticks
    ax0.tick_params(right=True, top=True, direction='inout')
    ax1.tick_params(right=True, top=True, direction='inout')
    ax1.tick_params(labelbottom=False)
    cax.xaxis.tick_top()

    # labels
    for i in range(2):
        ax = [ax0, ax1][i]
        lab = [r'\textit{Gaia} 5D stars', r'\textit{Gaia} 6D stars'][i]
        bbox = dict(facecolor='white', alpha=0.85, edgecolor='teal')
        t = ax.text(0.95, 0.95, lab, transform=ax.transAxes,
                    ha='right', va='top', bbox=bbox)

    # save
    plt.savefig("figX_median_vlos_map.pdf")

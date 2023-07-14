#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D map of median RV in disc plane.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

from os.path import exists
from scipy.ndimage import gaussian_filter
from tqdm import trange

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_pos
from src.constants import D_GC


def calc_vlos_map(X, Y, V, N_bins, X_min, X_max, Y_min, Y_max):

    # bin edges
    X_edges = np.linspace(X_min, X_max, N_bins + 1)
    Y_edges = np.linspace(Y_min, Y_max, N_bins + 1)

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
    for i in trange(N_bins):
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


def create_plot_data(dfile, N_bins, X_min, X_max, Y_min, Y_max):

    # load XYV data
    print("Loading XYV data:")
    ddir = get_datadir()
    data = np.load(ddir + "figures/fig8_median_vlos_map_XYV_data.npz")
    X5 = data['X_5D']
    Y5 = data['Y_5D']
    V5 = data['V_5D']
    X6 = data['X_6D']
    Y6 = data['Y_6D']
    V6 = data['V_6D']
    print(">>>Done.\n")

    # construct maps
    print("5D map:")
    mu5 = calc_vlos_map(
        X5, Y5, V5, N_bins, X_min, X_max, Y_min, Y_max
    )
    print(">>>Done.\n")
    print("6D map:")
    mu6 = calc_vlos_map(
        X6, Y6, V6, N_bins, X_min, X_max, Y_min, Y_max
    )
    print(">>>Done.\n")

    # load training set, calculate 99% distance bounds
    print("99% spatial bounds:")
    df = pd.read_hdf(get_datadir() + 'DR3_6D/train.hdf5')
    d = np.mean(np.stack([df[f'd{i}'] for i in range(10)], axis=-1), axis=-1)
    X, Y, Z = convert_pos(df['ra'].to_numpy(), df['dec'].to_numpy(), d).T
    lon = np.arctan2(Y, X + D_GC)
    lon[lon < 0] += 2 * np.pi
    N_bins = 32
    l_edges = np.linspace(0, 2 * np.pi, N_bins + 1)
    l_cens = 0.5 * (l_edges[1:] + l_edges[:-1])
    inds = np.digitize(lon, l_edges) - 1
    d99 = np.array([np.percentile(d[inds == i], q=(99)) for i in range(N_bins)])
    l_cens = np.append(l_cens, l_cens[0])
    d99 = np.append(d99, d99[0])
    X99 = d99 * np.cos(l_cens) - D_GC
    Y99 = d99 * np.sin(l_cens)
    print(">>>Done.\n")

    np.savez(dfile, mu5=mu5, mu6=mu6, X99=X99, Y99=Y99)
    return


if __name__ == "__main__":

    # plot region
    N_bins = 360
    X_min = -17
    X_max = 1
    Y_min = -9
    Y_max = 9

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig8_median_vlos_map_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, N_bins, X_min, X_max, Y_min, Y_max)
    data = np.load(dfile)
    mu5 = data['mu5']
    mu6 = data['mu6']
    X99 = data['X99']
    Y99 = data['Y99']

    # smooth data
    mu5 = gaussian_filter(data['mu5'], sigma=3)
    mu6 = gaussian_filter(data['mu6'], sigma=3)

    # bin edges
    X_edges = np.linspace(X_min, X_max, N_bins + 1)
    Y_edges = np.linspace(Y_min, Y_max, N_bins + 1)
    X_cens = 0.5 * (X_edges[1:] + X_edges[:-1])
    Y_cens = 0.5 * (Y_edges[1:] + Y_edges[:-1])

    # plot settings
    extent = [X_min, X_max, Y_min, Y_max]
    clist = ['teal', '#ffffff', 'goldenrod']
    cmap = LSCmap.from_list("", clist)
    imargs = dict(
        origin='lower', vmax=110, vmin=-110, extent=extent,
        cmap=cmap, interpolation='none'
    )
    gridargs = dict(c='k', ls='dotted', lw=0.5)

    # set up figure
    asp = 3.35 / 6.8
    fig = plt.figure(figsize=(3.35, 3.35 / asp))
    X0 = 0.13
    X1 = 0.95
    dX = X1 - X0
    dY = asp * dX
    Y0 = 0.06
    Y1 = Y0 + dY
    Y2 = Y1 + dY
    cdY = 0.025
    fig.suptitle("5D predictions: radial velocity map", x=X0 + 0.5 * dX)
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    cax = fig.add_axes([X0, Y2, dX, cdY])

    # plot maps
    im0 = ax0.imshow(mu5.T, **imargs)
    im1 = ax1.imshow(mu6.T, **imargs)

    # contours
    ax0.contour(X_cens, Y_cens, mu5.T, levels=[0], colors='k', linewidths=1)
    ax1.contour(X_cens, Y_cens, mu6.T, levels=[0], colors='k', linewidths=1)

    # colourbar
    plt.colorbar(im0, cax=cax, orientation='horizontal')

    # solar position
    ax0.scatter([-8.122], [0], c='k', marker='x', s=20)
    ax1.scatter([-8.122], [0], c='k', marker='x', s=20)

    # training set spatial bounds
    ax0.plot(X99, Y99, c='k', ls='dotted')
    ax1.plot(X99, Y99, c='k', ls='dotted')

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
        ax.set_xlim(X_min, X_max)
        ax.set_ylim(Y_min, Y_max)

    # axis labels
    ax0.set_xlabel(r'$X\ [\mathrm{kpc}]$', usetex=True)
    ax0.set_ylabel(r'$Y\ [\mathrm{kpc}]$', usetex=True)
    ax1.set_ylabel(r'$Y\ [\mathrm{kpc}]$', usetex=True)
    cax.set_xlabel(r"Median radial velocity [km/s]")
    cax.xaxis.set_label_position('top')

    # ticks
    ax0.tick_params(right=True, top=True, direction='inout')
    ax1.tick_params(right=True, top=True, direction='inout')
    ax1.tick_params(labelbottom=False)
    cax.xaxis.tick_top()

    # labels
    for i in range(2):
        ax = [ax0, ax1][i]
        lab = [r'Gaia 5D stars', 'Gaia 6D stars'][i]
        bbox = dict(facecolor='white', alpha=0.85, edgecolor='teal')
        t = ax.text(0.05, 0.95, lab, transform=ax.transAxes,
                    ha='left', va='top', bbox=bbox)
    ax1.text(0.825, 0.15, r"$v_\mathrm{los} = 0$", transform=ax1.transAxes, usetex=True)
    ax1.arrow(0.81, 0.16, -0.03, 0, width=0.004, transform=ax1.transAxes, fc='k', ec='none', alpha=0.8)
    ax1.arrow(0.83, 0.185, -0.08, 0.32, width=0.004, transform=ax1.transAxes, fc='k', ec='none', alpha=0.8)
    ax1.text(0.2, 0.725, "bounds\n99% of\ntraining set", transform=ax1.transAxes, ha='center')
    ax1.arrow(0.225, 0.71, 0.09, -0.07, width=0.004, transform=ax1.transAxes, fc='k', ec='none', alpha=0.8)

    # save
    plt.savefig("fig8_median_vlos_map.pdf", dpi=800)

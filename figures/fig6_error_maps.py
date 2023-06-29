#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disc plane map and sky map of error rate on DR3 6D test set.
Adapted from script by Axel Widmark.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

from os.path import exists
from scipy.stats import binned_statistic_2d as bin2d

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_pos
from src.gmm import gmm_cdf_batch


def calc_skymap(F, X, Y, Z, N_side, N_bins, N_phi, N_theta):

    # progress message
    print(">>>Calculating sky map:")

    # hp pixels
    N_pix = hp.nside2npix(N_side)
    pix = hp.pixelfunc.vec2pix(N_side, X, Y, Z, nest=False)

    # loop over pixels; calc histogram
    y = np.zeros((N_pix, N_bins))
    bin_edges = np.linspace(0, 1, N_bins + 1)
    for i in range(N_pix):
        hist, bin_edges = np.histogram(F[pix == i], bin_edges, density=True)
        y[i, :] = hist

    # calculate error map
    errormap = 0.5 * np.mean(np.abs(y - 1), axis=1)

    # angle coordinate arrays
    phi = np.linspace(-np.pi, np.pi, N_phi)
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_theta)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

    # convert to Cartesian unit vectors. Note: phi has minus sign in order to
    # 'flip' figure so (east is left)
    hpx = np.cos(theta) * np.cos(-phi)
    hpy = np.cos(theta) * np.sin(-phi)
    hpz = np.sin(theta)

    # get map values on grid
    y = errormap[hp.vec2pix(16, hpx, hpy, hpz)]

    return y, phi, theta


def calc_discmap(F, X, Y, N_bins, N_disc_bins, d_max):

    # progress message
    print(">>>Calculating disc map:")

    # set up bins
    x_edges = np.linspace(-d_max, d_max, N_disc_bins + 1)
    e = [x_edges, x_edges]

    # function to calculate error rate from 1D array of F
    def calc_error_rate(F):
        bin_edges = np.linspace(0, 1, N_bins + 1)
        h = np.histogram(F, bin_edges, density=True)[0]
        return 0.5 * np.mean(np.abs(h - 1))

    # calculate counts and error in 2d bins
    n = bin2d(X, Y, F, bins=e, statistic='count')[0]
    z = bin2d(X, Y, F, bins=e, statistic=calc_error_rate)[0]

    # construct 2d x y grids
    x_grid, y_grid = np.meshgrid(x_edges - 8.122, x_edges, indexing='ij')
    return z, n, x_grid, y_grid


def get_FXYZ():

    # data directory
    print(">>>Loading data:")
    ddir = get_datadir()
    gaia_file = ddir + "DR3_6D/test.hdf5"
    gmm_file = ddir + "DR3_predictions/6D_test_GMM_catalogue.hdf5"
    test_set = pd.read_hdf(gaia_file)
    test_set = test_set.merge(pd.read_hdf(gmm_file), on='source_id')

    # calculate GMM posterior positions
    print(">>>Get posterior positions:")
    weights = np.stack([test_set[f'w{i}'] for i in range(4)], axis=-1)
    means = np.stack([test_set[f'mu{i}'] for i in range(4)], axis=-1)
    variances = np.stack([test_set[f'var{i}'] for i in range(4)], axis=-1)
    x = test_set['v_los'].to_numpy()
    F = gmm_cdf_batch(x, weights, means, variances)

    # get Cartesian positions
    print(">>>Converting coordinates:")
    d = np.mean([test_set[f'd{i}'] for i in range(10)], axis=0)
    ra = test_set['ra'].to_numpy()
    dec = test_set['dec'].to_numpy()
    X, Y, Z = (convert_pos(ra, dec, d) - np.array([-8.122, 0, 0.0208])).T
    return F, X, Y, Z


def create_plot_data(dfile, N_side, N_bins, N_phi, N_theta, N_disc, d_max):

    # get test set positions and quantiles
    F, X, Y, Z = get_FXYZ()

    # calculate sky map
    z_sky, phi, theta = calc_skymap(F, X, Y, Z, N_side, N_bins, N_phi, N_theta)

    # calculate disc map
    z_disc, n, x, y = calc_discmap(F, X, Y, N_bins, N_disc, d_max)

    # save
    print(">>>Saving:")
    np.savez(
        dfile,
        z_sky=z_sky, phi=phi, theta=theta,
        z_disc=z_disc, n=n, x=x, y=y,
    )
    return


if __name__ == "__main__":

    # plot params
    N_side = 16    # healpix Nside parameter for sky map
    N_bins = 10    # number of bins for F histogram
    N_phi = 512    # number of phi pixels in sky map
    N_theta = 512  # number of theta pixels in sky map
    N_disc = 60    # number of bins in each direction in disc map
    d_max = 12     # half-width of disc map (kpc)

    # load plot data
    ddir = get_datadir()
    dfile = ddir + "figures/fig6_error_maps_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, N_side, N_bins, N_phi, N_theta, N_disc, d_max)
    data = np.load(dfile)
    z_sky = data['z_sky']
    phi = data['phi']
    theta = data['theta']
    z_disc = data['z_disc']
    n = data['n']
    x = data['x']
    y = data['y']

    # plot settings
    vmax = 0.25
    clist = [
        '#ffffff', '#f0f6f6', '#e2eded', '#d4e5e4',
        '#c5dcdb', '#b7d3d3', '#a9cbca', '#9ac2c1',
        '#8cbab9', '#7db1b0', '#6ea9a8', '#5fa09f',
        '#4f9897', '#3d908f', '#288787', '#007f7f'
    ]
    cmap = LSCmap.from_list("", clist)

    # set up figure
    aspect = 7 / 4.1
    left = 0.1
    right = 0.965
    bottom = 0.09
    hgap = 0.03
    vgap = 0.04
    dX = (right - left - hgap) / 2
    dY = aspect * dX
    cdY = 0.025
    Y0 = bottom
    Y1 = bottom + cdY + vgap
    X0 = left
    X1 = left + dX + hgap
    fig = plt.figure(figsize=(7, 7 / aspect))
    fig.suptitle("6D test set: error rate maps", x=left + 0.5 * (right - left))
    ax0 = fig.add_axes([X0, Y1, dX, dY])
    ax1 = fig.add_axes([X1, Y1, dX, dY], projection='aitoff')
    cax = fig.add_axes([X0, Y0, 2 * dX + hgap, 0.05])

    # plot disc map
    z_disc[n < 50] = np.nan
    ax0.pcolormesh(x, y, z_disc, vmin=0, vmax=vmax, cmap=cmap, rasterized=True)

    # plot sky map
    ax1.pcolormesh(phi, theta, z_sky, vmin=0, vmax=vmax, cmap=cmap, rasterized=True)
    mappable = ScalarMappable(Normalize(vmin=0, vmax=vmax), cmap=cmap)
    plt.colorbar(mappable, orientation='horizontal', cax=cax)

    # solar position
    ax0.scatter([-8.122], [0], c='k', marker='x', s=20)
    
    # aximuthal grid lines
    gridargs = dict(c='k', ls='dotted', lw=0.5)
    for R in np.arange(2, 30, 2):
        phi = np.linspace(0, 2 * np.pi, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)

    # radial grid lines
    for phi in np.linspace(0, 2 * np.pi, 17):
        R = np.linspace(0, 30, 500)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        ax0.plot(X, Y, **gridargs)

    # grid, ticks, labels etc
    ax0.set_xlim(-8.122 - d_max, -8.122 + d_max)
    ax0.set_ylim(-d_max, d_max)
    ax0.tick_params(right=True, top=True, direction='inout', labelbottom=False, labeltop=True)
    ax1.tick_params(labelright=True, labelleft=False)
    ax0.set_xlabel(r"$X$ [kpc]", usetex=True)
    ax0.set_ylabel(r"$Y$ [kpc]", usetex=True)
    ax0.xaxis.set_label_position('top')
    ax1.grid()
    ax1.set_xticklabels(ax1.get_xticklabels()[::-1])
    cax.set_xlabel("Error rate")

    # save figure
    fig.savefig("fig6_error_maps.pdf", dpi=800)

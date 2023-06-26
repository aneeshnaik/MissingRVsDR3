#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sky map of error rate. Adapted from a script made by A. Widmark.

Created: June 2023
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
import healpy as hp

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

from os.path import exists

sys.path.append("..")
from src.utils import get_datadir
from src.coords import convert_pos
from src.gmm import gmm_cdf_batch


def create_plot_data(dfile, N_side, N_bins, N_phi, N_theta):

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

    # hp pixels
    print(">>>Performing healpix decomposition:")
    N_pix = hp.nside2npix(N_side)
    pix = hp.pixelfunc.vec2pix(N_side, X, Y, Z, nest=False)

    # loop over pixels; calc histogram
    print(">>>Looping over pixels:")
    y = np.zeros((N_pix, N_bins))
    bin_edges = np.linspace(0, 1, N_bins + 1)
    for i in range(N_pix):
        hist, bin_edges = np.histogram(F[pix == i], bin_edges, density=True)
        y[i, :] = hist

    # calculate error map
    print(">>>Calculating error map:")
    errormap = 0.5 * np.mean(np.abs(y - 1), axis=1)

    # angle coordinate arrays
    print(">>>Constructing spherical grid:")
    phi = np.linspace(-np.pi, np.pi, N_phi)
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_theta)
    phi, theta = np.meshgrid(phi, theta, indexing='ij')

    # convert to Cartesian unit vectors. Note: phi has minus sign in order to
    # 'flip' figure so (east is left)
    hpx = np.cos(theta) * np.cos(-phi)
    hpy = np.cos(theta) * np.sin(-phi)
    hpz = np.sin(theta)

    # get map values on grid
    print(">>>Getting map values on grid:")
    y = errormap[hp.vec2pix(16, hpx, hpy, hpz)]

    # save
    print(">>>Saving:")
    np.savez(dfile, y=y, phi=phi, theta=theta)
    return


if __name__ == "__main__":

    # plot params
    N_side = 16    # healpix Nside parameter
    N_bins = 10    # number of bins for F histogram
    N_phi = 512    # number of phi pixels in map
    N_theta = 512  # number of theta pixels in map

    # load plot data
    ddir = get_datadir()
    dfile = ddir + "figures/figX3_sky_accuracy_data.npz"
    if not exists(dfile):
        create_plot_data(dfile, N_side, N_bins, N_phi, N_theta)
    data = np.load(dfile)
    y = data['y']
    phi = data['phi']
    theta = data['theta']

    # custom colour map
    clist = [
        '#ffffff', '#f0f6f6', '#e2eded', '#d4e5e4',
        '#c5dcdb', '#b7d3d3', '#a9cbca', '#9ac2c1',
        '#8cbab9', '#7db1b0', '#6ea9a8', '#5fa09f',
        '#4f9897', '#3d908f', '#288787', '#007f7f'
    ]
    cmap = LSCmap.from_list("", clist)

    # set up figure
    left = 0.02
    right = 0.98
    dX = right - left
    fig = plt.figure(figsize=(3.35, 2.35))
    ax = fig.add_axes([left, 0.1, dX, 1], projection='aitoff')
    cax = fig.add_axes([left + 0.02, 0.15, dX - 0.04, 0.035])

    # plot
    cm = ax.pcolormesh(phi, theta, y, vmin=0, vmax=0.25, cmap=cmap, rasterized=True)

    # colorbar
    plt.colorbar(cm, cax=cax, orientation='horizontal')

    # grid, ticks, labels etc
    ax.grid()
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.set_xticklabels(ax.get_xticklabels()[::-1])
    cax.set_xlabel("Error rate")

    # save
    fig.savefig("figX3_sky_accuracy.pdf", dpi=800)

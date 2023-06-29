#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot of spatial distribution of 6D training set.

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
from src.coords import convert_pos


def create_plot_data(dfile):

    # load data
    print(">>>Loading data:")
    ddir = get_datadir()
    train = pd.read_hdf(ddir + 'DR3_6D/train.hdf5')
    N = len(train)

    # get ra/dec
    ra = train['ra'].to_numpy()
    dec = train['dec'].to_numpy()

    # get random distance sample
    print(">>>Getting distance samples:")
    d = np.stack([train[f'd{i}'] for i in range(10)], axis=-1)
    d = d[np.arange(N), np.random.choice(np.arange(10), replace=True, size=N)]

    # convert to Cartesian
    print(">>>Converting to Cartesians:")
    X, Y, Z = convert_pos(ra, dec, d).T

    # calculate histograms
    print(">>>Calculating histograms:")
    edges = np.linspace(-3.5, 3.5, 40)
    cens = 0.5 * (edges[1:] + edges[:-1])
    nX, _ = np.histogram(X, -8.122 + edges, density=True)
    nY, _ = np.histogram(Y, edges, density=True)
    nZ, _ = np.histogram(Z, edges, density=True)

    # save
    print(">>>Saving:")
    np.savez(dfile, cens=cens, nX=nX, nY=nY, nZ=nZ)
    return


if __name__ == "__main__":

    # load plot data
    ddir = get_datadir()
    dfile = ddir + 'figures/fig3_training_set_extent_data.npz'
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    cens = data['cens']
    nX = data['nX']
    nY = data['nY']
    nZ = data['nZ']

    # setup figure
    fig = plt.figure(figsize=(3.35, 4))
    left = 0.13
    right = 0.99
    bottom = 0.06
    top = 0.94
    dX = right - left
    dY = (top - bottom) / 3
    axX = fig.add_axes([left, bottom + 2 * dY, dX, dY])
    axY = fig.add_axes([left, bottom + dY, dX, dY])
    axZ = fig.add_axes([left, bottom, dX, dY])
    fig.suptitle("DR3 training set: spatial distribution", x=left + 0.5 * dX)

    axX.barh(-8.122 + cens, nX, height=np.diff(cens)[0], color='teal', rasterized=True)
    axY.barh(cens, nY, height=np.diff(cens)[0], color='teal', rasterized=True)
    axZ.barh(cens, nZ, height=np.diff(cens)[0], color='teal', rasterized=True)

    # plot verticals
    largs = {'ls': 'dashed', 'c': 'k', 'lw': 0.5, 'alpha': 0.5}
    axX.plot([0, 1.5], [-8.122, -8.122], **largs)
    axY.plot([0, 1.5], [0, 0], **largs)
    axZ.plot([0, 1.5], [0.0208, 0.0208], **largs)

    # ticks, labels, etc.
    axX.set_ylim(-8.122 - 3.5, -8.122 + 3.5)
    axY.set_ylim(-3.5, 3.5)
    axZ.set_ylim(-3.5, 3.5)
    axX.set_ylabel("$X$ [kpc]", usetex=True)
    axY.set_ylabel("$Y$ [kpc]", usetex=True)
    axZ.set_ylabel("$Z$ [kpc]", usetex=True)
    axZ.set_xlabel("Density [arbitrary units]")
    for ax in [axX, axY, axZ]:
        ax.set_xlim(0, 1.5)
        ax.tick_params(direction='inout', labelbottom=False, top=True)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_label_coords(-0.1, 0.5)

    # save
    fig.savefig("fig3_training_set_extent.pdf", dpi=800)

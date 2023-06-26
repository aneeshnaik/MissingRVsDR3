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
    edges = np.linspace(-4, 4, 40)
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
    dfile = ddir + 'figures/figX4_training_set_distribution.npz'
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    cens = data['cens']
    nX = data['nX']
    nY = data['nY']
    nZ = data['nZ']

    # setup figure
    fig = plt.figure(figsize=(3.35, 1.7))
    left = 0.08
    right = 0.98
    bottom = 0.22
    top = 0.98
    dX = (right - left) / 3
    dY = top - bottom
    axX = fig.add_axes([left, bottom, dX, dY])
    axY = fig.add_axes([left + dX, bottom, dX, dY])
    axZ = fig.add_axes([left + 2 * dX, bottom, dX, dY])

    axX.bar(-8.122 + cens, nX, width=np.diff(cens)[0], color='teal', rasterized=True)
    axY.bar(cens, nY, width=np.diff(cens)[0], color='teal', rasterized=True)
    axZ.bar(cens, nZ, width=np.diff(cens)[0], color='teal', rasterized=True)

    # plot verticals
    largs = {'ls': 'dashed', 'c': 'k', 'lw': 0.5, 'alpha': 0.5}
    axX.plot([-8.122, -8.122], [0, 1.5], **largs)
    axY.plot([0, 0], [0, 1.5], **largs)
    axZ.plot([0.0208, 0.0208], [0, 1.5], **largs)

    # ticks, labels, etc.
    for ax in [axX, axY, axZ]:
        ax.set_ylim(0, 1.5)
        ax.tick_params(direction='inout', labelleft=False, right=True, top=True)
    axX.set_xlabel("$X$ [kpc]")
    axY.set_xlabel("$Y$ [kpc]")
    axZ.set_xlabel("$Z$ [kpc]")
    axX.set_ylabel("Density [arbitrary units]")

    # save
    plt.savefig("figX4_training_set_distribution.pdf", dpi=800)

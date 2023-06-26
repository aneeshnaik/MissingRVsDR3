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
from os.path import exists

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap as LSCmap
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir


def create_plot_data(dfile):
    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/figX6_W_map_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    mask_N = data["mask_N"]
    mask_S = data["mask_S"]
    mask_SandN = data["mask_SandN"]
    w_grid_600north = data["w_grid_600north"]
    w_grid_600south = data["w_grid_600south"]
    w_grid_z600pc = data["w_grid_z600pc"]
    spat_vec = data["spat_vec"]
    V0 = mask_SandN * w_grid_z600pc
    V1 = mask_N * w_grid_600north - mask_S * w_grid_600south

    # set up figure
    asp = 3.35 / 5.2
    left = 0.13
    right = 0.83
    top = 0.9
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
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    cax = fig.add_axes([CX, CY, CdX, CdY])

    # plot
    clist = ['teal', '#ffffff', 'goldenrod']
    cmap = LSCmap.from_list("", clist)
    extent = (18.2, -1.8, 10, -10)
    imargs = {'extent': extent, 'vmin': -6, 'vmax': 6, 'cmap': cmap}
    im0 = ax0.imshow(V0.T, **imargs)
    im1 = ax1.imshow(V1.T, **imargs)

    # grid lines, limits, ticks
    for ax in [ax0, ax1]:
        for r in [4, 8, 12, 16, 20]:
            th = np.linspace(0., 2 * np.pi, 1000)
            ax.plot(r * np.sin(th), r * np.cos(th), "k:", alpha=0.2)
        for th in np.arange(0, 2 * np.pi, np.pi / 6):
            r = np.linspace(0., 24, 10)
            ax.plot(r * np.sin(th), r * np.cos(th), "k:", alpha=0.2)
        ax.set_xlim(4.5, 17.5)
        ax.set_ylim(6.5, -6.5)
        ax.tick_params(direction='inout', top=True, right=True)

    # titles
    bbox_props = dict(
        boxstyle=patches.BoxStyle.Round(pad=0.4, rounding_size=0.2),
        facecolor='w',
        alpha=0.6,
        edgecolor='k'
    )
    targs = {'fontsize': 12, 'va': 'top', 'ha': 'left', 'bbox': bbox_props}
    t0 = r"$\overline{W}$"
    t1 = r"$\overline{W}_N - \overline{W}_S$"
    ax0.text(0.05, 0.95, t0, transform=ax0.transAxes, **targs)
    ax1.text(0.05, 0.95, t1, transform=ax1.transAxes, **targs)

    # axis labels
    ax0.set_xlabel(r"$X$ [kpc]")
    ax1.set_xlabel(r"$X$ [kpc]")
    ax0.set_ylabel(r"$Y$ [kpc]")
    ax1.set_ylabel(r"$Y$ [kpc]")

    # colourbar
    plt.colorbar(im0, cax=cax)
    cax.set_ylabel("Velocity [km/s]")

    # save
    plt.savefig("figX6_W_map.pdf")

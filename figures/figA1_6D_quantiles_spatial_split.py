#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantile plots in spatial bins.
Adapted from a script made by A. Widmark.

Created: June 2023
Author: A. P. Naik
"""
import numpy as np
import pandas as pd
import sys
from os.path import exists

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir
from src.gmm import gmm_cdf_batch
from src.coords import convert_pos


def create_plot_data(dfile):

    print("Creating plot data file:")

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

    # convert RA/dec/d to disc plane distance/angle
    print(">>>Get disc-plane distances/angles:")
    ra = test_set['ra'].to_numpy()
    dec = test_set['dec'].to_numpy()
    d = np.mean([test_set[f'd{i}'] for i in range(10)], axis=0)
    X, Y, Z = convert_pos(ra, dec, d).T
    X += 8.122
    XY_d = np.sqrt(X**2 + Y**2)
    XY_ang = np.arctan2(Y, X) % (2 * np.pi)

    # nested loop over angle then distance bins; get bin counts
    print(">>>Calculating bin counts:")
    N_bins = 100
    F_edges = np.linspace(0, 1, N_bins + 1)
    dist_edges = np.array([0, 1, 2, 4, 1e+3])
    ang_edges = np.linspace(0., 2 * np.pi, 7)
    N_dist_bins = len(dist_edges) - 1
    N_ang_bins = len(ang_edges) - 1
    counts = np.zeros((N_ang_bins, N_dist_bins, 100), dtype=int)
    for i in range(N_ang_bins):
        for j in range(N_dist_bins):
            ang0 = ang_edges[i]
            ang1 = ang_edges[i + 1]
            d0 = dist_edges[j]
            d1 = dist_edges[j + 1]
            m = (XY_d >= d0) & (XY_d < d1) & (XY_ang >= ang0) & (XY_ang < ang1)
            counts[i, j] = np.histogram(F[m], F_edges)[0]

    # save plot data
    print(">>>Saving")
    np.save(dfile, counts)
    print(">>>Done.\n")

    return


def add_inset(fig, X, Y, dX, dY, aspect, th_min, th_max, r_min, r_max):

    dXc = 0.3 * dX
    dYc = aspect * dXc
    Xc = X + 0.75 * dX - 0.5 * dXc
    Yc = Y + 0.25 * dY - 0.5 * dYc
    axc = fig.add_axes([Xc, Yc, dXc, dXc], projection='polar')
    axc.set_theta_offset(np.pi)
    axc.set_theta_direction(-1)

    axc.tick_params(labelleft=False, labelbottom=False)
    axc.patch.set_alpha(0)

    th = np.linspace(th_min, th_max, 200)
    r1 = np.linspace(r_min, r_min, 200)
    r2 = np.linspace(r_max, r_max, 200)
    axc.fill_between(th, r1, r2, color=c2)

    axc.set_ylim(0, 5)
    axc.set_thetagrids(np.arange(0, 360, 60))
    axc.set_rgrids([1, 2, 4])
    axc.patch.set_alpha(0.0)
    axc.spines['polar'].set_visible(False)
    axc.grid(linewidth=0.75, color='k', alpha=0.5)

    return


if __name__ == "__main__":

    # load plot data
    ddir = get_datadir()
    dfile = ddir + 'figures/figA1_6D_quantiles_spatial_split_data.npy'
    if not exists(dfile):
        create_plot_data(dfile)
    counts = np.load(dfile)

    # plot settings
    c1 = 'teal'
    c2 = 'goldenrod'

    # make figure
    N_rows = 7
    N_cols = 5
    left = 0.05
    right = 0.87
    top = 0.92
    bottom = 0.05
    dX = (right - left) / N_cols
    dY = (top - bottom) / N_rows
    aspect = 6.9 / 8
    fig = plt.figure(figsize=(6.9, 6.9 / aspect))
    fig.suptitle("6D test set: quantile distributions in spatial bins", x=left + 0.5 * (right - left))

    # loop over panels
    for row in range(N_rows):
        for col in range(N_cols):

            # create axes
            X = left + col * dX
            Y = top - (row + 1) * dY
            ax = fig.add_axes([X, Y, dX, dY])

            # add inset axes
            th_min = [0, 60, 120, 180, 240, 300, 0][row] * np.pi / 180
            th_max = [60, 120, 180, 240, 300, 360, 360][row] * np.pi / 180
            r_min = [0, 0, 1, 2, 4][col]
            r_max = [5, 1, 2, 4, 5][col]
            axc = add_inset(fig, X, Y, dX, dY, aspect, th_min, th_max, r_min, r_max)

            # plot
            x = np.linspace(0.005, 0.995, 100)
            if (row == N_rows - 1) and (col == 0):
                y = counts.sum(axis=0).sum(axis=0)
            elif (row == N_rows - 1):
                y = counts.sum(axis=0)[col - 1]
            elif (col == 0):
                y = counts.sum(axis=1)[row]
            else:
                y = counts[row, col - 1]
            N = y.sum()
            y = 100 * (y / N)
            a = 1 - 0.3 * (row < N_rows - 1) * (col > 0)
            ax.bar(x, y, width=0.01, alpha=a, color=c1, rasterized=True)

            # row/col headings
            if row == 0:
                srow = [
                    r'All distances',
                    r'$\sqrt{X^2+Y^2}<1$',
                    r'$1 \leq\sqrt{X^2+Y^2}< 2$',
                    r'$2 \leq\sqrt{X^2+Y^2} < 4 $',
                    r'$4 \leq \sqrt{X^2+Y^2}$',
                ][col]
                if col == 0:
                    usetex = False
                else:
                    usetex = True
                ax.text(0.5, 1.45, srow, ha='center', va='bottom', usetex=usetex)
            if col == N_cols - 1:
                scol = [
                    r'$l \in [0^\circ, 60^\circ]$',
                    r'$l \in [60^\circ, 120^\circ]$',
                    r'$l \in [120^\circ, 180^\circ]$',
                    r'$l \in [180^\circ, 240^\circ]$',
                    r'$l \in [240^\circ, 300^\circ]$',
                    r'$l \in [300^\circ, 360^\circ]$',
                    r'All $l$'
                ][row]
                if row == 6:
                    usetex = False
                else:
                    usetex = True
                ax.text(1.05, 0.5, scol, ha='left', va='center', usetex=usetex)

            # count label
            label = f"{N} stars"
            ax.text(0.075, 0.925, label, ha='left', va='top', transform=ax.transAxes)

            # plot y=1 line
            ax.plot([0, 1], [1, 1], c=c2, linestyle='dashed')

            # ticks
            ax.set_xticks([0, 0.5, 1])
            ax.tick_params(
                direction="inout",
                bottom=True, top=True, left=False, right=False
            )
            ax.tick_params(labelleft=False)
            if (row < 6) or ((row == 6) and (col != 2)):
                ax.tick_params(labelbottom=False)

            # limits
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.4])
    
            # axis labels
            if row == 3 and col == 0:
                ax.set_ylabel("Frequency Density [arbitrary units]")
            if row == 6 and col == 2:
                t = r'$F(v_\mathrm{true}|\mathrm{model})' \
                    r'= \int_{-\infty}^{v_\mathrm{true}}\mathrm{posterior}(v)dv$'
                ax.set_xlabel(t, usetex=True)

    fig.savefig("figA1_6D_quantiles_spatial_split.pdf", dpi=800)

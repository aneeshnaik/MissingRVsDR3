#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Plot prediction residuals against other stellar quantities.

Created: July 2022
Author: A. P. Naik
"""
import sys
import numpy as np
from os.path import exists

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

sys.path.append("..")
from src.utils import get_datadir


def create_plot_data(dfile):

    # data directory
    ddir = get_datadir()

    # get prediction residuals
    data = np.load(ddir + 'EDR3_predictions/EDR3_prediction_results.npz')
    res = (data['v_true'] - data['mu']) / data['sig']

    # get magnitude and temp
    data = np.load(ddir + 'EDR3_predictions/EDR3_prediction_aux.npz')
    G = data['G']
    T_eff = data['T_eff']

    # percentiles to get
    q = (16, 50, 84)

    # PANEL 1: residuals by G mag
    G_edges = np.append(np.arange(6, 12, 0.4), np.linspace(12, 14.5, 20))
    x0 = 0.5 * (G_edges[1:] + G_edges[:-1])
    N = len(x0)
    inds = np.digitize(G, G_edges) - 1
    lo, y0, hi = np.array([np.percentile(res[inds == i], q) for i in range(N)]).T
    y0err = np.stack((y0 - lo, hi - y0))

    # PANEL 2: residuals by T_eff
    x1 = np.unique(T_eff)
    lo, y1, hi = np.array([np.percentile(res[T_eff == T], q) for T in x1]).T
    y1err = np.stack((y1 - lo, hi - y1))

    # save
    np.savez(dfile, x0=x0, y0=y0, y0err=y0err, x1=x1, y1=y1, y1err=y1err)

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    dfile = "fig2_residual_trends_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    x0 = data['x0']
    y0 = data['y0']
    y0err = data['y0err']
    x1 = data['x1']
    y1 = data['y1']
    y1err = data['y1err']

    # set up figure
    c1 = 'teal'
    c2 = 'goldenrod'
    X0 = 0.1
    X2 = 0.98
    dX = (X2 - X0) / 2
    X1 = X0 + dX
    Y0 = 0.12
    Y1 = 0.98
    dY = Y1 - Y0
    fig = plt.figure(figsize=(6.9, 3.1), dpi=150)
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X1, Y0, dX, dY])

    # plot
    eargs = dict(c=c1, zorder=9, fmt='.')
    sargs = dict(fc=c2, ec='k', zorder=10, s=8)
    ax0.errorbar(x0, y0, y0err, **eargs)
    ax0.scatter(x0, y0, **sargs)
    ax1.errorbar(x1, y1, y1err, **eargs)
    ax1.scatter(x1, y1, **sargs)

    # fiducial lines
    ax0lim = ax0.get_xlim()
    ax1lim = ax1.get_xlim()
    largs = {'c': 'k', 'ls': 'dotted'}
    for i in range(2):
        ax = [ax0, ax1][i]
        lim = [ax0lim, ax1lim][i]
        ax.plot(lim, [0, 0], **largs)
        ax.plot(lim, [1, 1], **largs)
        ax.plot(lim, [-1, -1], **largs)

    # axis labels
    ylab = (r"Residuals $\equiv \displaystyle\frac"
            r"{v_\mathrm{true} - \mu_\mathrm{pred.}}"
            r"{\sigma_\mathrm{pred.}}$")
    ax0.set_xlabel(r"$G$")
    ax1.set_xlabel(r"$T_\mathrm{eff}\ [\mathrm{K}]$")
    ax0.set_ylabel(ylab)

    # limits
    ax0.set_xlim(ax0lim)
    ax1.set_xlim(ax1lim)
    for ax in fig.axes:
        ax.set_ylim(-1.4, 1.4)

    # ticks
    ax1.tick_params(labelleft=False)
    for ax in fig.axes:
        ax.tick_params(direction='inout', right=True, left=True, top=True)

    # save figure
    plt.savefig("fig2_residual_trends.pdf")

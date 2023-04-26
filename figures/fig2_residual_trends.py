#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Plot prediction residuals against other stellar quantities.

Created: July 2022
Author: A. P. Naik
"""
import sys
import numpy as np
import pandas as pd
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
    df = pd.read_hdf(ddir + 'EDR3_predictions/EDR3_prediction_aux.hdf5')
    G = df['phot_g_mean_mag']
    T_eff = df['rv_template_teff']
    d = df['mean_dist']

    # percentiles to get
    q = (2.3, 15.9, 50, 84.1, 97.7)

    # PANEL 1: residuals by G mag
    N_bins = 30
    G_edges = pd.qcut(G, N_bins, retbins=True)[1]
    y0 = 0.5 * (G_edges[1:] + G_edges[:-1])
    inds = np.digitize(G, G_edges) - 1
    vlo, lo, x0, hi, vhi = np.array([np.percentile(res[inds == i], q) for i in range(N_bins)]).T
    x0err1 = np.stack((x0 - lo, hi - x0))
    x0err2 = np.stack((x0 - vlo, vhi - x0))

    # PANEL 2: residuals by T_eff
    y1 = np.unique(T_eff)
    vlo, lo, x1, hi, vhi = np.array([np.percentile(res[T_eff == T], q) for T in y1]).T
    x1err1 = np.stack((x1 - lo, hi - x1))
    x1err2 = np.stack((x1 - vlo, vhi - x1))

    # PANEL 3: residuals by d
    N_bins = 30
    d_edges = pd.qcut(d, N_bins, retbins=True)[1]
    y2 = 0.5 * (d_edges[1:] + d_edges[:-1])
    inds = np.digitize(d, d_edges) - 1
    vlo, lo, x2, hi, vhi = np.array([np.percentile(res[inds == i], q) for i in range(N_bins)]).T
    x2err1 = np.stack((x2 - lo, hi - x2))
    x2err2 = np.stack((x2 - vlo, vhi - x2))

    # save
    np.savez(dfile,
             x0=x0, y0=y0, x0err1=x0err1, x0err2=x0err2,
             x1=x1, y1=y1, x1err1=x1err1, x1err2=x1err2,
             x2=x2, y2=y2, x2err1=x2err1, x2err2=x2err2)

    return


if __name__ == "__main__":

    # load plot data (create if not present)
    ddir = get_datadir()
    dfile = ddir + "figures/fig2_residual_trends_data.npz"
    if not exists(dfile):
        create_plot_data(dfile)
    data = np.load(dfile)
    x0 = data['x0']
    y0 = data['y0']
    x0err1 = data['x0err1']
    x0err2 = data['x0err2']
    x1 = data['x1']
    y1 = data['y1']
    x1err1 = data['x1err1']
    x1err2 = data['x1err2']
    x2 = data['x2']
    y2 = data['y2']
    x2err1 = data['x2err1']
    x2err2 = data['x2err2']

    # set up figure
    c1 = 'teal'
    c2 = 'goldenrod'
    X0 = 0.17
    X1 = 0.98
    dX = X1 - X0
    Y0 = 0.08
    Y3 = 0.99
    dY = (Y3 - Y0) / 3
    Y1 = Y0 + dY
    Y2 = Y1 + dY
    asp = 3.35 / 6.4
    fig = plt.figure(figsize=(3.35, 3.35 / asp), dpi=150)
    ax0 = fig.add_axes([X0, Y0, dX, dY])
    ax1 = fig.add_axes([X0, Y1, dX, dY])
    ax2 = fig.add_axes([X0, Y2, dX, dY])

    # plot
    eargs1 = dict(fc=c1, ec='k', zorder=2)
    eargs2 = dict(fc='lightgrey', ec='darkgrey', zorder=1)
    sargs = dict(fc=c2, ec='k', zorder=10, s=8)
    ax0.fill_betweenx(y0, x0 - x0err1[0], x0 + x0err1[1], **eargs1)
    ax0.fill_betweenx(y0, x0 - x0err2[0], x0 + x0err2[1], **eargs2)
    ax1.fill_betweenx(y1, x1 - x1err1[0], x1 + x1err1[1], **eargs1)
    ax1.fill_betweenx(y1, x1 - x1err2[0], x1 + x1err2[1], **eargs2)
    ax2.fill_betweenx(y2, x2 - x2err1[0], x2 + x2err1[1], **eargs1)
    ax2.fill_betweenx(y2, x2 - x2err2[0], x2 + x2err2[1], **eargs2)

    # limits
    ax0lim = [y0[0], y0[-1]]
    ax1lim = [y1[0], y1[-1]]
    ax2lim = [y2[0], y2[-1]]
    ax0.set_ylim(ax0lim)
    ax1.set_ylim(ax1lim)
    ax2.set_ylim(ax2lim)
    for ax in fig.axes:
        ax.set_xlim(-2.8, 2.8)

    # fiducial lines
    largs = {'c': 'k', 'ls': 'dotted'}
    for i in range(3):
        ax = [ax0, ax1, ax2][i]
        lim = [ax0lim, ax1lim, ax2lim][i]
        ax.plot([0, 0], lim, **largs)
        ax.plot([1, 1], lim, **largs)
        ax.plot([-1, -1], lim, **largs)
        ax.plot([2, 2], lim, **largs)
        ax.plot([-2, -2], lim, **largs)

    # axis labels
    xlab = (r"Residuals $\equiv \displaystyle\frac"
            r"{v_\mathrm{true} - \mu_\mathrm{pred.}}"
            r"{\sigma_\mathrm{pred.}}$")
    ax0.set_ylabel(r"$G$")
    ax1.set_ylabel(r"$T_\mathrm{eff}\ [\mathrm{K}]$")
    ax2.set_ylabel(r"$\mathrm{Distance}\ [\mathrm{kpc}]$")
    ax0.set_xlabel(xlab)

    # ticks
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    for ax in fig.axes:
        ax.tick_params(direction='inout', right=True, left=True, top=True)

    # save figure
    plt.savefig("fig2_residual_trends.pdf")

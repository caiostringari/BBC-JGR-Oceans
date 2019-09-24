# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_timestack.py
# pourpose : plot a timestack example
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
import sys
import warnings

# data I/O
import numpy as np
import pandas as pd
import xarray as xr
from simpledbf import Dbf5

from scipy import interpolate
from scipy.signal import find_peaks

from pywavelearn.image import construct_rgba_vector

from pywavelearn.utils import process_timestack, ellapsedseconds
from pywavelearn.tracking import optimal_wavepaths

# image tools
from skimage.transform import resize
from skimage.filters import gaussian

from scipy.signal import savgol_filter

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2

# quite skimage warningss
warnings.filterwarnings("ignore")


def surfzone_edge(t, x, T, X, sigma=1, kind="gaussian", smooth=0.1):
    """
    Find the surf zone edge based on wave path data.

    ----------
    Args:
        t (Mandatory [np.array]): target array of times (seconds).

        x (Mandatory [np.array]): target array of cross-shore locations (m).

        T (Mandatory [list of np.array]): arrays of time (seconds)
                                          of wave breaking location.

        X (Mandatory [list of np.array]): arrays of wave breaking locations.

        Use optimal_wavepaths() to get X and T.

    ----------
    Return:
        tsurf (Mandatory [np.array]): array of time events.

        xsurf (Mandatory [np.array]): array of cross-shore locations events
    """
    # process edge of the surf zone
    Tall = []
    Xall = []
    for t1, x1, in zip(T, X):
        for i, j in zip(t1, x1):
            Tall.append(i)
            Xall.append(j)
    Tall = np.array(Tall)
    Xall = np.array(Xall)
    sorts = np.argsort(Tall)
    Tall = Tall[sorts]
    Xall = Xall[sorts]

    # rasterize
    S, xedges, yedges = np.histogram2d(Tall, Xall,
                                       bins=(t, x),
                                       normed=True)
    # bring back to original dimension
    S = resize(S, (t.shape[0], x.shape[0]))
    S[S > 0] = 1

    # get the edge
    S = gaussian(S, sigma=1)
    S[S > 0] = 1

    # loop over time, get the edge(t, x) position
    tsurf = []
    xsurf = []
    for i, _t in enumerate(t):
        strip = S[i, :][::-1]
        idx = np.where(strip == 1)[0]
        if idx.size > 0:
            tsurf.append(_t)
            xsurf.append(x[::-1][idx[0]])
    tsurf = np.array(tsurf)
    xsurf = np.array(xsurf)

    # sort
    sorts = np.argsort(tsurf)
    tsurf[sorts], xsurf[sorts]

    # detect outer edges
    peaks, _ = find_peaks(xsurf-xsurf.mean())
    tpeaks = tsurf[peaks]
    xpeaks = xsurf[peaks]

    # interp
    f = interpolate.Rbf(tpeaks, xpeaks, kind=kind, smooth=smooth)
    xsurf = f(tsurf)

    return tsurf, xsurf


if __name__ == '__main__':
    # main()

    # files
    timestack = "Raw_Data/Timestacks/SevenMileBeach/20180614-001.nc"
    breaking = "Raw_Data/Breaking/SevenMileBeach/20180614-001.dbf"
    shoreline = "Raw_Data/Swash/SevenMileBeach/20180614-001.csv"
    merging = "Raw_Data/WaveOverruning/SevenMileBeach/20180614-001.csv"

    # process timestack
    t, sx, rgb = process_timestack(xr.open_dataset(timestack))
    sx -= sx.min()
    # sx = np.abs(sx)[::-1]
    t = ellapsedseconds(t)

    # process wavepaths
    wp = Dbf5(breaking, codec='utf-8').to_dataframe()
    if "newtime" in wp.columns.values:
        wp = wp[["newtime", "newspace", "wave"]]
        wp.columns = ["t", "x", "wave"]
    T, X, L = optimal_wavepaths(wp, order=2, min_wave_period=1, N=250,
                                project=False, t_weights=1)

    # process shoreline
    df = pd.read_csv(shoreline)
    tshore = df["time"].values
    xshore = df["shoreline"].values
    ushore = df["upper_ci"].values
    lshore = df["lower_ci"].values

    # process surf zone edge
    tsurf, xsurf = surfzone_edge(t, sx, T, X, smooth=0.05)

    # interp to the same domain as shoreline
    f = interpolate.interp1d(tsurf, xsurf, kind="linear")
    xsurfp = f(tshore)

    # process mergings
    df = pd.read_csv(merging)
    xmerge = df["intersection"].values
    tmerge = df["time"].values

    # open a new figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # plot stack
    im = ax.pcolormesh(t, sx, rgb.mean(axis=2).T)
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(construct_rgba_vector(np.rollaxis(rgb, 1, 0)))

    # plot wavepaths

    n = 8
    k = 0
    cmcolors = plt.cm.get_cmap("Dark2", n)
    colors = []
    for n in range(len(X)):
        if k > n:
            k = 0
        colors.append(cmcolors(k))
        k += 1

    k = 0
    for _t, _x in zip(T, X):
        if k == 0:
            ax.plot(_t, _x, lw=3, label="Wave paths", color=colors[k], ls="--")
        else:
            ax.plot(_t, _x, lw=3, ls="--")
        ax.plot(_t-(0.15*_t.std()), _x, lw=2, color="k", ls="--")
        ax.plot(_t+(0.15*_t.std()), _x, lw=2, color="k", ls="--")
        k += 1

    # plot mergings
    ax.scatter(tmerge, xmerge, edgecolor="lawngreen", s=150, marker="s",
               zorder=100, facecolor="none", linewidths=3,
               label="Capure")

    # plot shoreline
    ax.plot(tshore, xshore, color="orangered", lw=4, label="Shoreline")

    # plot surf zone boundary
    ax.plot(tsurf, xsurf, color="dodgerblue", lw=3, label="Surf zone")

    # plot X
    ax.fill_between(tshore, xshore, xsurfp, facecolor="none",
                    edgecolor="k", lw=2, hatch="//", alpha=0.5,
                    label=r"$\mathcal{X}$")

    # legend
    lg = ax.legend(loc=3, ncol=7, fontsize=14)
    lg.get_frame().set_color("w")

    ax.set_ylim(0, 190)
    ax.set_xlim(0, 300)

    ax.set_xlabel(r"Time $[s]$")
    ax.set_ylabel(r"Cross-shore Distance $[m]$")

    sns.despine(ax=ax)

    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : detect_bore_bore_capture.py
# POURPOSE : detect wave overruning (bore capture) given a wavepath dataset.
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     : 24/05/2018 [Caio Stringari] - Code base forked from previous
#                                          scripts.
#
# v1.1     : 26/10/2018 [Caio Stringari] - add shoreline.
#
# TODO:
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# System
import os
import sys
import json
import warnings

# Dates
import datetime
from matplotlib.dates import date2num, num2date

# Data
import xarray as xr
import pandas as pd
from simpledbf import Dbf5

# Arguments
import argparse

# Numpy
import numpy as np

# image tools
from skimage.transform import resize
from skimage.filters import gaussian

# Utils
from scipy import interpolate
from scipy.signal import find_peaks

# Least-square fits
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (RANSACRegressor,
                                  LinearRegression)

from pywavelearn.tracking import optimal_wavepaths
from pywavelearn.utils import (intersection, process_timestack,
                               ellapsedseconds, monotonic)

# Plotting
import seaborn as sns
from matplotlib import path
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=2)
sns.set_style("ticks", {'axes.linewidth': 2.0,
                        'legend.frameon': True,
                        'axes.facecolor': 'w',
                        'grid.color': '0'})

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


def projet_fowards(t, x, order=2, time=1, N=10, t_weights=1):
    """
    Project optimal wavepaths forward in time.

    ----------
    Args:
        t, x (Mandatory [1D np.ndarray]): time-space coordinates.

        min_wave_period (Mandatory [pandas.DataFrame]): input dataframe.

        order (Optional [int]): OLS polynomial order.

        time (Optional [float]): time project in seconds.

        t_weights (Optional [float]): weights for the projection.

    ----------
    Return:
        tpred, xpred (Mandatory [1D of np.ndarray]): projected time and
                                                     space coordinates.
    """

    # build the higher-order model
    model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                      ('ols', LinearRegression(fit_intercept=False))])

    # give the shore-most points of the wavepath more weight
    weights = np.ones(len(t))
    weights[0] = 100  # fisrt seaward-point
    tsrc = t
    tidx = t.max()-t_weights
    idx = np.argmin(np.abs(tsrc-tidx))
    weights[idx:] = 100

    # fit
    model.fit(t.reshape(-1, 1), x,
              **{'ols__sample_weight': weights})

    # predict
    tpred = np.linspace(t.max(), t.max()+time, N)
    xpred = model.predict(tpred.reshape(-1, 1))

    return tpred, xpred


def plot(stk_secs, stk_dist, rgb, T, X, Tproj, Xproj):
    """Plot the results."""
    # open a new figure
    fig, ats = plt.subplots(figsize=(12, 6))

    colors = plt.cm.get_cmap("tab20", len(X))

    # plot wavepaths - insiders
    kk = 0
    for t, x in zip(T, X):
        ats.plot(t, x, color="k", zorder=20, lw=2, ls="-")
        ats.plot(t+(t.std()*0.25), x, color=colors(kk), ls="--", zorder=20)
        ats.plot(t-(t.std()*0.25), x, color=colors(kk), ls="--", zorder=20)
        kk += 1

    # plot wavepaths - projections
    kk = 0
    for t, x in zip(Tproj, Xproj):
        ats.plot(t, x, color="orangered", lw=2, ls="--", zorder=25)
        kk += 1

    # plot timestack
    ats.pcolormesh(stk_secs, stk_dist, rgb.mean(axis=2).T, cmap="Greys_r",
                   zorder=10)

    # set axes
    ats.set_xlabel("Time [s]")
    ats.set_ylabel("Distance [m]")
    ats.set_xlim(stk_secs.min(), stk_secs.max())
    ats.set_ylim(stk_dist.min(), np.max(X))

    # set grids
    ats.set_axisbelow(False)
    # ats.grid(color="cyan", ls="--")

    # finalize
    sns.despine(ax=ats)

    return fig, ats


def main():
    """Call the main program."""
    # read parameters from JSON
    with open(args.input[0], 'r') as f:
        H = json.load(f)

    # input file names
    wavepaths = H["data"]["breaking"]
    timestack = H["data"]["timestack"]

    # shoreline
    has_shoreline = False
    try:
        shoreline = H["data"]["shoreline"]
        has_shoreline = True
    except Exception:
        has_shoreline = False

    # load shoreline
    if has_shoreline:

        # read variables
        shore = pd.read_csv(shoreline)
        tshore = shore["time"].values
        xshore = shore["shoreline"].values
    else:
        raise IOError("Sorry, you must provide a shoreline.")

    # regression parameters
    order = H["parameters"]["OLS_order"]
    weights_time_threshold = H["parameters"]["max_time_threshold"]

    # read timestack
    ds = xr.open_dataset(timestack)
    stk_time, stk_dist, rgb = process_timestack(ds)

    # time in seconds
    stk_secs = ellapsedseconds(stk_time)
    if not monotonic(stk_secs.tolist()):
        stk_secs = np.linspace(0, stk_secs.max(), rgb.shape[0])

    # fix distance offset
    stk_dist -= stk_dist.min()

    # projection time
    proj = H["parameters"]["projection"]

    # read waverays
    df = Dbf5(wavepaths, codec='utf-8').to_dataframe()
    if "newtime" in df.columns.values:
        df = df[["newtime", "newspace", "wave"]]
        df.columns = ["t", "x", "wave"]
    dfpts = df
    # else:

    # calculate optimal wave paths
    N = H["parameters"]["wavepath_npoints"]
    T, X, L = optimal_wavepaths(df, order=order, min_wave_period=1, N=N,
                                project=False,
                                t_weights=weights_time_threshold)

    # loop over each wave path and get possible intersections
    TF = []  # for plotting
    XF = []  # for plotting
    TP = []  # projections
    XP = []  # projections
    TI = []  # intersections
    XI = []  # intersections
    WID = []
    k = 0
    for t1, x1, in zip(T, X):

        # extend the first wave path "t" seconds,
        tf, xf = projet_fowards(t1, x1, order=order, time=proj, N=N)

        # add projection to the current wavepath
        ta = np.hstack([t1, tf])
        xa = np.hstack([x1, xf])

        # append to output
        TF.append(t1)
        XF.append(x1)
        TP.append(tf)
        XP.append(xf)

        # look for intersections
        for t2, x2 in zip(T, X):
            # if they are the same, do nothing
            if np.array_equal(x1, x2):
                pass
            else:
                ti, xi = intersection(ta, xa, t2, x2)
                if ti.any():
                    TI.append(ti[0])
                    XI.append(xi[0])
                    WID.append(k)
        k += 1

    # process surf zone edge
    tsurf, xsurf = surfzone_edge(stk_secs, stk_dist, T, X)

    # interp shoreline and swash to the same time interval
    tmax = min(tsurf.max(), tshore.max())
    tmin = max(tsurf.min(), tshore.min())
    tfinal = np.arange(tmin, tmax, 0.1)
    f1 = interpolate.interp1d(tsurf, xsurf, kind="linear")
    f2 = interpolate.interp1d(tshore, xshore, kind="linear")
    xsurf = f1(tfinal)
    xshore = f2(tfinal)

    # intersect shoreline and surf zone edge
    #  with each overruning event⁠
    TShI = []
    XShI = []
    TSfI = []
    XSfI = []
    for ti, xi in zip(TI, XI):

        # find nearest shoreline and surf zone position in time
        idx1 = np.argmin(np.abs(tfinal-ti))

        # append
        TShI.append(tfinal[idx1])
        XShI.append(xshore[idx1])
        TSfI.append(tfinal[idx1])
        XSfI.append(xsurf[idx1])

    # to dataframe
    df = pd.DataFrame(np.vstack([TI, XI,
                                 XSfI, XShI,
                                 [len(X)]*len(XI),
                                 WID]).T,
                      columns=["time", "intersection",
                               "surfzone_position",
                               "shoreline_position",
                               "n_waves", "wave_id"])
    df = df.drop_duplicates()

    # dump to csv
    print(" --- Detected {} overruning events".format(len(df)))
    print(df)
    df.to_csv(H["data"]["output"])

    # plot projections results
    fig, ax = plot(stk_secs, stk_dist, rgb, TF, XF, TP, XP)

    # scatter mergings
    ax.scatter(df["time"], df["intersection"], 120, marker="s",
               edgecolor="lawngreen", facecolor="none", lw=3, zorder=50,
               label="Detected bore merging")

    for t, xsf, xsh in zip(TI, XSfI, XShI):
        ax.axvline(t, color="r", lw=2, ls="--", zorder=20)
        ax.scatter(t, xsf, s=100, marker="o", zorder=50, edgecolor="r",
                   facecolor="none", lw=3)
        ax.scatter(t, xsh, s=100, marker="o", zorder=50, edgecolor="r",
                   facecolor="none", lw=3)

    # ax.scatter(tpeaks, xpeaks, 120, marker="o", zorder=100, color="r")

    # plot shoreline and surf zone
    ax.plot(tfinal, xshore, lw=3, ls="-",
            color="dodgerblue", zorder=25)

    ax.plot(tfinal, xsurf, lw=3, ls="-",
            color="dodgerblue", zorder=25)

    ax.fill_between(tfinal, xshore, xsurf, facecolor="none",
                    edgecolor="w", lw=2, hatch="//", alpha=0.5,
                    label=r"$\mathcal{X}$", zorder=20)

    # finalize
    if H["parameters"]["savefig"]:
        plt.savefig(H["parameters"]["savefig"], dpi=120)
    if H["parameters"]["show_plot"]:
        plt.show()
    if args.force_show:
        plt.show()

    plt.close("all")


if __name__ == '__main__':
    print("\nDetecting the wave overruning events, please wait...\n")
    try:
        inp = sys.argv[1]
    except Exception:
        raise IOError("Usage: learn_shoreline path/to/hyper.json")
    if inp in ["-i", "--input"]:
        # argument parser
        parser = argparse.ArgumentParser()
        # input data
        parser.add_argument('--input', '-i',
                            nargs=1,
                            action='store',
                            dest='input',
                            help="JSON hyper parameter input file.",
                            required=True)
        parser.add_argument('--force-show',
                            action='store_true',
                            dest='force_show',
                            help="Force plot show.",
                            required=False)
        args = parser.parse_args()
    else:
        # argument parser
        parser = argparse.ArgumentParser()
        # input data
        parser.add_argument('--input', '-i',
                            nargs=1,
                            action='store',
                            dest='input',
                            help="JSON hyper parameter input file.",)
        parser.add_argument('--force-show',
                            action='store_true',
                            dest='force_show',
                            help="Force plot show.",
                            required=False)
        # parser
        args = parser.parse_args(["-i", sys.argv[1]])
    main()
    print("\nMy work is done!\n")

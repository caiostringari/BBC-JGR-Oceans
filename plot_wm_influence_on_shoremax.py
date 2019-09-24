# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_wm_influence_on_shoremax.py
# pourpose : plot wave merging influence on extreme shoreline excursions.
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import sys

import datetime

# data I/O
import numpy as np
import pandas as pd
import xarray as xr
from simpledbf import Dbf5

from scipy.spatial import KDTree

import textwrap

from scipy.signal import find_peaks

from pywavelearn.image import construct_rgba_vector
from pywavelearn.tracking import optimal_wavepaths
from pywavelearn.utils import (process_timestack, ellapsedseconds)

from random import choice

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2


def split_events(df1, df2, reversed_shoreline=True):
    """
    Split runup events into wave-merging or non-wave merging generated.

    ----------
    Args:
        df1 (Mandatory [pd.DataFarme]): shoreline data. One column needs
                                        to be label "shoreline" and
                                        another "time", at the very
                                        least.
        df1 (Mandatory [pd.DataFarme]): wave merging data. One column needs
                                        to be label "time", at the very
                                        least.

        reversed_shoreline (bool [str]): Flag to indicate that runup
                                         is NEGATIVE and rundown is POSITIVE.

    ----------
    Return:
        T__, X__, (Mandatory [lists]): time (T__) and position
                                       (X__)of runup events.
    """

    shoreline = df1["shoreline"].values
    shoretime = df1["time"].values
    mergetime = df2["time"].values

    # remove the mean
    # shoreline -= shoreline.mean()

    # find peaks
    peaks, _ = find_peaks(-shoreline)
    tpeaks = np.array(shoretime[peaks])
    xpeaks = np.array(shoreline[peaks])

    # cut shoreline
    shoreline = shoreline[peaks[0]: peaks[-1]]
    shoretime = shoretime[peaks[0]: peaks[-1]]

    # build the searching tree
    Tree = KDTree(np.vstack([tpeaks, np.ones(len(tpeaks))]).T)

    # loop over merging events
    Twm = []
    Xwm = []
    Iwm = []
    for tm in mergetime:
        d, idx = Tree.query([tm, 1], 2)
        Twm.append(tpeaks[max(idx)])
        Xwm.append(xpeaks[max(idx)])
        Iwm.append(max(idx))
    _, iudx = np.unique(Twm, return_index=True)
    Twm = np.array(Twm)[iudx]
    Xwm = np.array(Xwm)[iudx]
    Iwm = np.array(Iwm)[iudx]

    # Non-mergings
    Inm = []
    for i in np.arange(0, len(tpeaks), 1):
        if i not in Iwm:
            Inm.append(i)
    Inm = np.array(Inm)
    Tnm = tpeaks[Inm]
    Xnm = xpeaks[Inm]

    # extreme events
    trx = shoreline.mean() - (2 * shoreline.std())
    Txx = []
    Xxx = []
    for t, x in zip(tpeaks, xpeaks):
        if x < trx:
            Txx.append(t)
            Xxx.append(x)

    return Twm, Xwm, Tnm, Xnm, Txx, Xxx


def plot_timestack():
    """Plot timestack. All variables are global."""

    # plot timestack
    im = ax.pcolormesh(t, x, rgb.mean(axis=2).T, cmap="Greys_r")
    im.set_array(None)
    im.set_edgecolor('none')
    im.set_facecolor(construct_rgba_vector(np.rollaxis(rgb, 1, 0)))

    # plot wave paths
    colors = plt.cm.get_cmap("tab20", len(X))
    k = 0
    for _t, _x in zip(T, X):
        ax.plot(_t, _x, lw=3, color=colors(k), ls="--")
        ax.plot(_t - (0.3 * _t.std()), _x, lw=1, color="k", ls="--")
        ax.plot(_t + (0.3 * _t.std()), _x, lw=1, color="k", ls="--")
        k += 1

    # plot shoreline
    ax.plot(ds["time"], ds["shoreline"], lw=3, color="k", label="Shoreline")

    # plot stats
    ax.axhline(xshore.mean(), lw=3, color="darkgreen", ls="-",
               label=r"Shoreline $\mu$")
    ax.axhline(xshore.mean() - (1 * xshore.std()), lw=3, color="navy", ls="-",
               label=r"$\mu$+$\sigma$")
    ax.axhline(xshore.mean() - (2 * xshore.std()), lw=3, color="r", ls="-",
               label=r"$\mu$+$2\sigma$")

    ax.axhline(Xwm.mean(), lw=3, color="gold", ls="--",
               label=r"Capture $\mu$")
    ax.axhline(Xnm.mean(), lw=3, color="dodgerblue", ls="--",
               label=r"Non-capture $\mu$")

    # plot mergers
    ax.scatter(tm, xm, 120, marker="s", edgecolor="r",
               facecolor="none", linewidths=3, zorder=50,
               label="Captured wave")

    # plot extreme events
    ax.scatter(Twm, Xwm, 120, marker="v", edgecolor="gold",
               facecolor="none", linewidths=3, zorder=55,
               label="Capture maxima")

    ax.scatter(Tnm, Xnm, 120, marker="o", edgecolor="dodgerblue",
               facecolor="none", linewidths=3, zorder=50,
               label="Non-capure maxima")

    ax.scatter(Txx, Xxx, 280, marker="o", edgecolor="crimson",
               facecolor="none", linewidths=3, zorder=50,
               label="Extreme events")

    lg = ax.legend(loc=1, ncol=5, fontsize=13)
    lg.get_frame().set_color("0.75")
    lg.get_frame().set_edgecolor('k')
    for handle in lg.legendHandles:
        try:
            handle.set_sizes([120.0])
        except Exception:
            pass

    ax.set_xlim(0, t.max())
    ax.set_ylim(0, 75)
    sns.despine(ax=ax)

    ax.set_xlabel(r"Time $[s]$")
    ax.set_ylabel(r"Distance $[m]$")

    # add letter
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    ax.text(0.010, 0.965, "a)", transform=ax.transAxes, ha="left",
            va="top", bbox=bbox, zorder=100)


def plot_avg_increase():

    # compute stats
    Location = []
    WM_Average = []
    NM_Average = []
    WM_Variance = []
    NM_Variance = []
    colors = []

    for loc, gdf in df.groupby("order"):

        # figure out color
        if gdf["location"].values[0] == "Werri Beach":
            colors.append('#8c564b')
        elif gdf["location"].values[0] == "One Mile Beach":
            colors.append('#d62728')
        elif gdf["location"].values[0] == "Frazer Beach":
            colors.append('#1f77b4')
        elif gdf["location"].values[0] == "Moreton Island":
            colors.append('#ff7f0e')
        elif gdf["location"].values[0] == "Seven Mile Beach":
            colors.append('#9467bd')
        elif gdf["location"].values[0] == "Nobbys Beach":
            colors.append('#2ca02c')
        else:
            print("Huston, we have a problem.")
            print(gdf["location"].values[0], "is the problem!")

        WM_Average.append(gdf["mean wave merging maxima"].mean())
        NM_Average.append(gdf["mean non wave merging maxima"].mean())
        WM_Variance.append(gdf["mean wave merging maxima"].std())
        NM_Variance.append(gdf["mean non wave merging maxima"].std())
        Location.append(gdf["location"].values[0])

    Location.append("Average")
    WM_Average.append(np.mean(WM_Average))
    WM_Variance.append(np.mean(WM_Variance))
    NM_Average.append(np.mean(NM_Average))
    NM_Variance.append(np.mean(NM_Variance))
    # Average = np.array(Average)
    # Variance = np.array(Variance)
    # colors.append("0.5")

    dx = 0.4
    x1 = np.arange(0, len(Location), 1)
    x2 = x1 + dx
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    bx1.bar(x1, WM_Average, width=dx, color="gold",
            edgecolor="k", alpha=0.75, label="Capture")
    bx1.bar(x2, NM_Average, width=dx, color="dodgerblue",
            edgecolor="k", alpha=0.75, label="Non-capture")
    bx1.errorbar(x1, WM_Average, WM_Variance, fmt="none",
                 color="k", capsize=5, lw=2)
    bx1.errorbar(x2, NM_Average, NM_Variance, fmt="none",
                 color="k", capsize=5, lw=2)

    lg = bx1.legend(ncol=2, fontsize=12)
    lg.get_frame().set_color("w")

    bx1.grid(color="w", lw=2, ls="-")

    bx1.set_xticks(x1)
    bx1.set_xticklabels(Location)
    bx1.set_ylabel(r"Swash zone extent $[-]$")

    # fix labels
    labels = []
    for label in Location:
        lbl = "\n".join(textwrap.wrap(label, 10))
        labels.append(lbl)
    labels.append("Average")
    bx1.set_xticklabels(labels, fontsize=14, rotation=90)

    bx1.set_ylim(0, 1)

    sns.despine(ax=bx1)

    # add letter
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    bx1.text(0.025, 0.965, "b)", transform=bx1.transAxes, ha="left",
             va="top", bbox=bbox, zorder=100)


def plot_probabilities():

    # compute stats
    colors = []
    Location = []
    Average = []
    Omega = []
    Irribaren = []

    # add Elizabeth data
    Location.append("Elizabeth Beach")
    Average.append(0.1489702368687787)
    Omega.append(1.7)
    Irribaren.append(4.71)
    colors.append('#e377c2')
    for loc, gdf in de.groupby("order"):

        # figure out color
        if gdf["location"].values[0] == "Werri Beach":
            colors.append('#8c564b')
        elif gdf["location"].values[0] == "One Mile Beach":
            colors.append('#d62728')
        elif gdf["location"].values[0] == "Frazer Beach":
            colors.append('#1f77b4')
        elif gdf["location"].values[0] == "Moreton Island":
            colors.append('#ff7f0e')
        elif gdf["location"].values[0] == "Seven Mile Beach":
            colors.append('#9467bd')
        elif gdf["location"].values[0] == "Nobbys Beach":
            colors.append('#2ca02c')
        else:
            print("Huston, we have a problem.")
            print(gdf["location"].values[0], "is the problem!")

        # compute stats
        Average.append(gdf["number of extreme events"].mean() /
                       gdf["number of wave mergings"].mean())
        # Variance.append(gdf["maximun increase ratio"].std())
        Location.append(gdf["location"].values[0])
        Omega.append(gdf["Omega"].values[0])
        Irribaren.append(gdf["Irribaren"].values[0])
    # print(np.mean(Average))

    # add average
    Location.append("Average")
    Average.append(np.mean(Average))
    colors.append('0.5')
    Average[0] = 0

    dx = 0.4
    x1 = np.arange(0, len(Location), 1)
    x2 = x1 + dx
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    bx2.bar(x1, Average, width=dx + dx, color=colors,
            edgecolor="k", alpha=0.75)
    for k, omega in enumerate(Omega):
        bx2.text(x1[k] - (dx * 0.9), Average[k] + 0.01,
                 r"$\Omega$=${0:.2f}$".format(omega),
                 rotation=90, fontsize=14, va="bottom", ha="left")
        bx2.text(x1[k], Average[k] + 0.01,
                 r"$\xi$=${0:.2f}$".format(Irribaren[k]),
                 rotation=90, fontsize=14, va="bottom", ha="left")

    bx2.grid(color="w", lw=2, ls="-")

    bx2.set_xticks(x1)
    bx2.set_xticklabels(Location)

    # fix labels
    labels = []
    for label in Location:
        lbl = "\n".join(textwrap.wrap(label, 10))
        labels.append(lbl)
    labels.append("Average")
    bx2.set_xticklabels(labels, fontsize=14, rotation=90)

    bx2.set_ylim(0, 0.55)

    bx2.set_ylabel(r"Probability $[-]$")

    sns.despine(ax=bx2)

    # add letter
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    bx2.text(0.025, 0.965, "c)", transform=bx2.transAxes, ha="left",
             va="top", bbox=bbox, zorder=100)


if __name__ == '__main__':

    # Colors
    # ('Frazer Beach', '#1f77b4')
    # ('Moreton Island', '#ff7f0e')
    # ('Nobbys Beach', '#2ca02c')
    # ('One Mile Beach', '#d62728')
    # ('Seven Mile Beach', '#9467bd')
    # ('Werri Beach', '#8c564b')
    # ('Elizabeth Beach', '#e377c2')

    # load a timestack
    f = "Raw_Data/Timestacks/WerriBeach/20140816-003.nc"
    t, x, rgb = process_timestack(xr.open_dataset(f))
    offset = x.min()
    x -= offset
    t = ellapsedseconds(t)

    # load wave paths
    f = "Raw_Data/WaveBreaking/WerriBeach/20140816-003.dbf"
    wp = Dbf5(f, codec='utf-8').to_dataframe()
    if "newtime" in wp.columns.values:
        wp = wp[["newtime", "newspace", "wave"]]
        wp.columns = ["t", "x", "wave"]
    T, X, _ = optimal_wavepaths(wp, order=2, min_wave_period=1, N=250,
                                project=False, t_weights=1)

    # load coast line
    f1 = "Raw_Data/Shoreline/WerriBeach/20140816-003.csv"
    ds = pd.read_csv(f1)
    xshore = ds["shoreline"].values
    tshore = ds["time"].values

    # load merge data
    f2 = "Raw_Data/BoreBoreCapture/WerriBeach/20140816-003.csv"
    dm = pd.read_csv(f2)
    tm = dm["time"].values
    xm = dm["intersection"].values

    # split events into merging and non-merging
    Twm, Xwm, Tnm, Xnm, Txx, Xxx = split_events(pd.read_csv(f1), dm)

    # load full analysis
    df = pd.read_csv("Proc_Data/wm_influence_on_shoremax.csv")
    de = pd.read_csv("Proc_Data/wm_influence_on_extreme_shoremax.csv")

    # --- plot everything ---

    gs = gridspec.GridSpec(2, 2)
    ax = plt.subplot(gs[0, :])
    bx1 = plt.subplot(gs[1, 0])
    bx2 = plt.subplot(gs[1, 1])
    fig = plt.gcf()
    fig.set_size_inches(12, 10)

    plot_timestack()
    plot_avg_increase()
    plot_probabilities()

    # finish up
    fig.tight_layout()
    plt.show()

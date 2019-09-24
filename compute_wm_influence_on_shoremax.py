# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : compute_wm_influence_on_shoremax.py
# pourpose : analyse wave merging influence on extreme shoreline excursions.
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import os
import sys

import datetime

# data I/O
import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale

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
        T__, X__ (Mandatory [lists]): time (T__) and position
                                      (X__)of runup events.
    """

    shoreline = df1["shoreline"].values
    shoretime = df1["time"].values
    mergetime = df2["time"].values

    # remove the mean
    shoreline -= shoreline.mean()

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

    return Twm, Xwm, Tnm, Xnm, np.array(Txx), np.array(Xxx)


if __name__ == '__main__':

    # data
    main_data = "Raw_Data/"
    swash = "Shoreline"
    merging = "BoreBoreCapture"

    # dates
    Dates = ["20140816",
             "20140807",
             "20161220",
             "20180614",
             "20171102",
             "20180424"]

    # folder location names
    Locations = ["WerriBeach",
                 "OneMileBeach",
                 "MoretonIsland",
                 "SevenMileBeach",
                 "NobbysBeach",
                 "FrazerBeach"]

    # location names
    Names = ["Werri Beach",
             "One Mile Beach",
             "Moreton Island",
             "Seven Mile Beach",
             "Nobbys Beach",
             "Frazer Beach"]

    Omega = [1.89, 3.39, 3.47, 3.77, 6.15, 6.35]
    Irribaren = [4.49, 1.32, 2.35, 1.21, 1.66, 0.90]

    Order = ["1", "2", "3", "4", "5", "6"]

    # Output variables
    WM_Mean = []
    NM_Mean = []
    WM_Max = []
    NM_Max = []
    XM = []
    Nxx = []
    Nwm = []
    Locs = []
    Ord = []
    Omg = []
    Irb = []

    NRUNS = 12
    N = len(Locations) * NRUNS

    # loop
    k = 0
    for loc, date, name, order, omg, irb in zip(Locations,
                                                Dates,
                                                Names,
                                                Order,
                                                Omega,
                                                Irribaren):

        # loop over runs
        for i in range(NRUNS):

            # shoreline data
            f = os.path.join(main_data, swash, loc,
                             date + "-" + str(i).zfill(3) + ".csv")
            ds = pd.read_csv(f)

            # wave merging data
            f = os.path.join(main_data, merging, loc,
                             date + "-" + str(i).zfill(3) + ".csv")
            dm = pd.read_csv(f)

            # fix issues with first runup
            ds = ds.loc[ds["time"] >= 10]

            # split events
            Twm, Xwm, Tnm, Xnm, Txx, Xxx = split_events(ds, dm)

            # normalized shoreline
            shore = ds["shoreline"].values
            shoreline = minmax_scale(shore)
            shoretime = ds["time"].values

            # find nearest mergings
            Xwm_n = []
            for t in Twm:
                idx = np.argmin(np.abs(t - shoretime))
                Xwm_n.append(shoreline[idx])

            # find neares non-mergings
            Xnm_n = []
            for t in Tnm:
                idx = np.argmin(np.abs(t - shoretime))
                Xnm_n.append(shoreline[idx])

            # fig, ax = plt.subplots()
            # ax.plot(shoretime, shoreline)
            # ax.scatter(Twm, Xwm_n, color="r")
            # ax.scatter(Tnm, Xnm_n, color="b")
            # plt.show()

            # find number of extremes
            trx = ds["shoreline"].mean() - (2 * ds["shoreline"].std())
            Txx = []
            for t, x in zip(np.hstack([Twm, Tnm]),
                            np.hstack([Xwm, Xnm])):
                if x < trx:
                    Txx.append(t)
            Nxx.append(len(Txx))

            # verify if an extreme event was generated from a merging
            flag = 0
            if len(Txx) > 0:
                for t in Txx:
                    # print(t)
                    if np.round(t, 2) in np.round(Twm, 2):
                        flag = 1
            XM.append(flag)

            # Calculate stats
            WM_Mean.append(np.mean(Xwm_n))
            WM_Max.append(-np.min(Xwm))
            NM_Mean.append(np.mean(Xnm_n))
            NM_Max.append(-np.min(Xnm))
            Nwm.append(len(Twm))
            Locs.append(name)
            Ord.append(order)
            Omg.append(omg)
            Irb.append(irb)

            # break
            k = + 1
        # break

    # build final dataframe
    df = pd.DataFrame()
    df["location"] = Locs
    df["order"] = Ord
    df["Omega"] = Omg
    df["Irribaren"] = Irb
    df["mean wave merging maxima"] = 1 - np.array(WM_Mean)
    df["mean non wave merging maxima"] = 1 - np.array(NM_Mean)

    WM_mean = df.groupby("location")["mean wave merging maxima"].mean()
    NM_mean = df.groupby("location")["mean non wave merging maxima"].mean()

    # build extreme events dataframe
    de = pd.DataFrame()
    de["location"] = Locs
    de["order"] = Ord
    de["Omega"] = Omg
    de["Irribaren"] = Irb
    de["number of extreme events"] = Nxx
    de["number of wave mergings"] = Nwm
    de["extreme event from wave merging"] = XM

    df.to_csv("Proc_Data/wm_influence_on_shoremax.csv")
    de.to_csv("Proc_Data/wm_influence_on_extreme_shoremax.csv")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_profiles.py
# pourpose : plot beach profiles
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# data I/O
import numpy as np
import pandas as pd
import xarray as xr

# linear regression
from scipy import interpolate
from scipy.stats import linregress
from scipy.stats import gaussian_kde

from pywavelearn.utils import ellapsedseconds, dffs
from matplotlib.dates import date2num, num2date

import warnings

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

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    # Colors
    # ('Frazer Beach', '#1f77b4')
    # ('Moreton Island', '#ff7f0e')
    # ('Nobbys Beach', '#2ca02c')
    # ('One Mile Beach', '#d62728')
    # ('Seven Mile Beach', '#9467bd')
    # ('Werri Beach', '#8c564b')
    # ('Elizabeth Beach', '#e377c2')

    beaches = ["Elizabeth Beach",
               "Werri Beach",
               "One Mile Beach",
               "Frazer Beach",
               "Moreton Island",
               "Seven Mile Beach"]

    profiles = ["Raw_Data/Profiles/13052019_Elizabeth_Beach.csv",
                "Raw_Data/Profiles/20140816_Werri_Beach.csv",
                "Raw_Data/Profiles/20140807_One_Mile_Beach.csv",
                "Raw_Data/Profiles/20180424_Frazer_Beach.csv",
                "Raw_Data/Profiles/20161220_Moreton_Island.csv",
                "Raw_Data/Profiles/20180616_Seven_Mile_Beach.csv"]

    WL = [1., 0.8, 1.0, 0.3, 1.6, 1.2]

    # colors
    colors = ["#e377c2", "#8c564b", "#d62728", "#1f77b4",
              "#ff7f0e", "#9467bd"]

    gs = gridspec.GridSpec(4, 2)

    ax1 = plt.subplot(gs[0, 0])  # FB
    ax2 = plt.subplot(gs[0, 1])  # OMB

    ax3 = plt.subplot(gs[1, 0])  # WB
    ax4 = plt.subplot(gs[1, 1])  # EB

    ax5 = plt.subplot(gs[2, :])  # MI
    ax6 = plt.subplot(gs[3, :])  # SMB

    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]

    # loop and plot
    k = 0
    for ax, prof, color, location, wl in zip(axs,
                                             profiles,
                                             colors,
                                             beaches,
                                             WL):

        # read profile
        df = pd.read_csv(prof)
        x = df["x"].values
        dx = df["x"].values.min()
        z = df["z"].values
        x -= dx

        # fix water level
        z -= -wl

        # plot the profile
        ax.plot(x, z, "-", lw=3,
                color=colors[k], zorder=12,)

        # fill sand
        ax.fill_between(x, -10, z, interpolate=True,
                        color='#ded6c4', zorder=10, )

        # fill water
        ax.fill_between(x, z, 0, interpolate=True,
                        color="#c9e7ff", zorder=5, alpha=0.5)

        ax.axhline(0, lw=3, color="navy", zorder=6)

        # grids
        sns.despine(ax=ax)
        ax.grid(color="w", ls="-", lw=2, zorder=10)
        for _, spine in ax.spines.items():
            spine.set_zorder(300)

        # axis limits
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(ymin=-4)
        ax.set_ylabel(r"Depth $[m]$")
        ax.set_xlabel(r"Cross-shore distance $[m]$")

        ax.set_title(location)

        # aspect ratio
        ax.set_aspect(4)

        k += 1

    # letters
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    axs[0].text(0.025, 0.065, "a)", transform=axs[0].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)
    axs[1].text(0.025, 0.1, "b)", transform=axs[1].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)
    axs[2].text(0.025, 0.065, "c)", transform=axs[2].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)
    axs[3].text(0.035, 0.05, "d)", transform=axs[3].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)
    axs[4].text(0.015, 0.05, "e)", transform=axs[4].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)
    axs[5].text(0.015, 0.05, "f)", transform=axs[5].transAxes, ha="left",
                va="bottom", bbox=bbox, zorder=100)

    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : compute_ig_influence_on_wm.py
# pourpose : compute bound IG influecence on wave merging
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import warnings

import numpy as np

import datetime

import pandas as pd
import xarray as xr
from simpledbf import Dbf5

import pycwt as wavelet
from pycwt.helpers import find

from skimage.color import rgb2gray

from sklearn.preprocessing import minmax_scale

from pywavelearn.tracking import optimal_wavepaths
from pywavelearn.utils import (process_timestack,
                               ellapsedseconds,
                               align_signals,
                               read_pressure_data)
from pywavelearn.stats import HM0, TM01

from matplotlib.dates import date2num

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 3.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "w",
                        'grid.color': "k"})
mpl.rcParams['axes.linewidth'] = 2
warnings.filterwarnings("ignore")


def wavelet_transform(dat, mother, s0, dj, J, dt, lims=[20, 120], t0=0):
    """
    Plot the continous wavelet transform for a given signal.

    Make sure to detrend and normalize the data before calling this funcion.

    This is a function wrapper around the pycwt simple_sample example with
    some modifications.

    ----------
    Args:
        dat (Mandatory [array like]): input signal data.

        mother (Mandatory [str]): the wavelet mother name.

        s0 (Mandatory [float]): starting scale.

        dj (Mandatory [float]): number of sub-octaves per octaves.

        j (Mandatory [float]):  powers of two with dj sub-octaves.

        dt (Mandatory [float]): same frequency in the same unit as the input.

        lims (Mandatory [list]): Period interval to integrate the local
                                 power spectrum.

        label (Mandatory [str]): the plot y-label.

        title (Mandatory [str]): the plot title.
    ----------
    Return:
        fig (plt.figure): the plot itself.
    """

    # also create a time array in years.
    N = dat.size
    t = np.arange(0, N) * dt + t0

    # write the following code to detrend and normalize the input data by its
    # standard deviation. Sometimes detrending is not necessary and simply
    # removing the mean value is good enough. However, if your dataset has a
    # well defined trend, such as the Mauna Loa CO\ :sub:`2` dataset available
    # in the above mentioned website, it is strongly advised to perform
    # detrending. Here, we fit a one-degree polynomial function and then
    # subtract it from the
    # original data.
    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    var = std ** 2  # Variance
    dat_norm = dat_notrend / std  # Normalized dataset
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    # the following routines perform the wavelet transform and inverse wavelet
    # transform using the parameters defined above. Since we have normalized
    # our input time-series, we multiply the inverse transform by the standard
    # deviation.
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(
        dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    # calculate the normalized wavelet and Fourier power spectra, as well as
    # the Fourier equivalent periods for each wavelet scale.
    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    # inverse transform but only considering lims
    idx1 = np.argmin(np.abs(period - LIMS[0]))
    idx2 = np.argmin(np.abs(period - LIMS[1]))
    _wave = wave.copy()
    _wave[0:idx1, :] = 0
    igwave = wavelet.icwt(_wave, scales, dt, dj, mother) * std

    # could stop at this point and plot our results. However we are also
    # interested in the power spectra significance test. The power is
    # significant where the ratio ``power / sig95 > 1``.
    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                             significance_level=0.95,
                                             wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    # calculate the global wavelet spectrum and determine its
    # significance level.
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    return t, dt, power, period, coi, sig95, iwave, igwave


def roundTime(dt=None, roundTo=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt is None:
        dt = datetime.datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


if __name__ == '__main__':

    print("\nAnalysing wave group influence on bore-merging, please wait...\n")

    # walet constants
    MOTHER = wavelet.MexicanHat()
    DT = 1  # 1 second
    S0 = 0.25 * DT  # starting scale, in this case 0.25*1 = 0.25 seconds
    DJ = 1 / 12  # twelve sub-octaves per octaves
    J = 8 / DJ  # eight powers of two with dj sub-octaves

    # infragravity wave period range
    Ta = 300
    NRUNS = 11
    LIMS = [25, 250]

    # data
    main_data = "Raw_Data/"

    breaking = "WaveBreaking/"
    timestacks = "Timestacks/"
    overrunning = "BoreBoreCapture/"
    pressure = "PressureTransducer/"

    # dates
    Dates = ["20180424",
             "20180614",
             "20140807",
             "20140816",
             "20161220"]

    # folder location names
    Locations = ["FrazerBeach/",
                 "SevenMileBeach/",
                 "OneMileBeach/",
                 "WerriBeach/",
                 "MoretonIsland/"]

    # PT data
    PTs = [["20170706.nc", "HP1"],
           ["20180614.nc", "HP2"],
           ["20140807.nc", "TB_19"],
           ["20140816.nc", "UQ1"],
           ["20161220.nc", "HP5"]]

    # names
    Names = ["Frazer Beach",
             "Seven Mile Beach",
             "One Mile Beach",
             "Werri Beach",
             "Moreton Island"]

    # read overrun final tabular data
    df = pd.read_csv("data/final_tabular_data.csv")

    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    # loop over locations
    phases = []
    dominances = []
    Locs = []
    Hs = []
    Tp = []
    R = []
    RDT = []
    print("Looping over locations, please wait...")
    for loc, locn, date, prs in zip(Locations, Names, Dates, PTs):

        print("\n -- Analysing {}".format(locn))

        # loop over timestacks
        for i in range(NRUNS - 1):
            print("   - run {} of {}".format(i + 1, NRUNS - 1), end="\r")
            i += 1  # skip the first run

            # open a figure
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1,
                                                          figsize=(9, 12),
                                                          sharex=True)

            # open and process timestack
            f = main_data + timestacks + loc + date + "-{}.nc".format(
                str(i).zfill(3))
            t, x, rgb = process_timestack(xr.open_dataset(f))
            gray = rgb2gray(rgb)
            dtime = t
            t = ellapsedseconds(t)
            x -= x.min()
            pxint = gray[:, int(len(x) / 2)]

            # plot timestack
            ax1.pcolormesh(t, x, gray.T, cmap="Greys_r")

            # open and process wavepaths
            f = main_data + breaking + loc + date + "-{}.dbf".format(
                str(i).zfill(3))
            wp = Dbf5(f, codec='utf-8').to_dataframe()
            if "newtime" in wp.columns.values:
                wp = wp[["newtime", "newspace", "wave"]]
                wp.columns = ["t", "x", "wave"]
            T, X, _ = optimal_wavepaths(wp, order=2, min_wave_period=1,
                                        N=50, project=False, t_weights=1)
            # plot wavepaths
            for t, x in zip(T, X):
                ax1.plot(t, x, lw=3, zorder=10)
                ci = 0.25 * t.std()
                ax1.plot(t + ci, x, lw=1, zorder=10, ls="--", color="k")
                ax1.plot(t - ci, x, lw=1, zorder=10, ls="--", color="k")

            # open and process merging event file
            f = main_data + overrunning + loc + date + "-{}.csv".format(
                str(i).zfill(3))
            dfm = pd.read_csv(f)
            tmerge = dfm["time"].values
            xmerge = dfm["intersection"].values

            # plot merging events
            ax1.scatter(tmerge, xmerge, marker="s", s=140,
                        edgecolor="lawngreen", facecolor="none",
                        lw=4, zorder=20)
            ax1.set_ylabel(r"Distance $[m]$")
            for _t in tmerge:
                ax1.axvline(_t, color="r", ls="-", lw=3)

            # read and process PTs
            f = main_data + pressure + prs[0]
            t, time, eta = read_pressure_data(f, prs[1])
            df_eta = pd.DataFrame(eta, index=time, columns=["eta"])
            # df_int = pd.DataFrame(pxint, index=dtime, columns=["int"])

            # resample to 1Hz
            df_eta = df_eta.resample("1S").bfill()
            # df_int = df_int.resample("1S").bfill()

            # select period
            tmin = dtime.min() - datetime.timedelta(seconds=LIMS[1])
            tmax = dtime.max() + datetime.timedelta(seconds=LIMS[1])
            df_eta = df_eta.between_time(tmin.time(),
                                         tmax.time(),
                                         include_start=True,
                                         include_end=True)
            eta = df_eta["eta"].values - df_eta["eta"].values.mean()

            # plot PT data
            lines = []
            labels = []
            s = ellapsedseconds(df_eta.index.to_pydatetime()) - LIMS[1]
            ll = ax2.plot(s, eta, color="dodgerblue")
            lines.append(ll[0])
            labels.append("Sea-swell")
            for _t in tmerge:
                ax2.axvline(_t, color="r", ls="-", lw=3)
            ax2.set_ylabel(r"$\eta - \overline{\eta}$ $[m]$")

            # compute the wavelet transform
            t, dt, power, period, coi, sig95, ieta, igeta = wavelet_transform(
                eta, MOTHER, S0, DJ, J, DT, lims=LIMS)
            t -= LIMS[1]

            # integrate the local spectrum
            idx = np.argmin(np.abs(period - LIMS[0]))
            sw = np.trapz(power[0:idx, :], dx=dt, axis=0) / power.max()
            ig = np.trapz(power[idx::, :], dx=dt, axis=0) / power.max()
            tt = np.trapz(power, dx=dt, axis=0) / power.max()

            # compute dominace and phase at the merging time
            for _t in tmerge:
                idx = np.argmin(np.abs(t - _t))
                # IG phase
                if igeta[idx] >= 0:
                    phases.append("Positive")
                else:
                    phases.append("Negative")
                # wave dominance
                if ig[idx] >= sw[idx]:
                    dominances.append("Infragravity")
                else:
                    dominances.append("Sea-Swell")

                # append values
                Hs.append(HM0(eta, 1))
                Tp.append(TM01(eta, 1))
                R.append(np.median(ig / sw))

                # append location
                Locs.append(locn)

                # append run datetime
                # fmt =
                RDT.append(roundTime(dtime[0], roundTo=60).strftime(
                    "%Y-%m-%d %H:%M:%S"))
                # print(time)

            # plot the normalized wavelet power spectrum and significance
            # level contour lines and cone of influece hatched area.

            levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
            m = ax3.contourf(t, np.log2(period), np.log2(power),
                             np.log2(levels),
                             extend='both', cmap="cividis")
            extent = [t.min(), t.max(), 0, max(period)]
            ax3.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                                     t[:1] - dt, t[:1] - dt]),
                     np.concatenate([np.log2(coi), [1e-9],
                                     np.log2(period[-1:]),
                                     np.log2(period[-1:]), [1e-9]]),
                     'k', alpha=0.3, hatch='x')
            ticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                   np.ceil(np.log2(period.max())))
            ax3.set_yticks(np.log2(ticks))
            ax3.set_yticklabels(ticks)
            ax3.invert_yaxis()
            ax3.set_ylim(np.log2(512), np.log2(2))
            ax3.set_ylabel(r'Period $[s]$')
            ax3.axhline(np.log2(LIMS[0]), color="cyan", lw=3, ls="-")
            ax3.axhline(np.log2(LIMS[1]), color="cyan", lw=3, ls="-")
            for _t in tmerge:
                ax3.axvline(_t, color="r", ls="-", lw=3)

            # draw a colorbar
            cmap = plt.cm.cividis
            cax = inset_axes(ax3, width="2%", height="90%", loc=4,
                             bbox_to_anchor=(0.05, 0.0, 1, 1),
                             bbox_transform=ax3.transAxes)
            bounds = minmax_scale(np.log2(levels))
            bounds *= np.max(levels)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                           norm=norm,
                                           # # to use 'extend', you must
                                           # # specify two extra boundaries:
                                           # boundaries=bounds,
                                           extend='max',
                                           # ticks=bounds,  # optional
                                           spacing='proportional',
                                           orientation='vertical')
            cb.set_label(r"$|W|^2$ $[m^{2}/s]$", fontsize=12)

            # plot inverse transforms
            ll = ax2.plot(t, ieta, lw=3, ls="--", color="0.35")
            lines.append(ll[0])
            labels.append("Inverse Transform")
            ax2a = ax2.twinx()
            ll = ax2a.plot(t, igeta, lw=3, ls="-", color="k")
            lines.append(ll[0])
            labels.append("Infragravity")

            lg = ax2a.legend(lines, labels, loc=3, fontsize=14, ncol=3)
            lg.get_frame().set_color("w")
            lg.get_frame().set_edgecolor("k")
            lg.set_zorder(300)

            label = r"$\frac{\eta_{ig}-\overline{\eta_{ig}}}{std(\eta)} $"
            ax2a.set_ylabel(label + r" $[-]$")
            yticks = [-0.025, 0, 0.025]
            ax2a.set_yticks(yticks)
            ax2a.set_yticklabels(yticks, rotation='vertical', va="center")

            # plot
            ax4.plot(t, sw, lw=3, color="dodgerblue", label="Sea-swell")
            ax4.plot(t, ig, lw=3, color="k", label="Infragravity")
            # ax4.fill_between(t, tt, 0, color="k", label="Total", alpha=0.5)
            for _t in tmerge:
                ax4.axvline(_t, color="r", ls="-", lw=3)
            lg = ax4.legend(loc=1, fontsize=14, ncol=3)
            lg.get_frame().set_color("w")
            lg.get_frame().set_edgecolor("k")
            lg.set_zorder(100)

            # ax4.set_xlabel(r"Time $[s]$")
            ax4.set_ylabel(r"Power $[m^{2}/s]$")

            # plot normalized ratio
            P0 = minmax_scale(tt)
            PSW = sw / tt.max()
            PIG = ig / tt.max()

            NPIG = PIG / (PSW + PIG)
            NPSW = PSW / (PSW + PIG)

            ax5.fill_between(t, NPIG / NPSW, 0, color="k", zorder=15)
            ax5.fill_between(t, -(NPSW / NPIG), 0, color="k", zorder=15)

            for _t in tmerge:
                ax5.axvline(_t, color="r", ls="-", lw=3, zorder=20)

            ax5.set_xlabel(r"Time $[s]$")
            ax5.set_ylabel(r"$E_{ig}>E_{sw}$ $[\%]$")

            ax5.set_ylim(-50, 50)
            sns.despine(ax=ax5)

            # hide not used intervals and fix grids
            for ax in [ax1, ax2, ax3, ax4, ax5]:
                ax.axvspan(-LIMS[1], 0, color="0.5", alpha=0.5, zorder=50,
                           hatch="x")
                ax.axvspan(Ta, Ta + LIMS[1], color="0.5", alpha=0.5, zorder=50,
                           hatch="x")

                ax.grid(color="k", ls="--", lw=1, zorder=10)
                ax.set_axisbelow(True)
                for _, spine in ax.spines.items():
                    spine.set_zorder(300)
                sns.despine(ax=ax)
            for _, spine in ax2a.spines.items():
                spine.set_zorder(300)
            sns.despine(ax=ax2a,
                        top=True, left=False, right=False, bottom=False)

            # set axis limits
            ax1.set_xlim(-LIMS[0], Ta + LIMS[0])
            ax2.set_ylim(-1.25, +1.25)
            # ax2.set_yticks([-0.5, 0, 0.5])
            ax4.set_ylim(0, np.max(tt) * 1.1)
            ax2a.set_ylim(-0.03, 0.03)
            ax1.set_ylim(np.min(X) - (np.min(X) * 0.2),
                         np.max(X) + (np.max(X) * 0.2))

            ax1.text(0.015, 0.975, "a)", transform=ax1.transAxes, ha="left",
                     va="top", bbox=bbox, zorder=100)
            ax2.text(0.015, 0.975, "b)", transform=ax2.transAxes, ha="left",
                     va="top", bbox=bbox, zorder=100)
            ax3.text(0.015, 0.975, "c)", transform=ax3.transAxes, ha="left",
                     va="top", bbox=bbox, zorder=100)
            ax4.text(0.015, 0.975, "d)", transform=ax4.transAxes, ha="left",
                     va="top", bbox=bbox, zorder=100)
            ax5.text(0.015, 0.975, "e)", transform=ax5.transAxes, ha="left",
                     va="top", bbox=bbox, zorder=100)

            # finalise
            fig.tight_layout()
            # plt.savefig("plot/" + loc + str(i).zfill(3) + ".png")
            # plt.show()
            plt.close("all")
            # break
        print(" ")
        # break

    # organize output
    df = pd.DataFrame()
    df["location_name"] = Locs
    df["run_datetime"] = Locs
    df["infragravity_phase"] = phases
    df["energy_dominance"] = dominances
    df["run_datetime"] = RDT
    df["energy_ratio"] = R
    df["Hs"] = Hs
    df["Tp"] = Tp
    df.to_csv("Proc_Data/wave_group_influence.csv")

    print("\nMy work is done!\n")

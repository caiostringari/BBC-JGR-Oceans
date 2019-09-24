# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_merging_predictor.py
# pourpose : plot overrun location
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

import warnings

# data I/O
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# from sklearn.metrics import r2_score
from scipy.stats import pearsonr

import xgboost

import statsmodels.api as sm

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    # load data
    df = pd.read_csv("Proc_Data/final_tabular_data.csv")
    df.drop_duplicates(inplace=True)

    ig = pd.read_csv("Proc_Data/wave_group_influence.csv")

    beaches = ["Frazer Beach",
               "Moreton Island",
               "Nobbys Beach",
               "One Mile Beach",
               "Seven Mile Beach",
               "Werri Beach",
               "Elizabeth Beach"]

    colors = sns.color_palette("tab10", len(beaches)).as_hex()
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)
    letters = ["a)", "b)", "c)", "d)", "e)", "f)"]

    fig, axs = plt.subplots(2, 4, figsize=(14, 7))
    axs = axs.flatten()
    axs = [axs[0], axs[1], axs[2],
           axs[4], axs[5], axs[6],
           axs[3], axs[7]]

    bins = np.arange(0., 1.1, 0.1)
    rug_kws = {"color": "k", "facecolor": "k"}
    hist_kws = hist_kws = {"edgecolor": "none", "alpha": 0.7}

    k = 0
    alpha = []
    Slopes = []
    alpha_mean = []
    for loc, dfg in df.groupby("location_name"):
        print(loc)

        # overrun locations
        X = dfg["norm_overrun_location"].values
        X[X <= 0] = 0
        X[X >= 1] = 1

        sns.distplot(X, color="0.5", norm_hist=True, bins=bins,
                     ax=axs[k], kde=False, rug=True,
                     rug_kws=rug_kws, hist_kws=hist_kws,
                     label=r"All data")
        j = 0
        _a = []
        yFit = []
        for r, rdf in dfg.groupby("run_datetime"):

            # overrun locations
            X = rdf["norm_overrun_location"].values
            X[X <= 0] = 0
            X[X >= 1] = 1

            # fit
            params = stats.expon.fit(X)
            xfit = np.linspace(0.05, 1.1, 100)
            yfit = stats.expon.pdf(xfit, *params)
            yFit.append(yfit)

            if j > 0 and j < 11:
                if loc != "Nobbys Beach":
                    alpha.append(params[1])
                # Slopes.append(slopes[k])

            _a.append(alpha)

            # plot
            if j == 0:

                # plot model
                axs[k].plot(xfit, yfit,  label=loc, alpha=0.5,
                            color=colors[k], lw=3, zorder=100)
            else:
                # plot model
                axs[k].plot(xfit, yfit, alpha=0.5,
                            color=colors[k], lw=3, zorder=100)

            j += 1

        # plot means
        axs[k].plot(xfit, np.array(yFit).mean(axis=0), color="k",
                    lw=3, zorder=200, label="Average")
        alpha_mean.append(np.mean(_a))

        # plot alpha values
        axs[k].text(0.95, 0.5,
                    r"$\lambda_{avg}$ = "+str(round(np.mean(_a), 2)),
                    transform=axs[k].transAxes,
                    ha="right", va="bottom", bbox=bbox, zorder=100,
                    fontsize=14)
        axs[k].text(0.95, 0.35,
                    r"$\lambda_{std}$ = "+str(round(np.std(_a), 2)),
                    transform=axs[k].transAxes,
                    ha="right", va="bottom", bbox=bbox, zorder=100,
                    fontsize=14)

        lg = axs[k].legend(fontsize=12)
        lg.get_frame().set_color("w")
        lg.set_zorder(100)

        axs[k].grid(color="w", lw=2, ls="-")

        axs[k].set_xlim(0, 0.9)
        axs[k].set_ylim(0, 10)

        axs[k].set_xticks([0, 0.25, 0.5, 0.75, 1])

        sns.despine(ax=axs[k])

        axs[k].text(0.95, 0.05, letters[k], transform=axs[k].transAxes,
                    ha="right", va="bottom", bbox=bbox, zorder=100)

        axs[k].set_ylabel(" ")
        axs[k].set_ylabel(" ")

        k += 1

    axs[0].set_ylabel(r"$p(c)$ $[-]$")
    axs[3].set_ylabel(r"$p(c)$ $[-]$")

    axs[3].set_xlabel(r"$\chi$ $[-]$")
    axs[4].set_xlabel(r"$\chi$ $[-]$")
    axs[5].set_xlabel(r"$\chi$ $[-]$")

    # PREDICTION #

    include_ratio = True
    nielsen_hanslow = False

    # build alpha predictor
    Hs = []
    Tp = []
    Rt = []
    for loc, dfg in ig.groupby("location_name"):
        for r, rdf in dfg.groupby("run_datetime"):
            Hs.append(rdf["Hs"].values.mean())
            Tp.append(rdf["Tp"].values.mean())
            Rt.append(rdf["energy_ratio"].values.mean())
    Hs = np.array(Hs)
    Tp = np.array(Tp)
    R = np.array(Rt)

    if include_ratio:
        df = pd.DataFrame(np.vstack([Tp, Hs, R]).T,
                          columns=["Tp", "Hs", R])
        y = np.array(alpha)
    else:
        df = pd.DataFrame(np.vstack([Tp, Hs]).T,
                          columns=["Tp", "Hs"])
        y = np.array(alpha)
    if nielsen_hanslow:
        L = (9.81/2*3.14)*(Tp**2)
        df = pd.DataFrame(np.vstack([np.sqrt(Hs*L)]).T,
                          columns=["Tp/Hs"])
        y = np.array(alpha)

    # predictor
    if nielsen_hanslow:
        order = 1
    else:
        order = 2
    X = df.values
    model = Pipeline(steps=[("poly",
                             PolynomialFeatures(degree=order,
                                                interaction_only=False,
                                                include_bias=True)),
                            ("reg",
                             LinearRegression(fit_intercept=True))])
    model.fit(X, y)

    # predict
    ytrue = np.array(alpha)
    ypred = model.predict(X)
    r, p = pearsonr(ypred, alpha)

    # plot
    xb = np.linspace(0, 1, 10)
    axs[-1].plot(xb, xb, color="k", ls="--", )
    axs[-1].scatter(ytrue, ypred, 50, c="k", zorder=20, alpha=0.5)
    sns.regplot(ytrue, ypred, ci=95, color="r", scatter_kws={"alpha": 0},
                robust=False)

    sns.despine(ax=axs[-1])
    axs[-1].grid(color="w", lw="2", ls="-")

    axs[-1].set_xlim(0, 0.4)
    axs[-1].set_ylim(0, 0.4)
    axs[-1].set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    axs[-1].set_yticks([0, 0.1, 0.2, 0.3, 0.4])

    axs[-1].set_ylabel(r"$\hat{\lambda}$ $[-]$")
    axs[-1].set_xlabel(r"$\lambda$ $[-]$")

    axs[-1].text(0.125, 0.875, "g)", transform=axs[-1].transAxes,
                 ha="right", va="bottom", bbox=bbox, zorder=100)

    t = r"$r_{xy}$" + "={0:.2f}, $p \ll 0.05$".format(r)
    axs[-1].text(0.975, 0.05, t,
                 transform=axs[-1].transAxes,
                 ha="right", va="bottom", bbox=bbox, zorder=100,
                 fontsize=14,)

    i = model.steps[1][1].intercept_
    c = model.steps[1][1].coef_
    # print the equation
    if nielsen_hanslow:
        print("\n Model:\n")
        print("yhat = {0:.2f} {1:.3f}x".format(i, c[1]))
        print("r = {}".format(round(r, 2)))
    else:
        if include_ratio:
            print("\n Model:\n")
            print("yhat={0:.2f}+{1:.2f}Tp+{2:.2f}Hs+{3:.2f}R+{4:.2f}Tp^2"
                  "+{5:.2f}TpHs+{6:.2f}TpR+{7:.2f}Hs^2+{8:.2f}HsR"
                  "+{9:.2f}R^2".format(
                       i, c[0], c[1], c[2], c[3], c[4],
                       c[5], c[6], c[7], c[8]))
            print("r = {}".format(round(r, 2)))
        else:
            print("\n Model:\n")
            print("yhat = {0:.2f} + {1:.2f}Tp + {2:.2f}Hs + {3:.2f}Tp^2"
                  " +{4:.2f}TpHs + {5:.2f}Hs^2".format(
                       i, c[0], c[1], c[2], c[3], c[4]))
            print("r = {}".format(round(r, 2)))

    fig.delaxes(axs[-2])
    l, b, w, h = axs[-1].get_position().bounds
    axs[-1].set_position([l+0.035, b+0.25, w, h], "both")

    plt.show()

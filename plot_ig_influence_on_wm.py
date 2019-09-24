# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : plot_ig_influence_on_merging.py
# pourpose : analyse wavegroup influence on bore-merging
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# data I/O
import numpy as np
import pandas as pd

import textwrap

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
sns.set_context("paper", font_scale=1.75, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})

mpl.rcParams['axes.linewidth'] = 2
pd.options.display.float_format = '{:,.2f}%'.format


def create_df():
    """Create a table."""
    df = pd.read_csv("Proc_Data/wave_group_influence.csv")
    # print(df.keys())

    Positives = []
    Negatives = []
    IG = []
    SW = []
    Location = []
    Ravg = []
    Rstd = []
    for g, gdf in df.groupby("location_name"):

        # phase
        positive = len(
            gdf.loc[gdf["infragravity_phase"] == "Positive"])/len(gdf)
        negative = len(
            gdf.loc[gdf["infragravity_phase"] == "Negative"])/len(gdf)

        # dominance
        ig = len(gdf.loc[gdf["energy_dominance"] == "Infragravity"])/len(gdf)
        sw = len(gdf.loc[gdf["energy_dominance"] == "Sea-Swell"])/len(gdf)

        # ratio

        # append
        Positives.append(np.round(positive*100, 2))
        Negatives.append(np.round(negative*100, 2))
        IG.append(np.round(ig*100, 2))
        SW.append(np.round(sw*100, 2))
        Ravg.append(gdf["energy_ratio"].mean())
        Rstd.append(gdf["energy_ratio"].std())
        Location.append(g)
    # Location.append("Average")
    # print(len(Ravg), len(IG))

    # build the final dataframe
    df = pd.DataFrame()
    df["Positive phase"] = Positives
    df["Negative phase"] = Negatives
    df["IG dominance"] = IG
    df["SW dominance"] = SW
    df["Ravg"] = Ravg
    df["Rstd"] = Rstd
    # df["Average"] =
    df = df.T
    df.columns = Location
    df["Average"] = [np.mean(Positives), np.mean(Negatives),
                     np.mean(IG), np.mean(SW), np.mean(Ravg), np.mean(Rstd)]
    df["Deviation"] = [np.std(Positives), np.std(Negatives),
                       np.std(IG), np.std(SW), np.std(Ravg), np.std(Rstd)]

    df.T.to_latex("data/wavegroup_analysis.tex")
    df.T.to_csv("data/wavegroup_analysis.csv")

    return df.T


def plot(df):
    """Plot the results."""

    df["order"] = [3, 2, 4, 1, 5, 6, 7]
    df = df.sort_values("order")
    df.drop("order", axis=1, inplace=True)
    deviation = df.iloc[-1].values
    df = df.iloc[:-1]

    print("\nResults:\n")
    print(df)
    print("")

    colors = np.array(["#9467bd", "#ff7f0e", "#1f77b4",
                       "#d62728", "#8c564b", "0.5"])

    # plot
    dx = 0.4
    x1 = np.arange(1, len(colors)+1, 1)
    x2 = x1 + dx
    bbox = dict(boxstyle="square", ec="none", fc="1", lw=1, alpha=0.7)

    fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6),
                                        sharex=True, sharey=False)

    # plot dominance
    ax1.bar(x1, df["IG dominance"].values, width=dx, label="Infragravity",
            facecolor="navy", edgecolor="k", alpha=0.75)
    ax1.bar(x2, df["SW dominance"].values, width=dx, label="Sea-swell",
            facecolor="darkgreen", edgecolor="k", alpha=0.75)

    # plot IG phase
    ax2.bar(x1, df["Positive phase"].values, width=dx, label="Positive phase",
            facecolor="orangered", edgecolor="k", alpha=0.75)
    ax2.bar(x2, df["Negative phase"].values, width=dx, label="Negative phase",
            facecolor="dodgerblue", edgecolor="k", alpha=0.75)

    # plot IG/SW ratio
    ax3.bar(x1, df["Ravg"].values, width=dx+dx, label="Positive phase",
            color=colors, edgecolor="k", alpha=0.75)
    ax3.errorbar(x1, df["Ravg"].values, df["Rstd"].values, fmt="none",
                 color="k", capsize=5, lw=2)

    # add numbers to first plot
    for xa, xb, ya, yb in zip(x1, x2,
                              df["IG dominance"].values,
                              df["SW dominance"].values):
        ax1.text(xa, ya*1.05, str(round(ya, 1))+"%", ha="center", va="bottom",
                 rotation=90, fontsize=12, zorder=100, bbox=bbox)
        ax1.text(xb, yb*1.05, str(round(yb, 1))+"%", ha="center", va="bottom",
                 rotation=90, fontsize=12, zorder=100, bbox=bbox)

    # add numbers to second plot
    for xa, xb, ya, yb in zip(x1, x2,
                              df["Positive phase"].values,
                              df["Negative phase"].values):
        ax2.text(xa, ya*1.05, str(round(ya, 1))+"%", ha="center", va="bottom",
                 rotation=90, fontsize=12, zorder=100, bbox=bbox)
        ax2.text(xb, yb*1.05, str(round(yb, 1))+"%", ha="center", va="bottom",
                 rotation=90, fontsize=12, zorder=100, bbox=bbox)

    # 50% line
    ax1.axhline(50, lw=3, ls="--", color="r")
    ax2.axhline(50, lw=3, ls="--", color="r")

    # legends
    lg = ax2.legend()
    lg.get_frame().set_color("w")
    lg = ax1.legend()
    lg.get_frame().set_color("w")

    # fix ticks
    ax1.set_xticks(x1+(dx/8))

    # set labels
    labels = []
    for label in df.index.values:
        lbl = "\n".join(textwrap.wrap(label, 10))
        labels.append(lbl)
    ax1.set_xticklabels(labels, fontsize=14, rotation=90)
    ax2.set_xticklabels(labels, fontsize=14, rotation=90)
    ax3.set_xticklabels(labels, fontsize=14, rotation=90)
    ax1.set_ylabel(r"Perc. of occurrence $[\%]$")
    ax2.set_ylabel(r"Perc. of occurrence $[\%]$")
    ax3.set_ylabel(r"$E_{ig}/E_{sw}$ $[-]$")

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    ax1.grid(color="w", lw=2, ls="-")
    ax2.grid(color="w", lw=2, ls="-")
    ax3.grid(color="w", lw=2, ls="-")

    sns.despine(ax=ax1)
    sns.despine(ax=ax2)
    sns.despine(ax=ax3)

    ax1.text(0.025, 0.965, "b)", transform=ax1.transAxes, ha="left",
             va="top", bbox=bbox, zorder=100)
    ax2.text(0.025, 0.965, "c)", transform=ax2.transAxes, ha="left",
             va="top", bbox=bbox, zorder=100)
    ax3.text(1-0.025, 0.965, "a)", transform=ax3.transAxes, ha="right",
             va="top", bbox=bbox, zorder=100)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Colors
    # ('Frazer Beach', '#1f77b4')
    # ('Moreton Island', '#ff7f0e')
    # ('Nobbys Beach', '#2ca02c')
    # ('One Mile Beach', '#d62728')
    # ('Seven Mile Beach', '#9467bd')
    # ('Werri Beach', '#8c564b')
    # ('Elizabeth Beach', '#e377c2')
    df = create_df()
    plot(df)

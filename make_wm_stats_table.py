# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# script   : make_overrun_stats_table.py
# pourpose : overrun frequency analysis
# author   : caio eadi stringari
# email    : caio.eadistringari@uon.edu.au
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# data I/O
import numpy as np
import pandas as pd

from pywavelearn.utils import ellapsedseconds

from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})
mpl.rcParams['axes.linewidth'] = 2

if __name__ == '__main__':

    df = pd.read_csv("Proc_Data/final_tabular_data.csv")
    df.drop_duplicates(inplace=True)

    # keys
    keys1 = ["number_of_waves",
             "number_of_events",
             "location", "run_id"]
    keys2 = ["avg_wave_period",
             "avg_overrun_period",
             "overrun_probability",
             "location"]

    beaches = np.sort(df["location_name"].unique())

    # count events
    n_waves = []
    n_merges = []
    for loc, grp in df[keys1].groupby("location"):
        dfr = grp.groupby("run_id").first().sum()
        n_waves.append(dfr["number_of_waves"])
        n_merges.append(dfr["number_of_events"])
    grp1 = pd.DataFrame(np.vstack([n_waves, n_merges]).T, columns=["Nw", "Nm"])

    # group by location and get means
    grp2 = df[keys2].groupby("location").mean()

    # final table
    totals = [grp1["Nw"].sum(),
              grp1["Nm"].sum(),
              np.round(grp2["avg_wave_period"].mean(), 2),
              np.round(grp2["avg_overrun_period"].mean(), 2),
              np.round(grp2["overrun_probability"].mean(), 2)]

    dff = pd.DataFrame()
    dff["Number of waves"] = grp1["Nw"].values.astype(int)
    dff["Number of merges"] = grp1["Nm"].values.astype(int)
    dff["Mean wave period"] = np.round(grp2["avg_wave_period"].values, 2)
    dff["Mean overrun period"] = np.round(grp2["avg_overrun_period"].values, 2)
    dff["Mean overrun probability"] = np.round(
        grp2["overrun_probability"].values, 2)
    dff = dff.T
    dff.columns = beaches
    dff["All beaches"] = totals

    print(dff)
    # dff.to_latex("Proc_Data/stats.tex")
    # dff.to_csv("Proc_Data/stats.csv")
    # dff.to_excel("Proc_Data/stats.xlsx")

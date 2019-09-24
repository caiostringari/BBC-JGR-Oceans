# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
#
# SCRIPT   : make_final_data_table.py
# POURPOSE : organize wave overruning data into tabular form
#
# AUTHOR   : Caio Eadi Stringari
# EMAIL    : Caio.EadiStringari@uon.edu.au
#
# v1.0     :
#
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# System
import os
import sys
import subprocess

from glob import glob

from natsort import natsorted

import datetime
from matplotlib.dates import date2num

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# plotting
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_context("paper", font_scale=2.0, rc={"lines.linewidth": 2.0})
sns.set_style("ticks", {'axes.linewidth': 2,
                        'legend.frameon': True,
                        'axes.facecolor': "#E9E9F1",
                        'grid.color': "w"})

if __name__ == '__main__':

    main_datapath = "Raw_Data/"

    # data
    overruning = "BoreBoreCapture/"
    breaking = "WaveBreaking/"

    # folder location names
    Locations = ["OneMileBeach",
                 "WerriBeach",
                 "MoretonIsland",
                 "FrazerBeach",
                 "SevenMileBeach",
                 "NobbysBeach"]

    # names
    Names = ["One Mile Beach",
             "Werri Beach",
             "Moreton Island",
             "Frazer Beach",
             "Seven Mile Beach",
             "Nobbys Beach"]

    # dates
    Dates = [datetime.datetime(2014, 8, 7, 9, 0),
             datetime.datetime(2014, 8, 16, 9, 0),
             datetime.datetime(2017, 12, 20, 10, 0),
             datetime.datetime(2018, 12, 24, 11, 0),
             datetime.datetime(2018, 6, 14, 11, 30),
             datetime.datetime(2017, 11, 2, 11, 30)]

    # constants
    T = 300

    # initialize variables
    times = []
    datetimes = []
    intersections_norm = []
    intersections_true = []
    run_ids = []
    run_times = []
    occurances = []
    freq_occur = []
    n_waves = []

    avg_overrun_period = []
    avg_wave_period = []

    locations = []
    locations_names = []
    cross_shore_locations = []
    event_ids = []

    # timedelta
    dt = 5

    # loop
    for loc, date, name in zip(Locations, Dates, Names):

        # loop over overruning files
        files = natsorted(
            glob(os.path.join(main_datapath, overruning, loc, "*.csv")))

        # k = 0
        for k, f in enumerate(files):
            df = pd.read_csv(f)
            now = date+datetime.timedelta(minutes=k*dt)

            # loop over overruning events
            for (i, row) in df.iterrows():
                event_ids.append(i+1)

                # temporal
                times.append(row["time"])
                datetimes.append(now+datetime.timedelta(
                    seconds=int(row["time"])))

                # frequency
                n_waves.append(row["n_waves"])
                freq_occur.append(len(df)/row["n_waves"])

                # averaged period
                avg_overrun_period.append(T/len(df))
                avg_wave_period.append(T/row["n_waves"])

                # HKR
                X = np.array([row["surfzone_position"],
                              row["shoreline_position"]]).reshape(-1, 1)
                scaler = MinMaxScaler().fit(X)
                xnorm = scaler.transform(row["intersection"].reshape(1, -1))
                intersections_norm.append(np.squeeze(xnorm))
                intersections_true.append(row["intersection"])

                # Metadata
                locations.append(loc)
                locations_names.append(name)
                occurances.append(len(df))
                run_times.append(now)
                run_ids.append(k)

        # break

    # final dataframe
    dff = pd.DataFrame()

    dff["event_id"] = event_ids

    dff["norm_overrun_location"] = intersections_norm
    dff["true_overrun_location"] = intersections_true

    dff["overrun_probability"] = freq_occur
    dff["number_of_events"] = occurances
    dff["number_of_waves"] = n_waves

    dff["avg_overrun_period"] = avg_overrun_period
    dff["avg_wave_period"] = avg_wave_period
    dff["norm_overrun_period"] = np.array(avg_overrun_period) / \
        np.array(avg_wave_period)

    dff["overrun_time"] = times
    dff["overrun_datetime"] = datetimes

    dff["run_id"] = run_ids
    dff["run_datetime"] = run_times
    dff["location"] = locations
    dff["location_name"] = locations_names

    # dump to file
    dff.to_csv("Proc_Data/final_tabular_data.csv")

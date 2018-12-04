import os
import re
import sys
import glob
import pickle

import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt

from constants import ROCK, PAPER, SCISSORS
from constants import OUTPUT_ROOT_DIR

# from constants import INTERACTION_OUTPUT_DIR, PLOTS_OUTPUT_DIR
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_plots")
INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_interactions")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_plots")

print("Globbing files from {:s}... ".format(INTERACTION_OUTPUT_DIR), end="")

interaction_files = glob.glob(INTERACTION_OUTPUT_DIR + "/rps_microbe_species*.pickle")
n_files = len(interaction_files)

print("{:d} files found.".format(n_files))

max_period = 0
max_hour = {}

for fpath in interaction_files:
    period = int(re.search("p\d\d\d\d", fpath).group(0)[1:])
    hour = int(re.search("h\d\d\d", fpath).group(0)[1:])

    max_period = max(max_period, period)

    if period in max_hour:
        max_hour[period] = max(max_hour[period], hour)
    else:
        max_hour[period] = hour

time = []
n_rocks = []
n_papers = []
n_scissors = []

for period in range(max_period+1):
    for hour in range(max_hour[period]+1):
        print("Period {:}, hour {:}...".format(str(period).zfill(4), str(hour).zfill(3)))
        pickle_fname = "rps_microbe_species_p" + str(period).zfill(4) + "_h" + str(hour).zfill(3) + ".pickle"
        pickle_fpath = os.path.join(INTERACTION_OUTPUT_DIR, pickle_fname)
        with open(pickle_fpath, "rb") as f:
            microbes = pickle.load(f)
            species = np.array(microbes[:, 2])

        n_rocks.append(np.sum(species == ROCK))
        n_papers.append(np.sum(species == PAPER))
        n_scissors.append(np.sum(species == SCISSORS))

print("Plotting species count...")

fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

ax.plot(range(n_files), n_rocks, color='red', label='Rocks')
ax.plot(range(n_files), n_papers, color='green', label='Papers')
ax.plot(range(n_files), n_scissors, color='blue', label='Scissors')

plt.xlim([0, n_files])
plt.xlabel("Time (hours)")
plt.ylabel("Microbe species count")
plt.title("Rock, paper, scissors species count")
ax.legend()

png_fpath = os.path.join(PLOTS_OUTPUT_DIR, "species_count.png")
print("Saving figure: {:s}".format(png_fpath))
plt.savefig(png_fpath, dpi=300, format='png', transparent=False)
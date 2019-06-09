import matplotlib
matplotlib.use("Agg")

import os

import xarray as xr
from numpy import sum

import matplotlib.pyplot as plt

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from interactions import ROCK, PAPER, SCISSORS, ROCK_COLOR, PAPER_COLOR, SCISSORS_COLOR


def species_count_figure(output_dir, start_time, end_time, dt, png_filename="species_count.png"):
    iters = (end_time - start_time) // dt
    times = [start_time + n * dt for n in range(iters)]

    microbe_data = xr.open_dataset(os.path.join(output_dir, "microbe_data.nc"))

    n_rocks = []
    n_papers = []
    n_scissors = []

    logger.info("Calculating species count time series...")
    for i, t in enumerate(times):
        logger.info("Calculating species count time series... {:}".format(t))
        n_rocks.append(sum(microbe_data["species"][:, i] == ROCK))
        n_papers.append(sum(microbe_data["species"][:, i] == PAPER))
        n_scissors.append(sum(microbe_data["species"][:, i] == SCISSORS))

    logger.info("Plotting species count...")

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    ax.plot(times, n_rocks, color='red', label='Rocks')
    ax.plot(times, n_papers, color='green', label='Papers')
    ax.plot(times, n_scissors, color='blue', label='Scissors')

    plt.xlim([start_time, end_time])
    plt.xlabel("Time")
    plt.ylabel("Microbe species count")
    plt.title("Rock, paper, scissors species count")
    ax.legend()

    png_filepath = os.path.join(output_dir, png_filename)
    logger.info("Saving species count time series figure: {:s}".format(png_filepath))
    plt.savefig(png_filepath, dpi=300, format='png', transparent=False)

import matplotlib
matplotlib.use("Agg")

import os
import re
import datetime

import numpy as np
from numpy import datetime64, abs, argmin
import xarray as xr
import joblib

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import cartopy
import cartopy.util
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from velocity_fields import oscar_dataset
from interactions import ROCK, PAPER, SCISSORS, ROCK_COLOR, PAPER_COLOR, SCISSORS_COLOR


class MicrobePlotter:
    def __init__(
            self,
            N_procs=1,
            dark_theme=False,
            microbe_marker_size=10,
            input_dir=".",
            output_dir=".",
    ):
        # velocity_dataset = oscar_dataset(2018)
        #
        # # Choose subset of velocity field we want to use
        # nominal_depth = velocity_dataset["depth"].values[0]
        # self.velocity_subdataset = velocity_dataset.sel(depth=nominal_depth,
        #                                                 latitude=slice(60, 0), longitude=slice(180, 240))
        #
        # self.grid_times = self.velocity_subdataset["time"].values
        # self.grid_lats = self.velocity_subdataset["latitude"].values
        # self.grid_lons = self.velocity_subdataset["longitude"].values
        # self.grid_depth = np.array([nominal_depth])

        self.vector_crs = ccrs.PlateCarree()
        self.land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
                                                            edgecolor='face', facecolor='dimgray', linewidth=0)

        self.crs_sps = ccrs.PlateCarree(central_longitude=-150)
        self.crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

        self.N_procs = N_procs
        self.dark_theme = dark_theme
        self.microbe_marker_size = microbe_marker_size
        self.input_dir = input_dir
        self.output_dir = output_dir

    def plot_frames(self, start_time, end_time, dt):
        iters = (end_time - start_time) // dt
        times = [start_time + n*dt for n in range(iters)]

        logger.info("Plotting {:d} frames from {:}->{:} on {:d} processors."
                    .format(iters, start_time, end_time, self.N_procs))

        if self.N_procs == 1:
            for i, t in enumerate(times):
                self.plot_frame(i, t)
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.plot_frame)(i, t)
                for i, t in enumerate(times)
            )

    def plot_frame(self, i, frame_time):
        logger = logging.getLogger(__name__ + str(i))  # Give each tile/processor its own logger.

        nc_input_filepath = os.path.join(self.output_dir, "microbe_data.nc")
        microbe_data = xr.open_dataset(nc_input_filepath)

        if self.dark_theme:
            plt.style.use("dark_background")

        logger.info("Plotting frame {:d}...".format(i))

        microbe_lons = microbe_data["longitude"][:, i]
        microbe_lats = microbe_data["latitude"][:, i]
        species = microbe_data["species"][:, i]

        fig = plt.figure(figsize=(16, 9))
        matplotlib.rcParams.update({'font.size': 10})

        ax = plt.subplot(111, projection=self.crs_sps)
        ax.add_feature(self.land_50m)
        ax.set_extent([-180, -120, 0, 60], ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, linestyle="--",
                          color="darkgray", alpha=0.8)
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlocator = mticker.FixedLocator([-180, -170, -160, -150, -140, -130, -120])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # # Find index of closest velocity field (in time).
        # u_idx = argmin(abs(self.grid_times - datetime64(frame_time)))
        #
        # u_data = self.velocity_subdataset["u"][u_idx].values
        # v_data = self.velocity_subdataset["v"][u_idx].values
        # u_magnitude = np.sqrt(u_data*u_data + v_data*v_data)
        #
        # im = ax.pcolormesh(self.grid_lons, self.grid_lats, u_magnitude, transform=self.vector_crs,
        #                    vmin=0, vmax=1, cmap="Blues_r")
        #
        # clb = fig.colorbar(im, ax=ax, extend="max", fraction=0.046, pad=0.1)
        # clb.ax.set_title(r"m/s")

        n_microbes = len(species)
        colors = n_microbes * [""]

        for j in range(n_microbes):
            if species[j] == ROCK:
                colors[j] = ROCK_COLOR
            elif species[j] == PAPER:
                colors[j] = PAPER_COLOR
            elif species[j] == SCISSORS:
                colors[j] = SCISSORS_COLOR

        ms = matplotlib.markers.MarkerStyle(marker=".", fillstyle="full")
        plt.scatter(microbe_lons, microbe_lats, marker=ms, linewidths=0, c=colors, edgecolors=colors, facecolors=colors,
                    s=self.microbe_marker_size, transform=self.vector_crs)

        plt.title(frame_time)

        rock_patch = Patch(color=ROCK_COLOR, label="Rocks")
        paper_patch = Patch(color=PAPER_COLOR, label="Papers")
        scissor_patch = Patch(color=SCISSORS_COLOR, label="Scissors")
        ax.legend(handles=[rock_patch, paper_patch, scissor_patch])

        ax.outline_patch.set_edgecolor("white")

        png_filename = "lagrangian_microbes_" + str(i).zfill(5) + ".png"
        png_filepath = os.path.join(self.output_dir, png_filename)
        logger.info("Saving figure: {:s}".format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format="png", transparent=False)

        plt.close("all")

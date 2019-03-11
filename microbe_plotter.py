import os
import re
import glob
import pickle

import numpy as np
from numpy import datetime64, abs, argmin
import xarray as xr
import joblib

import matplotlib
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

from velocity_fields import oscar_dataset_opendap_url
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
        oscar_url = oscar_dataset_opendap_url(2017)
        logger.info("Accessing OSCAR dataset over OPeNDAP: {:s}".format(oscar_url))

        velocity_dataset = xr.open_dataset(oscar_url)

        # Choose subset of velocity field we want to use
        nominal_depth = velocity_dataset["depth"].values[0]
        self.velocity_subdataset = velocity_dataset.sel(depth=nominal_depth,
                                                        latitude=slice(60, 0), longitude=slice(180, 240))

        self.grid_times = self.velocity_subdataset["time"].values
        self.grid_lats = self.velocity_subdataset["latitude"].values
        self.grid_lons = self.velocity_subdataset["longitude"].values
        self.grid_depth = np.array([nominal_depth])

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

    def plot_frames(self, iter_start, iter_end):
        logger.info("Plotting frames {:d}->{:d} on {:d} processors.".format(iter_start, iter_end, self.N_procs))

        if self.N_procs == 1:
            for i in range(iter_start, iter_end+1):
                self.plot_frame(i)
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.plot_frame)(i) for i in range(iter_start, iter_end+1)
            )

    def plot_frame(self, iteration):
        logger = logging.getLogger(__name__ + str(iteration))  # Give each tile/processor its own logger.

        if self.dark_theme:
            plt.style.use("dark_background")

        input_filename = "microbe_properties_" + str(iteration).zfill(5) + ".pickle"
        input_filepath = os.path.join(self.input_dir, input_filename)

        logger.info("Plotting frame {:d} from {:s}...".format(iteration, input_filepath))

        with open(input_filepath, "rb") as f:
            microbe_output = joblib.load(f)

            frame_time = microbe_output["time"]
            microbe_lons = microbe_output["lon"]
            microbe_lats = microbe_output["lat"]
            microbe_properties = microbe_output["properties"]
            species = microbe_properties["species"]

        fig = plt.figure(figsize=(16, 9))
        matplotlib.rcParams.update({'font.size': 10})

        ax = plt.subplot(111, projection=self.crs_sps)
        ax.add_feature(self.land_50m)
        ax.set_extent([-180, -120, 0, 60], ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, linestyle="--",
                          color="white", alpha=0.8)
        gl.xlabels_top = False
        gl.ylabels_left = False
        gl.xlocator = mticker.FixedLocator([-180, -170, -160, -150, -140, -130, -120])
        gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # Find index of closest velocity field (in time).
        u_idx = argmin(abs(self.grid_times - datetime64(frame_time)))

        u_data = self.velocity_subdataset["u"][u_idx].values
        v_data = self.velocity_subdataset["v"][u_idx].values
        u_magnitude = np.sqrt(u_data*u_data + v_data*v_data)

        print(u_data.shape)
        print(v_data.shape)
        print(u_magnitude.shape)

        im = ax.pcolormesh(self.grid_lons, self.grid_lats, u_magnitude, transform=self.vector_crs,
                           vmin=0, vmax=1, cmap="Blues_r")

        clb = fig.colorbar(im, ax=ax, extend="max", fraction=0.046, pad=0.1)
        clb.ax.set_title(r"m/s")

        n_microbes = len(microbe_lons)
        colors = n_microbes * [""]

        for i in range(n_microbes):
            if species[i] == ROCK:
                colors[i] = ROCK_COLOR
            elif species[i] == PAPER:
                colors[i] = PAPER_COLOR
            elif species[i] == SCISSORS:
                colors[i] = SCISSORS_COLOR

        ms = matplotlib.markers.MarkerStyle(marker=".", fillstyle="full")
        plt.scatter(microbe_lons, microbe_lats, marker=ms, linewidths=0, c=colors, edgecolors=colors, facecolors=colors,
                    s=self.microbe_marker_size, transform=self.vector_crs)

        plt.title(frame_time)

        rock_patch = Patch(color=ROCK_COLOR, label="Rocks")
        paper_patch = Patch(color=PAPER_COLOR, label="Papers")
        scissor_patch = Patch(color=SCISSORS_COLOR, label="Scissors")
        ax.legend(handles=[rock_patch, paper_patch, scissor_patch])

        png_filename = "lagrangian_microbes_" + str(iteration).zfill(5) + ".png"
        png_filepath = os.path.join(self.output_dir, png_filename)
        logger.info("Saving figure: {:s}".format(png_filepath))
        plt.savefig(png_filepath, dpi=300, format="png", transparent=False)

        plt.close("all")

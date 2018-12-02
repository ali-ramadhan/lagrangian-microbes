import os
import re
import glob
import pickle

import numpy as np
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

from constants import INTERACTION_OUTPUT_DIR, PLOTS_OUTPUT_DIR
from constants import ROCK_COLOR, PAPER_COLOR, SCISSOR_COLOR, MICROBE_MARKER_SIZE
from constants import N, t, dt, tpd, n_periods
from utils import closest_hour


velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'
velocity_dataset = xr.open_dataset(velocity_dataset_filepath)

t_starts = n_periods * [None]
u_magnitude = n_periods * [None]

print("Reading in velocity fields...")

for period in range(n_periods):
    t_start = velocity_dataset["time"][period].values
    year_float = velocity_dataset["year"].values[period]
    depth_float = velocity_dataset["depth"].values[0]

    velocity_subdataset = velocity_dataset.sel(time=t_start, year=year_float, depth=depth_float,
        latitude=slice(60, 0), longitude=slice(-180, -120))

    # Grid longitudes and latitudes.
    glons = velocity_subdataset['longitude'].values
    glats = velocity_subdataset['latitude'].values
    depth = np.array([depth_float])

    u_data = velocity_subdataset['u'].values
    v_data = velocity_subdataset['v'].values

    t_starts[period] = t_start
    u_magnitude[period] = np.sqrt(u_data*u_data + v_data*v_data)


vector_crs = ccrs.PlateCarree()
land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
    edgecolor='face',facecolor='dimgray', linewidth=0)

crs_sps = ccrs.PlateCarree(central_longitude=-150)
crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

def plot_microbe_warfare_frame(fpath):
    period = int(re.search("p\d\d\d\d", fpath).group(0)[1:])
    hour = int(re.search("h\d\d\d", fpath).group(0)[1:])

    print("Plotting figure from {:s}...".format(fpath))

    with open(fpath, "rb") as f:
        microbes = pickle.load(f)

        # Microbe longitudes and latidues.
        mlons, mlats, species = microbes[:, 0], microbes[:, 1], microbes[:, 2]

    fig = plt.figure(figsize=(16, 9))
    matplotlib.rcParams.update({'font.size': 10})

    ax = plt.subplot(111, projection=crs_sps)
    ax.add_feature(land_50m)
    ax.set_extent([-180, -120, 0, 60], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='black',
        alpha=0.8, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-180, -170, -160, -150, -140, -130, -120])
    gl.ylocator = mticker.FixedLocator([0, 10, 20, 30, 40, 50, 60])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    im = ax.pcolormesh(glons, glats, u_magnitude[period], transform=vector_crs, vmin=0, vmax=1, cmap='Blues_r')

    clb = fig.colorbar(im, ax=ax, extend='max', fraction=0.046, pad=0.1)
    clb.ax.set_title(r'm/s')

    n_microbes = len(mlons)
    colors = n_microbes * [""]

    for i in range(n_microbes):
        if species[i] == 1:
            colors[i] = ROCK_COLOR
        elif species[i] == 2:
            colors[i] = PAPER_COLOR
        elif species[i] == 3:
            colors[i] = SCISSOR_COLOR

    ms = matplotlib.markers.MarkerStyle(marker=".", fillstyle="full")
    plt.scatter(mlons, mlats, marker=ms, linewidths=0, c=colors, edgecolors=colors, facecolors=colors, s=MICROBE_MARKER_SIZE, transform=vector_crs)

    plt.title(closest_hour(t_starts[period]) + hour*dt)

    rock_patch = Patch(color=ROCK_COLOR, label="Rocks")
    paper_patch = Patch(color=PAPER_COLOR, label="Papers")
    scissor_patch = Patch(color=SCISSOR_COLOR, label="Scissors")
    ax.legend(handles=[rock_patch, paper_patch, scissor_patch])

    png_fname = "microbe_warfare_ph" + str(period).zfill(4) + str(hour).zfill(3) + ".png"
    png_fpath = os.path.join(PLOTS_OUTPUT_DIR, png_fname)
    print("Saving figure: {:s}".format(png_fpath))
    plt.savefig(png_fpath, dpi=300, format='png', transparent=False)

    plt.close('all')

def renumber_files():
    frame_files = glob.glob(PLOTS_OUTPUT_DIR + "/microbe_warfare_ph*.png")
    nfiles = len(frame_files)

    print("Renaming {:d} files...".format(nfiles))

    old_i = 0
    new_i = 0
    while new_i < nfiles:
        fname = "microbe_warfare_ph" + str(old_i).zfill(7) + ".png"
        fpath = os.path.join(PLOTS_OUTPUT_DIR, fname)
        if os.path.isfile(fpath):
            new_fname = "microbe_warfare_ph" + str(new_i).zfill(7) + ".png"
            new_fpath = os.path.join(PLOTS_OUTPUT_DIR, new_fname)
            os.rename(fpath, new_fpath)
            print("Rename: {:s} -> {:s}".format(fpath, new_fpath))
            new_i += 1
        old_i += 1

if __name__ == "__main__":
    print("Found {:d} CPUs.".format(joblib.cpu_count()))

    print("Globbing files from {:s}... ".format(INTERACTION_OUTPUT_DIR), end="")
    interaction_files = glob.glob(INTERACTION_OUTPUT_DIR + "/rps_microbe_species*.pickle")
    print("{:d} files found.".format(len(interaction_files)))

    joblib.Parallel(n_jobs=-1)(joblib.delayed(plot_microbe_warfare_frame)(f) for f in interaction_files)
    
    renumber_files()

import pickle

import numpy as np
import xarray as xr
import joblib

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.util
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from constants import N, t, dt, tpd, n_days

def plot_frames_for_day(day):
    velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'
    velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
    velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0,
        latitude=slice(60, 0), longitude=slice(-180, -120))
    
    # Grid longitudes and latitudes.
    glons = velocity_subdataset['longitude'].values
    glats = velocity_subdataset['latitude'].values
    depth = np.array([15.0])

    u_data = velocity_subdataset['u'].values
    v_data = velocity_subdataset['v'].values

    u_magnitude = np.sqrt(u_data*u_data + v_data*v_data)

    for n in range(tpd):
        pickle_filepath = "rps_microbe_species_d" + str(day).zfill(4) + "_n" + str(n).zfill(2) + ".pickle"
        
        with open(pickle_filepath, "rb") as f:
            microbes = pickle.load(f)

        # Microbe longitudes and latidues.
        mlons, mlats, species = microbes[:, 0], microbes[:, 1], microbes[:, 2]

        vector_crs = ccrs.PlateCarree()
        land_50m = cartopy.feature.NaturalEarthFeature('physical', 'land', '50m',
            edgecolor='face',facecolor='dimgray', linewidth=0)

        fig = plt.figure(figsize=(16, 9))
        matplotlib.rcParams.update({'font.size': 10})

        crs_sps = ccrs.PlateCarree(central_longitude=-150)
        crs_sps._threshold = 1000.0  # This solves https://github.com/SciTools/cartopy/issues/363

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

        im = ax.pcolormesh(glons, glats, u_magnitude, transform=vector_crs, vmin=0, vmax=1, cmap='Blues_r')

        clb = fig.colorbar(im, ax=ax, extend='max', fraction=0.046, pad=0.1)
        clb.ax.set_title(r'm/s')

        rock_lats, rock_lons = [], []
        paper_lats, paper_lons = [], []
        scissors_lats, scissors_lons = [], []

        for i in range(len(species)):
            if species[i] == 1:
                rock_lats.append(mlats[i])
                rock_lons.append(mlons[i])
            elif species[i] == 2:
                paper_lats.append(mlats[i])
                paper_lons.append(mlons[i])
            elif species[i] == 3:
                scissors_lats.append(mlats[i])
                scissors_lons.append(mlons[i])

        ax.plot(rock_lons, rock_lats, marker='o', linestyle='', color='red', ms=2, label='Rocks', transform=vector_crs)
        ax.plot(paper_lons, paper_lats, marker='o', linestyle='', color='lime', ms=2, label='Papers', transform=vector_crs)
        ax.plot(scissors_lons, scissors_lats, marker='o', linestyle='', color='cyan', ms=2, label='Scissors', transform=vector_crs)

        plt.title(str(t))
        ax.legend()

        png_filename = "microbe_warfare_dn" + str(tpd*day + n).zfill(5) + ".png"
        print("Saving figure: {:s}".format(png_filename))
        plt.savefig(png_filename, dpi=300, format='png', transparent=False)

        plt.close('all')

if __name__ == "__main__":
    print("Found {:d} CPUs.".format(joblib.cpu_count()))
    joblib.Parallel(n_jobs=-1)(joblib.delayed(plot_frames_for_day)(d) for d in range(n_days))
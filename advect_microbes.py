import sys
import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels

from constants import N, t, dt, tpd, n_days
from utils import closest_hour

def initialize_microbes():
    velocity_dataset_filepath = "/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc"

    # Load velocity dataset
    velocity_dataset = xr.open_dataset(velocity_dataset_filepath)

    # Choose subset of velocity field we want to use
    velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0,
        latitude=slice(60, 0), longitude=slice(-180, -120))

    lats = velocity_subdataset['latitude'].values
    lons = velocity_subdataset['longitude'].values
    depth = np.array([15.0])

    u_data = velocity_subdataset['u'].values
    v_data = velocity_subdataset['v'].values

    u_field = parcels.field.Field(name='U', data=u_data,
        lon=lons, lat=lats, depth=depth, mesh='spherical')
    v_field = parcels.field.Field(name='V', data=v_data,
        lon=lons, lat=lats, depth=depth, mesh='spherical')

    u_magnitude = np.sqrt(u_data*u_data + v_data*v_data)

    fieldset = parcels.fieldset.FieldSet(u_field, v_field)

    # Microbe longitudes and latitudes
    mlons = np.repeat(np.linspace(-170, -130, 28), 28)
    mlats = np.tile(np.linspace(10, 50, 28), 28)

    return parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle, lon=mlons, lat=mlats)

def advect_microbes(pset, year, n):
    velocity_dataset_filepath = "/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel" + str(year) + "_180.nc"

    # Load velocity dataset
    velocity_dataset = xr.open_dataset(velocity_dataset_filepath)

    n_fields = len(velocity_dataset["time"])
    
    t_start = velocity_dataset["time"][n].values
    
    # If we're on the last field, then t_end will be midnight of next year.
    if n == n_fields:
        t_end = datetime(t_start.year + 1, 1, 1)
    else:
        t_end = velocity_dataset["time"][n+1].values
    
    t_start_ch = closest_hour(t_start)
    t_end_ch = closest_hour(t_end)

    advection_hours = round((t_end_ch - t_start_ch) / timedelta(hours=1))
    # print("{:} -> {:} ({:d} hours)".format(t_start, t_end, advection_hours))

    # Choose subset of velocity field we want to use
    year_float = velocity_dataset["year"].values[n]
    depth_float = velocity_dataset["depth"].values[0]

    velocity_subdataset = velocity_dataset.sel(time=t_start, year=year_float, depth=depth_float, latitude=slice(60, 0), longitude=slice(-180, -120))

    lats = velocity_subdataset['latitude'].values
    lons = velocity_subdataset['longitude'].values
    depth = np.array([depth_float])

    u_data = velocity_subdataset['u'].values
    v_data = velocity_subdataset['v'].values

    u_field = parcels.field.Field(name='U', data=u_data,
        lon=lons, lat=lats, depth=depth, mesh='spherical')
    v_field = parcels.field.Field(name='V', data=v_data,
        lon=lons, lat=lats, depth=depth, mesh='spherical')

    u_magnitude = np.sqrt(u_data*u_data + v_data*v_data)

    # pset.fieldset = parcels.fieldset.FieldSet(u_field, v_field)

    print("Advecting: {:} -> {:}... ".format(t_start_ch, t_end_ch), end="")

    nc_filename = "rps_microbe_locations_p" + str(n).zfill(4) + ".nc"

    tic = time.time()
    pset.execute(parcels.AdvectionRK4, runtime=advection_hours*dt, dt=dt, output_file=pset.ParticleFile(name=nc_filename, outputdt=dt))
    toc = time.time()
    print("({:g} s)".format(toc - tic))

    for i, p in enumerate(pset):
        if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
            print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
                .format(i, p.lat, p.lon))
            pset.remove(i)

if __name__ == "__main__":
    pset = initialize_microbes()
    year = 2017
    print("pset contains {:d} microbes.".format(len(pset)))

    for n in range(5):
        advect_microbes(pset, year, n)
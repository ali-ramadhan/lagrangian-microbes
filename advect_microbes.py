import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels

from constants import N, t, dt, tpd, n_days

velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'

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

lats_pset = np.tile(np.linspace(10, 50, 28), 28)
lons_pset = np.repeat(np.linspace(-170, -130, 28), 28)

pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle,
    lon=lons_pset, lat=lats_pset)

for n in range(n_days):
    print("Advecting: {:} -> {:}... ".format(t, t+tpd*dt), end="")

    nc_filename = "rps_microbe_locations_d" + str(n).zfill(4) + ".nc"

    t1 = time.time()
    pset.execute(parcels.AdvectionRK4, runtime=tpd*dt, dt=dt, output_file=pset.ParticleFile(name=nc_filename, outputdt=dt))
    t2 = time.time()
    print("({:g} s)".format(t2 - t1))

    for i, p in enumerate(pset):
        if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
            print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
                .format(i, p.lat, p.lon))
            pset.remove(i)

    t = t+dt
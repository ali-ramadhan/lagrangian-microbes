import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels
from scipy.spatial import KDTree

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

lats_pset = np.tile(np.linspace(10, 50, 91), 91)
lons_pset = np.repeat(np.linspace(-170, -130, 91), 91)

pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle,
    lon=lons_pset, lat=lats_pset)

t = datetime(2017, 1, 1)
dt = timedelta(hours=2)
tpd = 12  # time steps per day
n_days = 7  # number of days to advect microbes for

for n in range(n_days):
    print("Advecting: {:} -> {:}... ".format(t, t+tpd*dt))

    nc_filename = "advected_microbes_" + str(n).zfill(4) + ".nc"

    t1 = time.time()
    pset.execute(parcels.AdvectionRK4, runtime=tpd*dt, dt=dt, verbose_progress=True,
        output_file=pset.ParticleFile(name=nc_filename, outputdt=dt))
    t2 = time.time()
    print("({:g} s)".format(t2 - t1))

    for i, p in enumerate(pset):
        if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
            print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
                .format(i, p.lat, p.lon))
            pset.remove(i)

    t = t+dt
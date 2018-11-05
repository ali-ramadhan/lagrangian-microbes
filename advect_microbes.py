from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import parcels

velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'

# Load velocity dataset
velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
print(velocity_dataset)

# Choose subset of velocity field we want to use
velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0,
    latitude=slice(60, 0), longitude=slice(-180, -120))

print(velocity_subdataset)
print(velocity_subdataset['u'])

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
# fieldset.U.show()

lats_pset = np.tile(np.linspace(10, 50, 31), 31)
lons_pset = np.repeat(np.linspace(-170, -130, 31), 31)

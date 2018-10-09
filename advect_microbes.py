from datetime import timedelta

import numpy as np
import xarray as xr
import parcels

velocity_dataset_filepath = '/home/alir/nobackup/data/oscar_third_deg_180/oscar_vel2017_180.nc'

# Load velocity dataset
velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
print(velocity_dataset)

# Choose subset of velocity field we want to use
velocity_subdataset = velocity_dataset.sel(time=np.datetime64('2017-01-01'), year=2017.0, depth=15.0, latitude=slice(60, 0), longitude=slice(-180, -120))
print(velocity_subdataset)

print(velocity_subdataset['u'])

lons = velocity_subdataset['longitude'].values
lats = velocity_subdataset['latitude'].values

u_field = parcels.field.Field(name='u', data=velocity_subdataset['u'].values, lon=lons, lat=lats)
v_field = parcels.field.Field(name='v', data=velocity_subdataset['v'].values, lon=lons, lat=lats)

fieldset = parcels.fieldset.FieldSet(u_field, v_field)
fieldset.U.show()

pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle, lon=[-170, -160, -150, -140, -130], lat=[10, 20, 30, 40, 50])
print(pset)

pset.show(field=fieldset.V)

pset.execute(parcels.AdvectionRK4, runtime=timedelta(days=1), dt=timedelta(minutes=1), output_file=pset.ParticleFile(name="advected_microbes.nc", outputdt=timedelta(minutes=10)))
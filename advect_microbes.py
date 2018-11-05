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

# species_field = -1 * np.ones((11,11), dtype=np.int32)
# for i, lat in enumerate(np.linspace(10, 50, 11)):
#   for j, lon in enumerate(np.linspace(-170, -130, 11)):
#       pass

# species_pfield = parcels.field.Field(name='species', data=species_field,
#   lat=np.linspace(10, 50, 11), lon=np.linspace(-170, -130, 11), depth=depth, mesh='spherical')

class MicrobeParticle(parcels.JITParticle):
    species = parcels.Variable('species', dtype=np.int32, initial=-1)

pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=MicrobeParticle,
    lon=lons_pset, lat=lats_pset)

for i, particle in enumerate(pset):
    if 37.5 <= particle.lat <= 52.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 1
    elif 37.5 <= particle.lat <= 52.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 2
    elif 37.5 <= particle.lat <= 52.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 3
    elif 22.5 <= particle.lat <= 37.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 3
    elif 22.5 <= particle.lat <= 37.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 1
    elif 22.5 <= particle.lat <= 37.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 2
    elif 7.5 <= particle.lat <= 22.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 2
    elif 7.5 <= particle.lat <= 22.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 3
    elif 7.5 <= particle.lat <= 22.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 1
    print("Particle {:03d} @({:.2f},{:.2f}) [species={:d}]".format(i, particle.lat, particle.lon, particle.species))
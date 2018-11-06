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

def rock_paper_scissors_type(n):
    if n == 1:
        return "rock"
    elif n == 2:
        return "paper"
    elif n == 3:
        return "scissors"
    return None

t = datetime(2017, 1, 1)
dt = timedelta(hours=2)

for n in range(10000):
    print("Advecting: {:} -> {:}".format(t, t+dt))

    nc_filename = "advected_microbes_" + str(n).zfill(4) + ".nc"

    pset.execute(parcels.AdvectionRK4, runtime=dt, dt=dt, verbose_progress=True,
        output_file=pset.ParticleFile(name=nc_filename, outputdt=dt))

    print("Computing microbe interactions...")
    
    N = len(pset)
    
    for i, p1 in enumerate(pset):
        for j, p2 in enumerate(pset[i+1:]):
            if np.abs(p1.lat - p2.lat) < 1 and np.abs(p1.lon - p2.lon) < 1:
                p1_type = rock_paper_scissors_type(p1.species)
                p2_type = rock_paper_scissors_type(p2.species)

                winner = None

                if p1_type == "rock" and p2_type == "scissors":
                    winner = p1
                elif p1_type == "rock" and p2_type == "paper":
                    winner = p2
                elif p1_type == "paper" and p2_type == "rock":
                    winner = p1
                elif p1_type == "paper" and p2_type == "scissors":
                    winner = p2
                elif p1_type == "scissors" and p2_type == "rock":
                    winner = p2
                elif p1_type == "scissors" and p2_type == "paper":
                    winner = p1
                else:
                    winner = None

                if winner == p1:
                    p2.species = p1.species
                    print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                        .format(p1_type, i, p1.lat, p1.lon, p2_type, j+i, p2.lat, p2.lon, i))
                elif winner == p2:
                    p1.species = p2.species
                    print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                        .format(p1_type, i, p1.lat, p1.lon, p2_type, j+i, p2.lat, p2.lon, j+i))

    for i, p in enumerate(pset):
        if p.lat >= 59 or p.lat <= 1 or p.lon <= -179 or p.lon >= -121:
            print("Removing particle #{:d} @({:.2f},{:.2f}). Too close to boundary"
                .format(i, p.lat, p.lon))
            pset.remove(i)

    t = t+dt
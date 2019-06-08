import os
import argparse
from datetime import datetime, timedelta

from particle_advecter import ParticleAdvecter, uniform_particle_locations

parser = argparse.ArgumentParser(description="Advect some Lagrangian microbes in the Northern Pacific.")

parser.add_argument("-C", "--cores", type=int, required=True, help="Number of cores to use")
parser.add_argument("-N", "--N-particles", type=int, required=True, help="Number of Lagrangian microbes")
parser.add_argument("-K", "--Kh", type=float, required=True, help="Isotropic horizontal diffusivity")
parser.add_argument("-d", "--output_dir", type=str, required=True, help="Output directory")

args = parser.parse_args()
C, N, Kh = args.cores, args.N_particles, args.Kh

Kh = int(Kh) if Kh.is_integer() else Kh

# Output directories.
base_dir = args.output_dir
output_dir = os.path.join(base_dir, "N" + str(N) + "_Kh" + str(Kh))

# We'll advect 6 months at a time as we hit a weird joblib error around November if you try advecting for a full year.
start_time = datetime(2018, 1, 1)
mid_time = datetime(2018, 7, 1)
end_time = datetime(2019, 1, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

# Create a particle advecter that will the advect the particles we just generated in parallel.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=C, velocity_field="OSCAR", output_dir=output_dir, Kh=Kh)

# Advect the particles and save all the data to NetCDF.
pa.time_step(start_time, mid_time, dt)
pa.time_step(mid_time, end_time, dt)
pa.create_netcdf_file(start_time, end_time, dt)

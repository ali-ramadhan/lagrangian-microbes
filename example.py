from datetime import datetime, timedelta

import numpy as np
from numpy import int8

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interaction_simulator import InteractionSimulator

N = 10000  # Number of particles
output_dir = "/home/gridsan/aramadhan/microbes_output/"  # Output directory for everything.

start_time = datetime(2017, 1, 1)
end_time = datetime(2017, 2, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

# Create a particle advecter that will the advect the particles we just generated on 4 processors.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=4, velocity_field="OSCAR", output_dir=output_dir)

# Advect the particles from Jan 1 2017 to Feb 1 2017 using a time step of 1 hour.
pa.time_step(start_time=start_time, end_time=end_time, dt=dt)

# Representing rock, paper, and scissors with simple 8-bit integers.
ROCK, PAPER, SCISSORS = int8(1), int8(2), int8(3)

microbe_properties = {
    "species": np.random.choice([ROCK, PAPER, SCISSORS], N)
}

isim = InteractionSimulator(pa, microbe_properties, None, output_dir=output_dir)
isim.time_step(start_time=start_time, end_time=end_time, dt=dt)
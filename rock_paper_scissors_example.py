from datetime import datetime, timedelta

import numpy as np
from numpy import int8

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interaction_simulator import InteractionSimulator
from interactions import rock_paper_scissors

N = 10000  # Number of particles
output_dir = "/home/gridsan/aramadhan/microbes_output/"  # Output directory for everything.

start_time = datetime(2017, 1, 1)
end_time = datetime(2017, 2, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

# Create a particle advecter that will the advect the particles we just generated on 4 processors.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=4, velocity_field="OSCAR", output_dir=output_dir)

# Advect the particles.
pa.time_step(start_time=start_time, end_time=end_time, dt=dt)

# Create an interaction simulator that uses the rock-paper-scissors pair interaction.
isim = InteractionSimulator(pa, pair_interaction=rock_paper_scissors, interaction_radius=0.05, output_dir=output_dir)

# Simulate the interactions.
isim.time_step(start_time=start_time, end_time=end_time, dt=dt)

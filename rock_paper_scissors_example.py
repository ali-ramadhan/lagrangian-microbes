import os
import glob
from datetime import datetime, timedelta

import numpy as np
from numpy import int8
import ffmpeg

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interaction_simulator import InteractionSimulator
from microbe_plotter import MicrobePlotter
from interactions import rock_paper_scissors

N = 10000  # Number of particles
output_dir = "/home/alir/cnhlab004/lagrangian_microbes_output/"  # Output directory for everything.

start_time = datetime(2017, 1, 1)
end_time = datetime(2017, 2, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

# Create a particle advecter that will the advect the particles we just generated on 4 processors.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=4, velocity_field="OSCAR", output_dir=output_dir, Kh=100)

# Advect the particles and save all the data to NetCDF.
pa.time_step(start_time, end_time, dt)
pa.create_netcdf_file(start_time, end_time, dt)

# Create an interaction simulator that uses the rock-paper-scissors pair interaction.
rps_interaction = rock_paper_scissors(N_microbes=N, pRS=0.5, pPR=0.5, pSP=0.5)
isim = InteractionSimulator(pair_interaction=rps_interaction, interaction_radius=0.05, output_dir=output_dir)

# Simulate the interactions.
isim.time_step(start_time, end_time, dt)

# Create a microbe plotter that will produce a plot of all the microbes at a single iteration.
mp = MicrobePlotter(N_procs=-1, dark_theme=True, input_dir=output_dir, output_dir=output_dir)

# Plot the first 100 frames and save them to disk.
mp.plot_frames(start_time, end_time, dt)

# Make movie!
(
    ffmpeg
    .input(os.path.join(output_dir, "lagrangian_microbes_%05d.png"), framerate=30)
    .output(os.path.join(output_dir, "movie.mp4"), crf=15, pix_fmt='yuv420p')
    .run()
)

for fl in glob.glob(os.path.join(output_dir, "*.png")):
    os.remove(fl)

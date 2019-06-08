import os
import glob
import argparse
from datetime import datetime, timedelta

import numpy as np
from numpy import int8
import ffmpeg

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interactions import rock_paper_scissors
from interaction_simulator import InteractionSimulator
from microbe_plotter import MicrobePlotter
from analysis import species_count_figure

parser = argparse.ArgumentParser(description="Simulate some Lagrangian microbes in the Northern Pacific.")

parser.add_argument("-C", "--cores", type=int, required=True, help="Number of cores to use")
parser.add_argument("-N", "--N-particles", type=int, required=True, help="Number of Lagrangian microbes")
parser.add_argument("-K", "--Kh", type=float, required=True, help="Isotropic horizontal diffusivity")
parser.add_argument("-p", type=float, required=True, help="Interaction probability")
parser.add_argument("-a", type=float, required=True, help="Asymmetric factor in rock-paper interaction")
parser.add_argument("-d", "--output_dir", type=str, required=True, help="Output directory")

args = parser.parse_args()
C, N, Kh, p, a, base_dir = args.cores, args.N_particles, args.Kh, args.p, args.a, args.output_dir
pRS, pPR, pSP = p, p, p + a

Kh = int(Kh) if Kh.is_integer() else Kh

# Output directories.
output_dir = os.path.join(base_dir, "N" + str(N) + "_Kh" + str(Kh) + "_p" + str(p) + "_a" + str(a))

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

# Create an interaction simulator that uses the rock-paper-scissors pair interaction.
rps_interaction = rock_paper_scissors(N_microbes=N, pRS=0.5, pPR=0.5, pSP=0.5)
isim = InteractionSimulator(pair_interaction=rps_interaction, interaction_radius=0.05, output_dir=output_dir)

# Simulate the interactions.
isim.time_step(start_time, end_time, dt)

# Create a microbe plotter that will produce a plot of all the microbes at a single iteration.
mp = MicrobePlotter(N_procs=-1, dark_theme=True, input_dir=output_dir, output_dir=output_dir)

# Plot the first 100 frames and save them to disk.
mp.plot_frames(start_time, end_time, dt)

# Produce species count figure.
species_count_figure(output_dir, start_time, end_time, dt)

# Make movie!
(
    ffmpeg
    .input(os.path.join(output_dir, "lagrangian_microbes_%05d.png"), framerate=30)
    .output(os.path.join(output_dir, "movie.mp4"), crf=15, pix_fmt='yuv420p')
    .run()
)

for fl in glob.glob(os.path.join(output_dir, "*.png")):
    os.remove(fl)

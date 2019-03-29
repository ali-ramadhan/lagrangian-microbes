import os
import argparse
from datetime import datetime, timedelta

import numpy as np
from numpy import int8

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interaction_simulator import InteractionSimulator
from microbe_plotter import MicrobePlotter
from interactions import rock_paper_scissors

import argparse

parser = argparse.ArgumentParser(description="Simulate some Lagrangian microbes in the Northern Pacific.")

parser.add_argument("-N", "--N_particles", type=int, nargs=1, required=True, help="Number of Lagrangian microbes")
parser.add_argument("-p", type=float, nargs=1, required=True, help="Interaction probability")

args = parser.parse_args()
N, p = args.N_particles[0], args.p[0]
pRS, pPR, pSP = p, p, p

# Output directories.
base_output_dir = os.path.join("/home/alir/cnhlab004/lagrangian_microbes_output", "N" + str(N) + "_" + "p" + str(p))
advection_output_dir = os.path.join(base_output_dir, "particle_locations")
interaction_output_dir = os.path.join(base_output_dir, "microbe_interactions")
plots_output_dir = os.path.join(base_output_dir, "plots")

start_time = datetime(2018, 1, 1)
end_time = datetime(2018, 2, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=20, lat_max=50, lon_min=198, lon_max=208)

# Create a particle advecter that will the advect the particles we just generated in parallel.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=8, velocity_field="OSCAR", output_dir=advection_output_dir)

# Advect the particles.
pa.time_step(start_time=start_time, end_time=end_time, dt=dt)

# Create an interaction simulator that uses the rock-paper-scissors pair interaction.
rps_interaction = rock_paper_scissors(N_microbes=N, pRS=pRS, pPR=pPR, pSP=pSP)
isim = InteractionSimulator(pa, pair_interaction=rps_interaction, interaction_radius=0.05, output_dir=interaction_output_dir)

# Simulate the interactions.
isim.time_step(start_time=start_time, end_time=end_time, dt=dt)

# Create a microbe plotter that will produce a plot of all the microbes at a single iteration.
mp = MicrobePlotter(N_procs=8, dark_theme=True, input_dir=interaction_output_dir, output_dir=plots_output_dir)

# Plot the first 100 frames and save them to disk.
mp.plot_frames(0, isim.iteration)

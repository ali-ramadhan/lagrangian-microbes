from datetime import datetime, timedelta

import numpy as np
from numpy import int8

from particle_advecter import ParticleAdvecter, uniform_particle_locations
from interaction_simulator import InteractionSimulator

N = 10000  # Number of particles
output_dir = "/home/gridsan/aramadhan/microbes_output/"  # Output directory for everything.

# Representing rock, paper, and scissors with simple 8-bit integers.
ROCK, PAPER, SCISSORS = int8(1), int8(2), int8(3)

microbe_properties = {
    "species": np.random.choice([ROCK, PAPER, SCISSORS], N)
}

rock_paper_scissors_interaction_parameters = {
    "pRS": 0.50,  # Forward probability that rock beats scissors.
    "pPR": 0.50,  # Forward probability that paper beats rock.
    "pSP": 0.50   # Forward probability that scissors beats paper.
}


def rock_paper_scissors(parameters, microbe_properties, p1, p2):
    pRS, pPR, pSP = parameters["pRS"], parameters["pPR"], parameters["pSP"]
    species = microbe_properties["species"]

    if species[p1] != species[p2]:
        s1, s2 = species[p1], species[p2]

        r = np.random.rand()  # Random float from Uniform[0,1)

        winner = None

        if s1 == ROCK and s2 == SCISSORS:
            winner = p1 if r < pRS else p2
        elif s1 == ROCK and s2 == PAPER:
            winner = p2 if r < pPR else p1
        elif s1 == PAPER and s2 == ROCK:
            winner = p1 if r < pPR else p2
        elif s1 == PAPER and s2 == SCISSORS:
            winner = p2 if r < pSP else p1
        elif s1 == SCISSORS and s2 == ROCK:
            winner = p2 if r < pRS else p1
        elif s1 == SCISSORS and s2 == PAPER:
            winner = p1 if r < pSP else p2

        if winner == p1:
            species[p2] = species[p1]
        elif winner == p2:
            species[p1] = species[p2]


start_time = datetime(2017, 1, 1)
end_time = datetime(2017, 2, 1)
dt = timedelta(hours=1)

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=N, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

# Create a particle advecter that will the advect the particles we just generated on 4 processors.
pa = ParticleAdvecter(particle_lons, particle_lats, N_procs=4, velocity_field="OSCAR", output_dir=output_dir)

# Advect the particles from Jan 1 2017 to Feb 1 2017 using a time step of 1 hour.
pa.time_step(start_time=start_time, end_time=end_time, dt=dt)

isim = InteractionSimulator(pa, microbe_properties, pair_interaction=rock_paper_scissors,
                            pair_interaction_parameters=rock_paper_scissors_interaction_parameters,
                            interaction_radius=0.05, output_dir=output_dir)
isim.time_step(start_time=start_time, end_time=end_time, dt=dt)
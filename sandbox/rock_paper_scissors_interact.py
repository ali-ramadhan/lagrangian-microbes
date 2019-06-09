import os
import argparse
from datetime import datetime, timedelta

from interactions import rock_paper_scissors
from interaction_simulator import InteractionSimulator

parser = argparse.ArgumentParser(description="Simulate interactions between Lagrangian microbes in the Northern Pacific.")

parser.add_argument("-N", "--N-particles", type=int, required=True, help="Number of Lagrangian microbes")
parser.add_argument("-K", "--Kh", type=float, required=True, help="Isotropic horizontal diffusivity")
parser.add_argument("-p", type=float, required=True, help="Interaction probability")
parser.add_argument("-a", type=float, required=True, help="Asymmetric factor in rock-paper interaction")
parser.add_argument("-r", "--interaction-radius", type=float, required=True, help="Microbe interaction radius [deg²]")
parser.add_argument("-d", "--output_dir", type=str, required=True, help="Output directory")

args = parser.parse_args()
N, Kh, p, a, r = args.N_particles, args.Kh, args.p, args.a, args.interaction_radius
pRS, pPR, pSP = p, p, p + a

Kh = int(Kh) if Kh.is_integer() else Kh

# Output directories.
base_dir = args.output_dir
advection_dir = os.path.join(base_dir, "N" + str(N) + "_Kh" + str(Kh))
output_dir = os.path.join(base_dir, "N" + str(N) + "_Kh" + str(Kh) + "_p" + str(p) + "_a" + str(a) + "_r" + str(r))

start_time = datetime(2018, 1, 1)
mid_time = datetime(2018, 7, 1)
end_time = datetime(2019, 1, 1)
dt = timedelta(hours=1)

# Create an interaction simulator that uses the rock-paper-scissors pair interaction.
rps_interaction = rock_paper_scissors(N_microbes=N, pRS=pRS, pPR=pPR, pSP=pSP)
isim = InteractionSimulator(pair_interaction=rps_interaction, interaction_radius=r, advection_dir=advection_dir, output_dir=output_dir)

# Simulate the interactions.
isim.time_step(start_time, end_time, dt)

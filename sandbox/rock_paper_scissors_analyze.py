import os
import argparse
from datetime import datetime, timedelta

from analysis import species_count_figure

parser = argparse.ArgumentParser(description="Analyze Lagrangian microbe data.")

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
output_dir = os.path.join(base_dir, "N" + str(N) + "_Kh" + str(Kh) + "_p" + str(p) + "_a" + str(a) + "_r" + str(r))

start_time = datetime(2018, 1, 1)
end_time = datetime(2019, 1, 1)
dt = timedelta(hours=1)

png_filename = "species_count_N{:d}_Kh{:}_p{:}_a{:}_r{:}.png".format(N, Kh, p, a, r)
species_count_figure(output_dir, start_time, end_time, dt, png_filename)

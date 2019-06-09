import os
import glob
import argparse
from datetime import datetime, timedelta

import ffmpeg

from microbe_plotter import MicrobePlotter

parser = argparse.ArgumentParser(description="Plot and animate Lagrangian microbe data.")

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
mid_time = datetime(2018, 7, 1)
end_time = datetime(2019, 1, 1)
dt = timedelta(hours=1)

# Create a microbe plotter that will produce a plot of all the microbes at a single iteration.
mp = MicrobePlotter(N_procs=-1, dark_theme=True, input_dir=output_dir, output_dir=output_dir)

# Plot each frames and save them to disk.
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

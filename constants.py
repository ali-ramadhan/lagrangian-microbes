import os
from datetime import datetime, timedelta

ROCK, PAPER, SCISSORS = 1, 2, 3

# output directories.
OUTPUT_ROOT_DIR = os.path.join("/home", "alir", "nobackup", "lagrangian_microbe_output")

# ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_advection")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_plots")

ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_advection")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_plots")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_plots")
INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_pRS0.9_interactions")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_pRS0.9_plots")

for dir in [OUTPUT_ROOT_DIR, ADVECTION_OUTPUT_DIR, INTERACTION_OUTPUT_DIR, PLOTS_OUTPUT_DIR]:
    if not os.path.exists(dir):
        print("Creating directory: {:s}".format(dir))
        os.makedirs(dir)

# Domain boundary.
DOMAIN_LONS = slice(-180, -120)
DOMAIN_LATS = slice(60, 0)

Tx, Ty = 4, 7  # number of "tiles" in the x and y.

# Big patch (debug 2.8kp)
# lon_min, lon_max = -170, -130
# lat_min, lat_max = 10, 50
# NTx, NTy = 10, 10  # number of microbes in each tile (x and y directions)
# n_periods = 3 # number of periods to advect microbes for
# INTERACTION_LENGTH_SCALE = 0.5  # [deg]
# MICROBE_MARKER_SIZE = 40

# Small patch (490kp)
lon_min, lon_max = -155, -145
lat_min, lat_max = 25, 35
NTx, NTy = 175, 100  # number of microbes in each tile (x and y directions)
n_periods = 72 # number of periods to advect microbes for
INTERACTION_LENGTH_SCALE = 0.01  # [deg]
MICROBE_MARKER_SIZE = 1

N = Tx*Ty*NTx*NTy  # number of microbes

delta_mlon = (lon_max - lon_min) / (Tx * NTx)
delta_mlat = (lat_max - lat_min) / (Ty * NTy)

t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=1)  # advection time step
tpd = int(timedelta(days=1) / dt)   # time steps per day

# Interaction parameters
INTERACTION_NORM = 2
INTERACTION_pRS = 0.9  # Forward probability that rock beats scissors.
INTERACTION_pPR = 0.5  # Forward probability that paper beats rock.
INTERACTION_pSP = 0.5  # Forward probability that scissors beats paper.

# Plotting constants
ROCK_COLOR = "red"
PAPER_COLOR = "limegreen"
SCISSOR_COLOR = "blue"

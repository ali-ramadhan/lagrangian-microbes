import os
from datetime import datetime, timedelta

# output directories.
OUTPUT_ROOT_DIR = os.path.join("/home", "alir", "nobackup", "lagrangian_microbe_output")

# ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_advection")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_plots")

ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "small_patch_490kp_advection")
INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_interactions")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_plots")

for dir in [OUTPUT_ROOT_DIR, ADVECTION_OUTPUT_DIR, INTERACTION_OUTPUT_DIR, PLOTS_OUTPUT_DIR]:
    if not os.path.exists(dir):
        print("Creating directory: {:s}".format(dir))
        os.makedirs(dir)

# Domain boundary.
DOMAIN_LONS = slice(-180, -120)
DOMAIN_LATS = slice(60, 0)

# Square inside which we regularly place microbes.
# lon_min, lon_max = -170, -130
# lat_min, lat_max = 10, 50
lon_min, lon_max = -155, -145
lat_min, lat_max = 25, 35

Tx, Ty = 4, 7  # number of "tiles" in the x and y.
NTx, NTy = 175, 100  # number of microbes in each tile (x and y directions)
N = Tx*Ty*NTx*NTy  # number of microbes

t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=1)  # advection time step
tpd = int(timedelta(days=1) / dt)   # time steps per day

# number of periods to advect microbes for
# n_periods = 3
n_periods = 72

# Interaction parameters
# INTERACTION_LENGTH_SCALE = 0.5  # [deg]
INTERACTION_LENGTH_SCALE = 0.005  # [deg]

INTERACTION_NORM = 2
INTERACTION_p = 0.9


# Plotting constants
ROCK_COLOR = "red"
PAPER_COLOR = "limegreen"
SCISSOR_COLOR = "blue"

# MICROBE_MARKER_SIZE = 1
MICROBE_MARKER_SIZE = 40
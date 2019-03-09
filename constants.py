import os
from datetime import datetime, timedelta

# Representing rock, paper, and scissors with simple integers. Why use anything
# more complicated?
ROCK, PAPER, SCISSORS = 1, 2, 3

# Output directories. The advection and microbe interaction outputs can be pretty big.
# Three must be specified:
# 1. ADVECTION_OUTPUT_DIR for storing lat, lon data for each time step from Ocean Parcels.
# 2. INTERACTION_OUTPUT_DIR for storing lat, lon, species data from computing the microbe
#    interactions.
# 3. PLOTS_OUTPUT_DIR for storing plots.

OUTPUT_ROOT_DIR = os.path.join("/home", "alir", "nobackup", "lagrangian_microbe_output")

ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_advection")
INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_interactions")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_plots")

# ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_advection")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_plots")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.55_plots")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_pRS0.51_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_pRS0.51_plots")

# Create directories if they don't already exist.
# for dir in [OUTPUT_ROOT_DIR, ADVECTION_OUTPUT_DIR, INTERACTION_OUTPUT_DIR, PLOTS_OUTPUT_DIR]:
#     if not os.path.exists(dir):
#         print("Creating directory: {:s}".format(dir))
#         os.makedirs(dir)

# Domain boundary. This is used to select a subset of the velocity field.
DOMAIN_LONS = slice(-180, -120)
DOMAIN_LATS = slice(60, 0)

# Number of "tiles" in the x and y. This is used for parallelization as each core is
# responsible for executing a single tile. For single-core simulations just set Tx, Ty = 1, 1.
Tx, Ty = 4, 1

# Big patch (debug 2.8kp)
# lon_min, lon_max = -170, -130
# lat_min, lat_max = 10, 50
# NTx, NTy = 10, 10  # number of microbes in each tile (x and y directions)
# n_periods = 3 # number of periods to advect microbes for
# INTERACTION_LENGTH_SCALE = 0.5  # [deg]
# MICROBE_MARKER_SIZE = 40

# Small patch (490kp)
# lon_min, lon_max = -155, -145
# lat_min, lat_max = 25, 35
# NTx, NTy = 175, 100  # number of microbes in each tile (x and y directions)
# n_periods = 72 # number of periods to advect microbes for
# INTERACTION_LENGTH_SCALE = 0.01  # [deg]
# MICROBE_MARKER_SIZE = 1

lon_min, lon_max = -170, -130
lat_min, lat_max = 10, 50
NTx, NTy = 25, 10  # number of microbes in each tile (x and y directions)
n_periods = 3 # number of periods to advect microbes for
INTERACTION_LENGTH_SCALE = 0.5  # [deg]
MICROBE_MARKER_SIZE = 40

N = Tx*Ty*NTx*NTy  # Total number of microbes.

# Longitudinal and latitudinal separation between microbes if regularly spaced.
delta_mlon = (lon_max - lon_min) / (Tx * NTx)
delta_mlat = (lat_max - lat_min) / (Ty * NTy)

t = datetime(2017, 1, 1)           # initial time
dt = timedelta(hours=1)            # advection time step
tpd = int(timedelta(days=1) / dt)  # time steps per day

# Rock-paper-scissors interaction parameters.
INTERACTION_NORM = 2
INTERACTION_pRS = 0.51  # Forward probability that rock beats scissors.
INTERACTION_pPR = 0.50  # Forward probability that paper beats rock.
INTERACTION_pSP = 0.50  # Forward probability that scissors beats paper.

# Colors to use for plotting rock, paper, and scissors particles.
ROCK_COLOR = "red"
PAPER_COLOR = "limegreen"
SCISSOR_COLOR = "blue"

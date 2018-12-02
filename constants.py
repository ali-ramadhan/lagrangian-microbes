import os
from datetime import datetime, timedelta

OUTPUT_ROOT_DIR = os.path.join("/home", "alir", "nobackup", "lagrangian_microbe_output")

ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_advection")
INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_interactions")
PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "debug_2.8kp_p0.9_plots")

# ADVECTION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "small_patch_490kp_advection")
# INTERACTION_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_interactions")
# PLOTS_OUTPUT_DIR = os.path.join(OUTPUT_ROOT_DIR, "small_patch_490kp_p0.9_plots")

lon_min, lon_max = -170, -130
lat_min, lat_max = 10, 50

# lon_min, lon_max = -155, -145
# lat_min, lat_max = 25, 35

# N = 28**2  # number of microbes
Tx, Ty = 4, 7  # number of "tiles" in the x and y.
NTx, NTy = 10, 10  # number of microbes in each tile (x and y directions)
N = Tx*Ty*NTx*NTy  # number of microbes

t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=1)  # advection time step
tpd = int(timedelta(days=1) / dt)   # time steps per day

n_periods = 72  # number of periods to advect microbes for

# degrees lol...
INTERACTION_LENGTH_SCALE = 0.5
# INTERACTION_LENGTH_SCALE = 0.005

INTERACTION_NORM = 2
INTERACTION_p = 0.9

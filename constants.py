from datetime import datetime, timedelta

output_dir = "rps_proof_of_concept"

lon_min, lon_max = -170, -130
lat_min, lat_max = 10, 50

# N = 28**2  # number of microbes
Tx, Ty = 4, 7  # number of "tiles" in the x and y.
NTx, NTy = 5, 5  # number of microbes in each tile (x and y directions)
N = Tx*Ty*NTx*NTy  # number of microbes

t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=1)  # advection time step
tpd = int(timedelta(days=1) / dt)   # time steps per day

n_days = 7  # number of days to advect microbes for
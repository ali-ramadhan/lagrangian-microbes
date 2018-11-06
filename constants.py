from datetime import datetime, timedelta

output_dir = "rps_proof_of_concept"

N = 28**2  # number of microbes
t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=1)  # advection time step
tpd = 24  # time steps per day
n_days = 7  # number of days to advect microbes for
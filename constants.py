from datetime import datetime, timedelta

N = 28**2  # number of microbes
t = datetime(2017, 1, 1)  # initial time
dt = timedelta(hours=2)  # advection time step
tpd = 12  # time steps per day
n_days = 7  # number of days to advect microbes for
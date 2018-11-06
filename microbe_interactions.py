import time
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

def rock_paper_scissors_type(n):
    if n == 1:
        return "rock"
    elif n == 2:
        return "paper"
    elif n == 3:
        return "scissors"
    return None

# microbe_species = np.zeros(N)
# for i, m in enumerate(microbe_locations):
#     print("{:d}: {:}".format(i, m))
#     if 37.5 <= m.lat <= 52.5 and -172.5 <= m.lon <= -157.5:
#         m.species = 1
#     elif 37.5 <= m.lat <= 52.5 and -157.5 <= m.lon <= -142.5:
#         m.species = 2
#     elif 37.5 <= m.lat <= 52.5 and -142.5 <= m.lon <= -127.5:
#         m.species = 3
#     elif 22.5 <= m.lat <= 37.5 and -172.5 <= m.lon <= -157.5:
#         m.species = 3
#     elif 22.5 <= m.lat <= 37.5 and -157.5 <= m.lon <= -142.5:
#         m.species = 1
#     elif 22.5 <= m.lat <= 37.5 and -142.5 <= m.lon <= -127.5:
#         m.species = 2
#     elif 7.5 <= m.lat <= 22.5 and -172.5 <= m.lon <= -157.5:
#         m.species = 2
#     elif 7.5 <= m.lat <= 22.5 and -157.5 <= m.lon <= -142.5:
#         m.species = 3
#     elif 7.5 <= m.lat <= 22.5 and -142.5 <= m.lon <= -127.5:
#         m.species = 1

N = 28**2  # number of microbes
t = datetime(2017, 1, 1)
dt = timedelta(hours=2)
tpd = 12  # time steps per day
n_days = 7  # number of days to advect microbes for

for day in range(7):
    microbe_location_filepath = "rps_microbe_locations_" + str(day).zfill(4) + ".nc"
    microbe_location_dataset = xr.open_dataset(microbe_location_filepath)

    for n in range(tpd):
        print("{:} ".format(t), end="")
        lats = microbe_location_dataset["lat"][:, n].values
        lons = microbe_location_dataset["lon"][:, n].values
        microbe_locations = np.stack((lats, lons), axis=-1)

        print("Building k-d tree... ", end="")
        t1 = time.time()
        kd = KDTree(np.array(microbe_locations))
        t2 = time.time()
        print("({:g} s) ".format(t2 - t1), end="")

        print("Querying pairs... ", end="")
        t1 = time.time()
        microbe_pairs = kd.query_pairs(r=1, p=2)
        t2 = time.time()
        print("({:g} s) ".format(t2 - t1), end="")

        print(" {:d} interacting pairs.".format(len(microbe_pairs)))

        t = t + dt


#     print("Computing microbe interactions...", end="")

#     t1 = time.time()
    
#     particle_locations = np.zeros([N, 2])
#     for i, p in enumerate(pset):
#         particle_locations[i, :] = [p.lon, p.lat]

#     kd = KDTree(np.array(particle_locations))
#     interacting_pairs = kd.query_ball_tree(kd, r=1, p=1)
#     print("interacting_pairs: {:}".format(interacting_pairs))
    
#     t2 = time.time()
#     print(" ({:g} s)".format(t2 - t1))
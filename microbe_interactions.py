import time
import pickle
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.spatial import KDTree

from constants import N, t, dt, tpd, n_days

def rps_type(n):
    if n == 1:
        return "rock"
    elif n == 2:
        return "paper"
    elif n == 3:
        return "scissors"
    return None

def initialize_microbe_species():
    microbe_location_filepath = "rps_microbe_locations_p0000.nc"
    microbe_location_dataset = xr.open_dataset(microbe_location_filepath)
    
    lon0 = microbe_location_dataset["lon"][:, 0].values
    lat0 = microbe_location_dataset["lat"][:, 0].values
    microbe_locations = np.stack((lon0, lat0), axis=-1)
    
    microbe_species = np.zeros(N, dtype=int)

    for i, ml in enumerate(microbe_locations):
        lon, lat = ml
        if 37.5 <= lat <= 52.5 and -172.5 <= lon <= -157.5:
            microbe_species[i] = 1
        elif 37.5 <= lat <= 52.5 and -157.5 <= lon <= -142.5:
            microbe_species[i] = 2
        elif 37.5 <= lat <= 52.5 and -142.5 <= lon <= -127.5:
            microbe_species[i] = 3
        elif 22.5 <= lat <= 37.5 and -172.5 <= lon <= -157.5:
            microbe_species[i] = 3
        elif 22.5 <= lat <= 37.5 and -157.5 <= lon <= -142.5:
            microbe_species[i] = 1
        elif 22.5 <= lat <= 37.5 and -142.5 <= lon <= -127.5:
            microbe_species[i] = 2
        elif 7.5 <= lat <= 22.5 and -172.5 <= lon <= -157.5:
            microbe_species[i] = 2
        elif 7.5 <= lat <= 22.5 and -157.5 <= lon <= -142.5:
            microbe_species[i] = 3
        elif 7.5 <= lat <= 22.5 and -142.5 <= lon <= -127.5:
            microbe_species[i] = 1

        # print("#{:d}: lon={:.2f}, lat={:.2f}, species={:d}".format(i, lon, lat, microbe_species[i]))

    return microbe_species

microbe_species = initialize_microbe_species()

for period in range(5):
    microbe_location_filepath = "rps_microbe_locations_p" + str(period).zfill(4) + ".nc"
    microbe_location_dataset = xr.open_dataset(microbe_location_filepath)

    hours = len(microbe_location_dataset["time"][0, :])

    for n in range(hours):
        print("{:} ".format(t), end="")
        lons = microbe_location_dataset["lon"][:, n].values
        lats = microbe_location_dataset["lat"][:, n].values
        microbe_locations = np.stack((lons, lats), axis=-1)

        print("Building kd tree... ", end="")
        t1 = time.time()
        kd = KDTree(np.array(microbe_locations))
        t2 = time.time()
        print("({:g} s) ".format(t2 - t1), end="")

        print("Querying pairs... ", end="")
        t1 = time.time()
        microbe_pairs = kd.query_pairs(r=1, p=2)
        t2 = time.time()
        print("({:g} s) ".format(t2 - t1), end="")

        print(" {:d} pairs.".format(len(microbe_pairs)))

        for pair in microbe_pairs:
            p1, p2 = pair
            if microbe_species[p1] != microbe_species[p2]:
                s1, s2 = rps_type(microbe_species[p1]), rps_type(microbe_species[p2])
                
                winner = None
                if s1 == "rock" and s2 == "scissors":
                    winner = p1
                elif s1 == "rock" and s2 == "paper":
                    winner = p2
                elif s1 == "paper" and s2 == "rock":
                    winner = p1
                elif s1 == "paper" and s2 == "scissors":
                    winner = p2
                elif s1 == "scissors" and s2 == "rock":
                    winner = p2
                elif s1 == "scissors" and s2 == "paper":
                    winner = p1

                if winner == p1:
                    microbe_species[p2] = microbe_species[p1]
                    print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                        .format(s1, p1, lats[p1], lons[p1], s2, p2, lats[p2], lons[p2], p1))
                elif winner == p2:
                    microbe_species[p1] = microbe_species[p2]
                    print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                        .format(s1, p1, lats[p1], lons[p1], s2, p2, lats[p2], lons[p2], p2))

        pickle_filepath = "rps_microbe_species_p" + str(period).zfill(4) + "_h" + str(n).zfill(3) + ".pickle"
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(np.stack((lons, lats, microbe_species), axis=-1), f, pickle.HIGHEST_PROTOCOL)

        t = t + dt
import os
import time
import pickle
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
import joblib

from constants import N, Tx, Ty, NTx, NTy
from constants import t, dt, tpd, n_periods
from constants import output_dir
from constants import INTERACTION_LENGTH_SCALE, INTERACTION_NORM, INTERACTION_p

p = INTERACTION_p

def rps_type(n):
    if n == 1:
        return "rock"
    elif n == 2:
        return "paper"
    elif n == 3:
        return "scissors"
    return None

def initialize_microbe_species(microbe_locations):
    # microbe_location_filepath = "rps_microbe_locations_p0000.nc"
    # microbe_location_dataset = xr.open_dataset(microbe_location_filepath)

    # lon0 = microbe_location_dataset["lon"][:, 0].values
    # lat0 = microbe_location_dataset["lat"][:, 0].values
    # microbe_locations = np.stack((lon0, lat0), axis=-1)

    microbe_species = np.random.choice([1, 2, 3], N)

    # microbe_species = np.zeros(N, dtype=int)
    # for i, ml in enumerate(microbe_locations):
    #     lon, lat = ml
    #     if 37.5 <= lat <= 52.5 and -172.5 <= lon <= -157.5:
    #         microbe_species[i] = 1
    #     elif 37.5 <= lat <= 52.5 and -157.5 <= lon <= -142.5:
    #         microbe_species[i] = 2
    #     elif 37.5 <= lat <= 52.5 and -142.5 <= lon <= -127.5:
    #         microbe_species[i] = 3
    #     elif 22.5 <= lat <= 37.5 and -172.5 <= lon <= -157.5:
    #         microbe_species[i] = 3
    #     elif 22.5 <= lat <= 37.5 and -157.5 <= lon <= -142.5:
    #         microbe_species[i] = 1
    #     elif 22.5 <= lat <= 37.5 and -142.5 <= lon <= -127.5:
    #         microbe_species[i] = 2
    #     elif 7.5 <= lat <= 22.5 and -172.5 <= lon <= -157.5:
    #         microbe_species[i] = 2
    #     elif 7.5 <= lat <= 22.5 and -157.5 <= lon <= -142.5:
    #         microbe_species[i] = 3
    #     elif 7.5 <= lat <= 22.5 and -142.5 <= lon <= -127.5:
    #         microbe_species[i] = 1

        # print("#{:d}: lon={:.2f}, lat={:.2f}, species={:d}".format(i, lon, lat, microbe_species[i]))

    return microbe_species

microbe_species = None

for period in range(n_periods):
    # microbe_location_filepath = "rps_microbe_locations_p" + str(period).zfill(4) + ".nc"
    # microbe_location_dataset = xr.open_dataset(microbe_location_filepath)
    # hours = len(microbe_location_dataset["time"][0, :])

    lons, lats = None, None

    print("Reading files ({:s})... ".format(output_dir), end="")
    tic = time.time()
    for block in range(Tx*Ty):
        dump_filename = "rps_microbe_locations_p" + str(period).zfill(4) + "_block" + str(block).zfill(2) + ".joblib.pickle"
        dump_filepath = os.path.join(output_dir, dump_filename)
        latlon_store = joblib.load(dump_filepath)

        if block == 0:
            hours = latlon_store["hours"]
            lons = np.zeros((hours, Tx*Ty*NTx*NTy))
            lats = np.zeros((hours, Tx*Ty*NTx*NTy))

        i1 = block*(NTx*NTy)  # Particle starting index
        i2 = (block+1)*(NTx*NTy) # Particle ending index
        lons[:, i1:i2] = latlon_store["lon"]
        lats[:, i1:i2] = latlon_store["lat"]

    toc = time.time()
    print("({:g} s) ".format(toc - tic))

    for h in range(hours):
        print("{:} ".format(t), end="")

        # lons = microbe_location_dataset["lon"][:, n].values
        # lats = microbe_location_dataset["lat"][:, n].values
        # microbe_locations = np.stack((lons, lats), axis=-1)
        microbe_locations = np.stack((lons[h, :], lats[h, :]), axis=-1)

        if period == 0 and h == 0:
            microbe_species = initialize_microbe_species(microbe_locations)

        print("Building kd tree... ", end="")
        tic = time.time()
        kd = cKDTree(np.array(microbe_locations))
        toc = time.time()
        print("({:g} s) ".format(toc - tic), end="")

        print("Querying pairs... ", end="")
        tic = time.time()
        microbe_pairs = kd.query_pairs(r=INTERACTION_LENGTH_SCALE, p=INTERACTION_NORM)
        toc = time.time()
        print("({:g} s) ".format(toc - tic), end="")

        print(" {:d} pairs. ".format(len(microbe_pairs)), end="")

        tic = time.time()
        n_battles = 0
        for pair in microbe_pairs:
            p1, p2 = pair
            if microbe_species[p1] != microbe_species[p2]:
                s1, s2 = rps_type(microbe_species[p1]), rps_type(microbe_species[p2])

                r = np.random.rand()  # Random float from Uniform[0,1)

                winner = None

                if r < p:  # Forward interaction
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
                elif r > p:  # Reverse interaction
                    if s1 == "rock" and s2 == "scissors":
                        winner = p2
                    elif s1 == "rock" and s2 == "paper":
                        winner = p1
                    elif s1 == "paper" and s2 == "rock":
                        winner = p2
                    elif s1 == "paper" and s2 == "scissors":
                        winner = p1
                    elif s1 == "scissors" and s2 == "rock":
                        winner = p1
                    elif s1 == "scissors" and s2 == "paper":
                        winner = p2

                if winner == p1:
                    microbe_species[p2] = microbe_species[p1]
                    # print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                    #     .format(s1, p1, lats[h, p1], lons[h, p1], s2, p2, lats[h, p2], lons[h, p2], p1))
                elif winner == p2:
                    microbe_species[p1] = microbe_species[p2]
                    # print("[{:s}#{:d}] @({:.2f}, {:.2f}) vs. [{:s}#{:d}] @({:.2f}, {:.2f}): #{:d} wins!"
                    #     .format(s1, p1, lats[h, p1], lons[h, p1], s2, p2, lats[h, p2], lons[h, p2], p2))

                n_battles += 1

        toc = time.time()
        print("{:d} battles. ({:g} s)".format(n_battles, toc - tic))

        pickle_fname = "rps_microbe_species_p" + str(period).zfill(4) + "_h" + str(h).zfill(3) + ".pickle"
        pickle_fpath = os.path.join(output_dir, pickle_fname)
        with open(pickle_fpath, 'wb') as f:
            pickle.dump(np.stack((lons[h, :], lats[h, :], microbe_species), axis=-1), f, pickle.HIGHEST_PROTOCOL)

        t = t + dt

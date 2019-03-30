import os
import pickle
from time import time
from datetime import datetime, timedelta

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

import numpy as np
from numpy import float32, zeros, stack
from scipy.spatial import cKDTree
import joblib

from particle_advecter import ParticleAdvecter
from utils import pretty_time, pretty_filesize


class InteractionSimulator:
    def __init__(
            self,
            particle_advecter,
            pair_interaction,
            interaction_radius,
            interaction_norm=2,
            self_interaction=None,
            output_dir="."
    ):

        # Sanitize output_dir variable.
        output_dir = os.path.abspath(output_dir)
        logger.info("Microbe interactions output directory: {:s}".format(output_dir))

        # Create output directory if it doesn't exist.
        if not os.path.exists(output_dir):
            logger.info("Creating directory: {:s}".format(output_dir))
            os.makedirs(output_dir)

        pair_interaction_function, pair_interaction_parameters, microbe_properties = pair_interaction

        self.pa = particle_advecter
        self.microbe_properties = microbe_properties
        self.pair_interaction = pair_interaction_function
        self.pair_interaction_parameters = pair_interaction_parameters
        self.interaction_radius = interaction_radius
        self.interaction_norm = interaction_norm
        self.self_interaction = self_interaction
        self.output_dir = output_dir
        self.iteration = 0

    def time_step(self, start_time, end_time, dt):
        logger = logging.getLogger(__name__ + "interactions")
        t = start_time
        while t < end_time:
            iters_remaining = (end_time - t) // dt
            iters_to_do = min(self.pa.output_chunk_iters, iters_remaining)

            if iters_to_do == self.pa.output_chunk_iters:
                logger.info("Will simulate microbe interactions for {:d} iterations.".format(iters_to_do))
            else:
                logger.info("Will simulate microbe interactions for {:d} iterations to end of simulation."
                            .format(iters_to_do))

            start_iter_str = str(self.iteration).zfill(5)
            end_iter_str = str(self.iteration + iters_to_do).zfill(5)

            chunk_start_time = t
            chunk_end_time = t + iters_to_do*dt

            # Pre-allocate memory to store all the microbe locations.
            microbe_lons = zeros((iters_to_do, self.pa.N_particles), dtype=float32)
            microbe_lats = zeros((iters_to_do, self.pa.N_particles), dtype=float32)

            tic = time()
            for tile_id in range(self.pa.N_procs):
                input_filename = "particle_locations_" + start_iter_str + "_" + end_iter_str + \
                                 "_tile" + str(tile_id).zfill(2) + ".pickle"
                input_filepath = os.path.join(self.pa.output_dir, input_filename)
                particle_locations = joblib.load(input_filepath)

                times = particle_locations["time"]

                i1 = tile_id * self.pa.particles_per_tile        # Particle starting index
                i2 = (tile_id + 1) * self.pa.particles_per_tile  # Particle ending index
                microbe_lons[:, i1:i2] = particle_locations["lon"]
                microbe_lats[:, i1:i2] = particle_locations["lat"]

            toc = time()
            logger.info("Reading locations of {:d} particles from {:d} files took {:s}."
                        .format(self.pa.N_particles, self.pa.N_procs, pretty_time(toc - tic)))

            logger.info("Simulating interactions (iteration {:} -> {:}): {:} -> {:}..."
                        .format(start_iter_str, end_iter_str, chunk_start_time, chunk_end_time))

            for n in range(iters_to_do):
                microbe_locations = stack((microbe_lons[n, :], microbe_lats[n, :]), axis=-1)

                logger.info("Building kd-tree... ")
                tic = time()
                kdt = cKDTree(np.array(microbe_locations))
                build_time = time() - tic

                logger.info("Querying kd-tree for pairs... "
                            .format(self.interaction_radius, self.interaction_norm))
                tic = time()
                microbe_pairs = kdt.query_pairs(r=self.interaction_radius, p=self.interaction_norm)
                query_time = time() - tic

                n_interactions = len(microbe_pairs)
                logger.info("Simulating {:d} pair-wise interactions...".format(n_interactions))
                tic = time()
                for pair in microbe_pairs:
                    self.pair_interaction(self.pair_interaction_parameters, self.microbe_properties, pair[0], pair[1])
                interaction_time = time() - tic

                pickle_filename = "microbe_properties_" + str(self.iteration).zfill(5) + ".pickle"
                pickle_filepath = os.path.join(self.output_dir, pickle_filename)

                microbe_output = {
                    "time": times[n],
                    "lon": microbe_lons[n, :],
                    "lat": microbe_lats[n, :],
                    "properties": self.microbe_properties
                }

                tic = time()
                with open(pickle_filepath, "wb") as f:
                    logger.info("Saving microbe properties: {:s}".format(pickle_filepath))
                    joblib.dump(microbe_output, f, compress=("zlib", 3), protocol=pickle.HIGHEST_PROTOCOL)

                pickling_time = time() - tic
                pickle_filesize = os.path.getsize(pickle_filepath)

                logger.info("Building kd-tree:         {:s}.".format(pretty_time(build_time)))
                logger.info("Querying for pairs:       {:s}.".format(pretty_time(query_time)))
                logger.info("Simulating interactions:  {:s}.".format(pretty_time(interaction_time)))
                logger.info("Pickling and compressing: {:s}. ({:s}, {:s} per particle)"
                            .format(pretty_time(pickling_time), pretty_filesize(pickle_filesize),
                                    pretty_filesize(pickle_filesize / self.pa.N_particles)))

                t = t + dt
                self.iteration += 1

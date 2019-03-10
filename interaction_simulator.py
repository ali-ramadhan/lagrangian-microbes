import os
from time import time
from datetime import datetime, timedelta

import numpy as np
from numpy import zeros
from scipy.spatial import cKDTree
import joblib

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from particle_advecter import ParticleAdvecter
from utils import pretty_time, pretty_filesize


class InteractionSimulator:
    def __init__(
            self,
            particle_advecter,
            microbe_properties,
            interaction,
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

        self.pa = particle_advecter
        self.microbe_properties = microbe_properties
        self.interaction = interaction
        self.self_interaction = self_interaction
        self.output_dir = output_dir

    def time_step(self, start_time, end_time, dt):
        t = start_time
        iteration = 0
        while t < end_time:
            iters_remaining = (end_time - t) // dt
            iters_to_do = min(self.pa.output_chunk_iters, iters_remaining)

            if iters_to_do == self.pa.output_chunk_iters:
                logger.info("Will simulate microbe interactions for {:d} iterations.".format(iters_to_do))
            else:
                logger.info("Will simulate microbe interactions for {:d} iterations to end of simulation."
                            .format(iters_to_do))

            start_iter_str = str(iteration).zfill(5)
            end_iter_str = str(iteration + iters_to_do).zfill(5)

            chunk_start_time = t
            chunk_end_time = t + iters_to_do*dt

            microbe_lons = np.zeros((iters_to_do, self.pa.N_particles), dtype=np.float32)
            microbe_lats = np.zeros((iters_to_do, self.pa.N_particles), dtype=np.float32)

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
            logger.info("Reading location of {:d} particles from {:d} files took {:s}."
                        .format(self.pa.N_particles, self.pa.N_procs, pretty_time(toc - tic)))

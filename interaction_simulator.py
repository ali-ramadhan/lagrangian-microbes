import os
import pickle
from time import time
from datetime import datetime, timedelta

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

import numpy as np
import xarray as xr
from numpy import float32, int8, zeros, stack
from scipy.spatial import cKDTree
import joblib

from particle_advecter import ParticleAdvecter
from utils import pretty_time, pretty_filesize


class InteractionSimulator:
    def __init__(
            self,
            pair_interaction,
            interaction_radius,
            interaction_norm=2,
            self_interaction=None,
            advection_dir=".",
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

        self.microbe_properties = microbe_properties
        self.pair_interaction = pair_interaction_function
        self.pair_interaction_parameters = pair_interaction_parameters
        self.interaction_radius = interaction_radius
        self.interaction_norm = interaction_norm
        self.self_interaction = self_interaction
        self.advection_dir = advection_dir
        self.output_dir = output_dir
        self.iteration = 0

    def time_step(self, start_time, end_time, dt):
        logger = logging.getLogger(__name__ + "interactions")

        nc_input_filepath = os.path.join(self.advection_dir, "particle_data.nc")
        particle_data = xr.open_dataset(nc_input_filepath)
        N_particles, Nt = particle_data["longitude"].shape
        times = [start_time + n * dt for n in range(Nt)]

        mlons = zeros((N_particles, Nt), dtype=float32)
        mlats = zeros((N_particles, Nt), dtype=float32)
        species = zeros((N_particles, Nt), dtype=int8)

        # This dataset will store all the lat/lon/species positions of each Lagrangian microbe, and will be saved to
        # NetCDF.
        microbe_data = xr.Dataset({
                "longitude": (["particle number", "time"], mlons),
                "latitude":  (["particle number", "time"], mlats),
                "species":   (["particle number", "time"], species)
            },
            coords={
                "particle number": range(1, N_particles+1),
                "time": times
            }
        )

        logger.info("Simulating interactions for {:d} microbes over {:d} time steps ({:} -> {:})."
                    .format(N_particles, Nt, start_time, end_time))

        t = start_time
        while t < end_time:
            i = self.iteration

            logger.info("Simulating interactions for {:} -> {:}...".format(t, t+dt))

            microbe_locations = stack((particle_data["longitude"][:, i],
                                       particle_data["latitude"][:, i]), axis=-1)

            logger.info("Building kd-tree... ")
            tic = time()
            kdt = cKDTree(np.array(microbe_locations))
            build_time = time() - tic

            logger.info("Querying kd-tree for pairs... ")
            tic = time()
            microbe_pairs = kdt.query_pairs(r=self.interaction_radius, p=self.interaction_norm)
            query_time = time() - tic

            n_interactions = len(microbe_pairs)
            logger.info("Simulating {:d} pair-wise interactions...".format(n_interactions))
            tic = time()
            for pair in microbe_pairs:
                self.pair_interaction(self.pair_interaction_parameters, self.microbe_properties, pair[0], pair[1])
            interaction_time = time() - tic

            microbe_data["longitude"][:, i] = particle_data["longitude"][:, i]
            microbe_data["latitude"][:, i] = particle_data["latitude"][:, i]
            microbe_data["species"][:, i] = self.microbe_properties["species"]

            logger.info("Building kd-tree:        {:s}.".format(pretty_time(build_time)))
            logger.info("Querying for pairs:      {:s}.".format(pretty_time(query_time)))
            logger.info("Simulating interactions: {:s}.".format(pretty_time(interaction_time)))

            t = t + dt
            self.iteration += 1

        nc_output_filepath = os.path.join(self.output_dir, "microbe_data.nc")

        logger.info("Writing microbe data to {:s}...".format(nc_output_filepath))
        microbe_data.to_netcdf(nc_output_filepath)

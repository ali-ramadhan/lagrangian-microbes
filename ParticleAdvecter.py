import os
from datetime import datetime, timedelta
from itertools import product

import numpy as np
from numpy import linspace, repeat, tile
import joblib

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from utils import most_symmetric_integer_factorization


def generate_uniform_particle_locations(N_particles=1,
                                        lat_min_particles=10,
                                        lat_max_particles=50,
                                        lon_min_particles=-170,
                                        lon_max_particles=-130
                                        ):
    # Determine number of particles to generate along each dimension.
    N_particles_lat, N_particles_lon = most_symmetric_integer_factorization(N_particles)

    logger.info("Generating ({:d},{:d}) particles along each (lat, lon).".format(N_particles_lat, N_particles_lon))

    particle_lons = repeat(linspace(lon_min_particles, lon_max_particles, N_particles_lon), N_particles_lat)
    particle_lats = tile(linspace(lat_min_particles, lat_max_particles, N_particles_lat), N_particles_lon)

    return particle_lats, particle_lons


def distribute_particles_across_tiles(particle_lons, particle_lats, tiles):
    """
    Splits a list of particle longitudes and a list of particle latitudes into `tiles` equally sized lists.

    Args:
        particle_lons: List of particle longitudes.
        particle_lats: List of particle latitudes.
        tiles: Number of tiles or processors to split the particles into.

    Returns:
        particle_lons_tiled: A List containing `tiles` lists of particle longitudes for each processor.
        particle_lats_tiled: A List containing `tiles` lists of particle latitudes for each processor.

    """
    assert particle_lons.size == particle_lats.size
    N_particles = particle_lons.size

    assert (N_particles / tiles).is_integer()
    particles_per_tile = N_particles // tiles

    particle_lons_tiled = tiles * [None]
    particle_lats_tiled = tiles * [None]

    for i in range(tiles):
        particle_idx_start, particle_idx_end = i*particles_per_tile, (i+1)*particles_per_tile
        particle_lons_tiled[i] = particle_lons[particle_idx_start:particle_idx_end]
        particle_lats_tiled[i] = particle_lons[particle_idx_start:particle_idx_end]

    return particle_lons_tiled, particle_lats_tiled


def oscar_dataset_opendap_url(year):
    return r"https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel" \
           + str(year) + ".nc.gz"


class ParticleAdvecter:
    def __init__(
        self,
        velocity_field="OSCAR",
        particle_initial_distribution="uniform",
        dt=timedelta(hours=1),
        N_particles=1,
        N_procs=-1,
        lat_min_particle=10,
        lat_max_particle=50,
        lon_min_particle=-170,
        lon_max_particle=-130,
        oscar_dataset_dir=".",
        domain_lats=slice(60, 0),
        domain_lons=slice(-180, -120),
        start_time=datetime(2017, 1, 1),
        end_time=datetime(2018, 1, 1),
        output_dir="."
    ):
        assert velocity_field == "OSCAR", "OSCAR is the only supported velocity field right now."
        assert particle_initial_distribution == "uniform"

        # Figure out number of processors to use.
        assert 1 <= N_procs or N_procs == -1, "Number of processors N_procs must be a positive integer " \
                                              "or -1 (use all processors)."

        max_procs = joblib.cpu_count()
        N_procs = N_procs if 1 <= N_procs <= max_procs else max_procs

        logger.info("Requested {:d} processors. Found {:d}. Using {:d}.".format(N_procs, max_procs, N_procs))

        # Sanitize N_particles input.
        assert N_particles >= 1, "N_particles must be a positive integer."
        assert N_particles >= N_procs, "There must be at least one Lagrangian particle per processor."
        logger.info("Number of Lagrangian particles: {:d}".format(N_particles))

        # Ensure there are an equal number of particles per processor.
        if N_procs > 1:
            if not (N_particles / N_procs).is_integer():
                logger.warning("Cannot equally distribute {:d} Lagrangian particles across {:d} processor."
                               .format(N_particles, N_procs))

                N_particles = (N_particles // N_procs) * N_procs
                logger.warning("Using {:d} Lagrangian particles per processor.".format(N_particles))

        logger.info("Lagrangian particles per processor: {:}".format(N_particles / N_procs))

        # Sanitize output_dir variable.
        output_dir = os.path.abspath(output_dir)
        assert os.path.isdir(output_dir), "output_dir {:s} is not a directory.".format(output_dir)
        logger.info("Particle advection output directory: {:s}".format(output_dir))

        # Create output directory if it doesn't exist.
        if not os.path.exists(output_dir):
            logger.info("Creating directory: {:s}".format(output_dir))
            os.makedirs(output_dir)

        # Sanitize oscar_dataset_dir variable.
        oscar_dataset_dir = os.path.abspath(oscar_dataset_dir)
        assert os.path.isdir(oscar_dataset_dir), "oscar_dataset_Dir {:s} is not a directory.".format(oscar_dataset_dir)
        logger.info("Will read OSCAR velocity datasets from: {:s}".format(oscar_dataset_dir))

        logger.info("Generating locations for {:d} Lagrangian particles...".format(N_particles))

        # Generate initial locations for each particle.
        if particle_initial_distribution == "uniform":
            particle_lons, particle_lats = generate_uniform_particle_locations(N_particles=N_particles,
                                                                               lat_min_particles=lat_min_particle,
                                                                               lat_max_particles=lat_max_particle,
                                                                               lon_min_particles=lon_min_particle,
                                                                               lon_max_particles=lon_max_particle)
        else:
            raise ValueError("Only uniform initial particle distributions are supported.")

        logger.info("Distributing {:d} particles across {:d} processors...".format(N_particles, N_procs))
        particle_lons, particle_lats = distribute_particles_across_tiles(particle_lons, particle_lats, N_procs)

        self.velocity_field = velocity_field
        self.particle_initial_distribution = particle_initial_distribution
        self.N_procs = N_procs
        self.N_particles = N_particles
        self.output_dir = output_dir
        self.oscar_dataset_dir = oscar_dataset_dir
        self.particle_lons = particle_lons
        self.particle_lats = particle_lats

        self.dt = dt
        self.domain_lats = domain_lats
        self.domain_lons = domain_lons
        self.start_time = start_time
        self.end_time = end_time

    def time_step(self, Nt, dt):
        if self.N_procs == 1:
            self.time_step_tile(0)
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.time_step_tile)(tile_id) for tile_id in range(self.N_procs)
            )

    def time_step_tile(self, tile_id):
        velocity_dataset = xr.open_dataset(velocity_dataset_filepath)
        n_fields = len(velocity_dataset["time"])

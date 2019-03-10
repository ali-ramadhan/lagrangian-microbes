import os
import time
import pickle
from datetime import datetime, timedelta

import numpy as np
from numpy import linspace, repeat, tile
import xarray as xr
import joblib
import parcels

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from utils import most_symmetric_integer_factorization, closest_hour, pretty_time


def uniform_particle_locations(N_particles=1, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
    # Determine number of particles to generate along each dimension.
    N_particles_lat, N_particles_lon = most_symmetric_integer_factorization(N_particles)

    logger.info("Generating ({:d}, {:d}) particles along each (lat, lon).".format(N_particles_lat, N_particles_lon))

    particle_lons = repeat(linspace(lon_min, lon_max, N_particles_lon), N_particles_lat)
    particle_lats = tile(linspace(lat_min, lat_max, N_particles_lat), N_particles_lon)

    return particle_lons, particle_lats


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
        particle_lats_tiled[i] = particle_lats[particle_idx_start:particle_idx_end]

    return particle_lons_tiled, particle_lats_tiled


def oscar_dataset_opendap_url(year):
    return r"https://podaac-opendap.jpl.nasa.gov:443/opendap/allData/oscar/preview/L4/oscar_third_deg/oscar_vel" \
           + str(year) + ".nc.gz"


class ParticleAdvecter:
    def __init__(
        self,
        particle_lons,
        particle_lats,
        velocity_field="OSCAR",
        N_procs=-1,
        oscar_dataset_dir=".",
        output_dir="."
    ):
        assert velocity_field == "OSCAR", "OSCAR is the only supported velocity field right now."

        # Figure out number of processors to use.
        assert 1 <= N_procs or N_procs == -1, "Number of processors N_procs must be a positive integer " \
                                              "or -1 (use all processors)."

        max_procs = joblib.cpu_count()
        N_procs = N_procs if 1 <= N_procs <= max_procs else max_procs

        logger.info("Requested {:d} processors. Found {:d}. Using {:d}.".format(N_procs, max_procs, N_procs))

        assert particle_lons.size == particle_lats.size

        N_particles = particle_lons.size
        assert (N_particles / N_procs).is_integer()

        logger.info("Lagrangian particles per processor: {:}".format(N_particles // N_procs))

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

        logger.info("Distributing {:d} particles across {:d} processors...".format(N_particles, N_procs))
        particle_lons, particle_lats = distribute_particles_across_tiles(particle_lons, particle_lats, N_procs)

        self.particle_lons = particle_lons
        self.particle_lats = particle_lats
        self.velocity_field = velocity_field
        self.N_procs = N_procs
        self.output_dir = output_dir
        self.oscar_dataset_dir = oscar_dataset_dir

    def time_step(self, start_time, end_time, dt):
        if self.N_procs == 1:
            self.time_step_tile(0)
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.time_step_tile)(tile_id, start_time, end_time, dt) for tile_id in range(self.N_procs)
            )

    def time_step_tile(self, tile_id, start_time, end_time, dt):
        tilestamp = "[Tile {:02d}]".format(tile_id)

        particle_lons, particle_lats = self.particle_lons[tile_id], self.particle_lats[tile_id]
        particles_per_tile = particle_lons.size

        oscar_url = oscar_dataset_opendap_url(start_time.year)
        logger.info("{:s} Accessing OSCAR dataset over OPeNDAP: {:s}".format(tilestamp, oscar_url))

        velocity_dataset = xr.open_dataset(oscar_url)

        # Choose subset of velocity field we want to use
        nominal_depth = velocity_dataset["depth"].values[0]
        velocity_subdataset = velocity_dataset.sel(depth=nominal_depth)

        gtimes = velocity_subdataset["time"].values
        gtimes = np.array([(gtimes[i] - gtimes[0]) // np.timedelta64(1, "s") for i in range(gtimes.size)])

        glats = velocity_subdataset["latitude"].values
        glons = velocity_subdataset["longitude"].values
        depth = np.array([nominal_depth])

        u_data = velocity_subdataset["u"].values
        v_data = velocity_subdataset["v"].values

        u_field = parcels.field.Field(name="U", data=u_data, time=gtimes, lon=glons, lat=glats, depth=depth, mesh="spherical")
        v_field = parcels.field.Field(name="V", data=v_data, time=gtimes, lon=glons, lat=glats, depth=depth, mesh="spherical")

        fieldset = parcels.fieldset.FieldSet(u_field, v_field)

        pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle, lon=mlons, lat=mlats)

        dump_filename = "particle_locations_p0000" "_tile" + str(tile_id).zfill(2) + ".pickle"
        dump_filepath = os.path.join(self.output_dir, dump_filename)

        logger.info("{:s} Advecting: {:} -> {:}...".format(tilestamp, self.start_time, self.end_time))

        advection_hours = (self.end_time - self.start_time) // dt

        latlon_store = {
            "hours": advection_hours,
            "lat": np.zeros((advection_hours, particles_per_tile)),
            "lon": np.zeros((advection_hours, particles_per_tile))
        }

        tic = time.time()
        for h in range(advection_hours):
            print(h)
            pset.execute(parcels.AdvectionRK4, runtime=dt, dt=dt, verbose_progress=False, output_file=None)

            for i, p in enumerate(pset):
                latlon_store["lon"][h, i] = p.lon
                latlon_store["lat"][h, i] = p.lat

        with open(dump_filepath, "wb") as f:
            logger.info("{:s} Saving intermediate output: {:s}".format(tilestamp, dump_filepath))
            joblib.dump(latlon_store, f, compress=False, protocol=pickle.HIGHEST_PROTOCOL)

        # Create new mlon and mlat lists to create new particle set.
        # n_particles = len(pset)
        # mlons = np.zeros(n_particles)
        # mlats = np.zeros(n_particles)
        # for i, p in enumerate(pset):
        #     mlons[i] = p.lon
        #     mlats[i] = p.lat

        toc = time.time()
        logger.info("{:s} Advection and dumping took {:s}.".format(tilestamp, pretty_time(toc - tic)))

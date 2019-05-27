import os
import re
import pickle
import math
from time import time
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import joblib
import parcels
from numpy import float32, fabs, sqrt, transpose, linspace, repeat, tile, zeros, ones
from numpy.random import uniform

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)

from velocity_fields import oscar_dataset
from utils import most_symmetric_integer_factorization, closest_hour, pretty_time, pretty_filesize


def uniform_particle_locations(N_particles, lat_min, lat_max, lon_min, lon_max):
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


class ParticleAdvecter:
    def __init__(
        self,
        particle_lons,
        particle_lats,
        N_procs=-1,
        velocity_field="OSCAR",
        output_dir=".",
        output_chunk_iters=100,
        Kh=0
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
        logger.info("Particle advection output directory: {:s}".format(output_dir))

        # Create output directory if it doesn't exist.
        if not os.path.exists(output_dir):
            logger.info("Creating directory: {:s}".format(output_dir))
            os.makedirs(output_dir)

        logger.info("Distributing {:d} particles across {:d} processors...".format(N_particles, N_procs))
        particle_lons, particle_lats = distribute_particles_across_tiles(particle_lons, particle_lats, N_procs)

        assert output_chunk_iters >= 1

        self.iteration = 0
        self.particle_lons = particle_lons
        self.particle_lats = particle_lats
        self.velocity_field = velocity_field
        self.N_particles = N_particles
        self.N_procs = N_procs
        self.particles_per_tile = N_particles // N_procs
        self.output_dir = output_dir
        self.output_chunk_iters = output_chunk_iters
        self.Kh = Kh / 1e10  # Converting diffusivity from [m^2/s] -> [deg^2/s] assuming 1 deg = 100 km.

    def time_step(self, start_time, end_time, dt):
        logger.info("Starting time stepping: {:} -> {:} (dt={:}) on {:d} processors."
                    .format(start_time, end_time, dt, self.N_procs))

        if self.N_procs == 1:
            self.time_step_tile(0, start_time, end_time, dt)
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.time_step_tile)(tile_id, start_time, end_time, dt) for tile_id in range(self.N_procs)
            )

        iters = (end_time - start_time) // dt
        self.iteration += iters

    def time_step_tile(self, tile_id, start_time, end_time, dt):
        tilestamp = "[Tile {:02d}]".format(tile_id) if self.N_procs > 1 else ""
        logger = logging.getLogger(__name__ + tilestamp)  # Give each tile/processor its own logger.

        particle_lons, particle_lats = self.particle_lons[tile_id], self.particle_lats[tile_id]
        particles_per_tile = particle_lons.size

        velocity_dataset = oscar_dataset(start_time.year)

        # Choose subset of velocity field we want to use
        nominal_depth = velocity_dataset["depth"].values[0]
        velocity_subdataset = velocity_dataset.sel(depth=nominal_depth)

        grid_times = velocity_subdataset["time"].values
        grid_lats = velocity_subdataset["latitude"].values
        grid_lons = velocity_subdataset["longitude"].values
        grid_depth = np.array([nominal_depth])

        # Convert from an array of numpy datetime64's to an array of ints (seconds since start of dataset).
        grid_times = np.array([(grid_times[i] - grid_times[0]) // np.timedelta64(1, "s")
                               for i in range(grid_times.size)])

        logger.info("{:s} Building parcels grid, fields, and particle set...".format(tilestamp))

        grid = parcels.grid.RectilinearZGrid(grid_lons, grid_lats, depth=grid_depth, time=grid_times, mesh="spherical")

        u_data = velocity_subdataset["u"].values
        v_data = velocity_subdataset["v"].values

        u_field = parcels.field.Field(name="U", data=u_data, grid=grid, interp_method="linear")
        v_field = parcels.field.Field(name="V", data=v_data, grid=grid, interp_method="linear")
        fieldset = parcels.fieldset.FieldSet(u_field, v_field)

        pset = parcels.ParticleSet.from_list(fieldset=fieldset, pclass=parcels.JITParticle,
                                             lon=particle_lons, lat=particle_lats)

        t = start_time
        iteration = self.iteration
        while t < end_time:
            iters_remaining = (end_time - t) // dt
            iters_to_do = min(self.output_chunk_iters, iters_remaining)

            start_iter_str = str(iteration).zfill(5)
            end_iter_str = str(iteration + iters_to_do).zfill(5)

            chunk_start_time = t
            chunk_end_time = t + iters_to_do*dt

            dump_filename = "particle_locations_" + start_iter_str + "_" + end_iter_str + \
                            "_tile" + str(tile_id).zfill(2) + ".pickle"
            dump_filepath = os.path.join(self.output_dir, dump_filename)

            logger.info("{:s} Advecting particles (iteration {:} -> {:}): {:} -> {:}..."
                        .format(tilestamp, start_iter_str, end_iter_str, chunk_start_time, chunk_end_time))

            # Parcels only uses float32 to keep track of particle locations so we don't lose anything by saving lat/lon
            # using float32. The exception is on a C-grid (might be relevant for LLC4320).
            intermediate_output = {
                "time": iters_to_do * [None],
                "lat": zeros((iters_to_do, particles_per_tile), dtype=np.float32),
                "lon": zeros((iters_to_do, particles_per_tile), dtype=np.float32)
            }

            advection_time = 0
            storing_time = 0
            diffusion_time = 0

            for n in range(iters_to_do):
                tic = time()
                pset.execute(parcels.AdvectionRK4,
                             runtime=dt, dt=dt, verbose_progress=False, output_file=None)
                toc = time()

                t = t + dt
                iteration = iteration + 1

                advection_time += toc - tic

                tic = time()
                intermediate_output["time"][n] = t
                for i, p in enumerate(pset):
                    intermediate_output["lon"][n, i] = p.lon
                    intermediate_output["lat"][n, i] = p.lat
                toc = time()
                storing_time += toc - tic

                tic = time()
                for p in pset:
                    p.lat += uniform(-1, 1) * sqrt(6 * fabs(p.dt) * self.Kh)
                    p.lon += uniform(-1, 1) * sqrt(6 * fabs(p.dt) * self.Kh)
                toc = time()
                diffusion_time += toc - tic

            tic = time()
            with open(dump_filepath, "wb") as f:
                logger.info("{:s} Dumping intermediate output: {:s}".format(tilestamp, dump_filepath))
                joblib.dump(intermediate_output, f, compress=("zlib", 3), protocol=pickle.HIGHEST_PROTOCOL)

            toc = time()
            pickling_time = toc - tic
            pickle_filesize = os.path.getsize(dump_filepath)

            logger.info("{:s} Advecting particles:         {:s}.".format(tilestamp, pretty_time(advection_time)))
            logger.info("{:s} Storing intermediate output: {:s}.".format(tilestamp, pretty_time(storing_time)))
            logger.info("{:s} Diffusing particles:         {:s}.".format(tilestamp, pretty_time(diffusion_time)))
            logger.info("{:s} Pickling and compressing:    {:s}. ({:s}, {:s} per particle per iteration)"
                        .format(tilestamp, pretty_time(pickling_time), pretty_filesize(pickle_filesize),
                                pretty_filesize(pickle_filesize / (iters_to_do * particles_per_tile))))

        logger.info("{:s} Saving particle locations...".format(tilestamp))
        for i, p in enumerate(pset):
            self.particle_lons[tile_id][i] = p.lon
            self.particle_lats[tile_id][i] = p.lat

    def create_netcdf_file(self, start_time, end_time, dt):
        logger = logging.getLogger(__name__ + "netcdf")

        iters = (end_time - start_time) // dt
        times = [start_time + n*dt for n in range(iters)]

        plons = zeros((self.N_particles, iters), dtype=float32)
        plats = zeros((self.N_particles, iters), dtype=float32)

        # This dataset will store all the lat/lon positions of each Lagrangian microbe, and will be saved to NetCDF.
        particle_data = xr.Dataset({
                "longitude": (["particle number", "time"], plons),
                "latitude":  (["particle number", "time"], plats)
            },
            coords={
                "particle number": range(1, self.N_particles+1),
                "time": times
            }
        )

        pkl_files = glob(os.path.join(self.output_dir, "particle_locations_*.pickle"))
        pkl_files.sort()

        for pkl_filepath in pkl_files:
            logger.info("Collecting particle locations from {:s}...".format(pkl_filepath))

            filename = os.path.basename(pkl_filepath)
            t1 = int(filename.split("_")[2])  # Starting iteration
            t2 = int(filename.split("_")[3])  # Ending iteration
            tile_id = int(filename.split("_")[4][4:6])

            particle_locations_pkl = joblib.load(pkl_filepath)

            i1 = tile_id * self.particles_per_tile  # Particle starting index
            i2 = (tile_id + 1) * self.particles_per_tile  # Particle ending index

            particle_data["longitude"][i1:i2, t1:t2] = transpose(particle_locations_pkl["lon"])
            particle_data["latitude"][i1:i2, t1:t2] = transpose(particle_locations_pkl["lat"])

        nc_filepath = os.path.join(self.output_dir, "particle_data.nc")
        logger.info("Writing particle locations to {:s}...".format(nc_filepath))
        particle_data.to_netcdf(nc_filepath)

        for pkl_filepath in pkl_files:
            logger.debug("Deleting {:s}...".format(pkl_filepath))
            os.remove(pkl_filepath)

import os
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import joblib

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig("logging.ini")
logger = logging.getLogger(__name__)


class ParticleAdvecter:
    def __init__(
        self,
        velocity_field="OSCAR",
        particle_initial_distribution="uniform",
        start_time=datetime(2017, 1, 1),
        end_time=datetime(2018, 1, 1),
        dt=timedelta(hours=1),
        N_particles=1,
        N_procs=-1,
        Tx=-1,
        Ty=-1,
        lat_min_particle=10,
        lat_max_particle=50,
        lon_min_particle=-170,
        lon_max_particle=-130,
        delta_lat_particle=1,
        delta_lon_particle=1,
        oscar_dataset_dir=".",
        domain_lats=slice(60, 0),
        domain_lons=slice(-180, -120),
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
        assert 1 <= N_particles, "N_particles must be a positive integer."
        logger.info("Number of Lagrangian particles: {:d}".format(N_particles))

        # Sanitize Lagrangian particle spacing inputs.
        assert 0 < delta_lat_particle, "delta_lat_particle must be a positive number."
        assert 0 < delta_lon_particle, "delta_lon_particle must be a positive number."
        logger.info("Lagrangian particle spacing: delta_lat={:.3f}°, delta_lon={:.3f}°"
                    .format(delta_lat_particle, delta_lon_particle))

        # Figure out the number of parallelization tiles to use.
        assert 1 <= Tx or Tx == -1, "Number of tiles Tx must be a positive integer or -1 (choose automatically)."
        assert 1 <= Ty or Ty == -1, "Number of tiles Ty must be a positive integer or -1 (choose automatically)."

        # If -1 is given then we automatically choose the number of tiles. To avoid having to find a nice factorization,
        # we just choose N_procs tiles in the x direction.
        if Tx == -1 and Ty == -1:
            Tx, Ty = N_procs, 1

        # Sanitize Tx, Ty input.
        assert Tx*Ty == N_procs, "Total number of tiles Tx*Ty must equal N_procs."

        if Tx*Ty == 1:
            particles_per_tile = N_particles  # for the case of 1 tile.
        elif Tx*Ty > 1:
            # If we're using multiple tiles, print some tile information.
            logger.info("Number of tiles: Tx={:d}, Ty={:d}, total={:d}".format(Tx, Ty, Tx*Ty))
            logger.info("Lagrangian particles per tile: {:}".format(N_particles / (Tx*Ty)))

            # Number of integer particles in each tile.
            particles_per_tile = N_particles // (Tx * Ty)

            # If we cannot have an integer number of particles per tile, round down and use that many per tile.
            if particles_per_tile*Tx*Ty != N_particles:
                logger.warning("N_particles does not factor across {:d} tiles. "
                               "Using {:d} Lagrangian particles per tile.".format(Tx*Ty, particles_per_tile))

                N_particles = particles_per_tile*Tx*Ty
                logger.warning("N_particles has been set to {:d}.".format(N_particles))
        else:
            raise ValueError("Number of tiles Tx*Ty must be a positive integer.")

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

        logger.info("Generating locations for {:d} Lagrangian particles across {:d} tiles..."
                    .format(N_particles, Tx*Ty))

        # Create a storage space for particle locations on each tile.
        particle_lons = Tx * Ty * [None]
        particle_lats = Tx * Ty * [None]

        # The (lat, lon) spacing between each tile.
        delta_lon_tile = (lon_max_particle - lon_min_particle) / Tx
        delta_lat_tile = (lat_max_particle - lat_min_particle) / Ty
        logger.debug("Tile spacing: delta_lat_tile={:.2f}, delta_lon_tile={:.2f}°"
                     .format(delta_lat_tile, delta_lon_tile))

        # Number of uniformly spaced particles in each direction per tile.
        NTx = particles_per_tile // Tx
        NTy = particles_per_tile // Ty

        for i, j in product(range(Tx), range(Ty)):
            # Calculate the (lat, lon) bounding box for each tile.
            lon_min_tile = lon_min_particle + i*delta_lon_tile
            lon_max_tile = lon_min_particle + (i+1)*delta_lon_tile
            lat_min_tile = lat_min_particle + j*delta_lat_tile
            lat_max_tile = lat_min_particle + (j+1)*delta_lat_tile

            logger.debug("Tile (i,j,i*Ty+j) = ({:d},{:d},{:d})".format(i, j, i * Ty + j))
            logger.debug("Tile (Tx={:d}, Ty={:d}): {:.2f}-{:.2f} °E, {:.2f}-{:.2f} °N"
                         .format(i, j, lon_min_tile, lon_max_tile, lat_min_tile, lat_max_tile))

            # Generate NTx*NTy uniformly spaced (lat, lon) locations in the tile, and store them.
            particle_lons[i*Ty + j] = np.repeat(np.linspace(lon_min_tile, lon_max_tile - delta_lon_tile, NTx), NTy)
            particle_lats[i*Ty + j] = np.tile(np.linspace(lat_min_tile, lat_max_tile - delta_lat_tile, NTy), NTx)

        self.velocity_field = velocity_field
        self.particle_initial_distribution = particle_initial_distribution
        self.N_procs = N_procs
        self.N_particles = N_particles
        self.delta_lat_particle = delta_lat_particle
        self.delta_lon_particle = delta_lon_particle
        self.Tx = Tx
        self.Ty = Ty
        self.output_dir = output_dir
        self.oscar_dataset_dir = oscar_dataset_dir
        self.particle_lons = particle_lons
        self.particle_lats = particle_lats

    def oscar_dataset_filepath(self):
        fp = os.path.join(self.oscar_dataset_dir, "oscar_vel" + str(self.time.year) + "_180.nc")
        assert os.path.isfile(fp), "OSCAR dataset {:s} could not be found (or is not a file).".format(fp)
        return fp

    def time_step(self, Nt, dt):
        if self.N_procs == 1:
            pass
        else:
            joblib.Parallel(n_jobs=self.N_procs)(
                joblib.delayed(self.time_step_tile)(tile_id, self.lons[tile_id], self.lats[tile_id])
                for tile_id in range(self.Tx * self.Ty)
            )

    def time_step_tile(self, Nt, dt):
        pass

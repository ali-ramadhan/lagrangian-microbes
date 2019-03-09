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
        start_time=datetime(2017, 1, 1),
        end_time=datetime(2018, 1, 1),
        dt = timedelta(hours=1),
        domain_lats=slice(60, 0),
        domain_lons=slice(-180, -120),
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
        output_dir="."
    ):
        assert 1 <= N_procs or N_procs == -1, "N_procs must be a positive integer or -1 (use all processors)."
        max_procs = joblib.cpu_count()
        self.N_procs = N_procs if 1 <= N_procs <= max_procs else max_procs
        logger.info("Requested {:d} processors. Found {:d}. Using {:d}.".format(N_procs, max_procs, self.N_procs))

        assert 1 <= N_particles, "N_particles must be a positive integer."
        self.N_particles = N_particles
        logger.info("Number of Lagrangian particles: {:d}".format(N_particles))

        assert 0 < delta_lat_particle, "delta_lat_particle must be a positive number."
        assert 0 < delta_lon_particle, "delta_lon_particle must be a positive number."
        self.delta_lat_particle, self.delta_lon_particle = delta_lat_particle, delta_lon_particle
        logger.info("Lagrangian particle spacing: delta_lat={:.3f}°, delta_lon={:.3f}°"
                    .format(delta_lat_particle, delta_lon_particle))

        # Automatically choose number of tiles. To avoid having to find a nice factorization, just choose N_procs tiles
        # in the x direction.
        if Tx == -1 and Ty == -1:
            Tx, Ty = N_procs, 1

        assert Tx*Ty == N_procs, "Total number of tiles Tx*Ty must equal N_procs."
        if Tx*Ty > 1:
            logger.info("Number of tiles: Tx={:d}, Ty={:d}".format(Tx, Ty))
            logger.info("Lagrangian particles per tile: {:d}".format(N_particles//(Tx*Ty)))

        NTx = N_particles // Tx
        NTy = N_particles // Ty

        logger.info("Particle advection output directory: {:s}".format(os.path.abspath(output_dir)))
        if not os.path.exists(output_dir):
            logger.info("Creating directory: {:s}".format(output_dir))
            os.makedirs(output_dir)

        mlon_blocks = Tx * Ty * [None]
        mlat_blocks = Tx * Ty * [None]

        delta_lon_tile = (lon_max_particle - lon_min_particle) / Tx
        delta_lat_tile = (lat_max_particle - lat_min_particle) / Ty

        for i, j in product(range(Tx), range(Ty)):
            lon_min_tile = lon_min_particle + i*delta_lon_tile
            lon_max_tile = lon_min_particle + (i+1)*delta_lon_tile
            lat_min_tile = lat_min_particle + j*delta_lat_tile
            lat_max_tile = lat_min_particle + (j+1)*delta_lat_tile

            logger.debug("(Tx={:d}, Ty={:d}) {:.2f}-{:.2f} E, {:.2f}-{:.2f} N"
                         .format(i, j, lon_min_tile, lon_max_tile, lat_min_tile, lat_max_tile))
            logger.debug("(i,j,i*Ty+j) = ({:d},{:d},{:d})".format(i, j, i*Ty+j))

            mlon_blocks[i*Ty + j] = np.repeat(np.linspace(lon_min_tile, lon_max_tile - delta_lon_tile, NTx), NTy)
            mlat_blocks[i*Ty + j] = np.tile(np.linspace(lat_min_tile, lat_max_tile - delta_lat_tile, NTy), NTx)

    def time_step(self, Nt, dt):
        pass
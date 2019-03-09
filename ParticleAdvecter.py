import joblib

# Configure logger first before importing any sub-module that depend on the logger being already configured.
import logging.config

logging.config.fileConfig('logging.ini')
logger = logging.getLogger(__name__)

class ParticleAdvecter:
    def __init__(
        self,
        N_procs=-1,
        N_particles=1,
        delta_lat_particle=1,
        delta_lon_particle=1
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
        logger.info("Lagrangian particle spacing: delta_lat={:.3f}°, delta_lon={:.3f}°".format(delta_lat_particle, delta_lon_particle))


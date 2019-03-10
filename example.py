from datetime import datetime, timedelta

from particle_advecter import ParticleAdvecter, uniform_particle_locations

# Generate initial locations for each particle.
particle_lons, particle_lats = uniform_particle_locations(N_particles=1000, lat_min=25, lat_max=35, lon_min=205, lon_max=215)

p = ParticleAdvecter(particle_lons, particle_lats, N_procs=4, output_dir="/home/gridsan/aramadhan/microbes_output/")
p.time_step(start_time=datetime(2017, 1, 1), end_time=datetime(2017, 2, 1), dt=timedelta(hours=1))



import ParticleAdvecter

p = ParticleAdvecter.ParticleAdvecter(N_procs=4, N_particles=1000,
                                      output_dir="/home/gridsan/aramadhan/microbes_output/")
p.time_step()

class MicrobeParticle(parcels.JITParticle):
    species = parcels.Variable('species', dtype=np.int32, initial=-1)

def rock_paper_scissors_type(n):
    if n == 1:
        return "rock"
    elif n == 2:
        return "paper"
    elif n == 3:
        return "scissors"
    return None

for i, particle in enumerate(pset):
    if 37.5 <= particle.lat <= 52.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 1
    elif 37.5 <= particle.lat <= 52.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 2
    elif 37.5 <= particle.lat <= 52.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 3
    elif 22.5 <= particle.lat <= 37.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 3
    elif 22.5 <= particle.lat <= 37.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 1
    elif 22.5 <= particle.lat <= 37.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 2
    elif 7.5 <= particle.lat <= 22.5 and -172.5 <= particle.lon <= -157.5:
        particle.species = 2
    elif 7.5 <= particle.lat <= 22.5 and -157.5 <= particle.lon <= -142.5:
        particle.species = 3
    elif 7.5 <= particle.lat <= 22.5 and -142.5 <= particle.lon <= -127.5:
        particle.species = 1
    # print("Particle {:03d} @({:.2f},{:.2f}) [species={:d}]".format(i, particle.lat, particle.lon, particle.species))

    print("Computing microbe interactions...", end="")

    t1 = time.time()
    
    particle_locations = np.zeros([N, 2])
    for i, p in enumerate(pset):
        particle_locations[i, :] = [p.lon, p.lat]

    kd = KDTree(np.array(particle_locations))
    interacting_pairs = kd.query_ball_tree(kd, r=1, p=1)
    print("interacting_pairs: {:}".format(interacting_pairs))
    
    t2 = time.time()
    print(" ({:g} s)".format(t2 - t1))
from subprocess import Popen

MODULE_CMD = "module add engaging/ffmpeg/20160616 engaging/x264/20160616"

CD_CMD = "cd ~/lagrangian-microbes/"
ENV_CMD = "source activate lagrangian_microbes"

OUTPUT_DIR = "~/cnhlab004/lagrangian_microbes_output/"


def rps_advect_cmd(C, N, Kh, dir):
    return "python rock_paper_scissors_advect.py" \
           + " -C " + str(C)  \
           + " -N " + str(N)  \
           + " -K " + str(Kh) \
           + " -d " + str(dir)


def rps_interact_cmd(N, Kh, p, a, r, dir):
    return "python rock_paper_scissors_interact.py" \
           + " -N " + str(N)  \
           + " -K " + str(Kh) \
           + " -p " + str(p)  \
           + " -a " + str(a)  \
           + " -r " + str(r)  \
           + " -d " + str(dir)


def rps_analyze_cmd(N, Kh, p, a, r, dir):
    return "python rock_paper_scissors_analyze.py" \
           + " -N " + str(N)  \
           + " -K " + str(Kh) \
           + " -p " + str(p)  \
           + " -a " + str(a)  \
           + " -r " + str(r)  \
           + " -d " + str(dir)


def rps_movie_cmd(N, Kh, p, a, r, dir):
    return "python rock_paper_scissors_movie.py" \
           + " -N " + str(N)  \
           + " -K " + str(Kh) \
           + " -p " + str(p)  \
           + " -a " + str(a)  \
           + " -r " + str(r)  \
           + " -d " + str(dir)


def ensemble_advection():
    hours = {
        10000: 1,
        100000: 3,
        1000000: 12
    }

    def slurm_script_lines(C, N, Kh, hours):
        lines = \
            ["#!/bin/bash",
             "#",
             "#SBATCH --job-name=advect_microbes_N{:d}_Kh{:d}".format(N, Kh),
             "#SBATCH --output=advect_microbes_N{:d}_Kh{:d}_%j.log".format(N, Kh),
             "#SBATCH --mail-type=ALL",
             "#SBATCH --mail-user=alir@mit.edu",
             "#SBATCH --partition=sched_mit_darwin2",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=28",
             "#SBATCH --time={:d}:00:00".format(hours),
             "#SBATCH --mem=100gb",
             "",
             CD_CMD,
             ENV_CMD,
             rps_advect_cmd(C, N, Kh, OUTPUT_DIR)]

        return lines

    C = 25
    for N in [10000, 100000, 1000000]:
        for Kh in [0, 20, 100, 500]:
            slurm_script_filename = "advect_rps_N{:d}_Kh{:d}.slurm".format(N, Kh)

            with open(slurm_script_filename, "w") as f:
                script_lines = slurm_script_lines(C, N, Kh, hours[N])
                f.writelines("{:s}\n".format(l) for l in script_lines)

            Popen("sbatch {:s}".format(slurm_script_filename), shell=True)

def ensemble_interaction():
    hours = {
        10000: 2,
        100000: 12,
        1000000: 12
    }

    interaction_lengthscale = {
        10000: 0.05,    # ~5 km
        100000: 0.0015, # ~1.5 km
        1000000: 0.005  # ~500 m
    }

    def slurm_script_lines(N, Kh, p, a, r, hours):
        lines = \
            ["#!/bin/bash",
             "#",
             "#SBATCH --job-name=interact_microbes_N{:d}_Kh{:d}_p{:}_a{:}".format(N, Kh, p, a),
             "#SBATCH --output=interact_microbes_N{:d}_Kh{:d}_p{:}_a{:}_%j.log".format(N, Kh, p, a),
             "#SBATCH --mail-type=ALL",
             "#SBATCH --mail-user=alir@mit.edu",
             "#SBATCH --partition=sched_mit_darwin2",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=1",
             "#SBATCH --time={:d}:00:00".format(hours),
             "#SBATCH --mem=20gb",
             "",
             CD_CMD,
             ENV_CMD,
             rps_interact_cmd(N, Kh, p, a, r, OUTPUT_DIR)]

        return lines

    for N in [100000]:
        for Kh in [0, 20, 100, 500]:
            for p in [0.5, 0.7]:
                for a in [0, 0.1, 0.01, 0.001]:
                    r = interaction_lengthscale[N]

                    slurm_script_filename = "advect_rps_N{:d}_Kh{:d}_p{:}_a{:}.slurm".format(N, Kh, p, a)

                    with open(slurm_script_filename, "w") as f:
                        script_lines = slurm_script_lines(N, Kh, p, a, r, hours[N])
                        f.writelines("{:s}\n".format(l) for l in script_lines)

                    Popen("sbatch {:s}".format(slurm_script_filename), shell=True)

def ensemble_analysis():
    minutes = {
        10000: 20,
        100000: 120,
        1000000: 360
    }

    interaction_lengthscale = {
        10000: 0.05,    # ~5 km
        100000: 0.0015, # ~1.5 km
        1000000: 0.005  # ~500 m
    }

    def slurm_script_lines(N, Kh, p, a, r, mins):
        lines = \
            ["#!/bin/bash",
             "#",
             "#SBATCH --job-name=analyze_microbes_N{:d}_Kh{:d}_p{:}_a{:}".format(N, Kh, p, a),
             "#SBATCH --output=analyze_microbes_N{:d}_Kh{:d}_p{:}_a{:}_%j.log".format(N, Kh, p, a),
             "#SBATCH --mail-type=ALL",
             "#SBATCH --mail-user=alir@mit.edu",
             "#SBATCH --partition=sched_mit_darwin2",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=1",
             "#SBATCH --time={:d}:00".format(mins),
             "#SBATCH --mem=10gb",
             "",
             CD_CMD,
             ENV_CMD,
             rps_analyze_cmd(N, Kh, p, a, r, OUTPUT_DIR)]

        return lines

    for N in [10000]:
        for Kh in [0, 20, 100, 500]:
            for p in [0.5, 0.7]:
                for a in [0, 0.1, 0.01, 0.001]:
                    r = interaction_lengthscale[N]

                    slurm_script_filename = "analyze_rps_N{:d}_Kh{:d}_p{:}_a{:}.slurm".format(N, Kh, p, a)

                    with open(slurm_script_filename, "w") as f:
                        script_lines = slurm_script_lines(N, Kh, p, a, r, minutes[N])
                        f.writelines("{:s}\n".format(l) for l in script_lines)

                    Popen("sbatch {:s}".format(slurm_script_filename), shell=True)

def ensemble_movies():
    interaction_lengthscale = {
        10000: 0.05,    # ~5 km
        100000: 0.0015, # ~1.5 km
        1000000: 0.005  # ~500 m
    }

    def slurm_script_lines(N, Kh, p, a, r):
        lines = \
            ["#!/bin/bash",
             "#",
             "#SBATCH --job-name=movie_microbes_N{:d}_Kh{:d}_p{:}_a{:}".format(N, Kh, p, a),
             "#SBATCH --output=movie_microbes_N{:d}_Kh{:d}_p{:}_a{:}_%j.log".format(N, Kh, p, a),
             "#SBATCH --mail-type=ALL",
             "#SBATCH --mail-user=alir@mit.edu",
             "#SBATCH --partition=sched_mit_darwin2",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             "#SBATCH --cpus-per-task=28",
             "#SBATCH --time=3:00:00",
             "#SBATCH --mem=100gb",
             "",
             MODULE_CMD,
             CD_CMD,
             ENV_CMD,
             rps_movie_cmd(N, Kh, p, a, r, OUTPUT_DIR)]

        return lines

    for N in [10000]:
        for Kh in [0, 20, 100, 500]:
            for p in [0.5, 0.7]:
                for a in [0, 0.1, 0.01, 0.001]:
                    r = interaction_lengthscale[N]

                    slurm_script_filename = "movie_rps_N{:d}_Kh{:d}_p{:}_a{:}.slurm".format(N, Kh, p, a)

                    with open(slurm_script_filename, "w") as f:
                        script_lines = slurm_script_lines(N, Kh, p, a, r)
                        f.writelines("{:s}\n".format(l) for l in script_lines)

                    Popen("sbatch {:s}".format(slurm_script_filename), shell=True)


if __name__ == "__main__":
    ensemble_movies()


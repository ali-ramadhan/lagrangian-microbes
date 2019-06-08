from subprocess import Popen

MODULE_CMD = "module add engaging/ffmpeg/20160616 engaging/x264/20160616"

CD_CMD = "cd ~/lagrangian-microbes/"
ENV_CMD = "source activate lagrangian_microbes"

OUTPUT_DIR = "~/cnhlab004/lagrangian_microbes_output/"


def rps_advect_cmd(C, N, Kh, dir):
    return "python rock_paper_scissors_advect.py" \
           + " -C " + str(C) \
           + " -N " + str(N) \
           + " -K " + str(Kh) \
           + " -d " + str(dir)


def slurm_script_lines(C, N, Kh, hours):
    lines = \
        ["#!/bin/bash",
         "#",
         "#SBATCH --job-name=advect_microbes_N{:d}_Kh{:d}".format(N, Kh),
         "#SBATCH --output=lagrangian_microbes_N{:d}_Kh{:d}_%j.log".format(N, Kh),
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


if __name__ == "__main__":
    hours = {
        10000: 1,
        100000: 3,
        1000000: 12
    }

    C = 25
    for N in [1000000]:
        for Kh in [0, 20, 100, 500]:
            slurm_script_filename = "advect_rps_N{:d}_Kh{:d}.slurm".format(N, Kh)

            with open(slurm_script_filename, "w") as f:
                script_lines = slurm_script_lines(C, N, Kh, hours[N])
                f.writelines("{:s}\n".format(l) for l in script_lines)

            p = Popen("sbatch {:s}".format(slurm_script_filename), shell=True)

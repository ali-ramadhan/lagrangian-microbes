import logging

from subprocess import Popen

logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(funcName)s:%(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

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


if __name__ == "__main__":
    C = 25
    N = 10000
    Kh = 33

    script_lines = \
        ["#!/bin/bash",
         "#SBATCH --job-name=advect_microbes_N{:d}_Kh{:d}".format(N, Kh),
         "#SBATCH --output=lagrangian_microbes_N{:d}_Kh{:d}_%j.log".format(N, Kh),
         "#SBATCH --mail-type=ALL",
         "#SBATCH --mail-user=alir@mit.edu",
         "#SBATCH --partition=sched_mit_darwin2",
         "#SBATCH --nodes=1"
         "#SBATCH --ntasks=1"
         "#SBATCH --cpus-per-task=28"
         "#SBATCH --time=60:00",
         "#SBATCH --mem=100gb",
         CD_CMD,
         ENV_CMD,
         rps_advect_cmd(C, N, Kh, OUTPUT_DIR)]

    slurm_script_filename = "advect_rps_N{:d}_Kh{:d}.slurm".format(N, Kh)
    with open(slurm_script_filename) as f:
        f.writelines("{:s}\n".format(l) for l in script_lines)

    p = Popen("sbatch {:s}".format(slurm_script_filename), shell=True)

#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=recirc
#SBATCH --output=output_r%j.out
#SBATCH --error=output_r%j.err
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12g
#SBATCH --time=36:00:00
#--exclusive


module load mpi/openmpi-x86_64



mpirun -n 128 python -u dfl_cbxfel_new_config.py

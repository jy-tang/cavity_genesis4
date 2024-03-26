#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=merge
#SBATCH --output=output_m%j.out
#SBATCH --error=output_m%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30g
#SBATCH --time=36:00:00
#--exclusive


module load mpi/openmpi-x86_64


mpirun -n 1 python -u merge_files_mpi.py


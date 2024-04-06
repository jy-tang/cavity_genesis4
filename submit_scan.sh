#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=scan
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2g
#SBATCH --time=01:30:00
#--exclude=sdfmilan[023-026.120]
#--exclusive

module load mpi/openmpi-x86_64
export PYTHONPATH=$PYTHONPATH:/sdf/group/ad/beamphysics/jytang/cavity_genesis4_20keV/cavity_genesis4

python -u  cavity_genesis4/genesis4/scan_param_20keV.py


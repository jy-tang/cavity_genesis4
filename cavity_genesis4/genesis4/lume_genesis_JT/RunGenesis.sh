#!/bin/bash
#SBATCH --account=ad:beamphysics
#SBATCH --partition=milano
#SBATCH --job-name=Genesis
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=120
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4g
#SBATCH --time=00:30:00
##SBATCH --exclude=sdfmilan[108]
##SBATCH --nodes=2
#--exclusive


module load mpi/openmpi-x86_64

#mpirun -n 120 /sdf/group/ad/beamphysics/software/genesis4/milano_openmpi/genesis4/build/genesis4 $1
mpirun -n 120 genesis4 $1

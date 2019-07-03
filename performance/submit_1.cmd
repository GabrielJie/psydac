#!/bin/bash

#SBATCH -D ./

#SBATCH -J bench

#SBATCH --partition=express

#SBATCH --time=00:30:00

### increase the number of MPI processes by adding more nodes

#SBATCH --nodes=30

### launch one MPI process per physical core

#SBATCH --ntasks-per-node=40

#SBATCH --mail-type=none
#SBATCH --mail-user=said.hadjout@ipp.mpg.de

### request fat memory nodes, if necessary

#SBATCH --mem=185000

module unload intel
module unload gcc/7
module load gcc/9 impi
module load anaconda/3
module load mpi4py

srun -N 30 -n 1024 python3 test_perf_3d_parallel.py > p3d.out
## bash test_weak_scaling.sh   > weak_scaling.ou

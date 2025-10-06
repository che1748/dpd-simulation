#!/bin/bash

#SBATCH --time=01:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "duo_dpd.py"   # job name




# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load python/3.9.18
module load numpy
python3 duo_dpd.py

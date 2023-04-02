#! /bin/bash -l

#SBATCH -J gef_p864
#SBATCH -o gef_p864.out
#SBATCH -e gef_p864.error
#SBATCH -N 11  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 864  #how many cores
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH --array=1-7

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

mpirun -np 864 python3 ./main_gef.py config/procs864/case6_cwy1s.ini

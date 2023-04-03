#! /bin/bash -l

#SBATCH -J gef_p4704
#SBATCH -o gef_p4704.out
#SBATCH -e gef_p4704.error
#SBATCH -N 59  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 4704  #how many cores
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH --array=1-7

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

mpirun -np 4704 python3 ./main_gef.py config/procs4704/case6_cwyne.ini

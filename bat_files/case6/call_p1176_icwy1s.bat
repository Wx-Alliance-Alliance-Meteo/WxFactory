#! /bin/bash -l

#SBATCH -J gef_p1176
#SBATCH -o gef_p1176.out
#SBATCH -e gef_p1176.error
#SBATCH -N 8  #how many cluster nodes #STEPHANE HAS TO CHANGE
#SBATCH -n 1176  #how many cores
#SBATCH --exclusive
#SBATCH -p short
#SBATCH --constraint=ib
#SBATCH -t 03:00:00
#SBATCH --array=1-7

conda activate gef3

#srun --mpi=pmi2 --cpu-bind=cores python3 ./main_gef.py config/case6.ini

mpirun -np 1176 python3 ./main_gef.py config/procs1176/case6_icwy1s.ini
